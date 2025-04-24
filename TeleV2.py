import streamlit as st
import pandas as pd
from pymongo import MongoClient
from wordcloud import WordCloud
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sentence_transformers import SentenceTransformer, util
from fpdf import FPDF
from io import BytesIO
import numpy as np

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["telegram_data"]
users_col = db["users"]
messages_col = db["messages"]
groups_col = db["groups"]

# AI Pipelines
sentiment_model = pipeline("sentiment-analysis")
ner_model = pipeline("ner", grouped_entities=True)
summarizer = pipeline("summarization")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit Setup
st.set_page_config(layout="wide")
st.title("🧠 Telegram AI Intelligence Dashboard")

# Select User
users = list(users_col.find({}, {"user_id": 1, "first_name": 1, "last_name": 1, "username": 1}))
user_map = {
    f"{u.get('first_name', '')} {u.get('last_name', '')} (@{u.get('username', 'N/A')})": u["user_id"]
    for u in users
}
selected_user = st.selectbox("Select a User", list(user_map.keys()))
user_id = user_map[selected_user]

# Get User Messages
user = users_col.find_one({"user_id": user_id})
messages = list(messages_col.find({"user_id": user_id}))
texts = [msg.get("text", "") for msg in messages if isinstance(msg.get("text", ""), str)]

# Sentiment Analysis
st.subheader("📊 Sentiment Analysis")
sentiments = [sentiment_model(t[:512])[0]['label'] for t in texts]
sentiment_df = pd.Series(sentiments).value_counts()
st.bar_chart(sentiment_df)

# Named Entity Recognition
st.subheader("🧠 Named Entities")
ner_results = []
for t in texts:
    try:
        ner_results.extend([(ent['word'], ent['entity_group']) for ent in ner_model(t[:512])])
    except:
        continue
ner_df = pd.DataFrame(ner_results, columns=["Entity", "Type"])
st.dataframe(ner_df.value_counts().reset_index(name="Count"))

# Topic Detection
st.subheader("📌 Topic Detection")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(texts)
model = NMF(n_components=5)
W = model.fit_transform(X)
H = model.components_
topics = []
for topic in H:
    words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-6:-1]]
    topics.append(", ".join(words))
for i, topic in enumerate(topics):
    st.write(f"**Topic {i+1}:** {topic}")

# Word Cloud
st.subheader("☁️ Word Cloud")
wc = WordCloud(width=800, height=300).generate(" ".join(texts))
st.image(wc.to_array(), use_column_width=True)

# Semantic Search
st.subheader("🔍 Semantic Search")
query = st.text_input("Search in user messages")
if query:
    corpus_embeddings = semantic_model.encode(texts, convert_to_tensor=True)
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_idx = np.argsort(scores.cpu().numpy())[::-1][:5]
    for i in top_idx:
        st.markdown(f"- {texts[i]}")

# Summary
st.subheader("📝 LLM Summary")
all_text = " ".join(texts)
summary = ""
for i in range(0, len(all_text), 2000):
    chunk = all_text[i:i+2000]
    try:
        result = summarizer(chunk[:1000], max_length=130, min_length=30, do_sample=False)
        summary += result[0]['summary_text'] + "\n"
    except:
        continue
st.text_area("Summary", summary, height=200)

# PDF Generation
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Telegram AI Summary Report", ln=True, align='C')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font("Arial", size=11)
        self.multi_cell(0, 8, body)
        self.ln()

def generate_pdf(title, info, messages, summary):
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_body(title)
    for line in info:
        pdf.chapter_body(line)
    pdf.chapter_body("\nSummary:\n" + summary)
    pdf.chapter_body("\nRecent Messages:")
    for m in messages[:10]:
        pdf.chapter_body(f"{m.get('date')} - {m.get('text', '')[:100]}")
    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

user_info = [
    f"Name: {user.get('first_name', '')} {user.get('last_name', '')}",
    f"Username: @{user.get('username', 'N/A')}",
    f"Phone: {user.get('phone', 'N/A')}",
    f"Total Messages: {len(messages)}",
    f"Top Sentiments: {', '.join(sentiment_df.index[:3])}"
]

if st.button("📥 Download PDF Report"):
    pdf_data = generate_pdf("User Profile Report", user_info, messages, summary)
    st.download_button("Download PDF", pdf_data, file_name="user_summary.pdf", mime="application/pdf")

# Extension to Groups
st.header("👥 Group Analysis")
groups = list(groups_col.find({}, {"chat_id": 1, "title": 1}))
group_map = {f"{g['title']} ({g['chat_id']})": g["chat_id"] for g in groups}
selected_group = st.selectbox("Select a Group", list(group_map.keys()))
group_id = group_map[selected_group]
group_messages = list(messages_col.find({"chat_id": group_id}))
group_texts = [msg.get("text", "") for msg in group_messages if isinstance(msg.get("text", ""), str)]

st.markdown(f"**Group ID:** {group_id}")
st.markdown(f"**Total Messages:** {len(group_messages)}")

if group_messages:
    group_summary = ""
    for i in range(0, len(" ".join(group_texts)), 2000):
        chunk = " ".join(group_texts)[i:i+2000]
        try:
            result = summarizer(chunk[:1000], max_length=130, min_length=30, do_sample=False)
            group_summary += result[0]['summary_text'] + "\n"
        except:
            continue
    st.text_area("Group Summary", group_summary, height=200)

    if st.button("📥 Download Group PDF"):
        group_info = [f"Group: {selected_group}", f"Total Messages: {len(group_messages)}"]
        pdf_data = generate_pdf("Group Summary Report", group_info, group_messages, group_summary)
        st.download_button("Download Group PDF", pdf_data, file_name="group_summary.pdf", mime="application/pdf")
