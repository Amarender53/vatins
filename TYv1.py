# Cybersecurity Intelligence Dashboard from Telegram and YouTube Data
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from transformers import pipeline
from langdetect import detect
from deep_translator import GoogleTranslator
import networkx as nx
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
tg_db = client['telegram_data']
yt_db = client['youtube_data']

# Load NLP models
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
ner_model = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl", grouped_entities=True)

# Helper Functions
@st.cache_data
def translate_text(text):
    try:
        if detect(text) != 'en':
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except:
        return text

@st.cache_data
def get_topics(texts, n_topics=5):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=1)
    W = nmf.fit_transform(X)
    H = nmf.components_
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic in H:
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        topics.append(", ".join(top_words))
    return topics

# UI Setup
st.set_page_config(layout="wide")
st.title("🔒 Cybersecurity Intelligence Dashboard")
data_source = st.radio("Select Data Source:", ["Telegram", "YouTube"])

# Load Data Based on Source
if data_source == "Telegram":
    raw_data = list(tg_db['messages'].find({"text": {"$exists": True, "$ne": None}}))
elif data_source == "YouTube":
    raw_data = list(yt_db['comments'].find({"text": {"$exists": True, "$ne": None}}))
else:
    st.error("⚠️ Invalid data source selected.")
    st.stop()

if not raw_data:
    st.error("❌ No valid 'text' field found in the selected data source.")
    st.stop()

data_df = pd.DataFrame(raw_data)
data_df["text"] = data_df["text"].astype(str)

# Limit for performance
texts = data_df["text"].tolist()[:500]

# Translation & NLP
st.info("🔄 Translating and analyzing text... This may take a few seconds.")
translated = [translate_text(t) for t in texts]
sentiments = [sentiment_model(t[:512])[0]["label"] for t in translated]
entities = [ner_model(t[:512]) for t in translated]

# Sentiment Visualization
st.subheader("📊 Sentiment Distribution")
sentiment_df = pd.Series(sentiments).value_counts()
st.bar_chart(sentiment_df)

# Named Entity Recognition
st.subheader("☑️ Named Entities")
ner_flat = [(e['word'], e['entity_group']) for sublist in entities for e in sublist]
ner_df = pd.DataFrame(ner_flat, columns=["Entity", "Type"])
st.dataframe(ner_df.value_counts().reset_index(name="Count"))

# Topic Detection
st.subheader("⚠️ Detected Topics")
topics = get_topics(translated)
for i, topic in enumerate(topics, 1):
    st.markdown(f"**Topic {i}:** {topic}")

# Word Cloud
st.subheader("🌍 Word Cloud")
all_text = " ".join(translated)
wc = WordCloud(width=800, height=300, background_color='white').generate(all_text)
fig, ax = plt.subplots()
ax.imshow(wc, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# Network Graph
st.subheader("🧡 Network Graph (Entity Co-occurrence)")
G = nx.Graph()
for ents in entities:
    words = [e['word'] for e in ents if e['entity_group'] in ["ORG", "PER", "LOC"]]
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            G.add_edge(words[i], words[j])
plt.figure(figsize=(8, 6))
nx.draw_networkx(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=1500, font_size=10)
st.pyplot(plt)

# PDF Export
st.subheader("📄 Export Intelligence Report")
def generate_pdf():
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Cyber Intelligence Report", styles['Title']),
        Spacer(1, 12),
        Paragraph(f"Data Source: {data_source}", styles['Normal']),
        Paragraph(f"Total Records Processed: {len(texts)}", styles['Normal']),
        Spacer(1, 12),
        Paragraph("Detected Topics:", styles['Heading2'])
    ]
    for topic in topics:
        elements.append(Paragraph(f"- {topic}", styles['Normal']))
    doc.build(elements)
    buf.seek(0)
    return buf

pdf_data = generate_pdf()
st.download_button("📃 Download PDF Report", data=pdf_data, file_name="cyber_report.pdf", mime="application/pdf")
