import streamlit as st
from pymongo import MongoClient
import pandas as pd
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from deep_translator import GoogleTranslator
from langdetect import detect
from transformers import pipeline
import io
from fpdf import FPDF

# Load models once
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
ner_pipeline = pipeline("ner", grouped_entities=True, model="Davlan/bert-base-multilingual-cased-ner-hrl")
nlp = spacy.load("en_core_web_sm")

# MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["youtube_data"]
channels_col = db["channels"]
videos_col = db["videos"]
comments_col = db["comments"]

# Streamlit setup
st.set_page_config(layout="wide", page_title="⚡ YouTube AI Analytics")
st.title("⚡ YouTube Channel AI Analytics (Optimized)")

# ---------- Cached Helpers ----------
@st.cache_data
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

@st.cache_data
def translate_text(text):
    try:
        lang = detect(text)
        return GoogleTranslator(source='auto', target='en').translate(text) if lang != "en" else text
    except:
        return text

@st.cache_data
def fast_sentiment(text):
    try:
        return sentiment_pipeline([text])[0]['label']
    except:
        return "NEUTRAL"

@st.cache_data
def fast_ner(text):
    try:
        results = ner_pipeline(text[:512])
        return [(r["word"], r["entity_group"]) for r in results]
    except:
        return []

@st.cache_data
def get_topics_from_comments(comments, n_topics=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(comments)
    model = NMF(n_components=n_topics, random_state=1)
    model.fit(X)
    words = vectorizer.get_feature_names_out()
    return [", ".join([words[i] for i in topic.argsort()[:-6:-1]]) for topic in model.components_]

def search_keywords(comments, query):
    return [c for c in comments if query.lower() in c.lower()]

# ---------- UI ----------
channel_titles = [ch["title"] for ch in channels_col.find()]
selected_channel = st.selectbox("Choose a YouTube Channel", channel_titles)

if selected_channel:
    channel = channels_col.find_one({"title": selected_channel})
    videos = list(videos_col.find({"channel_id": channel["channel_id"]}))
    video_ids = [v["video_id"] for v in videos]
    comments = list(comments_col.find({"video_id": {"$in": video_ids}}).limit(500))  # Limit to 500

    if not comments:
        st.warning("No comments available.")
        st.stop()

    comments_df = pd.DataFrame(comments).dropna(subset=["text"]).head(100)  # Limit to top 100
    st.subheader(f"🎬 Channel Summary: {selected_channel}")
    st.write(f"**Channel ID:** {channel['channel_id']}")
    st.write(f"**Total Videos:** {channel['video_count']}")

    st.subheader("📈 Views & Likes per Video")
    videos_df = pd.DataFrame(videos)
    if not videos_df.empty:
        st.bar_chart(videos_df.set_index("title")[["views", "likes"]])

    st.subheader("🔁 Processing Top Comments...")

    translated_texts, sentiments, entities_list = [], [], []
    for text in comments_df["text"]:
        translated = translate_text(text)
        translated_texts.append(translated)
        sentiments.append(fast_sentiment(translated))
        entities_list.extend(fast_ner(translated))

    comments_df["translated_text"] = translated_texts
    comments_df["sentiment"] = sentiments

    st.subheader("🔎 Named Entities (NER)")
    if entities_list:
        entity_df = pd.DataFrame(entities_list, columns=["Entity", "Type"])
        st.dataframe(entity_df.value_counts().reset_index(name="Count").sort_values("Count", ascending=False))
    else:
        st.info("No named entities found.")

    st.subheader("🗂 Topic Detection")
    topics = get_topics_from_comments(translated_texts)
    for i, topic in enumerate(topics, 1):
        st.write(f"**Topic {i}:** {topic}")

    st.subheader("🔍 Semantic Search")
    query = st.text_input("Search within comments:")
    if query:
        matches = search_keywords(translated_texts, query)
        st.write(matches[:10] if matches else "No matches found.")

    st.subheader("📊 Sentiment Distribution")
    st.bar_chart(pd.Series(sentiments).value_counts())

    st.subheader("☁️ Word Cloud")
    all_text = " ".join(translated_texts)
    wc = WordCloud(width=800, height=300, background_color="white").generate(all_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("📁 Download")
    csv_data = comments_df.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Download CSV", data=csv_data, file_name="comments.csv", mime="text/csv")

    class PDF(FPDF):
        def chapter(self, title, text):
            self.add_page()
            self.set_font("Arial", size=12)
            self.multi_cell(0, 10, text)

    # Use utf-8 encoding in buffer
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add your translated comments
    text = "\n".join(comments_df["translated_text"].head(10))
    pdf.chapter("Top Comments", text)

    # Generate PDF in memory
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer, 'F')  # 'F' = file-like object (BytesIO)
    pdf_buffer.seek(0)

    # Streamlit download button
    st.download_button("📄 Download PDF", pdf_buffer, file_name="report.pdf", mime="application/pdf")