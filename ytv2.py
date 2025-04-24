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
from googleapiclient.discovery import build
import os
import folium
from streamlit_folium import folium_static

# ---------------- SETUP ----------------
st.set_page_config(page_title="YouTube AI Analytics", layout="wide")

# NLP setup
nlp = spacy.load("en_core_web_sm")

# MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["youtube_data"]
channels_col = db["channels"]
videos_col = db["videos"]
comments_col = db["comments"]

# YouTube API
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# ---------------- UTIL FUNCTIONS ----------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != "en":
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except:
        return text

def analyze_sentiment(text):
    text = translate_to_english(text)
    return TextBlob(text).sentiment.polarity

def extract_named_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def get_topics_from_comments(comments, n_topics=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(comments)
    model = NMF(n_components=n_topics, random_state=1)
    model.fit(X)
    words = vectorizer.get_feature_names_out()
    topics = []
    for topic in model.components_:
        topic_words = [words[i] for i in topic.argsort()[:-6:-1]]
        topics.append(", ".join(topic_words))
    return topics

def search_keywords(comments, query):
    query = query.lower()
    return [c for c in comments if query in c.lower()]

def display_map_if_available(location):
    m = folium.Map(location=[location["lat"], location["lon"]], zoom_start=10)
    folium.Marker([location["lat"], location["lon"]], tooltip="Comment Location").add_to(m)
    folium_static(m)

# ---------------- DASHBOARD UI ----------------
st.title("📊 YouTube Channel AI Analytics")

# Channel Selection
channel_titles = [ch["title"] for ch in channels_col.find()]
selected_channel = st.selectbox("Choose a YouTube Channel", channel_titles)

if selected_channel:
    channel = channels_col.find_one({"title": selected_channel})
    videos = list(videos_col.find({"channel_id": channel["channel_id"]}))
    video_ids = [v["video_id"] for v in videos]
    comments = list(comments_col.find({"video_id": {"$in": video_ids}}))

    videos_df = pd.DataFrame(videos)
    comments_df = pd.DataFrame(comments)

    if videos_df.empty or comments_df.empty:
        st.warning("No videos or comments found for this channel.")
        st.stop()

    # Channel Summary
    st.subheader(f"🎬 Channel Summary: {selected_channel}")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Channel ID", channel["channel_id"])
    with col2: st.metric("Total Videos", channel["video_count"])
    with col3: st.metric("Total Comments", len(comments_df))

    # Video Metrics
    st.subheader("📈 Views & Likes per Video")
    videos_df.set_index("title")[["views", "likes"]].sort_values("views", ascending=False).plot(kind="bar", figsize=(10, 5))
    st.pyplot(plt)

    # Timeline
    st.subheader("🕒 Video Publishing Timeline")
    videos_df["published_at"] = pd.to_datetime(videos_df["published_at"])
    st.line_chart(videos_df.set_index("published_at")["views"])

    # Language Detection & Translation
    st.subheader("🌐 Language Detection & Translation")
    comments_df["language"] = comments_df["text"].apply(detect_language)
    comments_df["translated_text"] = comments_df["text"].apply(translate_to_english)
    lang_counts = comments_df["language"].value_counts()
    st.dataframe(lang_counts.reset_index().rename(columns={"index": "Language", "language": "Count"}))

    # Sentiment Analysis
    st.subheader("🧠 Sentiment Analysis")
    with st.spinner("Analyzing sentiments..."):
        comments_df["sentiment"] = comments_df["translated_text"].apply(analyze_sentiment)
        st.bar_chart(comments_df["sentiment"].value_counts(bins=10))
        st.dataframe(comments_df[["author", "text", "sentiment"]].head(10))

    # NER
    st.subheader("🔍 Named Entity Recognition")
    full_text = " ".join(comments_df["translated_text"].tolist())
    entities = extract_named_entities(full_text)
    if entities:
        entity_df = pd.DataFrame(entities, columns=["Entity", "Label"])
        st.dataframe(entity_df.value_counts().reset_index(name="Count").sort_values("Count", ascending=False).head(10))
    else:
        st.info("No entities detected.")

    # Topic Modeling
    st.subheader("🗂 Topic Detection")
    with st.spinner("Detecting discussion topics..."):
        topics = get_topics_from_comments(comments_df["translated_text"].tolist())
        for i, t in enumerate(topics, 1):
            st.markdown(f"**Topic {i}**: {t}")

    # Semantic Search
    st.subheader("🔍 Semantic Search in Comments")
    query = st.text_input("Search for a keyword/phrase:")
    if query:
        results = search_keywords(comments_df["translated_text"].tolist(), query)
        st.write(results[:10] if results else "No matching comments found.")

    # Geo Mapping
    st.subheader("🗺 Geo-Based Analysis (if available)")
    if "location" in comments_df.columns and isinstance(comments_df["location"].iloc[0], dict):
        display_map_if_available(comments_df["location"].iloc[0])
    else:
        st.info("No geo-location data available.")

    # WordCloud
    st.subheader("☁️ Comment Word Cloud")
    wordcloud = WordCloud(width=800, height=300, background_color='white').generate(full_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
