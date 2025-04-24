import streamlit as st
from pymongo import MongoClient
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from deep_translator import GoogleTranslator
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import spacy
import emoji

# Load models
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except:
    sentiment_pipeline = lambda x: [{"label": "NEUTRAL", "score": 0.0}]

ner_pipeline = pipeline("ner", grouped_entities=True, model="Davlan/bert-base-multilingual-cased-ner-hrl")
nlp = spacy.load("en_core_web_sm")

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["youtube_data"]
channels_col = db["channels"]
videos_col = db["videos"]
comments_col = db["comments"]

# Streamlit config
st.set_page_config(layout="wide", page_title="▶️ YouTube AI Analytics")
st.image("https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg", width=200)

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
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(comments)
        model = NMF(n_components=n_topics, random_state=1)
        model.fit(X)
        words = vectorizer.get_feature_names_out()
        return [", ".join([words[i] for i in topic.argsort()[:-6:-1]]) for topic in model.components_]
    except:
        return []

def search_keywords(comments, query):
    return [c for c in comments if query.lower() in c.lower()]

# UI
channel_titles = [ch["title"] for ch in channels_col.find()]
selected_channel = st.selectbox("Choose a YouTube Channel", channel_titles)

if selected_channel:
    channel = channels_col.find_one({"title": selected_channel})
    channel_id = channel.get("channel_id", "Unknown")
    video_count = channel.get("video_count", 0)

    videos = list(videos_col.find({"channel_id": channel_id}))
    video_ids = [v["video_id"] for v in videos]
    comments = list(comments_col.find({"video_id": {"$in": video_ids}}).limit(500))

    if not comments:
        st.warning("No comments available.")
        st.stop()

    comments_df = pd.DataFrame(comments).dropna(subset=["text"]).head(100)

    total_likes = sum(v.get("likes", 0) for v in videos)
    total_views = sum(v.get("views", 0) for v in videos)
    total_comments = len(comments)

    st.markdown("### 🎮 YouTube Channel Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📺 Channel Name", selected_channel)
        st.metric("🆑 Channel ID", channel_id)
    with col2:
        st.metric("🎮 Total Videos", video_count)
        st.metric("💬 Total Comments", total_comments)
    with col3:
        st.metric("👍 Total Likes", total_likes)
        st.metric("👁️ Total Views", total_views)

    st.subheader("📈 Views & Likes per Video")
    videos_df = pd.DataFrame(videos)
    if not videos_df.empty:
        videos_df = videos_df.dropna(subset=["title"])
        videos_df.set_index("title", inplace=True)
        st.bar_chart(videos_df[["views", "likes"]])

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

    st.subheader("💂 Topic Detection")
    topics = get_topics_from_comments(translated_texts)
    for i, topic in enumerate(topics, 1):
        st.write(f"**Topic {i}:** {topic}")

    st.subheader("🔍 Semantic Search")
    query = st.text_input("Search within comments:")
    if query:
        matches = search_keywords(translated_texts, query)
        st.write(matches[:10] if matches else "No matches found.")

    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = pd.Series(sentiments).value_counts()
    fig_sent, ax_sent = plt.subplots()
    sentiment_counts.plot(kind="bar", color="skyblue", ax=ax_sent)
    st.pyplot(fig_sent)

    st.subheader("☁️ Word Cloud")
    all_text = " ".join(translated_texts)
    wc = WordCloud(width=800, height=300, background_color="white").generate(all_text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    st.subheader("📁 Download CSV")
    csv_data = comments_df.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Download CSV", data=csv_data, file_name="comments.csv", mime="text/csv")

    wordcloud_img = io.BytesIO()
    wc.to_image().save(wordcloud_img, format="PNG")
    wordcloud_img.seek(0)

    sentiment_chart_img = io.BytesIO()
    fig_sent.savefig(sentiment_chart_img, format="PNG")
    sentiment_chart_img.seek(0)

    def generate_summary_pdf():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("YouTube Channel AI Summary", styles['Title']))
        elements.append(Spacer(1, 12))

        data = [
            ["Channel Name", selected_channel],
            ["Channel ID", channel_id],
            ["Total Videos", video_count],
            ["Total Views", f"{total_views:,}"],
            ["Total Likes", f"{total_likes:,}"],
            ["Total Comments", f"{total_comments:,}"]
        ]
        table = Table(data, hAlign='LEFT', colWidths=[150, 300])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica')
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

        elements.append(Paragraph("Word Cloud", styles['Heading2']))
        elements.append(RLImage(wordcloud_img, width=4*inch, height=2*inch))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Sentiment Distribution", styles['Heading2']))
        elements.append(RLImage(sentiment_chart_img, width=4*inch, height=2*inch))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Top Translated Comments", styles['Heading2']))

        def sanitize(text):
            text = text.replace("<br>", "<br/>")
            text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return text

        for comment in translated_texts[:10]:
            clean_comment = sanitize(comment)
            elements.append(Paragraph(f"- {clean_comment}", styles['Normal']))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf_bytes = generate_summary_pdf()
    st.download_button("📅 Download Summary PDF", data=pdf_bytes, file_name=f"{selected_channel}_summary.pdf", mime="application/pdf")
