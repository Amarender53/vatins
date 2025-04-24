import streamlit as st
from pymongo import MongoClient
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from transformers import pipeline
import io
import tempfile

# Setup
st.set_page_config(page_title="YouTube AI Analytics Dashboard", layout="wide")
st.title("📊 YouTube AI Analytics Dashboard")

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017")
db = client["youtube_db"]
channels_col = db["channels"]
comments_col = db["comments"]

# HuggingFace Pipelines
sentiment_pipeline = pipeline("sentiment-analysis")
translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
ner_pipeline = pipeline("ner", grouped_entities=True)
topic_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# UI - Channel Selection
channel_names = [c["channel_name"] for c in channels_col.find()]
selected_channel = st.selectbox("🎥 Select a Channel", channel_names)

if selected_channel:
    channel = channels_col.find_one({"channel_name": selected_channel})
    st.subheader(f"📺 Channel Summary: {channel['channel_name']}")
    st.markdown(f"- **Channel ID**: {channel['channel_id']}")
    st.markdown(f"- **Subscribers**: {channel.get('subscribers', 'N/A')}")
    st.markdown(f"- **Total Videos**: {channel.get('video_count', 'N/A')}")

    # Fetch Comments
    comments_cursor = comments_col.find({"channel_name": selected_channel})
    comments_df = pd.DataFrame(comments_cursor)

    if not comments_df.empty:
        # Limit to 100 comments for performance
        comments_df = comments_df.head(100)

        # Language Detection
        st.subheader("🌐 Language Detection")
        comments_df["language"] = comments_df["text"].apply(lambda x: translation_pipeline(x)[0]["src_text"] if x else "unknown")
        st.write(comments_df["language"].value_counts())

        # Translation to English
        st.subheader("🌍 Translation")
        comments_df["translated_text"] = comments_df["text"].apply(lambda x: translation_pipeline(x)[0]["translation_text"] if x else "")
        translated_texts = comments_df["translated_text"].tolist()

        # Sentiment Analysis
        st.subheader("📈 Sentiment Analysis")
        sentiments = [s["label"] for s in sentiment_pipeline(translated_texts)]
        comments_df["sentiment"] = sentiments
        sentiment_counts = pd.Series(sentiments).value_counts()
        fig_sent, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', color="skyblue", ax=ax)
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig_sent)

        # Word Cloud
        st.subheader("☁️ Word Cloud")
        all_text = " ".join(translated_texts)
        wc = WordCloud(width=800, height=300, background_color="white").generate(all_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # NER
        st.subheader("🧠 Named Entities")
        entities = ner_pipeline(all_text)
        entities_df = pd.DataFrame([{"Entity": e["word"], "Label": e["entity_group"]} for e in entities])
        st.dataframe(entities_df)

        # Topic Modeling
        st.subheader("🧩 Topic Modeling (Zero-shot)")
        candidate_labels = ["Education", "Entertainment", "Politics", "Sports", "Technology", "Health", "Finance"]
        topic_results = topic_pipeline(all_text, candidate_labels)
        topics = sorted(zip(topic_results["labels"], topic_results["scores"]), key=lambda x: -x[1])
        st.write("Top Topics:")
        for t in topics[:3]:
            st.markdown(f"- {t[0]} ({round(t[1]*100, 2)}%)")

        # Save images temporarily
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_sent:
            fig_sent.savefig(tmp_sent.name, bbox_inches='tight')
            sentiment_chart_path = tmp_sent.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_wc:
            wc.to_file(tmp_wc.name)
            wordcloud_path = tmp_wc.name

        # PDF Report
        st.subheader("📄 Generate PDF Report")
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        width, height = A4

        # Page 1 - Summary
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, f"YouTube Channel Report: {selected_channel}")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Channel ID: {channel['channel_id']}")
        c.drawString(50, height - 100, f"Total Videos: {channel.get('video_count', 'N/A')}")
        c.drawString(50, height - 120, f"Subscribers: {channel.get('subscribers', 'N/A')}")

        # Topics
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 150, "Top Topics:")
        c.setFont("Helvetica", 12)
        for i, topic in enumerate(topics[:5], start=1):
            c.drawString(60, height - 150 - (i * 20), f"Topic {i}: {topic[0]} ({round(topic[1]*100, 2)}%)")

        # Page 2 - Sentiment Chart
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 50, "Sentiment Distribution")
        c.drawImage(ImageReader(sentiment_chart_path), 50, height/2 - 100, width=500, preserveAspectRatio=True, mask='auto')

        # Page 3 - Word Cloud
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 50, "Word Cloud")
        c.drawImage(ImageReader(wordcloud_path), 50, height/2 - 100, width=500, preserveAspectRatio=True, mask='auto')

        c.save()
        pdf_buffer.seek(0)

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"{selected_channel}_analytics_report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("No comments found for this channel.")
