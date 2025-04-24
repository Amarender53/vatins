import streamlit as st
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from googleapiclient.discovery import build
from transformers import pipeline
from pymongo import MongoClient
import nltk
from sentence_transformers import SentenceTransformer
from datetime import datetime
from fpdf import FPDF
from langdetect import detect
from keybert import KeyBERT

# Setup
nltk.download('punkt')
sentiment_pipeline = pipeline("sentiment-analysis")
ner_pipeline = pipeline("ner", aggregation_strategy="simple")
summarizer_pipeline = pipeline("summarization")
translator_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
kw_model = KeyBERT(model=SentenceTransformer('all-MiniLM-L6-v2'))

YOUTUBE_API_KEY = "AIzaSyCKRTuJPZ1xw3NGuXwUgkXuYz8ZGpcdHE8"
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
client = MongoClient("mongodb://localhost:27017/")
db = client["youtube_data"]

# UI Config
st.set_page_config(layout="wide")
st.title("\U0001F4FA YouTube Intelligence Dashboard")

# Helper Functions

def get_channel_info(channel_id):
    try:
        data = youtube.channels().list(part="snippet,contentDetails", id=channel_id).execute()
        info = data['items'][0]
        return {
            "title": info['snippet']['title'],
            "upload_playlist": info['contentDetails']['relatedPlaylists']['uploads']
        }
    except:
        return None

def get_channel_videos(playlist_id):
    videos, token = [], None
    while True:
        res = youtube.playlistItems().list(part="snippet", playlistId=playlist_id, maxResults=10, pageToken=token).execute()
        for item in res["items"]:
            videos.append({
                "video_id": item["snippet"]["resourceId"]["videoId"],
                "title": item["snippet"]["title"]
            })
        token = res.get("nextPageToken")
        if not token: break
    return videos

def get_comments(video_id):
    comments = []
    try:
        res = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100).execute()
        for item in res["items"]:
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            if len(text.strip()) > 3:
                comments.append(text)
    except: pass
    return comments

def detect_languages(comments):
    langs = {}
    for c in comments:
        try:
            lang = detect(c)
            langs[lang] = langs.get(lang, 0) + 1
        except: pass
    return langs

def analyze_and_store(channel_id, video_id, comments):
    translated = [translator_pipeline(c)[0]['translation_text'] for c in comments[:30]]
    summary = summarizer_pipeline(" ".join(translated), max_length=250, min_length=80, do_sample=False)[0]['summary_text']
    sentiments = sentiment_pipeline(translated[:30])
    ner_data = ner_pipeline(" ".join(translated[:10]))

    sentiment_counts = {label: 0 for label in ["POSITIVE", "NEGATIVE", "NEUTRAL"]}
    for s in sentiments:
        label = s['label'].upper()
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

    wc = WordCloud(width=800, height=400).generate(" ".join(translated))
    wc_path = "wordcloud.png"
    wc.to_file(wc_path)

    topics = kw_model.extract_keywords(" ".join(translated), keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    topic_list = [t[0] for t in topics]

    db.reports.insert_one({
        "channel_id": channel_id,
        "video_id": video_id,
        "datetime": datetime.now(),
        "summary": summary,
        "sentiment_counts": sentiment_counts,
        "entities": list(set([e['word'] for e in ner_data])),
        "topics": topic_list
    })

    return summary, sentiment_counts, wc_path, list(set([f"{e['word']} ({e['entity_group']})" for e in ner_data])), topic_list

def export_to_pdf(channel_id, summary, sentiment_data, entities, topics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, f"YouTube Intelligence Report - {channel_id}", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Summary:\n{summary}")
    pdf.ln()
    pdf.multi_cell(0, 10, f"Sentiment Counts:\n{sentiment_data}")
    pdf.ln()
    pdf.multi_cell(0, 10, f"Named Entities:\n{', '.join(entities)}")
    pdf.ln()
    pdf.multi_cell(0, 10, f"Topics:\n{', '.join(topics)}")
    pdf.ln()
    if os.path.exists("wordcloud.png"):
        pdf.image("wordcloud.png", x=10, y=None, w=180)
    filename = f"YouTube_Report_{channel_id}_{datetime.now().strftime('%Y%m%d%H%M')}.pdf"
    pdf.output(filename)
    return filename

# --- UI ---
saved_channel_ids = [c["channel_id"] for c in db.channels.find({}, {"channel_id": 1})]

if not saved_channel_ids:
    st.warning("⚠️ No saved reports found in the database. Please run analysis to populate this list.")
    st.stop()

selected_channel = st.selectbox("Select a YouTube Channel from saved reports", saved_channel_ids)

if selected_channel:
    info = get_channel_info(selected_channel)
    if not info:
        st.error("Failed to retrieve channel info. Check if the channel ID is correct or YouTube API quota is exceeded.")
        st.stop()

    video_list = get_channel_videos(info['upload_playlist'])
    if not video_list:
        st.warning("No videos found in this channel playlist.")
        st.stop()

    video_titles = {v['title']: v['video_id'] for v in video_list}
    selected_video_title = st.selectbox("Select a Video", list(video_titles.keys()))
    selected_video_id = video_titles[selected_video_title]

    date_range = st.date_input("\U0001F4C5 Filter by Date", [])
    search_term = st.text_input("\U0001F50D Search Comments")

    if st.button("\U0001F4CA Generate Report"):
        with st.spinner("Fetching and analyzing data..."):
            comments = get_comments(selected_video_id)

            if search_term:
                comments = [c for c in comments if search_term.lower() in c.lower()]

            if comments:
                langs = detect_languages(comments)
                st.write("\U0001F30D Detected Languages:", langs)

                summary, sentiment_data, wc_path, entities, topics = analyze_and_store(selected_channel, selected_video_id, comments)
                st.subheader("\U0001F9E0 AI Summary")
                st.success(summary)
                st.subheader("\U0001F4CA Sentiment Chart")
                st.bar_chart(sentiment_data)
                st.subheader("\u2601\uFE0F Word Cloud")
                st.image(wc_path, width=600)
                st.subheader("\U0001F9FE Named Entities")
                st.code(", ".join(entities))
                st.subheader("\U0001F4CC Topics")
                st.write(topics)

                if st.button("\u2B07\uFE0F Export to PDF"):
                    file_path = export_to_pdf(selected_channel, summary, sentiment_data, entities, topics)
                    with open(file_path, "rb") as f:
                        st.download_button("Download Report PDF", f, file_name=file_path, mime="application/pdf")
            else:
                st.warning("No comments found.")

st.sidebar.subheader("\U0001F4DA Saved Reports")
for doc in db.reports.find().sort("datetime", -1).limit(10):
    if st.session_state.get("date_range"):
        if not (st.session_state.date_range[0] <= doc["datetime"].date() <= st.session_state.date_range[1]):
            continue
    st.sidebar.markdown(f"**{doc['channel_id']}** - {doc['datetime'].strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.markdown(f"`{doc['summary'][:80]}...`")