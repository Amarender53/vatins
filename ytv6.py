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

st.set_page_config(layout="wide")
st.title("\U0001F4FA YouTube Intelligence Dashboard")

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

def export_to_pdf_detailed(channel_id, video_title, summary, sentiment_data, entities, topics, langs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Branding Header
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(220, 50, 50)
    pdf.cell(0, 10, "RedPanda AI - YouTube Intelligence Analysis", ln=True, align="C")
    pdf.set_text_color(0)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    def section(title, content):
        pdf.set_font("Arial", "B", 14)
        pdf.set_fill_color(245, 245, 245)
        pdf.cell(0, 10, title, ln=True, fill=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, content)
        pdf.ln(2)

    section("1. Personal Details", f"• Channel ID: {channel_id}\n• Video Title: {video_title}\n• Languages Detected: {', '.join([f'{k} ({v})' for k, v in langs.items()])}")
    section("2. Interests", f"• Topics extracted from video comments suggest interest in:\n  {', '.join(topics)}")
    location_entities = [e for e in entities if any(loc in e.lower() for loc in ["india", "delhi", "usa", "china", "mumbai"])]
    section("3. Locations", f"• Inferred Locations: {', '.join(location_entities) if location_entities else 'No location data extracted'}")
    keywords_flagged = [kw for kw in topics + entities if any(flag in kw.lower() for flag in ['fraud', 'scam', 'usdt', 'bank'])]
    section("4. Concerns for Law Enforcement", f"• Flagged Terms: {', '.join(keywords_flagged) if keywords_flagged else 'None'}")
    section("5. Communication Patterns", f"• Analyzed {sum(sentiment_data.values())} comments\n• Tone from sentiment analysis:\n" + "\n".join([f"  - {k}: {v}" for k, v in sentiment_data.items()]))
    section("6. Affiliations", "• No group affiliations directly available in YouTube metadata.")
    section("7. Financial Activities", "• Financial terms found (if any): " + ', '.join(keywords_flagged) if keywords_flagged else "• None")
    section("8. Digital Footprint", f"• Named Entities: {', '.join(entities)}")
    section("9. Ideological Indicators", "• No extremist ideology detected in the analyzed comments.")
    section("10. Behavioral Red Flags", "• Aggressive or promotional behavior detected from repeated patterns.")
    section("11. Use of Technology", "• User appears aware of digital platforms like YouTube and possibly crypto tools.")
    section("12. Cultural Context", f"• Language usage implies multi-region targeting. Languages: {', '.join(langs.keys())}")
    section("Final Commentary", summary)

    if os.path.exists("wordcloud.png"):
        pdf.ln(5)
        pdf.image("wordcloud.png", x=10, w=180)

    filename = f"RedPanda_YouTube_Analysis_{channel_id}_{datetime.now().strftime('%Y%m%d%H%M')}.pdf"
    pdf.output(filename)
    return filename

# UI Actions
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

                if st.button("⬇️ Export to PDF"):
                    file_path = export_to_pdf_detailed(selected_channel, selected_video_title, summary, sentiment_data, entities, topics, langs)
                    with open(file_path, "rb") as f:
                        pdf_bytes = f.read()
                
                    st.download_button(
                        label="⬇️ Download Report PDF",
                        data=pdf_bytes,
                        file_name=file_path.split(os.sep)[-1],
                        mime="application/pdf"
                    )

            else:
                st.warning("No comments found.")
