import streamlit as st
from pymongo import MongoClient
from transformers import pipeline
from deep_translator import GoogleTranslator
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from fpdf import FPDF
import matplotlib.pyplot as plt
import pandas as pd
import re
from io import BytesIO

# ------------------ MongoDB Setup ------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["youtube_data"]
comments_col = db["comments"]
videos_col = db["videos"]
channels_col = db["channels"]

# ------------------ NLP Pipelines ------------------
summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
ner_model = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
kw_model = KeyBERT()

# ------------------ Text Processing ------------------
def fetch_video_comments(video_id):
    return [doc["text"] for doc in comments_col.find({"video_id": video_id}) if isinstance(doc.get("text"), str)]

def fetch_channel_comments(channel_id):
    vids = [v["video_id"] for v in videos_col.find({"channel_id": channel_id})]
    return [doc["text"] for doc in comments_col.find({"video_id": {"$in": vids}}) if isinstance(doc.get("text"), str)]

def summarize_texts(texts, lang="en"):
    chunk = "\n".join(texts[:100])
    summary = summarizer(chunk[:3000], max_length=300, min_length=50, do_sample=False)[0]["summary_text"]
    return GoogleTranslator(source="en", target=lang).translate(summary) if lang != "en" else summary

def analyze_sentiment(texts):
    return sentiment_model(texts[:50])

def extract_entities(text):
    return ner_model(text[:512])

def extract_topics(text):
    return [kw[0] for kw in kw_model.extract_keywords(text, top_n=10)]

def semantic_search(text, query="hate speech"):
    vect = TfidfVectorizer().fit_transform([text, query])
    sim = cosine_similarity(vect[0:1], vect[1:2])
    return float(sim[0][0])

def detect_incidents(text):
    keywords = ["bomb", "attack", "riot", "kill", "hack", "leak", "terror", "gun", "blast"]
    return [k for k in keywords if k in text.lower()] or ["None"]

def geo_locations(text):
    pattern = r"\b(Hyderabad|Delhi|New York|Mumbai|London|Paris|Pakistan|India|USA|Canada|Kabul)\b"
    return list(set(re.findall(pattern, text)))

# ------------------ Visuals ------------------
def word_cloud(text):
    wc = WordCloud(width=800, height=300, background_color='white', colormap='tab10', stopwords=WordCloud().stopwords).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def sentiment_pie(df):
    df = pd.DataFrame(df)
    counts = df["label"].value_counts()
    fig, ax = plt.subplots()
    counts.plot.pie(autopct="%1.1f%%", ax=ax, colors=["green", "red", "orange"])
    ax.set_ylabel("")
    return fig

def entity_bar(entities):
    df = pd.DataFrame(entities)
    fig, ax = plt.subplots()
    df["entity_group"].value_counts().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Named Entity Frequency")
    return fig

# ------------------ PDF Generator ------------------
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "YouTube AI Intelligence Report", ln=True, align="C")
        self.ln(10)

    def section(self, title, content):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 8, content)
        self.ln(4)

def generate_pdf(meta, summary, sentiment, entities, topics, incidents, geo, similarity):
    pdf = PDF()
    pdf.add_page()
    pdf.section("1. Meta Info", meta)
    pdf.section("2. Summary", summary)
    pdf.section("3. Sentiment", f"{sentiment['label']} (Score: {sentiment['score']:.2f})")
    pdf.section("4. Entities", "\n".join([f"{e['entity_group']}: {e['word']}" for e in entities]))
    pdf.section("5. Topics", ", ".join(topics))
    pdf.section("6. Geo", ", ".join(geo or ["None"]))
    pdf.section("7. Incidents", ", ".join(incidents or ["None"]))
    pdf.section("8. Semantic Similarity", f"{similarity:.2f} relevance to 'hate speech'")
    pdf.section("9. Remarks", "This AI-generated intelligence summary is based on public YouTube comments.")
    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

# ------------------ Streamlit App ------------------
st.set_page_config(layout="wide", page_title="YouTube Intelligence AI Dashboard")
st.title("🎥 YouTube Intelligence Dashboard (AI Powered)")

analysis_type = st.radio("Select Analysis Type", ["Video Level", "Channel Level"])

lang_map = {"English": "en", "Telugu": "te", "Urdu": "ur", "Malayalam": "ml", "Bengali": "bn", "Arabic": "ar"}
lang = st.selectbox("Select Report Language", list(lang_map.keys()))

if analysis_type == "Video Level":
    videos = list(videos_col.find({}, {"video_id": 1, "title": 1}))
    video_map = {f"{v['title']} ({v['video_id']})": v for v in videos}
    selected = st.selectbox("Select a YouTube Video", list(video_map.keys()))
    video = video_map[selected]
    comments = fetch_video_comments(video["video_id"])
    meta = f"Video Title: {video['title']}\nVideo ID: {video['video_id']}"
    file_id = video['title'][:50].replace(" ", "_")

else:
    channels = list(channels_col.find({}, {"channel_id": 1, "title": 1}))
    chan_map = {f"{c['title']} ({c['channel_id']})": c for c in channels}
    selected = st.selectbox("Select a YouTube Channel", list(chan_map.keys()))
    channel = chan_map[selected]
    comments = fetch_channel_comments(channel["channel_id"])
    meta = f"Channel Title: {channel['title']}\nChannel ID: {channel['channel_id']}"
    file_id = channel['title'][:50].replace(" ", "_")

if st.button("🧠 Run Analysis"):
    if not comments:
        st.warning("No comments found.")
        st.stop()

    text = " ".join(comments)
    summary = summarize_texts(comments, lang_map[lang])
    sentiment = analyze_sentiment(comments)[0]
    entities = extract_entities(text)
    topics = extract_topics(text)
    geo = geo_locations(text)
    incidents = detect_incidents(text)
    similarity = semantic_search(text)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Summary")
        st.write(summary)
        st.subheader("🧠 Topics")
        st.write(", ".join(topics))
        st.subheader("🌍 Geo-Locations")
        st.write(", ".join(geo or ["None"]))

    with col2:
        st.subheader("📊 Sentiment Distribution")
        st.pyplot(sentiment_pie(analyze_sentiment(comments)))
        st.subheader("☁️ Word Cloud")
        st.pyplot(word_cloud(text))
        st.subheader("📌 Named Entity Chart")
        st.pyplot(entity_bar(entities))

    st.subheader("🧠 Named Entities Table")
    st.write(pd.DataFrame(entities))

    if any(i for i in incidents if i != "None"):
        st.error(f"⚠️ ALERT Keywords Found: {', '.join(incidents)}")
    if similarity > 0.5:
        st.warning(f"⚠️ High semantic similarity to 'hate speech': {similarity:.2f}")

    pdf = generate_pdf(meta, summary, sentiment, entities, topics, incidents, geo, similarity)
    st.download_button("📄 Download PDF", pdf, file_name=f"{file_id}_ai_report.pdf", mime="application/pdf")
