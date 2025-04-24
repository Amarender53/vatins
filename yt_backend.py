# --- yt_backend.py ---

from pymongo import MongoClient
from transformers import pipeline
import re
from fpdf import FPDF
from wordcloud import WordCloud
import pandas as pd

client = MongoClient("mongodb://localhost:27017/")
db = client["youtube_data"]
channels_col = db["channels"]
videos_col = db["videos"]
comments_col = db["comments"]

sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", 
    revision="714eb0f"
)

summarizer = pipeline(
    "summarization", 
    model="sshleifer/distilbart-cnn-12-6", 
    revision="a4f8f3e"
)

flagged_keywords = [
    "betting", "casino", "porn", "18+", "xxx", "drugs",
    "OG", "Cp", "C-P", "c*p", "C^P", "C&P", "Indian cp"
]

def get_video_data(video_id):
    video = videos_col.find_one({"video_id": video_id})
    comments = list(comments_col.find({"video_id": video_id}))
    sentiments = [sentiment_pipeline(c["text"][:512])[0]['label'] for c in comments]
    
    return {
        "video": video,
        "comments": comments,
        "sentiments": sentiments
    }

def get_all_channels():
    return list(channels_col.find())

# In yt_backend.py
def get_channel_summary(channel_id):
    # Code to generate channel summary here
    channel_data = get_channel_data(channel_id)  # Reuse the existing function to get data
    summary = {
        "channel": channel_data["channel"]["title"],
        "risk_score": channel_data["channel_risk_score"],
        "top_keywords": channel_data["top_keywords"],
        "top_hashtags": channel_data["top_hashtags"],
        "sentiments": channel_data["sentiments"],
    }
    return summary

# --- yt_backend.py ---

def get_comments_data(video_id):
    # Assuming comments_col is the collection for comment data in your MongoDB
    comments = list(comments_col.find({"video_id": video_id}))
    
    # Optionally, you could perform sentiment analysis or any other processing here
    sentiments = [sentiment_pipeline(c["text"][:512])[0]['label'] for c in comments]

    # Return the comments data along with sentiments
    return {
        "comments": comments,
        "sentiments": sentiments
    }


def get_channel_data(channel_id):
    channel = channels_col.find_one({"channel_id": channel_id})
    videos = list(videos_col.find({"channel_id": channel_id}))
    video_ids = [v["video_id"] for v in videos]
    comments = list(comments_col.find({"video_id": {"$in": video_ids}}))
    comment_texts = [c["text"] for c in comments if isinstance(c.get("text"), str)]

    flagged = [k for k in flagged_keywords if k in " ".join(comment_texts).lower()]
    risk_comments = [c for c in comment_texts if any(k in c.lower() for k in flagged_keywords)]
    risk_score = round(len(risk_comments) / max(1, len(comment_texts)), 2)

    try:
        summary_text = summarizer(" ".join(comment_texts)[:3000])[0]['summary_text']
    except:
        summary_text = "Summary unavailable."

    hashtags = re.findall(r"#\w+", " ".join(comment_texts))
    words = re.findall(r"\b\w{5,}\b", " ".join(comment_texts).lower())
    top_keywords = pd.Series(words).value_counts().head(10).index.tolist()
    top_hashtags = pd.Series(hashtags).value_counts().head(10).index.tolist()

    sentiments = [sentiment_pipeline(c[:512])[0]['label'] for c in comment_texts[:100] if isinstance(c, str)]


    return {
        "channel": channel,
        "videos": videos,
        "comments": comments,
        "comment_texts": comment_texts,
        "summary_text": summary_text,
        "risk_comments": risk_comments,
        "channel_risk_score": risk_score,
        "csam_flags": flagged,
        "top_keywords": top_keywords,
        "top_hashtags": top_hashtags,
        "sentiments": sentiments,
        "words": words
    }

# Channel Summary Class
class YouTubeChannelSummaryPDF(FPDF):
    def __init__(self, channel, past_playlists, present_playlists, comments, sentiments, top_keywords, top_hashtags):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.add_font("DejaVu", "", "E:/VATINS/fonts/DejaVuSans.ttf", uni=True)
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("DejaVu", "", 12)
        self.set_left_margin(15)
        self.set_right_margin(15)
        self.channel = channel
        self.past_playlists = past_playlists
        self.present_playlists = present_playlists
        self.comments = comments
        self.sentiments = sentiments
        self.top_keywords = top_keywords
        self.top_hashtags = top_hashtags
    
    def generate_summary(self):
        self.title_page()
        self.channel_info()
        self.past_playlists_summary()
        self.present_playlists_summary()
        self.comparative_analysis()
        self.sentiment_analysis()
        self.keywords_and_hashtags()

    def title_page(self):
        self.set_font("DejaVu", "B", 16)
        self.cell(0, 10, f"YouTube Channel Summary - {self.channel['title']}", ln=True, align='L')
        self.ln(4)

    def channel_info(self):
        self.set_font("DejaVu", "", 12)
        self.cell(0, 10, f"Channel ID: {self.channel['channel_id']}", ln=True)
        self.cell(0, 10, f"Subscribers: {self.channel['subscribers']:,}", ln=True)
        self.cell(0, 10, f"Total Views: {sum([v['views'] for v in self.past_playlists]):,}", ln=True)
        self.ln(4)

    def past_playlists_summary(self):
        self.set_font("DejaVu", "", 12)
        self.cell(0, 10, f"Past Playlists Summary", ln=True)
        for playlist in self.past_playlists:
            self.multi_cell(0, 8, f"Playlist: {playlist['name']}")
            for video in playlist['videos']:
                self.multi_cell(0, 8, f"  Video Title: {video['title']} | Views: {video['views']:,}")
        self.ln(4)

    def present_playlists_summary(self):
        self.set_font("DejaVu", "", 12)
        self.cell(0, 10, f"Present Playlists Summary", ln=True)
        for playlist in self.present_playlists:
            self.multi_cell(0, 8, f"Playlist: {playlist['name']}")
            for video in playlist['videos']:
                self.multi_cell(0, 8, f"  Video Title: {video['title']} | Views: {video['views']:,}")
        self.ln(4)

    def comparative_analysis(self):
        self.set_font("DejaVu", "", 12)
        self.cell(0, 10, f"Comparative Analysis (Past vs Present)", ln=True)
        # You can add subscriber growth, view growth, engagement trends over time
        self.multi_cell(0, 8, f"Compare the growth from past playlists to present playlists in terms of views, likes, and content trends.")
        self.ln(4)

    def sentiment_analysis(self):
        self.set_font("DejaVu", "", 12)
        sentiment_summary = pd.Series(self.sentiments).value_counts().to_frame().reset_index()
        sentiment_summary.columns = ['Sentiment', 'Count']
        self.cell(0, 10, f"Sentiment Analysis", ln=True)
        self.multi_cell(0, 8, sentiment_summary.to_string(index=False))
        self.ln(4)

    def keywords_and_hashtags(self):
        self.set_font("DejaVu", "", 12)
        self.cell(0, 10, "Top Keywords", ln=True)
        self.multi_cell(0, 8, ", ".join(self.top_keywords))
        self.cell(0, 10, "Top Hashtags", ln=True)
        self.multi_cell(0, 8, ", ".join(self.top_hashtags))
        self.ln(4)

    def generate_wordcloud(self):
        wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(self.top_keywords))
        buf = BytesIO()
        wc.to_image().save(buf, format='PNG')
        buf.seek(0)
        return buf
    
    

import streamlit as st
# ✅ This must be the first Streamlit call
st.set_page_config(layout="wide")

import pandas as pd
import re
from wordcloud import WordCloud
from fpdf import FPDF
from io import BytesIO
from yt_backend import get_all_channels, get_channel_data, get_channel_summary

# Ensure set_page_config is the first Streamlit command
st.set_page_config(layout="wide")

# Emoji Stripper
def strip_emojis(text):
    return re.sub(r'[\U00010000-\U0010ffff]', '', text)

st.title(strip_emojis("\U0001F4FA YouTube Intelligence Dashboard"))

# Sidebar dropdown for channel selection
channels = get_all_channels()
channel_map = {f"{c['title']} ({c['channel_id']})": c['channel_id'] for c in channels}
selected_channel = st.sidebar.selectbox("Select a YouTube Channel", list(channel_map.keys()))
channel_id = channel_map[selected_channel]

# Fetch data for selected channel
channel_data = get_channel_data(channel_id)
channel = channel_data['channel']
videos = channel_data['videos']
comments = channel_data['comments']
summary_text = channel_data['summary_text']
top_keywords = channel_data['top_keywords']
top_hashtags = channel_data['top_hashtags']
risk_comments = channel_data['risk_comments']
channel_risk_score = channel_data['channel_risk_score']
csam_flags = channel_data['csam_flags']
sentiments = channel_data['sentiments']
words = channel_data['words']

channel_tab, video_tab, comment_tab, keyword_tab, ai_tab, risk_tab, export_tab = st.tabs([
    "\U0001F4CA Channel Overview", "\U0001F3AE Video Insights", "\U0001F4AC Comments Analysis",
    "\U0001F3F7\uFE0F Keywords & Hashtags", "\U0001F9E0 AI Analysis", "\U0001F6A8 Risk Assessment", "\U0001F4E5 Reports"
])

with channel_tab:
    st.subheader("\U0001F4CA Channel Overview")
    overview_data = {
        "Field": ["Channel ID", "Title", "Subscribers", "Video Count", "Total Views", "Total Comments", "Risk Score"],
        "Value": [
            channel.get('channel_id'),
            channel.get('title', 'N/A'),
            f"{channel.get('subscribers', 'N/A'):,}",
            f"{channel.get('video_count', 'N/A'):,}",
            f"{sum(v.get('views', 0) for v in videos):,}",
            f"{len(comments):,}",
            f"{channel_risk_score}"
        ]
    }
    st.table(pd.DataFrame(overview_data))
    if csam_flags:
        st.error(f"Flagged Keywords: {', '.join(csam_flags)}")

with video_tab:
    st.subheader("\U0001F3AE Videos")
    st.dataframe(pd.DataFrame(videos)[["video_id", "title", "views", "likes", "published_at"]])

with comment_tab:
    st.subheader("\U0001F4AC Comments")
    st.write(f"Total Comments: {len(comments):,}")
    st.dataframe(pd.DataFrame(comments)[["video_id", "author", "text"]].head(100))

with keyword_tab:
    st.subheader("\U0001F3F7\uFE0F Top Keywords")
    st.write(top_keywords)
    st.subheader("\U0001F4CC Top Hashtags")
    st.write(top_hashtags)
    st.subheader("\u2601\uFE0F Word Cloud")
    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words))
    st.image(wc.to_array(), use_column_width=True)

with ai_tab:
    st.subheader("\U0001F9E0 Comments Summary by AI")
    st.text_area("Summary", summary_text, height=300)
    st.subheader("\U0001F4CA Sentiment Analysis")
    st.bar_chart(pd.Series(sentiments).value_counts())

with risk_tab:
    st.subheader("\U0001F6A8 Risky Comments")
    st.write(risk_comments[:10])
    st.markdown(f"**Risk Score:** `{channel_risk_score}`")

with export_tab:
    st.subheader("\U0001F4E5 Download Reports")

    class SimplePDF(FPDF):
        def __init__(self):
            super().__init__()
            self.add_page()
            self.add_font("Noto", "", "fonts/NotoSans-Regular.ttf", uni=True)
            self.set_font("Noto", "", 12)

        def add_title(self, title):
            self.set_font("Noto", "", 16)
            self.cell(0, 10, title, ln=True, align="L")
            self.ln(5)
            self.set_font("Noto", "", 12)

        def add_section(self, heading, text):
            self.set_font("Noto", "", 12)
            self.cell(0, 10, heading, ln=True)
            self.multi_cell(0, 8, str(text))
            self.ln(4)

    def generate_pdf(channel_id: str) -> BytesIO:
        data = get_channel_data(channel_id)
        channel = data["channel"]
        videos = data["videos"]
        comments = data["comments"]
        sentiments = data["sentiments"]
        top_keywords = data["top_keywords"]
        top_hashtags = data["top_hashtags"]
        summary_text = data["summary_text"]
        risk_comments = data["risk_comments"]
        channel_risk_score = data["channel_risk_score"]

        pdf = SimplePDF()
        pdf.add_title(f"YouTube Channel Intelligence Report - {channel.get('title', '')}")
        pdf.add_section("Channel Info", f"Channel ID: {channel.get('channel_id')}\n"
                                        f"Subscribers: {channel.get('subscribers')}\n"
                                        f"Video Count: {len(videos)}\n"
                                        f"Total Views: {sum([v.get('views', 0) for v in videos])}\n"
                                        f"Risk Score: {channel_risk_score}")
        pdf.add_section("AI Summary", summary_text)
        pdf.add_section("Top Keywords", ", ".join(top_keywords))
        pdf.add_section("Top Hashtags", ", ".join(top_hashtags))
        pdf.add_section("Flagged Comments", "\n\n".join(risk_comments[:5]))

        buf = BytesIO()
        pdf.output(buf)
        buf.seek(0)
        return buf

    safe_title = re.sub(r'[\\/*?:"<>|]', "", channel.get('title', 'channel_summary')).replace(" ", "_")
    st.download_button("\U0001F4C4 Download PDF", generate_pdf(channel_id), file_name=f"{safe_title}_summary.pdf", mime="application/pdf")

    csv = pd.DataFrame(videos)[["video_id", "title", "views", "likes"]].to_csv(index=False).encode("utf-8")
    st.download_button("\U0001F5C2\uFE0F Download CSV", csv, file_name="video_summary.csv", mime="text/csv")
