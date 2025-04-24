import streamlit as st
import pandas as pd
from pymongo import MongoClient
from transformers import pipeline
from fpdf import FPDF
from wordcloud import WordCloud
from io import BytesIO
import os
import textwrap
import unicodedata
import re

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["telegram_data"]
users_col = db["users"]
messages_col = db["messages"]
groups_col = db["groups"]

# Load NLP models
sentiment_model = pipeline("sentiment-analysis")
ner_model = pipeline("ner", grouped_entities=True)
summarizer = pipeline("summarization")

# Font path
FONT_PATH = "E:/VATINS/fonts/DejaVuSans.ttf"
BOLD_FONT_PATH = FONT_PATH.replace(".ttf", "-Bold.ttf")

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📋 Telegram Intelligence Summary Report")

# Load users
user_data = list(users_col.find({}, {"user_id": 1, "first_name": 1, "last_name": 1, "username": 1}))
user_map = {
    f"{u.get('first_name', '')} {u.get('last_name', '')} (@{u.get('username', 'N/A')})": u["user_id"]
    for u in user_data
}

selected_user = st.selectbox("Select a User", list(user_map.keys()))
user_id = user_map[selected_user]

user = users_col.find_one({"user_id": user_id})
messages = list(messages_col.find({"user_id": user_id}))
texts = [msg.get("text", "") for msg in messages if isinstance(msg.get("text", ""), str)]

# Behavior & Analytics
st.header("🔍 Behavior Analysis")
st.markdown(f"**User ID:** {user_id}")
st.markdown(f"**Name:** {user.get('first_name', '')} {user.get('last_name', '')}")
st.markdown(f"**Username:** @{user.get('username', 'N/A')}")
st.markdown(f"**Phone:** {user.get('phone', 'N/A')}")

# Sentiment Analysis
st.subheader("📊 Sentiment Distribution")
sentiments = [sentiment_model(t[:300])[0]["label"] for t in texts]
sentiment_df = pd.Series(sentiments).value_counts()
st.bar_chart(sentiment_df)

# WordCloud
st.subheader("☁️ Word Cloud")
all_text = " ".join(texts)
wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)
st.image(wc.to_array(), use_column_width=True)

# NER
st.subheader("🧠 Named Entity Recognition (NER)")
ner_data = []
for t in texts:
    ner_data.extend([(ent['word'], ent['entity_group']) for ent in ner_model(t[:512])])
ner_df = pd.DataFrame(ner_data, columns=["Entity", "Type"])
st.dataframe(ner_df.value_counts().reset_index(name="Count"))

# LLM-based Summarization - Point-wise
st.subheader("📝 Summary by LLM")
chunk_size = 3000
raw_summary = ""
for i in range(0, len(all_text), chunk_size):
    chunk = all_text[i:i+chunk_size]
    try:
        summary_piece = summarizer(chunk[:1000], max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        raw_summary += summary_piece + "\n"
    except:
        continue

# Clean bullet points
bullet_points = textwrap.wrap(raw_summary.replace('\n', ' '), width=100)
point_wise_summary = "\n".join([f"• {line.strip().lstrip('•')}" for line in bullet_points])
st.text_area("Generated Point-wise Summary", point_wise_summary, height=300)

# PDF Export
class UnicodePDF(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
        self.page_width = self.w - 2 * self.l_margin

        self.add_font("DejaVu", "", FONT_PATH, uni=True)
        if os.path.exists(BOLD_FONT_PATH):
            self.add_font("DejaVu", "B", BOLD_FONT_PATH, uni=True)

        self.set_font("DejaVu", "", 11)

    def header(self):
        self.set_font("DejaVu", "", 14)
        self.cell(0, 10, "📄 Telegram User Intelligence Report", ln=True, align='C')
        self.ln(5)

    def clean_text(self, text):
        text = str(text)
        emoji_pattern = re.compile(
            "[" u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\u2600-\u26FF\u2700-\u27BF"
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        return unicodedata.normalize("NFKD", text).strip()

    def break_long_words(self, text, limit=100):
        words = text.split()
        broken = []
        for word in words:
            if len(word) > limit:
                broken.extend(textwrap.wrap(word, width=limit, break_long_words=True))
            else:
                broken.append(word)
        return ' '.join(broken)

    def add_section_title(self, title):
        self.set_font("DejaVu", "B", 12)
        self.set_x(self.l_margin)
        self.cell(0, 10, title, ln=True)
        self.set_font("DejaVu", "", 11)
        self.ln(1)

    def add_bullet_list(self, items):
        for item in items:
            text = self.clean_text(item)
            text = self.break_long_words(text)
            wrapped = textwrap.wrap(text, width=100)
            for line in wrapped:
                self.set_x(self.l_margin)
                self.multi_cell(self.page_width, 6, f"• {line}", align='L')
            self.ln(1)

    def add_paragraph(self, text):
        text = self.clean_text(text)
        text = self.break_long_words(text)
        wrapped = textwrap.wrap(text, width=100)
        for line in wrapped:
            self.set_x(self.l_margin)
            self.multi_cell(self.page_width, 6, line, align='L')
        self.ln(1)

def generate_pdf(title, info, messages, summary_text):
    pdf = UnicodePDF()
    pdf.add_page()

    # Title
    pdf.add_section_title(title)

    # Info section
    pdf.add_section_title("👤 User Information")
    pdf.add_bullet_list(info)

    # Summary section
    pdf.add_section_title("📝 Summary")
    bullet_lines = [line.strip("• ").strip() for line in summary_text.split("\n") if line.strip()]
    pdf.add_bullet_list(bullet_lines)

    # Messages section
    pdf.add_section_title("💬 Recent Messages")
    for msg in messages[:10]:
        date = str(msg.get("date", ""))
        text = msg.get("text", "").replace("\\n", "\n")
        clean_text = pdf.clean_text(f"{date} - {text}")
        wrapped = textwrap.wrap(clean_text, width=100)
        for line in wrapped:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.page_width, 6, line, align='L')
        pdf.ln(1)

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# Download PDF
if st.button("📥 Download Summary PDF"):
    user_info = [
        f"Name: {user.get('first_name', '')} {user.get('last_name', '')}",
        f"Username: @{user.get('username', 'N/A')}",
        f"Phone: {user.get('phone', 'N/A')}",
        f"Total Messages: {len(messages)}",
        f"Top Sentiments: {', '.join(sentiment_df.index[:3])}"
    ]
    pdf_data = generate_pdf("User Summary Report", user_info, messages, point_wise_summary)
    st.download_button("📄 Download PDF", data=pdf_data, file_name="user_summary.pdf", mime="application/pdf")
