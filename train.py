# main.py
import os
import pandas as pd
import streamlit as st
from telethon.sync import TelegramClient
from pymongo import MongoClient
from dotenv import load_dotenv
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from fpdf import FPDF
import datetime
import base64
from io import BytesIO
import openai

openai.api_key="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"

# Load environment variables
load_dotenv()
api_id = int(os.getenv("TG_API_ID"))
api_hash = os.getenv("TELEGRAM_TOKEN")
mongo_uri = os.getenv("MONGO_URI")
llm_model = os.getenv("LLM_MODEL")

# MongoDB connection
mongo_client = MongoClient(mongo_uri)
db = mongo_client["telegram_data"]
messages_col = db["messages"]

# Sentiment and NER pipelines
sentiment_pipe = pipeline("sentiment-analysis")
ner_pipe = pipeline("ner", grouped_entities=True)

def scrape_telegram_data(entity_name):
    with TelegramClient('anon', api_id, api_hash) as client:
        entity = client.get_entity(entity_name)
        messages = []
        for msg in client.iter_messages(entity, limit=500):
            if msg.message:
                messages.append({
                    "user_id": msg.sender_id,
                    "text": msg.message,
                    "date": msg.date
                })
        messages_col.insert_many(messages)
        return len(messages)

def analyze_data():
    data = list(messages_col.find({}, {"_id": 0}))
    df = pd.DataFrame(data)
    df['sentiment'] = df['text'].apply(lambda x: sentiment_pipe(x)[0]['label'])
    df['entities'] = df['text'].apply(lambda x: [ent['word'] for ent in ner_pipe(x)])
    return df

def generate_wordcloud(df):
    text = ' '.join(df['text'].tolist())
    wordcloud = WordCloud(width=800, height=400).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def generate_summary(df, model="gpt-3.5-turbo"):
    # Join and truncate text from the DataFrame
    input_text = "\n".join(df['text'].astype(str).tolist()[:200])
    
    # Create prompt
    prompt = f"Summarize the following Telegram messages into a brief intelligence report:\n\n{input_text}"

    # Call OpenAI Chat API
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in summarizing social media and Telegram conversations for intelligence purposes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=800
    )

    # Return generated summary
    return response['choices'][0]['message']['content']

class PDF(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Telegram Intelligence Report", ln=True, align="C")
        self.ln(10)

    def add_section(self, title, content):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.set_font("Arial", size=11)
        self.multi_cell(0, 10, content)
        self.ln(5)

    def add_image(self, fig):
        img_buf = BytesIO()
        fig.savefig(img_buf, format='PNG')
        img_buf.seek(0)
        self.image(img_buf, w=180)

def generate_pdf(df, summary, fig):
    pdf = PDF()
    pdf.add_page()
    pdf.add_section("Summary", summary)
    pdf.add_section("Sentiment Stats", str(df['sentiment'].value_counts()))
    pdf.add_section("Named Entities", str(df['entities'].explode().value_counts().head(10)))
    pdf.add_image(fig)
    output_path = "telegram_report.pdf"
    pdf.output(output_path)
    return output_path

def show_pdf(path):
    with open(path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit UI
st.title("Telegram Data Intelligence Dashboard")
option = st.selectbox("Choose Action", ["Scrape Data", "Analyze & Generate Report"])

if option == "Scrape Data":
    entity = st.text_input("Enter Telegram group/channel username")
    if st.button("Scrape") and entity:
        count = scrape_telegram_data(entity)
        st.success(f"Scraped and stored {count} messages.")

elif option == "Analyze & Generate Report":
    df = analyze_data()
    summary = generate_summary(df)
    st.subheader("Summary")
    st.write(summary)
    fig = generate_wordcloud(df)
    st.pyplot(fig)
    pdf_path = generate_pdf(df, summary, fig)
    show_pdf(pdf_path)
    st.download_button("Download PDF", data=open(pdf_path, "rb"), file_name="report.pdf")
