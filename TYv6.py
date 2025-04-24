# import streamlit as st
# import os, re, textwrap, unicodedata
# from io import BytesIO
# from pymongo import MongoClient
# from fpdf import FPDF
# from dotenv import load_dotenv
# from litellm import completion
# from transformers import pipeline
# import pandas as pd
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# from langdetect import detect
# from datetime import datetime

# # ---------------- Load .env ----------------
# load_dotenv()
# LLM_MODEL = os.getenv("LLM_MODEL")
# MONGO_URI = os.getenv("MONGO_URI")
# CHUNK_SIZE = 10000
# LANG = "English"
# FONT_PATH = "E:/VATINS/fonts"

# # ---------------- MongoDB Connection ----------------
# client = MongoClient(MONGO_URI)
# messages_col = client["telegram_data"]["messages"]
# users_col = client["telegram_data"]["users"]

# # ---------------- NLP Pipelines ----------------
# sentiment_model = pipeline("sentiment-analysis")
# ner_model = pipeline("ner", grouped_entities=True)
# flagged_words = ["porn", "xxx", "teen", "casino", "1xbet", "betting", "18+", "adult"]

# # ---------------- PDF Class ----------------
# # class PDF(FPDF):
# #     def __init__(self):
# #         super().__init__()
# #         self.add_font("DejaVu", "", os.path.join(FONT_PATH, "DejaVuSans.ttf"), uni=True)
# #         self.add_font("DejaVu", "B", os.path.join(FONT_PATH, "DejaVuSans-Bold.ttf"), uni=True)
# #         self.set_font("DejaVu", "", 12)
        
# #         # Set A4 size (210mm x 297mm)
# #         self.set_auto_page_break(auto=True, margin=15) # Ensure the page is set to A4 size

# #     def safe_text(self, text):
# #         text = unicodedata.normalize("NFKD", text)
# #         text = re.sub(r'(\S{80,})', r'\1 ', text)  # Handle long words
# #         text = ''.join(char if unicodedata.category(char)[0] != 'C' else ' ' for char in text)
# #         return text

# #     def adjust_font_size(self, text):
# #         """Dynamically adjust font size based on content length."""
# #         text_width = self.get_string_width(text)
# #         if text_width > (self.w - 2 * self.l_margin):  # If text is too wide
# #             self.set_font("DejaVu", "", 10)  # Reduce font size
# #         else:
# #             self.set_font("DejaVu", "", 12)  # Use default font size

# #     def body(self, text, line_height=8):
# #         if not text:
# #             text = "N/A"
# #         clean_text = self.safe_text(str(text))

# #         lines = textwrap.wrap(clean_text, width=70)  # Wrap text into lines with safe width
# #         max_width = self.w - 2 * self.l_margin  # Maximum width for the text

# #         for line in lines:
# #             # Check if the line is too long to fit the page width
# #             if self.get_string_width(line) > max_width:
# #                 # Reduce font size if necessary
# #                 self.set_font("DejaVu", "", 10)
# #             else:
# #                 self.set_font("DejaVu", "", 12)

# #             try:
# #                 self.multi_cell(150, line_height, txt=line, align="L")  # Wrap text into cells
# #             except Exception as e:
# #                 print(f"Error rendering text: {e}")
# #                 self.multi_cell(150, line_height, txt="[Content too large to render]", align="L")
    
# #     def insert_image(self, image_path, title):
# #         self.set_font("DejaVu", "B", 14)
# #         self.cell(80, 5, title, ln=True)
# #         self.image(image_path, w=150)  # Ensure this fits in the page width
# #         self.ln(3)


# # def generate_pdf(summary, sentiment_text, ner_results, flagged_keywords, sentiment_img, wordcloud_img):
# #     pdf = PDF()
# #     pdf.add_page(format='A4')

# #     sections = [
# #         ("Telegram Intelligence Summary", "", 16),
# #         ("LLM Summary", summary, 14),
# #         ("Sentiment Summary", sentiment_text, 14),
# #         ("Named Entities", "\n".join([f"{x[0]} ({x[1]})" for x in ner_results[:10]]) or "No entities found", 14),
# #         ("Flagged Words", ", ".join(flagged_keywords) if flagged_keywords else "No flagged content detected.", 14),
# #     ]

# #     for title, content, size in sections:
# #         pdf.set_font("DejaVu", "B", size)
# #         pdf.cell(150, 5, title, ln=True)
# #         pdf.set_font("DejaVu", "", 12)
# #         pdf.body(content)
# #         pdf.ln(3)

# #     if sentiment_img: pdf.insert_image(sentiment_img, "Sentiment Chart")
# #     if wordcloud_img: pdf.insert_image(wordcloud_img, "Word Cloud")

# #     buf = BytesIO()
# #     pdf.output(buf)
# #     buf.seek(0)
# #     return buf


# class PDF(FPDF):
#     def __init__(self):
#         super().__init__()
#         self.add_font("DejaVu", "", os.path.join(FONT_PATH, "DejaVuSans.ttf"), uni=True)
#         self.add_font("DejaVu", "B", os.path.join(FONT_PATH, "DejaVuSans-Bold.ttf"), uni=True)
#         self.set_font("DejaVu", "", 12)
#         self.set_auto_page_break(auto=True, margin=15)

#     def safe_text(self, text):
#         text = unicodedata.normalize("NFKD", text)
#         text = re.sub(r'(\S{80,})', r'\1 ', text)
#         text = ''.join(char if unicodedata.category(char)[0] != 'C' else ' ' for char in text)
#         return text

#     def section_title(self, title, size=14):
#         self.set_font("DejaVu", "B", size)
#         self.cell(150, 10, title, ln=True, align="L")
#         self.ln(3)

#     def body(self, text, line_height=8):
#         if not text:
#             text = "N/A"
#         clean_text = self.safe_text(str(text))
#         lines = textwrap.wrap(clean_text, width=450)
#         max_width = self.w - 3 * self.l_margin

#         for line in lines:
#             if self.get_string_width(line) > max_width:
#                 self.set_font("DejaVu", "", 10)
#             else:
#                 self.set_font("DejaVu", "", 12)
#             self.multi_cell(150, line_height, txt=line, align="L")
#         self.ln(5)

#     def insert_image(self, image_path, title):
#         self.section_title(title)
#         self.image(image_path, w=self.w - 2 * self.l_margin)
#         self.ln(5)

# def generate_pdf(summary, sentiment_text, ner_results, flagged_keywords, sentiment_img, wordcloud_img):
#     pdf = PDF()
#     pdf.add_page()

#     sections = [
#         ("Telegram Intelligence Summary", "", 16),
#         ("LLM Summary", summary, 14),
#         ("Sentiment Summary", sentiment_text, 14),
#         ("Named Entities", "\n".join([f"{x[0]} ({x[1]})" for x in ner_results[:10]]) or "No entities found", 14),
#         ("Flagged Words", ", ".join(flagged_keywords) if flagged_keywords else "No flagged content detected.", 14),
#     ]

#     for title, content, size in sections:
#         pdf.section_title(title, size)
#         pdf.body(content)

#     if sentiment_img:
#         pdf.insert_image(sentiment_img, "Sentiment Chart")
#     if wordcloud_img:
#         pdf.insert_image(wordcloud_img, "Word Cloud")

#     buf = BytesIO()
#     pdf.output(buf)
#     buf.seek(0)
#     return buf



# # ---------------- LLM Summarization ----------------
# def call_gpt_api(prompt):
#     try:
#         res = completion(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
#         return res.choices[0].message.content.strip()
#     except Exception as e:
#         return f"[LLM Error] {e}"

# def summarize_texts(texts):
#     chunks, chunk = [], ""
#     for line in texts:
#         if len(chunk) + len(line) < CHUNK_SIZE:
#             chunk += line + " "
#         else:
#             chunks.append(chunk.strip())
#             chunk = line + " "
#     if chunk: chunks.append(chunk.strip())
#     return "\n\n".join(call_gpt_api(f"Summarize in {LANG}:\n{c}") for c in chunks)

# # ---------------- Streamlit UI ----------------
# st.set_page_config(page_title="Telegram Summarizer", layout="wide")
# st.title("Telegram Intelligence Summarizer")

# # Dropdown for user
# user_map = {
#     f"{u.get('first_name', '')} {u.get('last_name', '')} (@{u.get('username', '')})": u["user_id"]
#     for u in users_col.find({}, {"user_id": 1, "first_name": 1, "last_name": 1, "username": 1})
# }
# selected_user = st.selectbox("Select Telegram User", list(user_map.keys()))
# user_id = user_map[selected_user]

# @st.cache_data
# def get_user_texts(user_id):
#     messages = list(messages_col.find({"user_id": user_id}))
#     texts = [m.get("text", "") for m in messages if isinstance(m.get("text", ""), str) and len(m.get("text", "")) > 10]
#     dates = [m.get("date") for m in messages if "date" in m]
#     return texts, dates, messages

# texts, dates, messages = get_user_texts(user_id)
# all_text = " ".join(texts)

# if not texts:
#     st.warning("No valid messages found.")
#     st.stop()

# st.markdown(f"**Total messages:** {len(texts)}")

# # --- LLM Summary ---
# st.subheader("LLM Summary")
# summary = summarize_texts(texts)
# st.success(summary)

# # --- Sentiment ---
# st.subheader("Sentiment Analysis")
# sentiments = [sentiment_model(t[:512])[0]["label"] for t in texts]
# sentiment_series = pd.Series(sentiments).value_counts()
# sentiment_chart = st.bar_chart(sentiment_series)
# sentiment_text = ", ".join([f"{k}: {v}" for k, v in sentiment_series.items()])

# # Save sentiment chart to image
# sentiment_img = "sentiment_chart.png"
# plt.figure(figsize=(6, 3))
# sentiment_series.plot(kind="bar", color="skyblue")
# plt.title("Sentiment Distribution")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.savefig(sentiment_img)
# plt.close()

# # --- Word Cloud ---
# st.subheader("Word Cloud")
# wordcloud_img = "wordcloud.png"
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.savefig(wordcloud_img)
# st.image(wordcloud_img)

# # --- Timeline ---
# st.subheader("Message Timeline")
# if dates:
#     df_dates = pd.to_datetime(dates)
#     st.line_chart(df_dates.value_counts().sort_index())

# # --- NER ---
# st.subheader("Named Entities")
# ner_results = list(set((ent['word'], ent['entity_group']) for t in texts[:50] for ent in ner_model(t[:512])))
# df_ner = pd.DataFrame(ner_results, columns=["Entity", "Type"])
# if not df_ner.empty:
#     st.dataframe(df_ner.value_counts().reset_index(name="Count"))
# else:
#     st.info("No named entities found.")

# # --- Flagged ---
# st.subheader("Flagged Content")
# flagged = [w for w in flagged_words if w in all_text.lower()]
# flagged_msgs = [t for t in texts if any(fw in t.lower() for fw in flagged_words)]
# if flagged:
#     st.error(f"Flagged keywords: {', '.join(flagged)}")
#     st.markdown("**Examples:**")
#     for msg in flagged_msgs[:5]:
#         st.warning(msg)
# else:
#     st.success("No flagged content.")

# # --- PDF Download ---
# st.download_button(
#     "Download PDF Report",
#     data=generate_pdf(summary, sentiment_text, ner_results, flagged, sentiment_img, wordcloud_img),
#     file_name="telegram_summary.pdf",
#     mime="application/pdf"
# )


import streamlit as st
from pymongo import MongoClient
from telegram import Bot
import pandas as pd
import spacy
import openai
from sentence_transformers import SentenceTransformer, util
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from langdetect import detect
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
from wordcloud import WordCloud
import google.generativeai as genai
from deep_translator import GoogleTranslator

# ---------- CONFIG ---------- #
st.set_page_config(page_title="Telegram AI Profiler", layout="wide")
TELEGRAM_TOKEN = "ceb559c6275cfe6fd2297547f0384da3"  # Replace with your token
GEMINI_API_KEY = "AIzaSyCmrkhBetcpDbunHe-Ih2uLFjo6zgUiw58"


# Log in with your token
login("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")

# Load the gated Mistral model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)

# ---------- INIT GEMINI ---------- #
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

def summarize_with_gemini(text: str) -> str:
    try:
        response = gemini_model.generate_content(f"Summarize the following Telegram messages:\n\n{text}")
        return response.text
    except Exception as e:
        return f"Gemini summarization failed: {e}"

# ---------- LOAD LOCAL LLM ---------- #
@st.cache_resource
def load_local_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    return tokenizer, model

tokenizer, model = load_local_llm()

def summarize_with_local_llm(text: str, max_len=512):
    prompt = f"Summarize this Telegram content:\n\n{text[:max_len]}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------- INIT NLP + DB ---------- #
nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
geolocator = Nominatim(user_agent="geoapi")

client = MongoClient("mongodb://localhost:27017")
db = client["telegram_data"]
messages_col = db["messages"]
users_col = db["users"]
groups_col = db["groups"]

# ---------- STREAMLIT UI ---------- #
st.title("📡 Telegram AI Intelligence Dashboard")
user_id = st.text_input("Enter Telegram User ID:", value="7295041924")

if user_id:
    user = users_col.find_one({"user_id": int(user_id)})
    messages = list(messages_col.find({"user_id": int(user_id)}))
    df = pd.DataFrame(messages)

    if df.empty:
        st.warning("No messages found for this user.")
        st.stop()

    df["text"] = df["text"].astype(str)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "👥 Group Behavior", "📆 Incident Timeline", "🌐 Network Graph"])

    with tab1:
        st.header("📊 Summary Insights")

        # Sentiment Analysis (basic)
        sentiments = {"positive": 0, "negative": 0, "neutral": 0}
        for text in df["text"]:
            if "good" in text: sentiments["positive"] += 1
            elif "fraud" in text or "scam" in text: sentiments["negative"] += 1
            else: sentiments["neutral"] += 1

        # Named Entity Recognition
        entities = []
        for text in df["text"].sample(min(50, len(df))):
            doc = nlp(text)
            entities.extend([ent.text for ent in doc.ents])

        # Topic Detection
        embeddings = embed_model.encode(df["text"].tolist(), convert_to_tensor=True)
        top_keywords = set()
        for i in range(len(df)):
            sim = util.semantic_search(embeddings[i:i+1], embeddings, top_k=3)
            for match in sim[0]:
                top_keywords.update(df.iloc[match["corpus_id"]]["text"].split()[:3])

        # ---------- Word Cloud ----------
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df["text"]))
        st.image(wordcloud.to_array(), caption="Word Cloud", use_column_width=True)

        # ---------- Language Detection ----------
        langs = [detect(t) for t in df["text"].sample(min(50, len(df))) if t.strip()]
        lang_count = pd.Series(langs).value_counts()
        st.bar_chart(lang_count)


        # Gemini Summary
        try:
            sample_text = " ".join(df["text"].sample(min(30, len(df))))
            gemini_summary = summarize_with_gemini(sample_text)
        except Exception as e:
            gemini_summary = f"Gemini summarization failed: {e}"

        # Local Model Summary
        try:
            local_llm_summary = summarize_with_local_llm(sample_text)
        except Exception as e:
            local_llm_summary = f"Local model summarization failed: {e}"

        # Geo Analysis
        locations = []
        for msg in messages:
            if "location" in msg:
                loc = msg["location"]
                coord = (loc["latitude"], loc["longitude"])
                try:
                    address = geolocator.reverse(coord, language="en")
                    locations.append(address.address)
                except:
                    pass

        st.markdown(f"**Username:** {user.get('username', 'N/A')}")
        st.markdown(f"**Total Messages:** {len(df)}")
        st.json({"Languages": lang_count, "Sentiment": sentiments})

        st.markdown("### 🧠 Gemini Summary")
        st.write(gemini_summary)

        st.markdown("### 🧠 Local LLM Summary")
        st.write(local_llm_summary)

        st.markdown("### 🏷️ Top Keywords")
        st.write(list(top_keywords)[:15])

        st.markdown("### 🧾 Entities")
        st.write(list(set(entities))[:15])

        if locations:
            st.markdown("### 📍 Detected Locations")
            st.write(locations)

        def generate_pdf():
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=A4)
            width, height = A4
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 50, f"User: {user_id} Summary Report")
            c.drawString(50, height - 80, "Top Entities:")
            y = height - 100
            for e in list(set(entities))[:10]:
                c.drawString(70, y, e)
                y -= 20
            c.drawString(50, y - 20, "Top Keywords:")
            y -= 40
            for k in list(top_keywords)[:10]:
                c.drawString(70, y, k)
                y -= 20
            c.save()
            buffer.seek(0)
            return buffer

        st.download_button("📥 Download PDF Report", data=generate_pdf(), file_name="telegram_user_summary.pdf", mime="application/pdf")

    with tab2:
        st.header("👥 Group Behavior")
        group_msgs = messages_col.find({"chat_id": {"$exists": True}, "user_id": int(user_id)})
        group_df = pd.DataFrame(group_msgs)
        group_df["date"] = pd.to_datetime(group_df["date"])
        st.write(group_df[["chat_id", "date", "text"]].head(50))

    with tab3:
        st.header("📆 Incident Timeline")
        df["date"] = pd.to_datetime(df["date"])
        timeline = df.groupby(df["date"].dt.date).size()
        st.line_chart(timeline)

    with tab4:
        st.header("🌐 Communication Graph")
        G = nx.Graph()
        for msg in messages[:100]:
            user_node = str(msg["user_id"])
            chat_node = str(msg.get("chat_id", "unknown"))
            G.add_node(user_node, label="User")
            G.add_node(chat_node, label="Chat")
            G.add_edge(user_node, chat_node)

        net = Network(notebook=False, height="400px", width="100%")
        net.from_nx(G)
        net.save_graph("graph.html")
        with open("graph.html", 'r', encoding='utf-8') as f:
            html = f.read()
            components.html(html, height=500)


# # ---------- LOAD LOCAL LLM ---------- #
# @st.cache_resource
# def load_local_llm():
#     model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
#     return tokenizer, model

# tokenizer, model = load_local_llm()

# def summarize_with_local_llm(text: str, max_len=512):
#     prompt = f"Summarize this Telegram content:\n\n{text[:max_len]}"
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # ---------- INIT NLP + DB ---------- #
# nlp = spacy.load("en_core_web_sm")
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# geolocator = Nominatim(user_agent="geoapi")

# client = MongoClient("mongodb://localhost:27017")
# db = client["telegram_data"]
# messages_col = db["messages"]
# users_col = db["users"]
# groups_col = db["groups"]

# # ---------- STREAMLIT UI ---------- #
# st.title("📡 Telegram AI Intelligence Dashboard")
# user_id = st.text_input("Enter Telegram User ID:")

# if user_id:
#     user = users_col.find_one({"user_id": int(user_id)})
#     messages = list(messages_col.find({"user_id": int(user_id)}))
#     df = pd.DataFrame(messages)

#     if df.empty:
#         st.warning("No messages found for this user.")
#         st.stop()

#     df["text"] = df["text"].astype(str)

#     tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "👥 Group Behavior", "📆 Incident Timeline", "🌐 Network Graph"])

#     with tab1:
#         st.header("📊 Summary Insights")

#         # ---------- Sentiment Analysis ----------
#         sentiments = {"positive": 0, "negative": 0, "neutral": 0}
#         for text in df["text"]:
#             if "good" in text: sentiments["positive"] += 1
#             elif "fraud" in text or "scam" in text: sentiments["negative"] += 1
#             else: sentiments["neutral"] += 1

#         fig, ax = plt.subplots()
#         ax.pie(sentiments.values(), labels=sentiments.keys(), autopct='%1.1f%%', startangle=90)
#         st.pyplot(fig)

#         # ---------- Named Entity Recognition ----------
#         entities = []
#         for text in df["text"].sample(min(50, len(df))):
#             doc = nlp(text)
#             entities.extend([ent.text for ent in doc.ents])

#         # ---------- Topic Detection ----------
#         embeddings = embed_model.encode(df["text"].tolist(), convert_to_tensor=True)
#         top_keywords = set()
#         for i in range(len(df)):
#             sim = util.semantic_search(embeddings[i:i+1], embeddings, top_k=3)
#             for match in sim[0]:
#                 top_keywords.update(df.iloc[match["corpus_id"]]["text"].split()[:3])

#         # ---------- Word Cloud ----------
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df["text"]))
#         st.image(wordcloud.to_array(), caption="Word Cloud", use_column_width=True)

#         # ---------- Language Detection ----------
#         langs = [detect(t) for t in df["text"].sample(min(50, len(df))) if t.strip()]
#         lang_count = pd.Series(langs).value_counts()
#         st.bar_chart(lang_count)

#         # ---------- GPT & Local LLM Summary ----------
#         try:
#             sample_text = " ".join(df["text"].sample(min(30, len(df))))
#             response = openai.ChatCompletion.create(
#                 model="gpt-3.5-turbo",
#                 messages=[{"role": "user", "content": f"Summarize this: {sample_text}"}],
#             )
#             gpt_summary = response['choices'][0]['message']['content']
#         except Exception:
#             gpt_summary = "GPT summarization failed."

#         try:
#             local_llm_summary = summarize_with_local_llm(sample_text)
#         except Exception as e:
#             local_llm_summary = f"Local model summarization failed: {e}"

#         # ---------- Geo Analysis ----------
#         locations = []
#         for msg in messages:
#             if "location" in msg:
#                 loc = msg["location"]
#                 coord = (loc["latitude"], loc["longitude"])
#                 try:
#                     address = geolocator.reverse(coord, language="en")
#                     locations.append(address.address)
#                 except:
#                     pass

#         # ---------- Display Results ----------
#         st.markdown(f"**Username:** {user.get('username', 'N/A')}")
#         st.markdown(f"**Total Messages:** {len(df)}")
#         st.markdown("### 🧠 GPT Summary")
#         st.write(gpt_summary)
#         st.markdown("### 🧠 Local LLM Summary")
#         st.write(local_llm_summary)
#         st.markdown("### 🏷️ Top Keywords")
#         st.write(list(top_keywords)[:15])
#         st.markdown("### 🧾 Entities")
#         st.write(list(set(entities))[:15])
#         if locations:
#             st.markdown("### 📍 Detected Locations")
#             st.write(locations)

#         # ---------- Translation Support ----------
#         with st.expander("🌐 Translate Top Messages"):
#             lang = st.selectbox("Translate to:", ["hi", "ur", "te", "ml", "ar", "bn"])
#             for text in df["text"].sample(min(10, len(df))):
#                 try:
#                     translated = GoogleTranslator(source='auto', target=lang).translate(text)
#                     st.markdown(f"**Original:** {text}\n\n**Translated:** {translated}")
#                 except Exception as e:
#                     st.warning(f"Translation failed: {e}")

#         # ---------- PDF Export ----------
#         def generate_pdf():
#             buffer = BytesIO()
#             c = canvas.Canvas(buffer, pagesize=A4)
#             width, height = A4
#             c.setFont("Helvetica", 12)
#             c.drawString(50, height - 50, f"User: {user_id} Summary Report")
#             c.drawString(50, height - 80, "Top Entities:")
#             y = height - 100
#             for e in list(set(entities))[:10]:
#                 c.drawString(70, y, e)
#                 y -= 20
#             c.drawString(50, y - 20, "Top Keywords:")
#             y -= 40
#             for k in list(top_keywords)[:10]:
#                 c.drawString(70, y, k)
#                 y -= 20
#             c.save()
#             buffer.seek(0)
#             return buffer

#         st.download_button("📥 Download PDF Report", data=generate_pdf(), file_name="telegram_user_summary.pdf", mime="application/pdf")

#     with tab2:
#         st.header("👥 Group Behavior")
#         group_msgs = messages_col.find({"chat_id": {"$exists": True}, "user_id": int(user_id)})
#         group_df = pd.DataFrame(group_msgs)
#         group_df["date"] = pd.to_datetime(group_df["date"])
#         st.write(group_df[["chat_id", "date", "text"]].head(50))

#     with tab3:
#         st.header("📆 Incident Timeline")
#         df["date"] = pd.to_datetime(df["date"])
#         timeline = df.groupby(df["date"].dt.date).size()
#         st.line_chart(timeline)

#     with tab4:
#         st.header("🌐 Communication Graph")
#         G = nx.Graph()
#         for msg in messages[:100]:
#             user_node = str(msg["user_id"])
#             chat_node = str(msg.get("chat_id", "unknown"))
#             G.add_node(user_node, label="User")
#             G.add_node(chat_node, label="Chat")
#             G.add_edge(user_node, chat_node)

#         net = Network(notebook=False, height="400px", width="100%")
#         net.from_nx(G)
#         net.save_graph("graph.html")
#         with open("graph.html", 'r', encoding='utf-8') as f:
#             html = f.read()
#             components.html(html, height=500)








