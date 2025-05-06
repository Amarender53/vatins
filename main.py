from fastapi import FastAPI, HTTPException, Request, Query, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pymongo import MongoClient, ASCENDING
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from deep_translator import GoogleTranslator
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Any, Callable
from fpdf import FPDF
import pandas as pd
import traceback
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
import spacy
import locale
import torch
import re
import os
from collections import defaultdict
from bson import ObjectId
from googleapiclient.discovery import build
from dotenv import load_dotenv
import warnings
import logging  # ✅ Standard Python logging
from transformers import logging as hf_logging
import pytz

ist = pytz.timezone('Asia/Kolkata')
timestamp_ist = datetime.now(ist).isoformat()

#router = APIRouter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ------------------- Setup ------------------- #

stop_words = set(stopwords.words('english'))
words = word_tokenize("This is a sample sentence.")
filtered = [w for w in words if w.isalpha() and w not in stop_words]

try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    locale.setlocale(locale.LC_ALL, '')

def format_number(n):
    try:
        return f"{int(n):,}"
    except:
        return n

def convert_objectid(doc):
    if isinstance(doc, dict):
        for k, v in doc.items():
            if isinstance(v, ObjectId):
                doc[k] = str(v)
            elif isinstance(v, datetime):
                doc[k] = v.isoformat()
    return doc
load_dotenv()
API_KEY = "12345"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost:3000",
    "http://localhost:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=50000, socketTimeoutMS=50000)
db = client["telegram_scraper"]
channels_collection = db["channels"]     # YouTube
videos_collection = db["videos"]         # YouTube
comments_collection = db["comments"]     # YouTube
groups_collection = db["telegram_groups"]         # Telegram
messages_collection = db["telegram_messages"]     # Telegram
users_collection = db["telegram_users"]           #Telegram

# pull entire collections into pandas
groups_df   = pd.DataFrame(list(groups_collection.find()))
users_df    = pd.DataFrame(list(users_collection.find()))
messages_df = pd.DataFrame(list(messages_collection.find()))

# ---- YouTube & Telegram Specific Sentiment Models ----
yt_sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tg_sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# ---- NER Pipelines ----
yt_ner_analyzer = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
tg_ner_analyzer = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl", grouped_entities=True)

analyzer = SentimentIntensityAnalyzer()

# ---- Summarizer ----
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")
MAX_LEN = tokenizer.model_max_length
MAX_TOKENS = 10000

# ---- Semantic Search Model ----
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Language Translator ----
translator = GoogleTranslator(source='auto', target='en')

# ---- VADER Sentiment (for rule-based English sentiment scoring) ----
sentiment_analyzer = SentimentIntensityAnalyzer()


fraud_kw   = ['scam','fraud','usdt','loan','bitcoin','crypto','investment','scheme', "rug pull", "pump and dump", "fake airdrop", "private key", "wallet drain"]
bank_kw    = ['sbi','canara','hdfc','bank','upi','ifsc']
betting_kw = ['bet','gamble','casino','wager','odds','betting']
porn_kw    = ['porn','xxx','nsfw','adult','sex','milf','dp','nude', 'naked', 'topless', 'thong','onlyfans', 'fml']
TECH_KEYWORDS = ["python", "docker", "kubernetes", "aws", "azure", "react", "nodejs", "terraform"]
lang_labels = {"en": "English", "zh-cn": "Chinese", "zh": "Chinese", "hi": "Hindi"}

# regex for URLs / handles
URL_RE     = re.compile(r"https?://\S+")
HANDLE_RE  = re.compile(r"@[\w\d_]+")
_money_rx = re.compile(r'\b\d{1,3}(?:[,\d{3}]*)(?:\.\d+)?\s?(?:usd|inr|usdt|   ^b   |\$)?\b', re.IGNORECASE)
_phone_rx = re.compile(r'\b(?:\+?\d{1,3})?[-.\s]?\d{3}[-.\s]?\d{3,4}\b')

# define the categories you care about
def detect_flags(texts: List[str], keywords: List[str]) -> List[str]:
    matched = set()
    for text in texts:
        for kw in keywords:
            if kw.lower() in text.lower():
                matched.add(kw)
    return list(matched)

# NLP model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Your predefined category labels
CATEGORY_LABELS = ["fraud", "banking", "gambling", "adult", "hate speech", "politics", "religion", "spam"]

def detect_categories(texts: List[str], threshold: float = 0.5) -> List[str]:
    """
    Uses zero-shot classification to assign categories to a group of texts.
    Returns all labels whose confidence exceeds the threshold.
    """
    if not texts:
        return []

    # Concatenate a sample of the first N messages to summarize the group
    snippet = " ".join(texts[:20])[:1024]  # BART models have a 1024-token limit

    result = classifier(snippet, candidate_labels=CATEGORY_LABELS, multi_label=True)
    return [label for label, score in zip(result["labels"], result["scores"]) if score >= threshold]

def generate_group_topic_summary(topics: List[str]) -> str:
    if not topics:
        return "No clear topics identified for summarization."

    lower_topics = [t.lower() for t in topics]

    narrative = []

    # Section 1: Fraud/Financial
    if any(k in lower_topics for k in ["sbi", "bank", "otp", "details", "upi"]):
        narrative.append("Financial terms such as bank names and OTPs were mentioned, indicating possible sharing of sensitive or suspicious information.")

    # Section 2: Casual Chat
    if any(k in lower_topics for k in ["hi", "messages", "content", "process", "bro"]):
        narrative.append("There are casual or conversational terms suggesting regular chatting among group members.")

    # Section 3: Betting/Promotions
    if any(k in lower_topics for k in ["betting", "promotions", "cheyu"]):
        narrative.append("Terms related to betting and promotions point to possible gambling-related discussions.")

    # Section 4: Violence/Concern
    if any(k in lower_topics for k in ["died", "injured", "terrorist", "attack", "morning", "men"]):
        narrative.append("Mentions of violent or traumatic events raise potential concerns for safety or extremism.")

    # Section 5: Planning
    if any(k in lower_topics for k in ["22nd", "monna", "gurinchi"]):
        narrative.append("Temporal and planning references suggest coordination or discussions around specific events or dates.")

    return " ".join(narrative) if narrative else "No significant narrative could be inferred from the current topics."


def translate(text: str) -> str:
    try:
        if detect(text) != 'en':
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except Exception:
        return text

def categorize_sentiment(texts: List[str]) -> Dict[str, List[str]]:
    analyzer = SentimentIntensityAnalyzer()
    categories = {"positive": [], "negative": [], "neutral": []}
    for text in texts:
        score = analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            categories["positive"].append(text)
        elif score <= -0.05:
            categories["negative"].append(text)
        else:
            categories["neutral"].append(text)
    return categories

def load_messages(chat_id: str) -> List[str]:
    client = MongoClient("mongodb://localhost:27017")
    db = client["telegram_scraper"]
    messages_collection = db["telegram_messages"]
    
    messages = messages_collection.find({"chat_id": chat_id})
    return [msg.get("text", "") for msg in messages if "text" in msg]

def assess_risk(report: Dict) -> str:
    flags = report.get("behavioral_flags", [])
    fraud = report.get("fraud_signals", [])
    if len(flags) > 2 or len(fraud) > 1:
        return "High"
    elif len(flags) > 0 or len(fraud) > 0:
        return "Medium"
    return "Low"

def summarize_group(chat_id: str):
    messages = load_messages(chat_id)
    translated_texts = [translate(msg) for msg in messages]
    categories = detect_categories(translated_texts, threshold=0.6)

    # Stub: Load group title if needed
    grp_title = f"Group {chat_id}"

    return {
        "detected_categories": categories,
        "overall_summary": (
            f"Group '{grp_title}' covers {', '.join(categories)} topics, "
            f"among others."
        )
    }

def get_user_group_details_by_chat_ids(chat_ids: List[Any]) -> List[Dict[str, Any]]:
    group_infos = []
    for group in groups_collection.find({"chat_id": {"$in": chat_ids + [str(cid) for cid in chat_ids if isinstance(cid, int)]}}):
        count = messages_collection.count_documents({"chat_id": group["chat_id"]})
        group_infos.append({
            "chat_id": str(group["chat_id"]),
            "title": group.get("title", "Unknown"),
            "total_messages": count
        })
    return group_infos

def generate_sentiment_wordclouds_from_categories(
    sentiments: Dict[str, List[str]], output_dir: str = "sentiment_clouds"
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    file_paths = {}

    for category, texts in sentiments.items():
        if texts:
            wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(texts))
            file_path = os.path.join(output_dir, f"{category}_wordcloud.png")
            wc.to_file(file_path)
            file_paths[category] = file_path

    return file_paths

def generate_creative_narrative_summary(sentiments: Dict[str, List[str]]) -> str:
    pos = sentiments["positive"]
    neg = sentiments["negative"]
    neu = sentiments["neutral"]

    if len(pos) > len(neg) + len(neu):
        return "The channel radiates positivity, with audiences consistently responding with praise, appreciation, and supportive sentiments. Emotional tone is generally uplifting."
    elif len(neg) > len(pos):
        return "A noticeable degree of criticism or discontent is evident among the audience, pointing to polarized reactions or controversial content themes."
    elif len(neu) > max(len(pos), len(neg)):
        return "The audience expresses themselves in neutral or ambiguous ways, suggesting informational engagement rather than emotional involvement."
    else:
        return "The sentiment around this channel is diverse—ranging from admiration to critique—reflecting a multifaceted viewer base with varied perspectives."

class TimeBoundRequest(BaseModel):
    channel_id: str
    time_filter: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class ProfileRequest(BaseModel):
    channel_id: str

def generate_group_paragraph_summary(topics: List[str], flags: List[str], languages: List[str], total_messages: int, participants_count: int) -> List[Dict[str, str]]:
    return [
        {
            "title": "Topic Overview",
            "content": f"The group frequently discusses: {', '.join(topics)}. These appear to be dominant themes."
        },
        {
            "title": "Communication Behavior",
            "content": f"The group has {participants_count} participants and {total_messages} messages. Flags detected: {', '.join(flags) or 'None'}."
        },
        {
            "title": "Linguistic Profile",
            "content": f"Languages used include: {', '.join(languages)}. This suggests possible multicultural interaction."
        }
    ]

# Request model
class ChannelRequest(BaseModel):
    channel_id: str

class VideoRequest(BaseModel):
    channel_id: str
    video_id: str

class TelegramDashboardResponse(BaseModel):
    total_groups: int
    total_users: int
    active_users: int
    total_messages: int
    message_rate: int
    message_rate_change: float
    group_propagation_avg: float
    total_media_files: int
    avg_views_per_message: float
    
    
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(200, 10, 'Telegram User Intelligence Report', ln=True, align='C')
        self.ln(10)

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(2)

    def section_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 10, body)
        self.ln()

class Comment(BaseModel):
    text: str
    likes: int
    commented_at: str
    channel_name: str

class UserSummary(BaseModel):
    telegram_id: str
    username: str
    full_name: str               #    ^f^p new
    first_name: Optional[str]
    last_name:  Optional[str]
    phone:      Optional[str]
    status:    Optional[str]
    bio:       str
    group_count: int
    user_total_messages: int
    groups:    List[str]
    last_active: Optional[datetime]
    languages: List[str]
    interests: List[str]
    suspicious_indicators: List[str]
    communication_topics: List[str]
    affiliations: List[str]
    digital_footprint: List[str]
    technology_usage: List[str]
    recent_messages: List[str]
    overall_summary: str

class MessageSummaryRequest(BaseModel):
    user_id:    Optional[str] = None
    time_filter: str
    start_date: Optional[str] = None
    end_date:   Optional[str] = None

def get_time_bounds(time_filter: str, start_date: Optional[str], end_date: Optional[str]) -> (Optional[datetime], Optional[datetime]):
    now = datetime.utcnow()
    if time_filter == "last_24_hours":
        return now - timedelta(days=1), now
    elif time_filter == "last_week":
        return now - timedelta(weeks=1), now
    elif time_filter == "last_month":
        return now - timedelta(days=    30), now
    elif time_filter == "custom":
        if not start_date or not end_date:
            raise HTTPException(status_code=400, detail="Custom range requires start_date and end_date.")
        return datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d")
    return None, None

def fetch_comments_from_db(channel_id: str) -> List[Dict]:
    comments = comments_collection.find({"channel_id": channel_id})
    return list(comments)

def fetch_comments_by_video_id(video_id: str) -> List[Dict]:
    comments = comments_collection.find({"video_id": video_id})
    return list(comments)

def channel_exists(channel_id: str) -> bool:
    return channels_collection.find_one({"channel_id": channel_id}) is not None

def generate_group_narrative_summary(topics, flags, langs):
    parts = []
    if topics:
        parts.append(f"The group actively discusses subjects like {', '.join(topics[:5])}.")
    if flags:
        parts.append(f"Behavioral signals include: {', '.join(flags)}.")
    if langs:
        parts.append(f"Users are communicating in {', '.join(langs)}, suggesting linguistic diversity.")
    return " ".join(parts) if parts else "No significant narrative could be generated."

def generate_group_overall_summary(title, users, messages, topics, flags, languages):
    summary = f"Group '{title}' contains {users} participants with {messages} messages exchanged."
    if topics:
        summary += f" Common topics involve: {', '.join(topics[:5])}."
    if flags:
        summary += f" Noteworthy behavioral flags include: {', '.join(flags)}."
    if languages:
        summary += f" Detected languages include {', '.join(languages)}."
    return summary


def preprocess_text(text: str) -> str:
    try:
        if not text.strip():
            return ""
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated.strip()
    except:
        return text.strip()

class ChannelInfo(BaseModel):
    channel_id: str
    channel_name: str
class AuthorSummaryResponse(BaseModel):
    author: str
    username: str
    total_comments: int
    total_likes: int
    average_likes: float
    max_likes: int
    risk_level: str
    recent_comments: List[Comment]
    sentiment: Dict[str, int]
    interests: List[str]
    summary: str
    languages_detected: List[str]
    flags: List[str]
    footprints: List[str]
    topics: List[str]
    recent_channels: List[ChannelInfo] = Field(default_factory=list, alias="channels_messaged")


def extract_interests(texts):
    nlp = spacy.load("en_core_web_sm")
    combined = " ".join(texts)
    doc = nlp(combined)
    nouns = [chunk.text.lower() for chunk in doc.noun_chunks]
    common = Counter(nouns).most_common(10)
    return [w for w, _ in common]

def translate_texts(texts):
    out = []
    for t in texts:
        try:
            if detect(t) != "en":
                out.append(translator.translate(t))
            else:
                out.append(t)
        except:  # Just catching the exception without using 'e'
            out.append(t)  # If translation fails, keep the original text
    return out

def extract_keywords(texts, top_k=5):
    if not texts:
        return []
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    X = tfidf.fit_transform(texts)
    scores = list(zip(tfidf.get_feature_names_out(), X.sum(axis=0).A1))
    return [kw for kw, _ in sorted(scores, key=lambda x: -x[1])[:top_k]]

def calculate_risk_score(sentiment_summary: Dict[str, int], texts: List[str]) -> int:
    score = 0
    score += sentiment_summary.get("negative", 0) * 2
    if any(re.search(r"\bcrypto|fraud|loan\b", t.lower()) for t in texts):
        score += 20
    return min(100, score)

def detect_digital_footprint(texts):
    footprints = set()
    for t in texts:
        footprints.update(URL_RE.findall(t))
        footprints.update(HANDLE_RE.findall(t))
    return list(footprints)

def generate_narrative_summary(user, dynamic_summary, label="Telegram user") -> str:
    sentiments = dynamic_summary.get("sentiments", {})
    topics = dynamic_summary.get("topics", [])
    footprints = dynamic_summary.get("footprints", [])

    # Default summary if sentiments are missing
    if not sentiments:
        return (
            f"{label} analysis for {user.get('first_name', 'Unknown')}:\n"
            f"- Topics of interest: {', '.join(topics) if topics else 'N/A'}\n"
            f"- Digital footprints detected: {', '.join(footprints) if footprints else 'None'}"
        )

    dominant = max(sentiments, key=lambda k: len(sentiments[k]))

    if dominant == "positive":
        sentiment_summary = (
            "The audience sentiment is largely appreciative, with expressions of praise, support, and satisfaction. "
            "The user fosters a positive emotional connection and resonates well with others."
        )
    elif dominant == "negative":
        sentiment_summary = (
            "The sentiment leans toward dissatisfaction. There are concerns, disapproval, or critical feedback, "
            "indicating possible misalignment with expectations or controversial topics."
        )
    else:
        sentiment_summary = (
            "The overall sentiment is neutral. Messages appear observational, reserved, or informational in tone, "
            "suggesting limited emotional engagement or a diverse context."
        )

    return (
        f"{label} summary for {user.get('first_name', 'Unknown')}:\n"
        f"- Topics of interest: {', '.join(topics) if topics else 'N/A'}\n"
        f"- Digital footprints: {', '.join(footprints) if footprints else 'None'}\n"
        f"- Sentiment analysis: {sentiment_summary}"
    )

def analyze_behavior_and_footprint(texts: List[str]) -> Dict[str, List[str]]:
    flags = []
    footprints = set()

    # Detect sentiment and posting frequency
    if any(analyzer.polarity_scores(t)["compound"] < -0.5 for t in texts):
        flags.append("Negative sentiment")
    if len(texts) > 100:
        flags.append("High frequency posting")

    # Extract URLs and handles
    for t in texts:
        footprints.update(URL_RE.findall(t))
        footprints.update(HANDLE_RE.findall(t))

    return {
        "behavioral_flags": flags,
        "digital_footprints": list(footprints)
    }

def extract_affiliations(texts: List[str], analyzer) -> Dict[str, List[str]]:
    if not texts:
        return {"organizations": [], "people": []}

    combined_text = " ".join(texts[:50])
    ents = analyzer(combined_text)

    orgs = {e["word"] for e in ents if e.get("entity_group") == "ORG"}
    pers = {e["word"] for e in ents if e.get("entity_group") == "PER"}

    return {
        "organizations": list(orgs),
        "people": list(pers)
    }

def detect_behavioral_flags(texts: List[str]) -> List[str]:
    flags=[]
    if any(_money_rx.search(t) for t in texts):
        flags.append("Monetary amounts mentioned")
    if any(_phone_rx.search(t) for t in texts):
        flags.append("Possible phone/ID numbers present")
    neg=[t for t in texts if sentiment_analyzer.polarity_scores(t)["compound"] < -0.7]
    if neg:
        flags.append(f"{len(neg)} highly negative messages")
    if len(texts)>100:
        flags.append("High frequency posting")
    return flags

def safe_str(value):
    return str(value) if value is not None else "unknown"

def extract_flags(texts: List[str], keywords: List[str]) -> List[str]:
    found=set()
    for t in texts:
        low=t.lower()
        for k in keywords:
            if k in low:
                found.add(k)
    return list(found)

def summarize_text(text: str, max_chunk_tokens: int = MAX_TOKENS) -> str:
    if len(text) < 50:
        return text

    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]

    chunks = []
    for i in range(0, len(input_ids), max_chunk_tokens):
        chunk_ids = input_ids[i:i + max_chunk_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
        summaries.append(summary)

    if len(summaries) == 1:
        return summaries[0]

    # Optionally summarize the combined result again
    final_summary = summarizer(" ".join(summaries), max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
    return final_summary

def chunk_text(text, max_tokens=1024):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

def extract_main_topic(texts: List[str], top_k: int = 1) -> str:
    all_text = " ".join(texts)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(all_text.lower())
    filtered_words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 2]
    most_common = Counter(filtered_words).most_common(top_k)
    return most_common[0][0] if most_common else "unknown"

def generate_paragraph_summaries(report: Dict) -> List[Dict[str, str]]:
    def describe_list(values: List[str], fallback: str = "No notable terms found.") -> str:
        return ", ".join(values) if values else fallback

    return [
        {
            "title": "User Interests",
            "content": (
                f"Detected interest-related keywords: {describe_list(report.get('fraud_signals', []) + report.get('banking_terms', []))}. "
                f"Detected overall topics: {describe_list(report.get('interests_detected', []))}."
            )
        },
        {
            "title": "Location Insight",
            "content": (
                f"Languages used: {describe_list(report.get('languages_detected', []))}. "
                f"Banking references detected: {describe_list(report.get('banking_terms', []))}."
            )
        },
        {
            "title": "Law Enforcement Indicators",
            "content": (
                f"Possible fraud terms identified: {describe_list(report.get('fraud_signals', []))}. "
                f"Aliases used: {describe_list(report.get('aliases_used', []))}."
            )
        },
        {
            "title": "Communication Patterns",
            "content": (
                f"Total messages sent: {report.get('total_messages', 0)}. "
                f"Recent message activity suggests behavioral patterns typical of high engagement."
            )
        },
        {
            "title": "Financial Behavior",
            "content": (
                f"Banking keywords: {describe_list(report.get('banking_terms', []))}. "
                f"Crypto-related terms: {describe_list(report.get('fraud_signals', []))}."
            )
        },
        {
            "title": "Anonymity & Aliases",
            "content": (
                f"Aliases detected: {describe_list(report.get('aliases_used', []))}. "
                f"Behavior suggests anonymity strategies if frequent alias use is present."
            )
        },
        {
            "title": "Behavioral Red Flags",
            "content": describe_list(report.get('behavioral_flags', []), "No major red flags detected.")
        },
        {
            "title": "Cultural & Linguistic Context",
            "content": (
                f"Languages used across messages: {describe_list(report.get('languages_detected', []))}. "
                f"Cross-cultural or multilingual engagement can be inferred based on linguistic diversity."
            )
        }
    ]

def summarize_messages(messages: List[str]) -> str:
    """
    Dynamically summarizes a list of messages using a transformer-based model.
    """
    if not messages:
        return "No messages available."

    combined = " ".join(messages[:100])[:3000]  # Limit input size for performance
    chunks = [combined[i:i + 1024] for i in range(0, len(combined), 1024)]

    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
            summaries.append(summary)
        except Exception:
            continue

    if len(summaries) == 1:
        return summaries[0]

    final_input = " ".join(summaries)
    final_summary = summarizer(final_input, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
    return final_summary

def detect_dynamic_topics(texts: List[str], top_n: int = 3) -> List[str]:
    vectorizer = CountVectorizer(stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=top_n, random_state=42)
    lda.fit(doc_term_matrix)
    topics = []
    for topic in lda.components_:
        words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:][::-1]]
        topics.append(", ".join(words))
    return topics

def generate_dynamic_summary(texts: List[str]) -> Dict[str, Any]:
    """
    Fully dynamic summary generator using summarization and topic modeling.
    """
    translated_texts = [translate(t) for t in texts if t.strip()]
    summary = summarize_messages(translated_texts)
    topics = detect_dynamic_topics(translated_texts)

    return {
        "summary": summary,
        "topics": topics,
        "languages_detected": detect_languages(translated_texts),
        "flags": detect_behavioral_flags(translated_texts),
        "footprints": detect_digital_footprint(translated_texts)
    }


def generate_topic_summary(themes: List[str]) -> str:
    if not themes:
        return "\n###    ^=       Topic Modeling Summary\n- No dominant discussion topics were detected."

    summary = "###    ^=       Topic Modeling Summary\nBased on topic modeling, the user's messages likely discuss:\n"
    for theme in set(themes):
        summary += f"\n- **{theme}**: This appears to be a significant area of focus in the user's conversations."
    return summary

def fetch_telegram_messages(group_id: Optional[str] = None,
                             user_id: Optional[str] = None,
                             start: Optional[datetime] = None,
                             end: Optional[datetime] = None) -> List[str]:
    query = {}
    if group_id:
        query["chat_id"] = group_id
    if user_id:
        query["user_id"] = user_id
    if start or end:
        query["timestamp"] = {}
        if start:
            query["timestamp"]["$gte"] = start
        if end:
            query["timestamp"]["$lte"] = end

    docs = list(messages_collection.find(query))
    return [d.get("text", "") for d in docs if d.get("text")]


def translate_text(text: str) -> str:
    try:
        if detect(text) != 'en':
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except Exception:
        return text

def generate_clean_narrative_summary(author: str, texts: List[str], dynamic: Dict[str, List[str]]) -> str:
    lang_part = ", ".join(dynamic["languages_detected"]) or "unknown languages"
    flag_part = ", ".join(dynamic["flags"]) or "no specific behavioral red flags"
    topic_part = "; ".join(dynamic["topics"]) or "general topics"
    
    sentiment_overview = analyze_sentiment_summary_vader(texts)
    sentiment_insights = sentiment_overview.get("Insights", [])
    insight_text = " ".join(sentiment_insights)

    return (
        f"The author '{author}' actively engages across YouTube discussions, mostly using {lang_part}. "
        f"The conversation topics revolve around {topic_part}. "
        f"Behavioral patterns observed include {flag_part}. "
        f"Sentiment analysis reveals: {insight_text or 'a balanced tone with mixed opinions'}."
    )

def generate_paragraph_summaries_from_author_data(user_data: Dict, texts: List[str]) -> List[Dict[str, str]]:
    summaries = []

    interests = user_data.get("interests_detected", [])
    summaries.append({
        "title": "User Interests",
        "content": f"Detected interest-related keywords: {', '.join(interests) or 'None'}."
    })

    languages = user_data.get("languages_detected", [])
    banking = user_data.get("banking_terms", [])
    summaries.append({
        "title": "Location Insight",
        "content": f"Languages used: {', '.join(languages) or 'unknown'}. Banking references detected: {', '.join(banking) or 'none'}."
    })

    fraud = user_data.get("fraud_signals", [])
    aliases = user_data.get("aliases_used", [])
    summaries.append({
        "title": "Law Enforcement Indicators",
        "content": f"Possible fraud terms identified: {', '.join(fraud) or 'None'}. Aliases used: {', '.join(aliases) or 'No notable terms found.'}"
    })

    summaries.append({
        "title": "Communication Patterns",
        "content": f"Total messages sent: {user_data.get('total_messages', 0)}. Recent message activity suggests behavioral patterns typical of high engagement."
    })

    summaries.append({
        "title": "Financial Behavior",
        "content": f"Banking keywords: {', '.join(banking) or 'none'}. Crypto-related terms: {', '.join([k for k in fraud if 'crypto' in k]) or 'none'}."
    })

    summaries.append({
        "title": "Anonymity & Aliases",
        "content": f"Aliases detected: {', '.join(aliases) or 'No notable terms found.'}. Behavior suggests anonymity strategies if frequent alias use is present."
    })

    flags = user_data.get("behavioral_flags", [])
    summaries.append({
        "title": "Behavioral Red Flags",
        "content": f"{', '.join(flags) or 'None'}"
    })

    summaries.append({
        "title": "Cultural & Linguistic Context",
        "content": f"Languages used across messages: {', '.join(languages) or 'unknown'}. Cross-cultural or multilingual engagement can be inferred based on linguistic diversity."
    })

    return summaries


# Function to filter comments based on date range and translate them
def filter_comments(comments: List[Dict], start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[str]:
    def in_range(published_at: str) -> bool:
        try:
            dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
            if start and dt < start:
                return False
            if end and dt > end:
                return False
            return True
        except Exception as e:
            logger.warning(f"Date parsing error: {e}")
            return False

    translated_texts = []
    for c in comments:
        try:
            text = c.get("text")
            published_at = c.get("commented_at")

            if text and isinstance(text, str) and text.strip() and published_at and in_range(published_at):
                translated = translator.translate(text, dest='en')
                translated_texts.append(translated.text)
        except Exception as e:
            logger.warning(f"Error processing comment: {e}")
            continue

    return translated_texts

def get_cumulative_stats(start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
    start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    date_filter = {}
    if start:
        date_filter["commented_at"] = {"$gte": start}
    if end:
        date_filter.setdefault("commented_at", {}).update({"$lte": end})

    video_channels = videos_collection.distinct("channel_id")
    total_channels = channels_collection.count_documents({})
    total_comments = comments_collection.count_documents(date_filter)
    unique_authors = comments_collection.distinct("author", date_filter)

    first_comment = comments_collection.find_one(
        {**date_filter, "commented_at": {"$exists": True}}, sort=[("commented_at", ASCENDING)]
    )
    last_comment = comments_collection.find_one(
        {**date_filter, "commented_at": {"$exists": True}}, sort=[("commented_at", -1)]
    )

    avg_comments_per_day = None
    if first_comment and last_comment:
        try:
            start_ts = first_comment["commented_at"]
            end_ts = last_comment["commented_at"]

            if isinstance(start_ts, str):
                start_ts = datetime.strptime(start_ts, "%Y-%m-%dT%H:%M:%SZ")
            if isinstance(end_ts, str):
                end_ts = datetime.strptime(end_ts, "%Y-%m-%dT%H:%M:%SZ")

            days = max((end_ts - start_ts).days, 1)
            avg_comments_per_day = format_number(round(total_comments / days, 2))
        except Exception as e:
            print(f"Error parsing comment timestamps: {e}")

    return {
        "total_channels": format_number(total_channels),
        "channels_with_videos": format_number(len(video_channels)),
        "total_comments": format_number(total_comments),
        "unique_commenters": format_number(len(unique_authors)),
        "avg_comments_per_day": avg_comments_per_day
    }

def detect_anomalies(texts: List[str]) -> List[str]:
    anomalies = []
    if any(_money_rx.search(t) for t in texts):
        anomalies.append("Monetary amounts mentioned")
    if any(_phone_rx.search(t) for t in texts):
        anomalies.append("Potential phone/ID numbers present")
    neg = [t for t in texts if sentiment_analyzer.polarity_scores(t)["compound"] < -0.7]
    if neg:
        anomalies.append(f"{len(neg)} highly negative messages")
    bets = detect_flags(texts, betting_kw)
    if bets:
        anomalies.append("Betting references: " + ", ".join(bets))
    porn = detect_flags(texts, porn_kw)
    if porn:
        anomalies.append("Adult content references: " + ", ".join(porn))
    fraud = detect_flags(texts, fraud_kw)
    if fraud:
        anomalies.append("Fraud keywords: " + ", ".join(fraud))
    bank = detect_flags(texts, bank_kw)
    if bank:
        anomalies.append("Bank keywords: " + ", ".join(bank))
    return anomalies


def fetch_messages(user_id=None, message_id=None, username=None, start=None, end=None):
    query = {}
    if user_id:
        query["user_id"] = user_id
    if message_id:
        query["message_id"] = message_id
    if username:
        query["username"] = username
    if start and end:
        query["timestamp"] = {"$gte": start, "$lte": end}
    return list(messages_collection.find(query, {"_id": 0}))

# Function for sentiment analysis of a list of comments
def analyze_sentiment_summary_vader(texts: List[str]) -> Dict:
    analyzer = SentimentIntensityAnalyzer()
    summary = {
        "Positive": {"count": 0, "examples": []},
        "Negative": {"count": 0, "examples": []},
        "Neutral": {"count": 0, "examples": []}
    }

    for text in texts:
        score = analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            summary["Positive"]["count"] += 1
            if len(summary["Positive"]["examples"]) < 3:
                summary["Positive"]["examples"].append(text)
        elif score <= -0.05:
            summary["Negative"]["count"] += 1
            if len(summary["Negative"]["examples"]) < 3:
                summary["Negative"]["examples"].append(text)
        else:
            summary["Neutral"]["count"] += 1
            if len(summary["Neutral"]["examples"]) < 3:
                summary["Neutral"]["examples"].append(text)

    total = len(texts)
    pos = summary["Positive"]["count"]
    neg = summary["Negative"]["count"]
    neu = summary["Neutral"]["count"]

    # Narrative analysis
    if total == 0:
        narrative = "No comment data was available to perform sentiment analysis for this channel."
    else:
        pos_ratio = pos / total
        neg_ratio = neg / total
        neu_ratio = neu / total

        tone = "mixed"
        if pos_ratio > 0.6:
            tone = "overwhelmingly positive"
        elif neg_ratio > 0.4:
            tone = "highly critical"
        elif neu_ratio > 0.5:
            tone = "neutral and detached"

        observations = []

        if tone == "overwhelmingly positive":
            observations.append("The channel is well-received, with audiences consistently expressing appreciation, praise, and positive emotional engagement.")
        elif tone == "highly critical":
            observations.append("Audience sentiment leans negative, suggesting recurring dissatisfaction or controversy around the channel's content or topics.")
        elif tone == "neutral and detached":
            observations.append("Many comments appear fact-based or indifferent, indicating viewers may be observing without strong emotional involvement.")
        else:
            observations.append("Sentiment distribution is fairly balanced, indicating a mix of support, criticism, and objective commentary from the viewers.")

        if neg > 0 and pos > 0:
            observations.append("While many comments express support, a notable volume of criticism or emotionally charged responses is also present.")
        if neu > 0 and (pos + neg) == 0:
            observations.append("The absence of emotionally expressive comments may indicate passive consumption or lack of engagement.")

        narrative = " ".join(observations)

    return {
        "Sentiment Summary": summary,
        "Narrative": narrative
    }

def semantic_search(corpus: list, query: str, top_k: int = 5):
    corpus_embeddings = semantic_model.encode(corpus, convert_to_tensor=True)
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(query_embedding, corpus_embeddings)
    top_results = torch.topk(cos_scores, k=min(top_k, len(corpus)))
    return [{"text": corpus[idx], "score": float(score)} for score, idx in zip(top_results.values, top_results.indices)]

def extract_hashtags(videos: List[dict]) -> List[str]:
    hashtags = []
    for video in videos:
        tags = video.get("hashtags", [])
        if tags:
            hashtags.extend(tags)
    return hashtags

def detect_languages(texts: List[str]) -> List[str]:
    langs = []
    for t in texts:
        try:
            lang = detect(t)
            langs.append(lang_labels.get(lang, lang))
        except:
            continue
    return list(set(langs))

def analyze_video_titles_and_generate_summary(videos: List[dict], titles: List[str]) -> Tuple[Dict, str]:
    if not titles:
        return {"positive": 0, "negative": 0, "neutral": 0}, "No titles available to analyze."

    results = sentiment_analyzer(titles[:200])
    pos = sum(r["label"] == "POSITIVE" for r in results)
    neg = sum(r["label"] == "NEGATIVE" for r in results)
    neu = len(results) - (pos + neg)

    hashtags = extract_hashtags(videos)
    common_hashtags = Counter(hashtags).most_common(5)
    top_hashtags = [h[0] for h in common_hashtags]

    # Generate natural language summary
    summary = f"Analyzed {len(titles)} video titles."
    summary += f" Sentiment distribution: {pos} positive, {neg} negative, and {neu} neutral titles."
    if top_hashtags:
        summary += f" Common hashtags include: {', '.join(top_hashtags)}."
    else:
        summary += " No common hashtags were found."

    return {"positive": pos, "negative": neg, "neutral": neu}, summary

def fetch_youtube_comments(channel_id: str, video_id: str, max_results: int = 100) -> List[str]:
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        comments = []
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=max_results
        ).execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comment_doc = {
                "channel_id": channel_id,
                "video_id": video_id,
                "author": comment.get("authorDisplayName"),
                "text": comment.get("textDisplay"),
                "published_at": comment.get("publishedAt")
            }
            comments_collection.update_one(
                {"channel_id": channel_id, "video_id": video_id, "text": comment.get("textDisplay")},
                {"$setOnInsert": comment_doc},
                upsert=True
            )
            comments.append(comment.get("textDisplay"))
        return comments
    except Exception as e:
        logger.error(f"Error fetching YouTube comments: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch YouTube comments.")

def generate_final_structured_summary_array(report: Dict) -> List[str]:
    def bullet_section(subpoints: Dict[str, str]) -> List[str]:
        lines = []
        for subtitle, content in subpoints.items():
            lines.append(f"• {subtitle}: {content.strip() if content else 'No summary available.'}")
        return lines

    summary = ["🧾 Final Structured Summary"]

    summary += ["1. User Identity & Activity Overview"]
    summary += bullet_section({
        "User ID": report.get('telegram_id', 'N/A'),
        "Username": report.get('username', 'N/A'),
        "Name": report.get('name', 'N/A'),
        "Total Messages": str(report.get('total_messages', 0)),
        "Groups Participated": ', '.join(report.get('groups', [])) or 'N/A',
        "Languages Detected": ', '.join(report.get('languages_detected', [])) or 'Unknown'
    })

    summary += ["2. User Interests"]
    summary += bullet_section({
        "Content Themes": report.get('interests_summary')
    })

    summary += ["3. Location Insight"]
    summary += bullet_section({
        "Geo/Lang Clues": report.get('location_summary')
    })

    summary += ["4. Law Enforcement Indicators"]
    summary += bullet_section({
        "Red Flags": report.get('law_enforcement_summary')
    })

    summary += ["5. Communication Patterns"]
    summary += bullet_section({
        "Behavior": report.get('communication_summary')
    })

    summary += ["6. Financial Behavior"]
    summary += bullet_section({
        "Finance Indicators": report.get('financial_summary')
    })

    summary += ["7. Anonymity & Operational Security"]
    summary += bullet_section({
        "Aliases / Private Channels": report.get('anonymity_summary')
    })

    summary += ["8. Behavioral Red Flags"]
    summary += bullet_section({
        "Traits": report.get('behavioral_summary')
    })

    summary += ["9. Cultural & Linguistic Context"]
    summary += bullet_section({
        "Cross-Cultural Behavior": report.get('cultural_summary')
    })

    return summary

def fetch_and_store_video_metadata(channel_id: str, video_id: str):
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        response = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        ).execute()

        if not response["items"]:
            raise HTTPException(status_code=404, detail="Video not found on YouTube.")

        video_data = response["items"][0]
        snippet = video_data["snippet"]
        stats = video_data.get("statistics", {})

        video_doc = {
            "channel_id": channel_id,
            "video_id": video_id,
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "published_at": snippet.get("publishedAt"),
            "hashtags": [tag for tag in snippet.get("tags", []) if tag.startswith("#")],
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0))
        }

        videos_collection.update_one(
            {"video_id": video_id},
            {"$set": video_doc},
            upsert=True
        )
    except Exception as e:
        logger.error(f"Error fetching video metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch video metadata.")

def get_channel_summary_from_db(channel_id: str) -> dict:
    # Get channel basic info
    channel = channels_collection.find_one({"channel_id": channel_id})
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    # Format helper
    def format_number(num):
        return locale.format_string("%d", num, grouping=True)

    # Aggregate from videos
    videos = list(videos_collection.find({"channel_id": channel_id}))
    total_views = sum(video.get("views", 0) for video in videos)
    total_likes = sum(video.get("likes", 0) for video in videos)
    total_videos_uploaded = len(videos)

    # Aggregate from comments
    video_ids = [video.get("video_id") for video in videos]
    total_comments = comments_collection.count_documents({"video_id": {"$in": video_ids}})

    return {
        "channel_id": channel.get("channel_id", ""),
        "channel_name": channel.get("title", "No name available"),
        "total_subscribers": format_number(channel.get("subscribers", 0)),
        "total_videos_uploaded": format_number(total_videos_uploaded),
        "total_likes": format_number(total_likes),
        "total_views": format_number(total_views),
        "total_comments": format_number(total_comments)
    }

def generate_topicwise_narrative(topics: List[str], texts: List[str]) -> str:
    if not topics or not texts:
        return "No significant topics detected to generate summaries."

    topic_map = defaultdict(list)
    for topic in topics:
        pattern = re.compile(re.escape(topic), re.IGNORECASE)
        for text in texts:
            if pattern.search(text):
                topic_map[topic].append(text.strip())

    summary = ""
    for topic, matched_texts in topic_map.items():
        example = matched_texts[0] if matched_texts else "No clear context."
        summary += f"\n- **{topic.title()}**: Mentioned in {len(matched_texts)} message(s). Example: \"{example[:120]}...\""
    return summary.strip()

def detect_interests(text: str) -> List[str]:
    interests = []
    ltext = text.lower()
    if any(kw in ltext for kw in fraud_kw): interests.append("Cryptocurrency / Fraud")
    if any(kw in ltext for kw in bank_kw): interests.append("Banking / Financial Services")
    if any(kw in ltext for kw in betting_kw): interests.append("Gaming / Betting")
    return list(set(interests))

def detect_additional_topics(text: str, num_topics: int = 3) -> List[str]:
    try:
        vectorizer = CountVectorizer(stop_words='english')
        doc_term_matrix = vectorizer.fit_transform([text])
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(doc_term_matrix)

        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]
            topics.append(" ".join(top_words))
        return list(set(topics))
    except Exception as e:
        return [f"LDA Error: {str(e)}"]



def extract_summary(messages: List[Dict], user_data: Dict) -> Dict:
    all_text = " ".join(m.get("text", "") for m in messages if m.get("text"))
    interests = detect_interests(all_text)
    langs = detect_languages([m.get("text", "") for m in messages if m.get("text")])
    groups = set(m.get("group_id") for m in messages if m.get("group_id"))
    group_names = [g.get("name") for g in groups_collection.find({"_id": {"$in": list(groups)}})]

    return {
        "telegram_id": safe_str(user_data.get("user_id", "unknown")),
        "username": safe_str(user_data.get("username")),
        "name": safe_str(user_data.get("name")),
        "bio": safe_str(user_data.get("bio")) if user_data.get("bio") else "Not available",
        "last_active": str(max((m.get("date") for m in messages if m.get("date")), default="unknown")),
        "total_messages": len(messages),
        "groups": group_names,
        "languages_detected": langs,
        "interests_detected": interests,
        "aliases_used": list(set(re.findall(r'@\w+', all_text))),
        "fraud_signals": [kw for kw in fraud_kw if kw in all_text.lower()],
        "banking_terms": [kw for kw in bank_kw if kw in all_text.lower()],
        "behavioral_flags": [
            "Aggressive Solicitation" if len(messages) > 200 else "Low Volume",
            "Multiple Aliases" if len(set(re.findall(r'@\w+', all_text))) > 1 else "Single Identity",
            "Multilingual" if len(langs) > 1 else "Single Language"
        ],
        "final_commentary": "This user shows patterns of financial engagement and potential fraud indicators based on message volume and keyword analysis. Further verification advised."
    }

def generate_narrative(report: Dict) -> str:
    name = report.get("name", "unknown")
    username = report.get("username", "unknown")
    telegram_id = report.get("telegram_id", "unknown")
    group_str = ", ".join(report.get("groups", [])) or "unknown groups"
    last_active = report.get("last_active", "an unknown date")
    total_msgs = report.get("total_messages", 0)
    interests = ", ".join(report.get("interests_detected", [])) or "none"
    langs = ", ".join(report.get("languages_detected", [])) or "unknown"
    aliases = ", ".join(report.get("aliases_used", [])) or "none"
    frauds = ", ".join(report.get("fraud_signals", [])) or "none"
    banks = ", ".join(report.get("banking_terms", [])) or "none"
    behavior = ", ".join(report.get("behavioral_flags", [])) or "not available"

    return (
        f"User Analysis Report for Telegram ID: {telegram_id}\n\n"
        f"The user, whose username is '{username}' and name is '{name}', has demonstrated notable activity with "
        f"{total_msgs} messages, primarily within the following groups: {group_str}. Their last recorded activity "
        f"was on {last_active}. While the user has no known biography, the content of their messages reveals strong interests in "
        f"{interests}. Language usage analysis suggests communication in {langs}, implying engagement with a diverse or international audience.\n\n"
        f"Behavioral analysis shows frequent references to aliases such as {aliases}, and flagged terms like {frauds} and banking terms like {banks}. "
        f"These patterns align with potential risks related to scams, financial deception, or unregulated cryptocurrency operations. "
        f"Further behavioral traits include: {behavior}.\n\n"
        f"In conclusion, the user exhibits multilingual financial communication with indicators of suspicious or high-risk activity. "
        f"This profile warrants closer examination, particularly in cases of cryptocurrency fraud or cross-border financial crime."
    )

def generate_sentiment_narrative(comments: List[str]) -> str:
    if not comments:
        return "No comments were available for sentiment analysis."

    analyzer = SentimentIntensityAnalyzer()
    categorized = {"positive": [], "negative": [], "neutral": []}

    for text in comments:
        score = analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            categorized["positive"].append(text)
        elif score <= -0.05:
            categorized["negative"].append(text)
        else:
            categorized["neutral"].append(text)

    summary_parts = []

    if categorized["positive"]:
        example = categorized["positive"][0][:150]
        summary_parts.append(
            f"The user expresses a positive attitude in certain comments, showing appreciation or support. "
            f"Example: _\"{example}...\"_"
        )

    if categorized["negative"]:
        example = categorized["negative"][0][:150]
        summary_parts.append(
            f"Some comments reflect a negative or critical tone, possibly indicating disagreement or dissatisfaction. "
            f"For instance: _\"{example}...\"_"
        )

    if categorized["neutral"]:
        example = categorized["neutral"][0][:150]
        summary_parts.append(
            f"There are also comments that remain neutral or ambiguous, with minimal emotional expression. "
            f"One such comment: _\"{example}...\"_"
        )

    if not summary_parts:
        return "The author's tone could not be clearly categorized from the available comments."

    return " ".join(summary_parts)


def extract_named_entities(texts: List[str], platform: str) -> Dict[str, List[str]]:
    if platform == "youtube":
        ner_results = yt_ner_analyzer(" ".join(texts))
    elif platform == "telegram":
        ner_results = tg_ner_analyzer(" ".join(texts))
    else:
        raise ValueError("Invalid platform specified")

    entities = {}
    for ent in ner_results:
        label = ent['entity_group'] if 'entity_group' in ent else ent['entity']
        entities.setdefault(label, []).append(ent['word'])

    # Remove duplicates and clean
    return {k: list(set(v)) for k, v in entities.items()}

def semantic_keywords(texts: List[str], keyword: str) -> List[str]:
    embeddings = semantic_model.encode(texts[:200])
    keyword_emb = semantic_model.encode([keyword])[0]
    return [text for text, emb in zip(texts, embeddings) if torch.cosine_similarity(torch.tensor(emb), torch.tensor(keyword_emb), dim=0) > 0.5]

def geo_based_analysis(texts: List[str]) -> List[str]:
    locations = ["India", "USA", "Delhi", "Mumbai", "New York", "Bangalore"]
    return [loc for loc in locations if any(loc.lower() in t.lower() for t in texts)]

# ------------------- API Endpoints ------------------- #
@app.get("/")
def read_root():
    return {"message": "API is running."}

@app.get("/", response_class=HTMLResponse)
async def home():
    return "<h2>Telegram + YouTube Intelligence API</h2><p>Use /docs to explore</p>"

@app.get("/dashboard/telegram/detailed/")
def telegram_dashboard(
    range_type: str = Query("all_time", enum=["all_time", "last_24_hours", "last_week", "last_month", "custom"]),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    try:
        now = datetime.utcnow()

        if range_type == "last_24_hours":
            start = now - timedelta(days=1)
            end = now
        elif range_type == "last_week":
            start = now - timedelta(days=7)
            end = now
        elif range_type == "last_month":
            start = now - timedelta(days=30)
            end = now
        elif range_type == "custom":
            if not start_date or not end_date:
                raise HTTPException(status_code=400, detail="Start and end dates must be provided for custom range.")
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            start = None
            end = None

        message_filter = {}
        if start:
            message_filter["timestamp"] = {"$gte": start}
        if end:
            message_filter.setdefault("timestamp", {}).update({"$lte": end})

        total_groups = groups_collection.count_documents({})
        total_users = users_collection.count_documents({})
        total_messages = messages_collection.count_documents(message_filter)

        active_user_ids = messages_collection.distinct("user_id", message_filter)
        active_users = len(active_user_ids)

        last_day_count = rate_change = None
        if range_type in ["last_24_hours", "last_week"]:
            last_day = now - timedelta(days=1)
            last_week = now - timedelta(days=7)
            last_day_count = messages_collection.count_documents({"timestamp": {"$gte": last_day}})
            last_week_count = messages_collection.count_documents({"timestamp": {"$gte": last_week}})
            daily_avg = round(last_week_count / 7, 2) if last_week_count else 0
            rate_change = round((last_day_count - daily_avg) / daily_avg * 100, 2) if daily_avg else 0

        group_user_map = messages_collection.aggregate([
            {"$match": message_filter},
            {"$group": {"_id": "$chat_id", "unique_users": {"$addToSet": "$user_id"}}},
            {"$project": {"chat_id": "$_id", "unique_user_count": {"$size": "$unique_users"}}}
        ])
        group_user_counts = [g["unique_user_count"] for g in group_user_map]
        propagation_avg = round(sum(group_user_counts) / total_groups, 2) if total_groups else 0

        total_media = messages_collection.count_documents({"media_type": {"$ne": None}, **message_filter})

        messages_with_views = messages_collection.find({"views": {"$exists": True, "$ne": None}, **message_filter}, {"views": 1})
        views_list = [m.get("views", 0) for m in messages_with_views if isinstance(m.get("views", 0), (int, float))]
        avg_views = round(sum(views_list) / len(views_list), 2) if views_list else 0

        return {
            "range_type": range_type,
            "start_date": start_date or "Not specified",
            "end_date": end_date or "Not specified",
            "report_generated_at": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
            "total_groups": total_groups,
            "total_users": total_users,
            "active_users": active_users,
            "total_messages": total_messages,
            "message_rate": last_day_count,
            "message_rate_change": rate_change,
            "group_propagation_avg": propagation_avg,
            "total_media_files": total_media,
            "avg_views_per_message": avg_views
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/Youtube_Dashboard/")
async def cumulative_statistics(
    range_type: str = Query("all_time", enum=["all_time", "last_24_hours", "last_week", "last_month", "custom"]),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    try:
        now = datetime.utcnow()

        if range_type == "last_24_hours":
            start = (now - timedelta(days=1)).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")
        elif range_type == "last_week":
            start = (now - timedelta(weeks=1)).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")
        elif range_type == "last_month":
            start = (now - timedelta(days=30)).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")
        elif range_type == "custom":
            if not start_date or not end_date:
                raise HTTPException(status_code=400, detail="Start and end dates are required for custom range.")
            start = start_date
            end = end_date
        else:
            start = None
            end = None

        stats = get_cumulative_stats(start, end)
        stats.update({
            "range_type": range_type,
            "start_date": start or "Not specified",
            "end_date": end or "Not specified",
            "report_generated_at": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()
        })
        return stats

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    
#@app.route('/telegram_user_summary/', methods=['GET', 'POST'])
@app.get("/telegram_user_summary/")
async def telegram_user_summary(user_id: str):
    try:
        user = users_collection.find_one({"user_id": int(user_id)})
        if not user:
            user = {}

        messages = list(messages_collection.find({"user_id": int(user_id)}).sort("date", -1))
        if not messages:
            raise HTTPException(status_code=404, detail="No messages found for user")

        message_texts = [m.get("text", "") for m in messages if m.get("text")]
        translated_texts = [translate(t) for t in message_texts if t.strip()]

        # Extract raw user-level metadata
        report = {
            "telegram_id": str(user.get("user_id", "unknown")),
            "username": user.get("username", "unknown"),
            "first_name": user.get("first_name", "unknown"),
            "last_name": user.get("last_name", "unknown"),
            "bio": user.get("bio", "Not available"),
            "total_messages": len(messages),
            "languages_detected": detect_languages(translated_texts),
            "aliases_used": list(set(re.findall(r'@\\w+', " ".join(translated_texts)))),
            "banking_terms": [kw for kw in bank_kw if kw in " ".join(translated_texts).lower()],
            "fraud_signals": [kw for kw in fraud_kw if kw in " ".join(translated_texts).lower()],
            "interests_detected": detect_interests(" ".join(translated_texts)),
            "behavioral_flags": detect_behavioral_flags(translated_texts),
            "digital_footprint": detect_digital_footprint(translated_texts)
        }

        # Add group participation info
        group_ids = list({m["chat_id"] for m in messages if "chat_id" in m})
        group_details = []
        if group_ids:
            groups = list(groups_collection.find({"chat_id": {"$in": group_ids}}))
            for group in groups:
                chat_id = group.get("chat_id")
                total_msgs = messages_collection.count_documents({"chat_id": chat_id})
                group_details.append({
                    "title": group.get("title", "Unknown"),
                    "chat_id": str(chat_id),
                    "total_messages": total_msgs
                })

        # Risk level logic (example)
        red_flags = report.get("behavioral_flags", [])
        risk_level = "High" if len(red_flags) >= 2 else "Moderate" if red_flags else "Low"

        # Dynamic summaries
        dynamic_summary = generate_dynamic_summary(translated_texts)
        paragraph_summaries = generate_paragraph_summaries(report)

        return JSONResponse({
            "user_profile": report,
            "risk_level": risk_level,
            "total_groups": len(group_details),
            "group_details": group_details,
            "summary": generate_narrative_summary(user, dynamic_summary, label="Telegram User"),
            "topics": dynamic_summary["topics"],
            "digital_footprint": dynamic_summary["footprints"],
            "paragraph_summaries": paragraph_summaries,
            "recent_messages": [
                {"timestamp": str(m["timestamp"]), "text": m["text"]}
                for m in messages[:5] if m.get("text") and m.get("timestamp")
            ],
        })
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/telegram/group_summary/")
def generate_group_summary(chat_id: str):
    try:
        try:
            chat_id_int = int(chat_id)
        except ValueError:
            chat_id_int = None

        group = groups_collection.find_one({
            "chat_id": {"$in": [chat_id, chat_id_int] if chat_id_int else [chat_id]}
        })
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")

        one_week_ago = datetime.utcnow() - timedelta(days=7)
        message_filter = {"chat_id": {"$in": [chat_id_int, chat_id] if chat_id_int else [chat_id]}}
        messages = list(messages_collection.find(message_filter, {"_id": 0}))
        recent_msgs = [m for m in messages if m.get("timestamp") and m["timestamp"] >= one_week_ago]

        active_users = list(set(m.get("user_id") for m in recent_msgs if m.get("user_id")))
        total_users = len(set(m.get("user_id") for m in messages if m.get("user_id")))
        total_messages = len(messages)

        texts, translated_map = [], []
        for msg in messages[:100]:
            text = msg.get("text", "")
            try:
                translated = GoogleTranslator(source='auto', target='en').translate(text)
                texts.append(translated)
                translated_map.append({
                    "text": translated,
                    "timestamp": msg.get("timestamp"),
                    "chat_title": msg.get("chat_title", "Unknown")
                })
            except:
                continue

        top_topics = detect_dynamic_topics(texts, top_n=5)
        topicwise_narrative = generate_topicwise_narrative(top_topics, texts)
        behavioral_flags = detect_behavioral_flags(texts)
        languages = detect_languages(texts)

        fraud_flag = any(any(k in t.lower() for k in fraud_kw) for t in texts)
        risk_score = min(100, len(behavioral_flags) * 20 + sum(1 for t in texts if any(k in t.lower() for k in fraud_kw)) * 10)
        risk_level = "High" if risk_score > 80 else "Moderate" if risk_score > 40 else "Low"

        recent_messages = sorted(translated_map, key=lambda x: x.get("timestamp", datetime.min), reverse=True)[:10]

        conversation_summary = (
            f"The group actively discusses subjects like {', '.join(top_topics)}. "
            f"Behavioral signals include: {', '.join(behavioral_flags) or 'None'}. "
            f"Users are communicating in {', '.join(languages) or 'unknown languages'}."
        )

        overall_summary = (
            f"Group '{group.get('title', 'N/A')}' contains {total_users} participants with {total_messages} messages exchanged. "
            f"Common topics involve: {', '.join(top_topics)}. Noteworthy behavioral flags include: {', '.join(behavioral_flags) or 'None'}. "
            f"Detected languages include {', '.join(languages) or 'unknown languages'}\n\n"
            f"\U0001f9e0 Topic-wise Breakdown:\n{topicwise_narrative}"
        )

        return {
            "group_profile": {
                "chat_id": group.get("chat_id"),
                "created_at": group.get("created_at"),
                "updated_at": group.get("updated_at"),
                "is_channel": group.get("is_channel"),
                "participants_count": group.get("participants_count", total_users),
                "chat_title": group.get("title", "N/A"),
                "username": group.get("username", "N/A"),
                "active_users": len(active_users),
                "total_messages": total_messages,
                "topics": top_topics,
                "potential_fraud_detected": fraud_flag,
                "risk_level": risk_level,
                "risk_score": risk_score
            },
            "conversation_summary": conversation_summary,
            "overall_summary": overall_summary,
            "paragraph_summaries": generate_group_paragraph_summary(top_topics, behavioral_flags, languages, total_messages, total_users),
            "topicwise_summary": topicwise_narrative,
            "recent_messages": recent_messages,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/telegram/message_summary/by_user/{user_id}")
def telegram_message_summary_by_user(
    user_id: str,
    # optional date filtering if you like   ^`^totherwise both start/end will be None
    time_filter: Optional[str] = None,
    start_date:  Optional[str] = None,
    end_date:    Optional[str] = None,
):
    # if the client passed a time_filter, honor it; otherwise fetch all
    if time_filter:
        start, end = get_time_bounds(time_filter, start_date, end_date)
    else:
        start, end = None, None

    # fetch_messages signature is (user_id, message_id, username, start, end)
    msgs = fetch_messages(user_id=user_id, message_id=None, username=None, start=start, end=end)
    if not msgs:
        raise HTTPException(status_code=404, detail="No messages found for that user")

    # build your summary exactly as in the POST handler
    summary = []
    convo = []
    for m in msgs:
        txt = translate(m.get("text",""))
        convo.append(txt)
        summary.append({
            "message_id": m.get("message_id"),
            "chat_id":    m.get("chat_id"),
            "chat_title": m.get("chat_title","Unknown"),
            "timestamp":  m.get("timestamp"),
            "text":       txt,
            "views":      m.get("views",0),
            "forwards":   m.get("forwards",0),
            "has_media":  m.get("has_media",False),
            "username":   m.get("username","N/A"),
        })

    return {
        "total_messages":    len(msgs),
        "recent_messages":   summary[-5:],               # last 5 messages
        "discussion_topics": detect_dynamic_topics(convo, top_n=5),       # re-use your zero-shot topic detector
        "message_snippets":  convo[:3],                  # e.g. first 3 translated texts
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )

    
@app.get("/channel_summary/")
async def channel_summary(channel_id: str):
    try:
        if not channel_id:
            raise HTTPException(status_code=400, detail="channel_id is required")

        channel = channels_collection.find_one({"channel_id": channel_id})
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")

        videos = list(videos_collection.find({"channel_id": channel_id}))
        video_ids = [v["video_id"] for v in videos if "video_id" in v]
        video_titles = [v.get("title", "") for v in videos if isinstance(v.get("title", ""), str)]

        comments_cursor = comments_collection.find({"video_id": {"$in": video_ids}})
        comments = [c.get("text", "") for c in comments_cursor if isinstance(c.get("text", ""), str) and c["text"].strip()]

        if not comments:
            raise HTTPException(status_code=404, detail="No comments found for this channel")

        sentiment_result = analyze_sentiment_summary_vader(comments)
        comment_summary = summarize_text(" ".join(comments[:100])) if comments else "No comments to summarize."
        title_summary = summarize_text(" ".join(video_titles[:20])) if video_titles else "No video titles available."

        total_views = sum(v.get("views", 0) for v in videos)
        total_likes = sum(v.get("likes", 0) for v in videos)
        total_videos = len(videos)
        total_comments = len(comments)

        topic_list = detect_dynamic_topics(comments, top_n=5)
        topicwise = generate_topicwise_narrative(topic_list, comments)

        risk_score = calculate_risk_score(sentiment_result, comments)
        if risk_score > 80:
            risk_level = "High"
        elif risk_score > 40:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        return JSONResponse(content={
            "channel_profile": {
                "channel_id": channel_id,
                "channel_name": channel.get("title", "No name"),
                "description": channel.get("description", "No description available."),
                "country": channel.get("country", "N/A"),
                "created_at": str(channel.get("created_at", "N/A")),
                "thumbnail_url": channel.get("thumbnail_url", "N/A"),
                "total_subscribers": format_number(channel.get("subscribers", 0)),
                "total_videos_uploaded": format_number(total_videos),
                "total_likes": format_number(total_likes),
                "total_views": format_number(total_views),
                "total_comments": format_number(total_comments),
            },
            "comment_summary": comment_summary,
            "video_title_summary": title_summary,
            "sentiment_summary": sentiment_result,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "overall_summary": (
                f"Channel '{channel.get('title', 'No name')}' has {total_videos} videos, {total_comments} comments, "
                f"{format_number(total_views)} views, and covers topics like {', '.join(topic_list)}."
            ),
            "topicwise_summary": topicwise,
            "timestamp": datetime.now(pytz.timezone('Asia/Kolkata')).isoformat(),
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"[channel_summary] Error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )

@app.post("/fetch_video_data/")
async def fetch_video_data(req: VideoRequest):
    fetch_and_store_video_metadata(req.channel_id, req.video_id)
    comments = fetch_youtube_comments(req.channel_id, req.video_id)
    return {"message": "Video data and comments fetched and stored successfully.", "total_comments": len(comments)}

@app.get("/sentiment_summary/")
async def sentiment_summary(channel_id: str):
    try:
        # Try fetching comments directly by channel_id
        comments = list(comments_collection.find({"channel_id": channel_id}))
        
        # Fallback: Fetch video_ids and then get comments by video_id
        if not comments:
            video_ids = [v["video_id"] for v in videos_collection.find({"channel_id": channel_id})]
            if not video_ids:
                raise HTTPException(status_code=404, detail="No videos found for this channel")
            comments = list(comments_collection.find({"video_id": {"$in": video_ids}}))

        # Clean text
        texts = [c.get("text", "") for c in comments if isinstance(c.get("text", ""), str) and c.get("text", "").strip()]
        if not texts:
            raise HTTPException(status_code=404, detail="No comments found for this channel")

        # Analyze sentiment
        sentiments = categorize_sentiment(texts)
        wordcloud_paths = generate_sentiment_wordclouds_from_categories(sentiments)
        narrative = generate_creative_narrative_summary(sentiments)

        return JSONResponse({
            "channel_id": channel_id,
            "sentiment_distribution": {
                "positive": len(sentiments["positive"]),
                "negative": len(sentiments["negative"]),
                "neutral": len(sentiments["neutral"])
            },
            "sample_comments": {
                "positive": sentiments["positive"][:3],
                "negative": sentiments["negative"][:3],
                "neutral": sentiments["neutral"][:3]
            },
            "wordclouds": wordcloud_paths,
            "narrative_summary": narrative,
            "timestamp": datetime.utcnow().isoformat()
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"[sentiment_summary] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/author_summary/{author}")
def get_author_summary(author: str):
    try:
        comments = list(comments_collection.find({"author": {"$regex": re.escape(author), "$options": "i"}}).sort("commented_at", -1))
        if not comments:
            raise HTTPException(status_code=404, detail="No comments found for this author")

        texts = [c.get("text", "") for c in comments if isinstance(c.get("text", ""), str) and c["text"].strip()]

        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        analyzer = SentimentIntensityAnalyzer()
        for text in texts:
            score = analyzer.polarity_scores(text)["compound"]
            if score >= 0.05:
                sentiment_counts["positive"] += 1
            elif score <= -0.05:
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1

        likes = [int(str(c.get("like_count") or c.get("likes") or 0).replace(",", "").strip()) for c in comments if (c.get("like_count") or c.get("likes"))]
        total_likes = sum(likes)
        total_comments = len(comments)
        average_likes = round(total_likes / total_comments, 2) if total_comments else 0
        max_likes = max(likes) if likes else 0

        user_data = {
            "total_messages": total_comments,
            "languages_detected": detect_languages(texts),
            "aliases_used": list(set(re.findall(r"@\\w+", " ".join(texts)))),
            "banking_terms": [k for k in ["sbi", "bank"] if k in " ".join(texts).lower()],
            "fraud_signals": [k for k in ["crypto"] if k in " ".join(texts).lower()],
            "interests_detected": extract_interests(texts),
            "behavioral_flags": detect_behavioral_flags(texts),
            "digital_footprint": detect_digital_footprint(texts),
        }

        dynamic_topics = detect_dynamic_topics(texts)
        topicwise_narrative = generate_topicwise_narrative(dynamic_topics, texts)
        paragraph_summaries = generate_paragraph_summaries_from_author_data(user_data, texts)

        flags = user_data["behavioral_flags"]
        risk_score = 100 if len(flags) >= 3 else 60 if len(flags) == 2 else 30 if flags else 0
        risk_level = "High" if risk_score > 80 else "Moderate" if risk_score > 40 else "Low"

        recent_comments = [
            {
                "text": c.get("text", ""),
                "likes": c.get("like_count", 0),
                "commented_at": str(c.get("published_at", "N/A")),
                "channel_name": c.get("channel_name", "Unknown")
            } for c in comments[:5]
        ]

        narrative_summary = (
            f"The user '{author}' has contributed {total_comments} comments. These comments have received a total of {total_likes} likes (avg: {average_likes}). "
            f"Sentiment distribution: {sentiment_counts['positive']} positive, {sentiment_counts['neutral']} neutral, {sentiment_counts['negative']} negative. "
            f"Languages detected: {', '.join(user_data['languages_detected']) or 'unknown'}. Interests: {', '.join(user_data['interests_detected']) or 'none'}."
        )

        return JSONResponse({
            "author": author,
            "total_comments": total_comments,
            "total_likes": total_likes,
            "average_likes": average_likes,
            "max_likes": max_likes,
            "sentiment": sentiment_counts,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "flags": user_data["behavioral_flags"],
            "topics": dynamic_topics,
            "interests": user_data["interests_detected"],
            "summary": narrative_summary + "\n\n" + topicwise_narrative,
            "paragraph_summaries": paragraph_summaries,
            "recent_comments": recent_comments
        })

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"[author_summary] Error processing author '{author}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        

@app.get("/youtube/ner/")
async def youtube_ner(channel_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    query = {"channel_id": channel_id}
    if start_date or end_date:
        query["published_at"] = {}
        if start_date:
            query["published_at"]["$gte"] = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            query["published_at"]["$lte"] = datetime.strptime(end_date, "%Y-%m-%d")

    comments = list(comments_collection.find(query))
    texts = [c.get("text", "") for c in comments if c.get("text")]

    if not texts:
        raise HTTPException(status_code=404, detail="No comments found for this channel")

    entities = extract_named_entities(texts, platform="youtube")
    return JSONResponse({"channel_id": channel_id, "entities": entities})


# --- Telegram NER Endpoint ---
@app.get("/telegram/ner/")
async def telegram_ner(chat_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    query = {"chat_id": chat_id}
    if start_date or end_date:
        query["timestamp"] = {}
        if start_date:
            query["timestamp"]["$gte"] = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            query["timestamp"]["$lte"] = datetime.strptime(end_date, "%Y-%m-%d")

    messages = list(messages_collection.find(query))
    texts = [m.get("text", "") for m in messages if m.get("text")]

    if not texts:
        raise HTTPException(status_code=404, detail="No messages found for this group")

    entities = extract_named_entities(texts, platform="telegram")
    return JSONResponse({"chat_id": chat_id, "entities": entities})

@app.get("/telegram/group_sentiment_summary/")
async def telegram_group_sentiment_summary(chat_id: str):
    try:
        group = groups_collection.find_one({"chat_id": chat_id})
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")

        messages = list(messages_collection.find({"chat_id": chat_id}))
        texts = [m.get("text", "") for m in messages if isinstance(m.get("text", ""), str) and m["text"].strip()]

        if not texts:
            raise HTTPException(status_code=404, detail="No messages found for this group")

        sentiments = categorize_sentiment(texts)
        wordcloud_paths = generate_sentiment_wordclouds_from_categories(sentiments)
        narrative = generate_creative_narrative_summary(sentiments)

        return JSONResponse({
            "chat_id": chat_id,
            "chat_title": group.get("title", "N/A"),
            "sentiment_distribution": {
                "positive": len(sentiments["positive"]),
                "negative": len(sentiments["negative"]),
                "neutral": len(sentiments["neutral"])
            },
            "sample_messages": {
                "positive": sentiments["positive"][:3],
                "negative": sentiments["negative"][:3],
                "neutral": sentiments["neutral"][:3]
            },
            "wordclouds": wordcloud_paths,
            "narrative_summary": narrative,
            "timestamp": datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/geo_analysis/")
async def geo_analysis(req: TimeBoundRequest):
    start, end = get_time_bounds(req.time_filter, req.start_date, req.end_date)
    comments = list(comments_collection.find({"channel_id": req.channel_id}))
    texts = filter_comments(comments, start, end)
    if not texts:
        raise HTTPException(status_code=404, detail="No text found for given time range.")
    return {"geo_mentions": geo_based_analysis(texts)}

@app.post("/get_video_hashtags/")
async def get_video_hashtags(req: VideoRequest):
    youtube_comments = fetch_youtube_comments(req.channel_id, req.video_id)
    if not youtube_comments:
        raise HTTPException(status_code=404, detail="No comments found for this video.")
    hashtags = []
    for comment in youtube_comments:
        hashtags.extend(extract_hashtags(comment))
    return {"hashtags": list(set(hashtags))}
