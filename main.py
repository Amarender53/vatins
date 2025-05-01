from fastapi import FastAPI, HTTPException, Query, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient, ASCENDING
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from deep_translator import GoogleTranslator
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Any
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
import logging
from googleapiclient.discovery import build
from dotenv import load_dotenv

# ------------------- Setup ------------------- #
load_dotenv()
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
words = word_tokenize("This is a sample sentence.")
filtered = [w for w in words if w.isalpha() and w not in stop_words]

try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    locale.setlocale(locale.LC_ALL, '')
    
def format_number(num: int) -> str:
    try:
        return locale.format_string("%d", num, grouping=True)
    except Exception:
        return str(num)

API_KEY = os.getenv("API_KEY")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

YOUTUBE_API_KEY = "AIzaSyCKRTuJPZ1xw3NGuXwUgkXuYz8ZGpcdHE8"
MONGO_URI = "mongodb+srv://vatins:Test123@dev.cjmwsi.mongodb.net/?retryWrites=true&w=majority&appName=dev"

app = FastAPI()
client = MongoClient(MONGO_URI)
db = client["telegram_scraper"]
channels_collection = db["channels"]     # YouTube
videos_collection = db["videos"]         # YouTube
comments_collection = db["comments"]     # YouTube
groups_collection = db["groups"]         # Telegram
messages_collection = db["messages"]     # Telegram
users_collection = db["users"]           #Telegram

# ---- YouTube & Telegram Specific Sentiment Models ----
yt_sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tg_sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# ---- NER Pipelines ----
yt_ner_analyzer = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
tg_ner_analyzer = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl", grouped_entities=True)
ner_analyzer = pipeline("ner", grouped_entities=True)  # fallback general-purpose NER

# ---- Summarizer ----
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")
MAX_LEN = tokenizer.model_max_length
MAX_TOKENS = 1000

# ---- Semantic Search Model ----
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Language Translator ----
translator = GoogleTranslator(source='auto', target='en')

# ---- Topic Classifier ----
topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ---- VADER Sentiment (for rule-based English sentiment scoring) ----
sentiment_analyzer = SentimentIntensityAnalyzer()

class TimeBoundRequest(BaseModel):
    channel_id: str
    time_filter: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class ProfileRequest(BaseModel):
    channel_id: str


# Request model
class ChannelRequest(BaseModel):
    channel_id: str

class VideoRequest(BaseModel):
    channel_id: str
    video_id: str

class CommentItem(BaseModel):
    text: str
    likes: Optional[int]
    commented_at: Optional[str]

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

class AuthorSummaryResponse(BaseModel):
    author: str
    total_comments: int
    total_likes: int
    average_likes: float
    max_likes: int
    recent_comments: List[CommentItem]
    sentiment: Dict[str, int]
    interests: List[str]
    summary: str

def extract_interests(texts):
    nlp = spacy.load("en_core_web_sm")
    combined = " ".join(texts)
    doc = nlp(combined)
    nouns = [chunk.text.lower() for chunk in doc.noun_chunks]
    common = Counter(nouns).most_common(10)
    return [w for w, _ in common]
    
def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text

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

def generate_summary(texts: List[str]) -> str:
    if not texts:
        return "No meaningful summary available."

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
    avg_score = sum(sentiment_scores) / len(sentiment_scores)

    if avg_score >= 0.5:
        tone = "supportive"
    elif avg_score <= -0.5:
        tone = "critical"
    else:
        tone = "neutral"

    topic = extract_main_topic(texts)
    return f"Most comments are {tone} and focused on {topic}."

def fetch_telegram_messages(group_id: Optional[str] = None, user_id: Optional[str] = None,
                            start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[str]:
    query = {}
    if group_id:
        query["group_id"] = group_id
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

def fetch_telegram_messages(group_id: Optional[str], user_id: Optional[str], start: Optional[datetime], end: Optional[datetime]) -> List[str]:
    query = {}
    if group_id:
        query["group_id"] = group_id
    if user_id:
        query["user_id"] = user_id
    if start or end:
        query["timestamp"] = {}
        if start:
            query["timestamp"]["$gte"] = start
        if end:
            query["timestamp"]["$lte"] = end
    messages = list(messages_collection.find(query))
    return [m.get("text", "") for m in messages if m.get("text")]

def translate_text(text: str) -> str:
    try:
        if detect(text) != 'en':
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except Exception:
        return text


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
    # Convert input strings to datetime objects
    start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    date_filter = {}
    if start:
        date_filter["commented_at"] = {"$gte": start}
    if end:
        date_filter.setdefault("commented_at", {}).update({"$lte": end})

    # Distinct channel IDs from videos
    video_channels = videos_collection.distinct("channel_id")

    # Total channels (regardless of comment activity)
    total_channels = channels_collection.count_documents({})

    # Comments-related stats
    total_comments = comments_collection.count_documents(date_filter)
    unique_authors = comments_collection.distinct("author", date_filter)

    # Find first and last comment with valid date
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
            avg_comments_per_day = round(total_comments / days, 2)
        except Exception as e:
            print(f"Error parsing comment timestamps: {e}")

    return {
        "total_channels": total_channels,
        "channels_with_videos": len(video_channels),
        "total_comments": total_comments,
        "unique_commenters": len(unique_authors),
        "avg_comments_per_day": avg_comments_per_day
    }



# Function for sentiment analysis of a list of comments
def analyze_sentiment_summary_vader(texts: List[str]) -> dict:
    analyzer = SentimentIntensityAnalyzer()
    categorized = {"positive": [], "negative": [], "neutral": []}

    for text in texts:
        score = analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            categorized["positive"].append(text)
        elif score <= -0.05:
            categorized["negative"].append(text)
        else:
            categorized["neutral"].append(text)

    insights = []
    if categorized["negative"]:
        insights.append("Minor Toxicity Alert: Some comments express strong or controversial opinions.")
    if len(categorized["positive"]) > len(texts) // 2:
        insights.append("Majority Positive: Comments mostly express admiration, patriotism, or appreciation.")
    if categorized["neutral"]:
        insights.append("Neutral/Mixed: Some comments are vague, metaphorical, or non-expressive.")

    return {
        "Sentiment Summary": {
            "Positive": {
                "count": len(categorized["positive"]),
                "examples": categorized["positive"][:3]
            },
            "Negative": {
                "count": len(categorized["negative"]),
                "examples": categorized["negative"][:3]
            },
            "Neutral": {
                "count": len(categorized["neutral"]),
                "examples": categorized["neutral"][:3]
            }
        },
        "Insights": insights
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


def named_entities(texts: List[str]) -> List[Dict]:
    return ner_analyzer(" ".join(texts[:10]))

def detect_topics(texts: List[str]) -> List[str]:
    labels = ["cybercrime", "scam", "finance", "technology", "geography", "banking", "bot", "fraud"]
    return topic_classifier(" ".join(texts[:10]), candidate_labels=labels)["labels"]

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

@app.get("/telegram/user_summary/")
async def telegram_user_summary():
    users = list(users_collection.find({}))
    return {"total_users": len(users), "users": users[:10]}

@app.get("/telegram/group_summary/")
async def telegram_group_summary():
    groups = list(groups_collection.find({}))
    return {"total_groups": len(groups), "groups": groups[:10]}

@app.post("/telegram/message_summary/")
async def telegram_message_summary(req: TimeBoundRequest):
    start, end = get_time_bounds(req.time_filter, req.start_date, req.end_date)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.get("/Dashboard/", tags=["Analytics"])
async def cumulative_statistics(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Get overall cumulative statistics including:
    - Total Channels
    - Total Comments
    - Unique Authors
    - Average Comments Per Day
    Optionally filter by start_date and end_date in "YYYY-MM-DD" format.
    """
    try:
        stats = get_cumulative_stats(start_date, end_date)
        return stats
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/channel_summary/")
async def channel_summary(request: ChannelRequest, x_api_key: str = Header(...)):
    try:
        # --- API Key Auth ---
        if x_api_key != API_KEY:
            raise HTTPException(status_code=403, detail="Unauthorized: Invalid API Key")

        # --- Get Channel ---
        channel = channels_collection.find_one({"channel_id": request.channel_id})
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")

        # --- Get Videos ---
        videos = list(videos_collection.find({"channel_id": request.channel_id}))
        video_ids = [v["video_id"] for v in videos if "video_id" in v]
        video_titles = [v.get("title", "") for v in videos if isinstance(v.get("title", ""), str)]

        # --- Get Comments ---
        comments_cursor = comments_collection.find({"video_id": {"$in": video_ids}})
        comments = [c.get("text", "") for c in comments_cursor if isinstance(c.get("text", ""), str) and c["text"].strip()]

        if not comments:
            raise HTTPException(status_code=404, detail="No comments found for this channel")

        # --- Sentiment Analysis ---
        sentiment_result = analyze_sentiment_summary_vader(comments)

        # --- Summaries ---
        comment_text = " ".join(comments[:100])
        if not comment_text.strip():
            comment_summary = "No comments to summarize."
        else:
            comment_summary = summarize_text(comment_text)
        title_summary = summarize_text(" ".join(video_titles[:20])) if video_titles else "No video titles available."

        # --- Aggregates ---
        total_views = sum(v.get("views", 0) for v in videos)
        total_likes = sum(v.get("likes", 0) for v in videos)
        total_videos = len(videos)
        total_comments = len(comments)

        return JSONResponse(content={
            "channel_id": request.channel_id,
            "channel_name": channel.get("title", "No name"),
            "total_subscribers": format_number(channel.get("subscribers", 0)),
            "total_videos_uploaded": format_number(total_videos),
            "total_likes": format_number(total_likes),
            "total_views": format_number(total_views),
            "total_comments": format_number(total_comments),
            "sentiment_summary": sentiment_result,
            "comment_summary": comment_summary,
            "video_title_summary": title_summary
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()  # Print full stack trace to console
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )

@app.post("/fetch_video_data/")
async def fetch_video_data(req: VideoRequest):
    fetch_and_store_video_metadata(req.channel_id, req.video_id)
    comments = fetch_youtube_comments(req.channel_id, req.video_id)
    return {"message": "Video data and comments fetched and stored successfully.", "total_comments": len(comments)}

@app.get("/video_sentiment_summary/")
async def video_sentiment_summary(video_id: str, start_date: str = None, end_date: str = None):
    try:
        # Convert start and end to datetime objects if provided
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        # Fetch comments from MongoDB based on video_id
        comments = fetch_comments_by_video_id(video_id)  # <-- You need this function in your DB layer

        # Filter and translate comments based on the date range
        translated_comments = filter_comments(comments, start=start_dt, end=end_dt)

        if not translated_comments:
            raise HTTPException(status_code=404, detail="No comments found or unable to translate")

        # Perform sentiment analysis
        result = analyze_sentiment_summary_vader(translated_comments)

        insights = result["Insights"]
        summary_data = result["Sentiment Summary"]

        positive_examples = summary_data["Positive"]["examples"]
        negative_examples = summary_data["Negative"]["examples"]
        neutral_examples = summary_data["Neutral"]["examples"]

        return {
            "video_sentiment_summary": {
                "positive_count": len(positive_examples),
                "negative_count": len(negative_examples),
                "neutral_count": len(neutral_examples),
                "positive_examples": positive_examples,
                "negative_examples": negative_examples,
                "neutral_examples": neutral_examples,
                "insights": insights,
            }
        }

    except Exception as e:
        print(f"Error processing video sentiment summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/author_summary/{author}", response_model=AuthorSummaryResponse)
def get_author_summary(author: str):
    comments_cursor = comments_collection.find(
        {"author": {"$regex": re.escape(author), "$options": "i"}}
    ).sort("commented_at", -1)

    comments = list(comments_cursor)

    if not comments:
        raise HTTPException(status_code=404, detail="No comments found for this author")

    total_comments = len(comments)

    # Recent 5 comments
    recent_comments = [{
        "text": c.get("text", ""),
        "likes": c.get("likes", 0),
        "commented_at": c.get("commented_at")
    } for c in comments[:5]]

    # Likes stats
    likes = [c.get("likes", 0) for c in comments if isinstance(c.get("likes", 0), (int, float))]
    total_likes = sum(likes)
    average_likes = round(total_likes / total_comments, 2)
    max_likes = max(likes) if likes else 0

    # Comment texts for analysis
    texts = [c.get("text", "") for c in comments if isinstance(c.get("text", ""), str) and c.get("text", "").strip()]

    # Sentiment analysis using VADER
    analyzer = SentimentIntensityAnalyzer()
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for text in texts:
        score = analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            sentiment_counts["positive"] += 1
        elif score <= -0.05:
            sentiment_counts["negative"] += 1
        else:
            sentiment_counts["neutral"] += 1

    # Optional extras (if you have these functions)
    interests = extract_interests(texts) if 'extract_interests' in globals() else []
    summary = generate_summary(texts) if 'generate_summary' in globals() else "Summary not available."

    return {
        "author": author,
        "total_comments": total_comments,
        "total_likes": total_likes,
        "average_likes": average_likes,
        "max_likes": max_likes,
        "recent_comments": recent_comments,
        "sentiment": sentiment_counts,
        "interests": interests,
        "summary": summary
    }



@app.get("/sentiment_summary/")
async def sentiment_summary(channel_id: str, start_date: str = None, end_date: str = None):
    try:
        # Convert start and end to datetime objects if provided
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        # Fetch comments from MongoDB based on channel_id
        comments = fetch_comments_from_db(channel_id)

        # Filter and translate comments based on the date range
        translated_comments = filter_comments(comments, start=start_dt, end=end_dt)

        if not translated_comments:
            raise HTTPException(status_code=404, detail="No comments found or unable to translate")

        # Perform sentiment analysis (returns structured summary + insights)
        result = analyze_sentiment_summary_vader(translated_comments)

        insights = result["Insights"]
        summary_data = result["Sentiment Summary"]

        positive_examples = summary_data["Positive"]["examples"]
        negative_examples = summary_data["Negative"]["examples"]
        neutral_examples = summary_data["Neutral"]["examples"]

        return {
            "sentiment_summary": {
                "positive_count": len(positive_examples),
                "negative_count": len(negative_examples),
                "neutral_count": len(neutral_examples),
                "positive_examples": positive_examples,
                "negative_examples": negative_examples,
                "neutral_examples": neutral_examples,
                "insights": insights,
            }
        }

    except Exception as e:
        print(f"Error processing sentiment summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Root
@app.get("/", response_class=HTMLResponse)
async def home():
    return "<h1>YouTube Intelligence API</h1><p>Use /docs for Swagger UI.</p>"


@app.post("/analyze_ner/")
async def ner_analysis(req: TimeBoundRequest):
    start, end = get_time_bounds(req.time_filter, req.start_date, req.end_date)
    comments = list(comments_collection.find({"channel_id": req.channel_id}))
    texts = filter_comments(comments, start, end)
    if not texts:
        raise HTTPException(status_code=404, detail="No text found for given time range.")
    return named_entities(texts)

@app.post("/detect_topics/")
async def topic_analysis(req: TimeBoundRequest):
    start, end = get_time_bounds(req.time_filter, req.start_date, req.end_date)
    comments = list(comments_collection.find({"channel_id": req.channel_id}))
    texts = filter_comments(comments, start, end)
    if not texts:
        raise HTTPException(status_code=404, detail="No text found for given time range.")
    return detect_topics(texts)

@app.get("/telegram_sentiment/")
async def telegram_sentiment(group_id: Optional[str] = None, user_id: Optional[str] = None,
                             start_date: Optional[str] = None, end_date: Optional[str] = None):
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        messages = fetch_telegram_messages(group_id, user_id, start, end)
        if not messages:
            raise HTTPException(status_code=404, detail="No messages found.")
        return analyze_sentiment_summary_vader(messages)
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
