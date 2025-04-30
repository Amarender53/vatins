from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pymongo import MongoClient, ASCENDING
from deep_translator import GoogleTranslator
from googletrans import Translator
from collections import Counter
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Any
import nltk
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

try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    locale.setlocale(locale.LC_ALL, '')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

YOUTUBE_API_KEY = "AIzaSyCKRTuJPZ1xw3NGuXwUgkXuYz8ZGpcdHE8"
MONGO_URI = "mongodb+srv://vatins:Test123@dev.cjmwsi.mongodb.net/?retryWrites=true&w=majority&appName=dev"

app = FastAPI()
client = MongoClient(MONGO_URI)
db = client["telegram_scraper"]
comments_collection = db["comments"]
channels_collection = db["channels"]
videos_collection = db["videos"]

# ------------------- NLP Models ------------------- #
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
sentiment_analyzer = pipeline("sentiment-analysis", device=-1)
ner_analyzer = pipeline("ner", aggregation_strategy="simple", device=-1)
topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
summarizer = pipeline("summarization")
sentiment_pipeline = pipeline("sentiment-analysis")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
# ------------------- Request Models ------------------- #
class TimeBoundRequest(BaseModel):
    channel_id: str
    time_filter: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class ProfileRequest(BaseModel):
    channel_id: str

translator = Translator()

# Request model
class ChannelRequest(BaseModel):
    channel_id: str

class VideoRequest(BaseModel):
    channel_id: str
    video_id: str



# Helper to translate
if "text" in c and c["text"].strip():
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(c["text"])
        texts.append(translated)
    except:
        texts.append(c["text"])%SZ")
        end = datetime.strptime(last_comment["published_at"], "%Y-%m-%dT%H:%M:%SZ")
        days = max((end - start).days, 1)
        avg_comments_per_day = round(total_comments / days, 2)
    else:
        avg_comments_per_day = 0.0
        start = end = None

    return {
        "tabs": [
            {
                "label": "Total Channels",
                "value": total_channels
            },
            {
                "label": "Channels with Videos",
                "value": len(video_channels)
            },
            {
                "label": "Total Comments",
                "value": total_comments
            },
            {
                "label": "Unique Authors",
                "value": len(unique_authors)
            },
            {
                "label": "Avg. Comments/Day",
                "value": avg_comments_per_day,
                "time_range_days": (end - start).days if start and end else 0
            }
        ]
    }
def get_time_bounds(time_filter: str, start_date: Optional[str], end_date: Optional[str]) -> (Optional[datetime], Optional[datetime]):
    now = datetime.utcnow()
    if time_filter == "last_24_hours":
        return now - timedelta(days=1), now
    elif time_filter == "last_week":
        return now - timedelta(weeks=1), now
    elif time_filter == "last_month":
        return now - timedelta(days=30), now
    elif time_filter == "custom":
        if not start_date or not end_date:
            raise HTTPException(status_code=400, detail="Custom range requires start_date and end_date.")
        return datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d")
    return None, None

def fetch_comments_from_db(channel_id: str) -> List[Dict]:
    comments = comments_collection.find({"channel_id": channel_id})
    return list(comments)

# Function to filter comments based on date range and translate them
def filter_comments(comments: List[Dict], start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[str]:
    def in_range(published_at: str) -> bool:
        try:
            dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")  # Example format
            if start and dt < start:
                return False
            if end and dt > end:
                return False
            return True
        except Exception as e:
            print(f"Error parsing date: {e}")
            return False


    translated_texts = []
    for c in comments:
        try:
            text = c.get("text")
            published_at = c.get("published_at")

            if text and isinstance(text, str) and text.strip():
                if published_at and in_range(published_at):
                    translated = translator.translate(text, dest='en')
                    translated_texts.append(translated.text)
        except Exception as e:
            print(f"Error processing comment {c}: {e}")
            continue

    return translated_texts


    translated_texts = []
    for c in comments:
        try:
            text = c.get("text")
            published_at = c.get("published_at")

            if text and isinstance(text, str) and text.strip():
                if published_at and in_range(published_at):
                    translated = translator.translate(text, dest='en')
                    translated_texts.append(translated.text)
        except Exception as e:
            print(f"Error processing comment {c}: {e}")
            continue

    return translated_texts

# Function for sentiment analysis of a list of comments
def analyze_sentiment(comments: List[str]) -> Dict[str, List[str]]:
    sentiment_summary = {"positive": [], "negative": [], "neutral": []}

    # Perform sentiment analysis for each comment
    for comment in comments:
        blob = TextBlob(comment)
        sentiment_score = blob.sentiment.polarity

        # Classify sentiment
        if sentiment_score > 0:
            sentiment_summary["positive"].append(comment)
        elif sentiment_score < 0:
            sentiment_summary["negative"].append(comment)
        else:
            sentiment_summary["neutral"].append(comment)

    return sentiment_summary
def generate_insights(sentiment_summary: Dict[str, List[str]]) -> str:
    positive_count = len(sentiment_summary["positive"])
    negative_count = len(sentiment_summary["negative"])
    neutral_count = len(sentiment_summary["neutral"])

    insights = []

    # Majority Positive
    if positive_count > negative_count:
        insights.append("Majority Positive: Most comments express admiration, patriotism, or appreciation.")

    # Minor Toxicity Alert
    if negative_count > 0:
        insights.append("Minor Toxicity Alert: Some comments imply political accusations or strong negative sentiments.")

    # Neutral/Mixed
    if neutral_count > 0:
        insights.append("Neutral/Mixed: Some comments are generic or unclear in sentiment, often referencing concepts or people.")

    return "\n".join(insights)
def analyze_sentiment(texts: list) -> dict:
    results = sentiment_pipeline(texts)
    pos = sum(1 for r in results if r["label"] == "POSITIVE")
    neg = sum(1 for r in results if r["label"] == "NEGATIVE")
    neu = len(results) - (pos + neg)
    return {"positive": pos, "negative": neg, "neutral": neu}

def analyze_sentiment_summary(texts: List[str]) -> dict:
    results = sentiment_pipeline(texts)

    categorized = {
        "positive": [],
        "negative": [],
        "neutral": []
    }

    for text, result in zip(texts, results):
        label = result["label"]
        if label == "POSITIVE":
            categorized["positive"].append(text)
        elif label == "NEGATIVE":
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

def summarize_text(text: str) -> str:
    if len(text) < 50:
        return text
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

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


@app.post("/fetch_video_data/")
async def fetch_video_data(req: VideoRequest):
    fetch_and_store_video_metadata(req.channel_id, req.video_id)
    comments = fetch_youtube_comments(req.channel_id, req.video_id)
    return {"message": "Video data and comments fetched and stored successfully.", "total_comments": len(comments)}

# Endpoint: Channel Sentiment and Summary
@app.post("/channel_summary/")
async def channel_summary(request: ChannelRequest, x_api_key: str = Header(...)):
    # Dummy auth for simplicity
    if x_api_key != "your_api_key":
        raise HTTPException(status_code=403, detail="Unauthorized")

    # Filter comments for this channel
    comments = list(comments_collection.find({"channel_id": request.channel_id}))

    if not comments:
        raise HTTPException(status_code=404, detail="No comments found for this channel ID")

    comment_texts = [translate_text(c["text"]) for c in comments if "text" in c]
    video_titles = list(set(c.get("video_title", "") for c in comments if "video_title" in c))

    # Analyze sentiment and summarize
    sentiment = analyze_sentiment(comment_texts)
    summary_input = " ".join(comment_texts[:10])  # limit input length
    comment_summary = summarize_text(summary_input)
    title_summary = summarize_text(" ".join(video_titles)) if video_titles else "No video titles."

    return JSONResponse(content={
        "channel_id": request.channel_id,
        "sentiment": sentiment,
        "comment_summary": comment_summary,
        "video_title_summary": title_summary
    })
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

        # Perform sentiment analysis
        sentiment_summary = analyze_sentiment(translated_comments)

        # Generate insights
        insights = generate_insights(sentiment_summary)

        # Preparare examples for each sentiment type
        positive_examples = sentiment_summary["positive"][:5]  # Example of positive comments
        negative_examples = sentiment_summary["negative"][:1]  # Example of negative comment
        neutral_examples = sentiment_summary["neutral"][:3]  # Example of neutral comments

        # Return the sentiment summary with insights and examples
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

