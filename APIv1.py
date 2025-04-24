from fastapi import FastAPI, Depends, HTTPException, Header, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import logging
import time

# --- Import backend functions ---
from yt_backend import get_channel_data, get_comments_data as yt_get_comments_data, get_video_data, generate_pdf

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("yt-api-service")

# --- FastAPI instance ---
app = FastAPI(
    title="YouTube Intelligence API",
    version="1.1.0",
    description="API for analyzing YouTube channels, videos, and comments for intelligence purposes"
)

# --- On Startup ---
@app.on_event("startup")
async def startup_event():
    logger.info("YouTube Intelligence API is starting up...")

# --- Handle favicon.ico (prevent 404) ---
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")  # You can place your icon in `static/` folder

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Key Setup ---
VALID_API_KEYS = {"YT-INTEL-2025-ACCESS-KEY"}

def verify_api_key(x_api_key: str = Header(..., description="API key for authentication")):
    if x_api_key not in VALID_API_KEYS:
        logger.warning(f"Unauthorized API key used: {x_api_key}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key.")
    return x_api_key

# --- Rate Limiting ---
RATE_LIMIT = {}
MAX_REQUESTS = 100
RATE_LIMIT_WINDOW = 60

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    now = time.time()

    if client_ip in RATE_LIMIT:
        reqs, last = RATE_LIMIT[client_ip]
        if now - last > RATE_LIMIT_WINDOW:
            RATE_LIMIT[client_ip] = [1, now]
        else:
            reqs += 1
            if reqs > MAX_REQUESTS:
                return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded."})
            RATE_LIMIT[client_ip] = [reqs, last]
    else:
        RATE_LIMIT[client_ip] = [1, now]

    return await call_next(request)

# --- Exception Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"status_code": exc.status_code, "detail": exc.detail, "timestamp": datetime.now().isoformat()}}
    )

@app.exception_handler(Exception)
async def general_exception_handler(_: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": {"status_code": 500, "detail": "Internal Server Error", "message": str(exc)}}
    )

# --- Response Models ---
class ChannelSummary(BaseModel):
    channel_id: str
    title: str
    subscribers: str
    video_count: str
    total_views: str
    total_comments: str
    risk_score: float
    flagged_words: List[str]

class CommentData(BaseModel):
    video_id: str
    author: str
    text: str
    sentiment: str

class KeywordsHashtags(BaseModel):
    channel_id: str
    top_keywords: List[str]
    top_hashtags: List[str]

class VideoData(BaseModel):
    video_id: str
    title: str
    views: int
    likes: Optional[int]
    published_at: Optional[str]
    description: Optional[str]
    comment_count: int
    sentiment_summary: Dict[str, int]

# --- Routes ---
@app.get("/", tags=["Meta"])
def root():
    return {"message": "Welcome to the YouTube Intelligence API", "version": "1.1.0", "documentation": "/docs"}

@app.get("/health", tags=["Meta"])
def health_check():
    return {"status": "ok", "message": "API is running", "timestamp": datetime.now().isoformat()}

@app.get("/channel_summary", response_model=ChannelSummary, tags=["Channel"])
def fetch_channel_summary(channel_id: str, _: str = Depends(verify_api_key)):
    logger.info(f"Fetching summary for channel: {channel_id}")
    data = get_channel_data(channel_id)
    if not data or not data.get("channel"):
        raise HTTPException(status_code=404, detail="Channel not found.")
    channel = data["channel"]
    return {
        "channel_id": channel.get("channel_id"),
        "title": channel.get("title", "N/A"),
        "subscribers": str(channel.get("subscribers", "N/A")),
        "video_count": str(channel.get("video_count", len(data.get("videos", [])))),
        "total_views": str(sum(v.get("views", 0) for v in data.get("videos", []))),
        "total_comments": str(len(data.get("comments", []))),
        "risk_score": data.get("channel_risk_score", 0.0),
        "flagged_words": data.get("csam_flags", [])
    }

@app.get("/comments", response_model=List[CommentData], tags=["Comments"])
def fetch_comments(video_id: str, _: str = Depends(verify_api_key)):
    logger.info(f"Fetching comments for video: {video_id}")
    result = yt_get_comments_data(video_id)
    comments = result.get("comments", [])
    sentiments = result.get("sentiments", [])
    return [
        {
            "video_id": c.get("video_id"),
            "author": c.get("author"),
            "text": c.get("text"),
            "sentiment": sentiments[i] if i < len(sentiments) else "NEUTRAL"
        }
        for i, c in enumerate(comments)
    ]

@app.get("/channel_keywords", response_model=KeywordsHashtags, tags=["Channel"])
def fetch_keywords(channel_id: str, _: str = Depends(verify_api_key)):
    logger.info(f"Fetching keywords for channel: {channel_id}")
    data = get_channel_data(channel_id)
    if not data or not data.get("top_keywords"):
        raise HTTPException(status_code=404, detail="Keywords not found.")
    return {
        "channel_id": channel_id,
        "top_keywords": data.get("top_keywords", []),
        "top_hashtags": data.get("top_hashtags", [])
    }

@app.get("/video", response_model=VideoData, tags=["Video"])
def fetch_video(video_id: str, _: str = Depends(verify_api_key)):
    logger.info(f"Fetching video data: {video_id}")
    video = get_video_data(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found.")
    return video

@app.get("/export-pdf", tags=["Export"])
def export_pdf(channel_id: str, _: str = Depends(verify_api_key)):
    logger.info(f"Exporting PDF for channel: {channel_id}")
    pdf_buffer = generate_pdf(channel_id)
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={channel_id}_report.pdf"}
    )
