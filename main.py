# ------------------- Imports ------------------- #
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from fastapi.responses import StreamingResponse
from typing import List, Dict
from datetime import datetime
import torch
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import nltk

# ------------------- Setup ------------------- #
nltk.download('punkt')

app = FastAPI(
    title="YouTube AI Intelligence API",
    description="Generates Channel/User Summary + PDF directly.",
    version="1.0.0"
)

# ------------------- MongoDB Connection ------------------- #
MONGO_URI = "mongodb+srv://vatins:Test123@dev.cjmwsi.mongodb.net/?retryWrites=true&w=majority&appName=dev"
try:
    client = MongoClient(MONGO_URI)
    db = client["telegram_scraper"]
    comments_collection = db["comments"]
except Exception as e:
    print(f"MongoDB connection error: {e}")
    raise

# ------------------- Load NLP Models ------------------- #
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis")
ner_analyzer = pipeline("ner", aggregation_strategy="simple")
topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------- Helper Functions ------------------- #

def fetch_channel_statistics(channel_id: str) -> Dict:
    total_comments = comments_collection.count_documents({"channel_id": channel_id})
    unique_users = len(comments_collection.distinct("author", {"channel_id": channel_id}))
    
    first_comment = comments_collection.find_one({"channel_id": channel_id}, sort=[("timestamp", 1)])
    last_comment = comments_collection.find_one({"channel_id": channel_id}, sort=[("timestamp", -1)])
    
    if first_comment and last_comment:
        total_days = max((last_comment["timestamp"] - first_comment["timestamp"]).days, 1)
    else:
        total_days = 1
        
    avg_comments_per_day = round(total_comments / total_days, 2)
    
    return {
        "total_comments": total_comments,
        "unique_users": unique_users,
        "avg_comments_per_day": avg_comments_per_day
    }

def summarize_text(text: str) -> str:
    if len(text) < 50:
        return text
    result = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return result[0]['summary_text']

def analyze_sentiment(texts: List[str]) -> Dict:
    results = sentiment_analyzer(texts)
    pos = sum(1 for r in results if r['label'] == 'POSITIVE')
    neg = sum(1 for r in results if r['label'] == 'NEGATIVE')
    neu = len(results) - (pos + neg)
    return {"positive": pos, "negative": neg, "neutral": neu}

def extract_named_entities(text: str) -> Dict:
    entities = ner_analyzer(text)
    grouped = {}
    for entity in entities:
        label = entity['entity_group']
        grouped.setdefault(label, []).append(entity['word'])
    return grouped

def detect_topic(text: str) -> Dict:
    candidate_topics = ["Politics", "Religion", "Finance", "Technology", "Crime", "Sports", "Education"]
    result = topic_classifier(text, candidate_labels=candidate_topics)
    return {"detected_topic": result['labels'][0], "confidence": round(result['scores'][0], 2)}

def semantic_search(corpus: List[str], query: str, top_k: int = 5) -> List[Dict]:
    corpus_embeddings = semantic_model.encode(corpus, convert_to_tensor=True)
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(query_embedding, corpus_embeddings)
    top_results = torch.topk(cos_scores, k=min(top_k, len(corpus)))

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        results.append({"text": corpus[idx], "score": round(float(score), 3)})
    return results

def build_pdf_report(data: Dict) -> bytes:
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50
    
    # Title
    p.setFont("Helvetica-Bold", 20)
    p.drawCentredString(width/2, y, "YouTube Channel Intelligence Report")
    y -= 40

    p.setFont("Helvetica", 12)

    # Section: Channel Statistics
    p.drawString(50, y, "1. Channel Statistics:")
    y -= 20
    for key, value in data['channel_stats'].items():
        p.drawString(70, y, f"{key.replace('_', ' ').capitalize()}: {value}")
        y -= 20

    # Section: User Profile Summary
    y -= 10
    p.drawString(50, y, "2. User Profile Summary:")
    y -= 20
    for line in data['profile_summary'].split('. '):
        p.drawString(70, y, line.strip())
        y -= 15

    # Section: Sentiment Analysis
    y -= 10
    p.drawString(50, y, "3. Sentiment Analysis:")
    y -= 20
    for k, v in data['sentiment'].items():
        p.drawString(70, y, f"{k.capitalize()}: {v}")
        y -= 20

    # Section: Named Entities
    y -= 10
    p.drawString(50, y, "4. Named Entities:")
    y -= 20
    for entity, values in data['named_entities'].items():
        p.drawString(70, y, f"{entity}: {', '.join(values[:5])}")
        y -= 20

    # Section: Topic Detection
    y -= 10
    p.drawString(50, y, "5. Topic Detection:")
    y -= 20
    p.drawString(70, y, f"Topic: {data['topic']['detected_topic']} ({data['topic']['confidence']*100:.1f}%)")
    y -= 30

    # Section: Semantic Search
    p.drawString(50, y, "6. Semantic Search (Top Matches):")
    y -= 20
    for item in data['semantic_results']:
        p.drawString(70, y, f"- {item['text'][:70]}... (Score: {item['score']})")
        y -= 20
        if y < 100:
            p.showPage()
            y = height - 50

    p.save()
    buffer.seek(0)
    return buffer

# ------------------- Request Models ------------------- #

class GenerateRequest(BaseModel):
    channel_id: str
    query: str

# ------------------- API Endpoint ------------------- #

@app.post("/generate_summary/")
async def generate_summary(req: GenerateRequest):
    comments = list(comments_collection.find({"channel_id": req.channel_id}))
    if not comments:
        raise HTTPException(status_code=404, detail="No comments found.")

    texts = [c.get('text', '') for c in comments if 'text' in c]
    combined_text = " ".join(texts)

    # Analyze
    channel_stats = fetch_channel_statistics(req.channel_id)
    profile_summary = summarize_text(combined_text)
    sentiment = analyze_sentiment(texts)
    named_entities = extract_named_entities(combined_text)
    topic = detect_topic(combined_text)
    semantic_results = semantic_search(texts, req.query)

    # Build report data
    report_data = {
        "channel_stats": channel_stats,
        "profile_summary": profile_summary,
        "sentiment": sentiment,
        "named_entities": named_entities,
        "topic": topic,
        "semantic_results": semantic_results
    }

    # Build PDF
    pdf_buffer = build_pdf_report(report_data)

    return StreamingResponse(pdf_buffer, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=channel_summary.pdf"})

