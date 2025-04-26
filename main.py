# # from fastapi import FastAPI, Request, Form
# # from fastapi.responses import HTMLResponse, FileResponse
# # from io import BytesIO
# # from datetime import datetime
# # import tempfile
# # import nltk
# # import os
# # from fpdf import FPDF
# # import matplotlib.pyplot as plt
# # from wordcloud import WordCloud
# # from pymongo import MongoClient
# # from transformers import pipeline
# # from sentence_transformers import SentenceTransformer
# # from langdetect import detect
# # from keybert import KeyBERT
# # from googleapiclient.discovery import build

# # # Setup
# # nltk.download('punkt')
# # sentiment_pipeline = pipeline("sentiment-analysis")
# # ner_pipeline = pipeline("ner", aggregation_strategy="simple")
# # summarizer_pipeline = pipeline("summarization")
# # translator_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
# # kw_model = KeyBERT(model=SentenceTransformer('all-MiniLM-L6-v2'))

# # YOUTUBE_API_KEY = "AIzaSyCKRTuJPZ1xw3NGuXwUgkXuYz8ZGpcdHE8"
# # youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
# # client = MongoClient("mongodb://localhost:27017/")
# # db = client["youtube_data"]

# # app = FastAPI()

# # @app.get("/", response_class=HTMLResponse)
# # async def form_page():
# #     return """
# #         <h2>YouTube Analyzer</h2>
# #         <form action="/analyze/" method="post">
# #             <label>Channel ID:</label><br>
# #             <input type="text" name="channel_id" required><br><br>
# #             <label>Video ID:</label><br>
# #             <input type="text" name="video_id" required><br><br>
# #             <input type="submit" value="Analyze">
# #         </form>
# #     """

# # # ---------- Helper Functions ----------

# # def get_channel_info(channel_id):
# #     try:
# #         data = youtube.channels().list(part="snippet,contentDetails", id=channel_id).execute()
# #         info = data['items'][0]
# #         return {
# #             "title": info['snippet']['title'],
# #             "upload_playlist": info['contentDetails']['relatedPlaylists']['uploads']
# #         }
# #     except:
# #         return None

# # def get_video_title(video_id):
# #     try:
# #         data = youtube.videos().list(part="snippet", id=video_id).execute()
# #         return data['items'][0]['snippet']['title']
# #     except:
# #         return "Unknown Video Title"

# # def get_comments(video_id):
# #     comments = []
# #     try:
# #         res = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100).execute()
# #         for item in res["items"]:
# #             text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
# #             if len(text.strip()) > 3:
# #                 comments.append(text)
# #     except:
# #         pass
# #     return comments

# # def safe_summarize(text, chunk_size=400):
# #     sentences = nltk.sent_tokenize(text)
# #     chunks = []
# #     current_chunk = ""

# #     for sentence in sentences:
# #         if len(current_chunk) + len(sentence) <= chunk_size:
# #             current_chunk += " " + sentence
# #         else:
# #             chunks.append(current_chunk.strip())
# #             current_chunk = sentence

# #     if current_chunk:
# #         chunks.append(current_chunk.strip())

# #     summaries = []
# #     for chunk in chunks:
# #         try:
# #             result = summarizer_pipeline(chunk, max_length=70, min_length=30, do_sample=False)
# #             summaries.append(result[0]['summary_text'])
# #         except:
# #             continue

# #     final_summary = " ".join(summaries)
# #     return final_summary

# # def perform_analysis(channel_id, video_id):
# #     comments = get_comments(video_id)
# #     if not comments:
# #         raise Exception("No comments found for this video.")

# #     translated = [translator_pipeline(c)[0]['translation_text'] for c in comments[:30]]
# #     text_for_summary = " ".join(translated)
# #     summary = safe_summarize(text_for_summary)

# #     sentiments = sentiment_pipeline(translated[:30])
# #     ner_data = ner_pipeline(" ".join(translated[:10]))

# #     sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
# #     for s in sentiments:
# #         label = s['label'].upper()
# #         sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

# #     topics = kw_model.extract_keywords(" ".join(translated), keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
# #     topic_list = [t[0] for t in topics]

# #     entities = list(set([f"{e['word']} ({e['entity_group']})" for e in ner_data]))

# #     return summary, sentiment_counts, entities, topic_list

# # def create_pdf(channel_id, channel_title, video_title, summary, sentiment_data, entities, topics):
# #     pdf = FPDF()
# #     pdf.add_page()
# #     pdf.set_auto_page_break(auto=True, margin=15)

# #     pdf.set_font("Arial", "B", 16)
# #     pdf.set_text_color(220, 50, 50)
# #     pdf.cell(0, 10, "RedPanda AI - YouTube Intelligence Analysis", ln=True, align="C")
# #     pdf.set_text_color(0)
# #     pdf.set_font("Arial", "", 12)
# #     pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
# #     pdf.ln(10)

# #     def section(title, content):
# #         pdf.set_font("Arial", "B", 14)
# #         pdf.cell(0, 10, title, ln=True)
# #         pdf.set_font("Arial", "", 12)
# #         pdf.multi_cell(0, 8, content)
# #         pdf.ln(3)

# #     section("1. Personal Details", f"Channel ID: {channel_id}\nChannel Title: {channel_title}\nVideo Title: {video_title}")
# #     section("2. Summary", summary)
# #     section("3. Sentiments", str(sentiment_data))
# #     section("4. Entities", ", ".join(entities))
# #     section("5. Topics", ", ".join(topics))

# #     pdf_bytes = BytesIO()
# #     pdf.output(pdf_bytes)
# #     pdf_bytes.seek(0)
# #     return pdf_bytes

# # # ---------- Updated Endpoints ----------

# # @app.post("/analyze/", response_class=HTMLResponse)
# # async def analyze_form(request: Request, channel_id: str = Form(...), video_id: str = Form(...)):
# #     try:
# #         channel_info = get_channel_info(channel_id)
# #         video_title = get_video_title(video_id)
# #         summary, sentiments, entities, topics = perform_analysis(channel_id, video_id)

# #         request.state.last_analysis = {
# #             "channel_id": channel_id,
# #             "channel_title": channel_info['title'] if channel_info else "Unknown Channel",
# #             "video_title": video_title,
# #             "summary": summary,
# #             "sentiments": sentiments,
# #             "entities": entities,
# #             "topics": topics
# #         }

# #         return f"""
# #             <h2>Analysis Complete!</h2>
# #             <p><b>Channel Title:</b> {request.state.last_analysis['channel_title']}</p>
# #             <p><b>Video Title:</b> {request.state.last_analysis['video_title']}</p>
# #             <form action="/generate_pdf/" method="post">
# #                 <input type="submit" value="Download PDF Report">
# #             </form>
# #         """

# #     except Exception as e:
# #         return f"<h3>Error:</h3><p>{str(e)}</p>"

# # @app.post("/generate_pdf/", response_class=FileResponse)
# # async def generate_pdf_endpoint(request: Request):
# #     try:
# #         analysis = request.state.last_analysis

# #         pdf_bytes = create_pdf(
# #             analysis['channel_id'],
# #             analysis['channel_title'],
# #             analysis['video_title'],
# #             analysis['summary'],
# #             analysis['sentiments'],
# #             analysis['entities'],
# #             analysis['topics']
# #         )

# #         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
# #             temp_pdf.write(pdf_bytes.read())
# #             temp_pdf_path = temp_pdf.name

# #         return FileResponse(temp_pdf_path, filename="YouTube_Analysis_Report.pdf", media_type="application/pdf")

# #     except Exception as e:
# #         return HTMLResponse(content=f"<h3>Error generating PDF:</h3><p>{str(e)}</p>")




# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse, FileResponse
# from io import BytesIO
# from datetime import datetime
# import tempfile
# import os
# import json
# import nltk
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# from pymongo import MongoClient
# from fpdf import FPDF
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# from langdetect import detect
# from keybert import KeyBERT
# from googleapiclient.discovery import build

# # --- Setup
# nltk.download('punkt')
# sentiment_pipeline = pipeline("sentiment-analysis")
# ner_pipeline = pipeline("ner", aggregation_strategy="simple")
# summarizer_pipeline = pipeline("summarization")
# translator_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
# kw_model = KeyBERT(model=SentenceTransformer('all-MiniLM-L6-v2'))

# YOUTUBE_API_KEY = "AIzaSyCKRTuJPZ1xw3NGuXwUgkXuYz8ZGpcdHE8"
# youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
# client = MongoClient("mongodb://localhost:27017/")
# db = client["youtube_data"]

# app = FastAPI()

# # --- Helper Functions
# def format_number(num):
#     return f"{num:,}"

# def get_channel_info(channel_id):
#     try:
#         data = youtube.channels().list(part="snippet,statistics,contentDetails", id=channel_id).execute()
#         info = data['items'][0]
#         return {
#             "title": info['snippet']['title'],
#             "channel_name": info['snippet']['title'],
#             "subscribers": info['statistics'].get('subscriberCount', "N/A"),
#             "total_views": info['statistics'].get('viewCount', "N/A"),
#             "video_count": info['statistics'].get('videoCount', "N/A"),
#             "total_comments": info['statistics'].get('commentCount', "N/A")
#         }
#     except:
#         return None

# def get_video_title(video_id):
#     try:
#         data = youtube.videos().list(part="snippet", id=video_id).execute()
#         return data['items'][0]['snippet']['title']
#     except:
#         return "Unknown Video Title"

# def get_comments(video_id):
#     comments = []
#     try:
#         res = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100).execute()
#         for item in res["items"]:
#             text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
#             if len(text.strip()) > 3:
#                 comments.append(text)
#     except:
#         pass
#     return comments

# def safe_summarize(text, chunk_size=400):
#     sentences = nltk.sent_tokenize(text)
#     chunks = []
#     current_chunk = ""
#     for sentence in sentences:
#         if len(current_chunk) + len(sentence) <= chunk_size:
#             current_chunk += " " + sentence
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence
#     if current_chunk:
#         chunks.append(current_chunk.strip())

#     summaries = []
#     for chunk in chunks:
#         try:
#             result = summarizer_pipeline(chunk, max_length=70, min_length=30, do_sample=False)
#             summaries.append(result[0]['summary_text'])
#         except:
#             continue
#     return " ".join(summaries)

# def perform_analysis(channel_id, video_id):
#     channel_info = get_channel_info(channel_id)
#     if not channel_info:
#         raise Exception("Failed to fetch channel details.")

#     comments = get_comments(video_id)
#     if not comments:
#         raise Exception("No comments found for this video.")

#     translated = [translator_pipeline(c)[0]['translation_text'] for c in comments[:30]]
#     text_for_summary = " ".join(translated)
#     summary = safe_summarize(text_for_summary)

#     sentiments = sentiment_pipeline(translated[:30])
#     ner_data = ner_pipeline(" ".join(translated[:10]))

#     sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
#     for s in sentiments:
#         label = s['label'].upper()
#         sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

#     topics = kw_model.extract_keywords(" ".join(translated), keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
#     topic_list = [t[0] for t in topics]

#     entities = list(set([f"{e['word']} ({e['entity_group']})" for e in ner_data]))

#     return summary, sentiment_counts, entities, topic_list, translated, channel_info

# def create_pdf(channel_id, channel_info, video_title, summary, sentiment_data, entities, topics, translated_text):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", "B", 16)
#     pdf.cell(0, 10, "YouTube Intelligence Report", ln=True, align="C")
#     pdf.set_font("Arial", "", 12)
#     pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
#     pdf.ln(10)

#     def section(title, content):
#         pdf.set_font("Arial", "B", 14)
#         pdf.cell(0, 10, title, ln=True)
#         pdf.set_font("Arial", "", 12)
#         pdf.multi_cell(0, 8, content)
#         pdf.ln(5)

#     section("Personal Details", f"Channel ID: {channel_id}\nChannel Name: {channel_info['channel_name']}\nVideo Title: {video_title}")
#     section("Channel Statistics",
#             f"Subscribers: {channel_info['subscribers']}\n"
#             f"Total Views: {channel_info['total_views']}\n"
#             f"Video Count: {channel_info['video_count']}\n"
#             f"Total Comments: {channel_info['total_comments']}"
#     )
#     section("Summary", summary)
#     section("Sentiments", str(sentiment_data))
#     section("Entities", ", ".join(entities))
#     section("Topics", ", ".join(topics))

#     pdf_bytes = BytesIO()
#     pdf.output(pdf_bytes)
#     pdf_bytes.seek(0)
#     return pdf_bytes

# # --- Routes
# @app.get("/", response_class=HTMLResponse)
# async def home():
#     return """
#         <h2>YouTube Analyzer</h2>
#         <form action="/analyze/" method="post">
#             <label>Channel ID:</label><br>
#             <input type="text" name="channel_id" required><br><br>
#             <label>Video ID:</label><br>
#             <input type="text" name="video_id" required><br><br>
#             <input type="submit" value="Analyze">
#         </form>
#     """

# @app.post("/analyze/", response_class=HTMLResponse)
# async def analyze(request: Request, channel_id: str = Form(...), video_id: str = Form(...)):
#     try:
#         video_title = get_video_title(video_id)
#         summary, sentiments, entities, topics, translated, channel_info = perform_analysis(channel_id, video_id)

#         # Save full analysis
#         full_report = {
#             "personal_details": {
#                 "channel_id": channel_id,
#                 "channel_name": channel_info['channel_name'],
#                 "video_title": video_title
#             },
#             "channel_statistics": {
#                 "subscribers": channel_info.get('subscribers', 'N/A'),
#                 "total_views": channel_info.get('total_views', 'N/A'),
#                 "video_count": channel_info.get('video_count', 'N/A'),
#                 "total_comments": channel_info.get('total_comments', 'N/A')
#             },
#             "summary": summary,
#             "sentiment_analysis": sentiments,
#             "named_entities": entities,
#             "topics_extracted": topics
#         }

#         with open("full_report.json", "w") as f:
#             json.dump(full_report, f, indent=4)

#         return f"""
#             <h2>Analysis Complete!</h2>
#             <p><b>Channel Name:</b> {channel_info['channel_name']}</p>
#             <p><b>Subscribers:</b> {channel_info['subscribers']}</p>
#             <p><b>Total Views:</b> {channel_info['total_views']}</p>
#             <p><b>Videos Uploaded:</b> {channel_info['video_count']}</p>
#             <p><b>Video Title:</b> {video_title}</p>
#             <form action="/generate_pdf/" method="post">
#                 <input type="submit" value="Download PDF Report">
#             </form>
#             <br><br>
#             <form action="/download_json/" method="post">
#                 <input type="submit" value="Download Full JSON Report">
#             </form>
#         """

#     except Exception as e:
#         return f"<h3>Error:</h3><p>{str(e)}</p>"

# @app.post("/download_json/", response_class=FileResponse)
# async def download_json():
#     try:
#         return FileResponse("full_report.json", filename="YouTube_Full_Report.json", media_type="application/json")
#     except Exception as e:
#         return HTMLResponse(content=f"<h3>Error downloading JSON:</h3><p>{str(e)}</p>")


# @app.post("/generate_pdf/", response_class=FileResponse)
# async def generate_pdf():
#     try:
#         with open("last_analysis.json", "r") as f:
#             analysis_data = json.load(f)

#         pdf_bytes = create_pdf(
#             analysis_data['channel_id'],
#             analysis_data['channel_info'],
#             analysis_data['video_title'],
#             analysis_data['summary'],
#             analysis_data['sentiments'],
#             analysis_data['entities'],
#             analysis_data['topics'],
#             analysis_data['translated']
#         )

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
#             temp_pdf.write(pdf_bytes.read())
#             temp_pdf_path = temp_pdf.name

#         return FileResponse(temp_pdf_path, filename="YouTube_Analysis_Report.pdf", media_type="application/pdf")

#     except Exception as e:
#         return HTMLResponse(content=f"<h3>Error generating PDF:</h3><p>{str(e)}</p>")




from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from io import BytesIO
from datetime import datetime
import tempfile
import os
import json
import re
import nltk
from collections import Counter
from fpdf import FPDF
from pymongo import MongoClient
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Setup
nltk.download('punkt')
sentiment_pipeline = pipeline("sentiment-analysis")
ner_pipeline = pipeline("ner", aggregation_strategy="simple")
summarizer_pipeline = pipeline("summarization")
translator_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
kw_model = SentenceTransformer('all-MiniLM-L6-v2')

YOUTUBE_API_KEY = "AIzaSyCKRTuJPZ1xw3NGuXwUgkXuYz8ZGpcdHE8"
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

app = FastAPI()

# --- Helper Functions
def get_channel_info(channel_id):
    try:
        data = youtube.channels().list(part="snippet,statistics,contentDetails", id=channel_id).execute()
        info = data['items'][0]
        return {
            "title": info['snippet']['title'],
            "subscribers": info['statistics'].get('subscriberCount', "N/A"),
            "total_views": info['statistics'].get('viewCount', "N/A"),
            "video_count": info['statistics'].get('videoCount', "N/A"),
            "total_comments": info['statistics'].get('commentCount', "N/A")
        }
    except:
        return None

def get_video_title(video_id):
    try:
        data = youtube.videos().list(part="snippet", id=video_id).execute()
        return data['items'][0]['snippet']['title']
    except:
        return "Unknown Video Title"

def get_comments(video_id):
    comments = []
    try:
        res = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100).execute()
        for item in res["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text = snippet["textDisplay"]
            author = snippet["authorDisplayName"]
            comments.append({"author": author, "text": text})
    except:
        pass
    return comments

def safe_summarize(text, chunk_size=400):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    summaries = []
    for chunk in chunks:
        try:
            result = summarizer_pipeline(chunk, max_length=70, min_length=30, do_sample=False)
            summaries.append(result[0]['summary_text'])
        except:
            continue
    return " ".join(summaries)

def extract_hashtags(text):
    return re.findall(r"#\w+", text)

# --- Main Analysis Function
def perform_analysis(channel_id, video_id):
    channel_info = get_channel_info(channel_id)
    if not channel_info:
        raise Exception("Failed to fetch channel details.")

    comments_data = get_comments(video_id)
    if not comments_data:
        raise Exception("No comments found.")

    translated_texts = []
    all_authors = []
    all_hashtags = []

    for c in comments_data:
        translated = translator_pipeline(c['text'])[0]['translation_text']
        translated_texts.append(translated)
        all_authors.append(c['author'])
        all_hashtags.extend(extract_hashtags(c['text']))

    text_for_summary = " ".join(translated_texts)
    summary = safe_summarize(text_for_summary)

    sentiments = sentiment_pipeline(translated_texts[:30])
    ner_data = ner_pipeline(" ".join(translated_texts[:10]))

    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for s in sentiments:
        label = s['label'].upper()
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

    topics = kw_model.encode(translated_texts[:10])

    # Messages Data
    messages_list = [{"author": c['author'], "content": c['text']} for c in comments_data]

    # Dashboard Data
    total_comments = len(comments_data)
    unique_authors = len(set(all_authors))
    top_authors = Counter(all_authors).most_common(5)
    top_hashtags = Counter(all_hashtags).most_common(5)

    dashboard_data = {
        "statistics": {
            "total_channels": 1,  # Only one channel analyzed per request
            "total_comments": total_comments,
            "unique_authors": unique_authors,
            "avg_comments_per_day": total_comments  # Assume per day for now
        },
        "most_active_users": [{"username": u[0], "comments": u[1]} for u in top_authors],
        "popular_keywords": [h[0] for h in top_hashtags]
    }

    return {
        "summary": summary,
        "sentiment_counts": sentiment_counts,
        "ner_data": list(set([f"{e['word']} ({e['entity_group']})" for e in ner_data])),
        "topics": [str(t) for t in topics],  # Just for now
        "channel_info": channel_info,
        "dashboard_data": dashboard_data,
        "messages_list": messages_list,
        "video_title": get_video_title(video_id)
    }

# --- PDF Generation
def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "YouTube Intelligence Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)

    def section(title, content):
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, content)
        pdf.ln(5)

    section("Channel Info", f"Channel Name: {data['channel_info']['title']}\nSubscribers: {data['channel_info']['subscribers']}")
    section("Video Title", data['video_title'])
    section("Summary", data['summary'])
    section("Sentiments", json.dumps(data['sentiment_counts'], indent=2))
    section("Named Entities", ", ".join(data['ner_data']))
    section("Topics", ", ".join(data['topics']))

    pdf_bytes = BytesIO()
    pdf.output(pdf_bytes)
    pdf_bytes.seek(0)
    return pdf_bytes

# --- Routes
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
        <h2>YouTube Analyzer</h2>
        <form action="/analyze/" method="post">
            <label>Channel ID:</label><br>
            <input type="text" name="channel_id" required><br><br>
            <label>Video ID:</label><br>
            <input type="text" name="video_id" required><br><br>
            <input type="submit" value="Analyze">
        </form>
    """

@app.post("/analyze/", response_class=HTMLResponse)
async def analyze(channel_id: str = Form(...), video_id: str = Form(...)):
    try:
        result = perform_analysis(channel_id, video_id)

        # Save files
        with open("full_report.json", "w") as f:
            json.dump(result, f, indent=4)
        with open("dashboard_data.json", "w") as f:
            json.dump(result['dashboard_data'], f, indent=4)
        with open("messages_data.json", "w") as f:
            json.dump(result['messages_list'], f, indent=4)
            
        return """
            <h2>Analysis Complete!</h2>
            <form action="/generate_pdf/" method="post">
                <input type="submit" value="Download PDF Report">
            </form><br><br>
            <form action="/download_json/" method="post">
                <input type="submit" value="Download JSON Report">
            </form><br><br>
            <form action="/dashboard_data/" method="get">
                <input type="submit" value="View Dashboard Data">
            </form><br><br>
            <form action="/messages_data/" method="get">
                <input type="submit" value="View Messages Data">
            </form>
        """


    except Exception as e:
        return f"<h3>Error:</h3><p>{str(e)}</p>"

@app.post("/generate_pdf/", response_class=FileResponse)
async def generate_pdf():
    with open("full_report.json", "r") as f:
        data = json.load(f)

    pdf_bytes = create_pdf(data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes.read())
        temp_pdf_path = temp_pdf.name

    return FileResponse(temp_pdf_path, filename="YouTube_Report.pdf", media_type="application/pdf")

@app.post("/download_json/", response_class=FileResponse)
async def download_json():
    try:
        return FileResponse("full_report.json", filename="YouTube_Full_Report.json", media_type="application/json")
    except Exception as e:
        return HTMLResponse(content=f"<h3>Error downloading JSON:</h3><p>{str(e)}</p>")

@app.get("/dashboard_data/", response_class=JSONResponse)
async def dashboard_data():
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["youtube_data"]  # <<== CHANGE your database name here

        channels_collection = db["channels"]
        comments_collection = db["comments"]
        users_collection = db["users"]

        # Total channels
        total_channels = channels_collection.count_documents({})

        # Total comments
        total_comments = comments_collection.count_documents({})

        # Unique authors
        unique_authors = users_collection.count_documents({})

        # Avg comments per day
        dates = comments_collection.find({}, {"timestamp": 1})
        date_list = []
        for d in dates:
            if "timestamp" in d:
                try:
                    # Assuming timestamp stored as ISODate or string
                    dt = d['timestamp']
                    if isinstance(dt, str):
                        dt = datetime.fromisoformat(dt)
                    date_list.append(dt.date())
                except:
                    continue

        if date_list:
            first_day = min(date_list)
            last_day = max(date_list)
            total_days = (last_day - first_day).days or 1  # avoid division by zero
            avg_comments_per_day = total_comments // total_days
        else:
            avg_comments_per_day = 0

        dashboard_data = {
            "statistics": {
                "total_channels": total_channels,
                "total_comments": total_comments,
                "unique_authors": unique_authors,
                "avg_comments_per_day": avg_comments_per_day
            }
        }

        return JSONResponse(content=dashboard_data)

    except Exception as e:
        return HTMLResponse(content=f"<h3>Error fetching dashboard data: {str(e)}</h3>")
    
@app.get("/messages_data/", response_class=JSONResponse)
async def messages_data():
    with open("messages_data.json", "r") as f:
        data = json.load(f)
    return JSONResponse(content=data)
