# 🚀 User Intelligence Summary Stack (Updated with Narrative Analysis using Falcon-7B)

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    PegasusTokenizer, PegasusForConditionalGeneration,
    pipeline, AutoModelForCausalLM
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig
from langdetect import detect_langs
from langcodes import Language
from typing import List, Dict
from pydantic import BaseModel
from keybert import KeyBERT
import re
import torch

app = FastAPI()

MONGO_URI = "mongodb+srv://vatins:Test123@dev.cjmwsi.mongodb.net/?retryWrites=true&w=majority&appName=dev"
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=50000, socketTimeoutMS=50000)
db = client["telegram_scraper"]
users_col = db["telegram_users"]
messages_col = db["telegram_messages"]

# Load summarization and NER models
pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

ner_tokenizer = AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")
ner_model = AutoModelForTokenClassification.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

model_id = "openchat/openchat-3.5-1210"

narration_tokenizer = AutoTokenizer.from_pretrained(model_id)
narration_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
narration_pipeline = pipeline("text-generation", model=narration_model, tokenizer=narration_tokenizer)

sentiment_analyzer = SentimentIntensityAnalyzer()
keyword_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_extractor = KeyBERT(model=keyword_model)

# Data Models
class UserSummaryRequest(BaseModel):
    user_id: int

class RiskFlag(BaseModel):
    type: str
    description: str

class ParagraphSummary(BaseModel):
    title: str
    content: str

class UserSummaryResponse(BaseModel):
    user_profile: Dict
    risk_level: str
    risk_flags: List[RiskFlag]
    summary: str
    topics: List[List[str]]
    paragraph_summaries: List[ParagraphSummary]

# Utility Functions

def generate_narrative_section(title: str, data: Dict[str, str]) -> str:
    prompt = f"<|system|>\nYou are a professional intelligence analyst.\n<|user|>\nWrite a narrative paragraph for the report section titled **{title}** based on:\n"
    for k, v in data.items():
        prompt += f"- {k}: {v}\n"
    prompt += "\n<|assistant|>\n"

    try:
        output = narration_pipeline(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )[0]['generated_text']
        return output.split("<|assistant|>")[-1].strip()
    except Exception as e:
        return f"Narrative generation failed: {e}"


# Utility Functions
def summarize_with_model(text: str) -> str:
    inputs = pegasus_tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    summary_ids = pegasus_model.generate(**inputs, max_length=140, num_beams=4, early_stopping=True)
    return pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def extract_entities(text: str) -> List[str]:
    return list(set(ent['word'] for ent in ner_pipeline(text)))

def analyze_sentiment(text: str) -> str:
    score = sentiment_analyzer.polarity_scores(text)
    compound = score['compound']
    dominant = "NEGATIVE" if compound < -0.05 else "POSITIVE" if compound > 0.05 else "NEUTRAL"
    return f"{dominant} (compound={compound:.2f}, pos={score['pos']:.2f}, neu={score['neu']:.2f}, neg={score['neg']:.2f})"

def extract_keywords(text: str) -> List[str]:
    return [kw[0] for kw in kw_extractor.extract_keywords(text, top_n=15)]

def translate(text: str) -> str:
    lang = detect_langs(text)[0].lang
    if lang != "en":
        return GoogleTranslator(source='auto', target='en').translate(text)
    return text

def detect_languages(texts: List[str]) -> List[str]:
    detected = set()
    for t in texts:
        try:
            langs = detect_langs(t)
            for lang in langs:
                full = Language.get(lang.lang).display_name()
                detected.add(full)
        except:
            continue
    return list(detected)

def detect_fraud_crypto_terms(text: str) -> List[str]:
    terms = ["scam", "fraud", "usdt", "loan", "bitcoin", "crypto", "investment"]
    return [term for term in terms if term in text.lower()]

def generate_risk_flags(profile: Dict) -> List[Dict[str, str]]:
    flags = []
    if "crypto" in profile.get("fraud_signals", []):
        flags.append({"type": "🔴 Suspicious Financial Behavior", "description": "Cryptocurrency mentions detected."})
    if "2 highly negative messages" in profile.get("behavioral_flags", []):
        flags.append({"type": "🧾 Aggressive Tone", "description": "Multiple strongly negative messages detected."})
    if not flags:
        flags.append({"type": "🔵 Normal Activity", "description": "No significant red flags found."})
    return flags


def paragraph_summaries(profile, keywords, langs, flags, banking_terms, fraud_terms):
    lang_str = ", ".join(langs)
    keywords_3 = ", ".join(keywords[:3]) or "None"
    interests = ", ".join(profile.get("interests_detected", [])) or "None"
    banking = ", ".join(banking_terms) or "None"
    fraud = ", ".join(fraud_terms) or "None"
    aliases = "No notable terms found."
    total_msgs = profile.get("total_messages", 0)
    flag_text = ", ".join(flags) or "None"

    return [
        {"title": "User Interests", "content": generate_narrative_section("User Interests", {"Keywords": keywords_3, "Interests": interests})},
        {"title": "Location Insight", "content": generate_narrative_section("Location Insight", {"Languages": lang_str})},
        {"title": "Law Enforcement Indicators", "content": generate_narrative_section("Law Enforcement Indicators", {"Fraud Terms": fraud, "Aliases": aliases})},
        {"title": "Communication Patterns", "content": generate_narrative_section("Communication Patterns", {"Total Messages": str(total_msgs)})},
        {"title": "Financial Behavior", "content": generate_narrative_section("Financial Behavior", {"Banking Keywords": banking, "Crypto Terms": fraud})},
        {"title": "Anonymity & Aliases", "content": generate_narrative_section("Anonymity & Aliases", {"Aliases": aliases})},
        {"title": "Behavioral Red Flags", "content": generate_narrative_section("Behavioral Red Flags", {"Flags": flag_text})},
        {"title": "Cultural & Linguistic Context", "content": generate_narrative_section("Cultural & Linguistic Context", {"Languages": lang_str})}
    ]

def summarize_narrative(name: str, langs: List[str], topics: List[List[str]], flags: List[str], digital: List[str]) -> str:
    topics_joined = [", ".join(t) for t in topics[:3]]
    lang_str = ", ".join(langs)
    flag_str = ", ".join(flags)
    digital_str = ", ".join(digital)
    return f"{name} participates in discussions using {lang_str}. Topics include {topics_joined[0]}, {topics_joined[1]}, and {topics_joined[2]}. Behavioral indicators include: {flag_str}. Digital traces found: {digital_str}."

def generate_overall_summary(profile, sentiment, risk_level, topics):
    name = profile.get("username", "The user")
    languages = ", ".join(profile.get("languages_detected", []))
    top_topics = [", ".join(t) for t in topics[:3]]
    flags = profile.get("behavioral_flags", [])
    digital = profile.get("digital_footprint", [])
    flag_text = ", ".join(flags) or "None"
    digital_text = ", ".join(digital) or "None"

    summary = (
        f"{name} presents a profile marked by {risk_level.lower()} risk and {sentiment.lower()} tone. \n"
        f"Messages span {languages}, engaging with topics such as {top_topics[0]}, {top_topics[1]}, {top_topics[2]}. \n"
        f"Flagged indicators include: {flag_text}. Digital presence includes {digital_text}."
    )
    return summary

@app.get("/user_summary/", response_model=UserSummaryResponse)
async def generate_user_summary(user_id: str = Query(...)):
    try:
        user = users_col.find_one({"user_id": int(user_id)}) or {}
        messages = list(messages_col.find({"user_id": int(user_id)}).sort("date", -1))
        if not messages:
            raise HTTPException(status_code=404, detail="No messages found for user")

        message_texts = [m.get("text", "") for m in messages if m.get("text")]
        recent_msgs = [{"timestamp": str(m.get("date")), "text": m.get("text", "")} for m in messages[:10]]

        translated = translate(" ".join(message_texts))
        langs = detect_languages(message_texts)
        entities = extract_entities(translated)
        sentiment = analyze_sentiment(translated)
        keywords = extract_keywords(translated) or ["None"]
        flags = ["Monetary amounts mentioned"] + (["2 highly negative messages"] if sum(sentiment_analyzer.polarity_scores(t)['compound'] < -0.5 for t in message_texts) >= 2 else [])
        fraud_terms = detect_fraud_crypto_terms(translated)
        banking_terms = [term for term in ["sbi", "bank"] if term in translated.lower()]

        interests = []
        if "crypto" in translated.lower(): interests.append("Cryptocurrency / Fraud")
        if "bet" in translated.lower(): interests.append("Gaming / Betting")
        if banking_terms: interests.append("Banking / Financial Services")

        digital = list(set(re.findall(r"@\w+", translated)))
        risk_level = "High" if "crypto" in fraud_terms or "2 highly negative messages" in flags else "Moderate" if flags else "Low"

        profile = {
            "telegram_id": user_id,
            "username": user.get("username", "unknown"),
            "name": user.get("name", "unknown"),
            "bio": user.get("bio", "Not available"),
            "total_messages": len(message_texts),
            "languages_detected": langs,
            "aliases_used": [],
            "banking_terms": banking_terms,
            "fraud_signals": fraud_terms,
            "interests_detected": interests,
            "behavioral_flags": flags,
            "digital_footprint": digital
        }

        topic_chunks = [keywords[i:i+5] for i in range(0, len(keywords), 5)] or [["None"]]
        overall_summary = generate_overall_summary(profile, sentiment, risk_level, topic_chunks)

        return JSONResponse(content={
            "user_profile": profile,
            "risk_level": risk_level,
            "risk_flags": generate_risk_flags({}),
            "summary": summarize_narrative(profile['username'], langs, topic_chunks, flags, digital),
            "topics": topic_chunks[:3],
            "paragraph_summaries": paragraph_summaries(profile, keywords, langs, flags, banking_terms, fraud_terms),
            "overall_summary": overall_summary,
            "recent_messages": recent_msgs
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
