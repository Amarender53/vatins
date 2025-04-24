import pymysql
import pandas as pd
import spacy
from collections import Counter
import json

# Load spaCy for NER
nlp = spacy.load("en_core_web_sm")

# DB Connection
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='5368',
    database='VATINS',
    cursorclass=pymysql.cursors.DictCursor
)

def extract_entities(text):
    doc = nlp(text)
    return [(ent.label_, ent.text) for ent in doc.ents]

def get_user_profiles():
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM Tipline")
        tips = pd.DataFrame(cursor.fetchall())

    if tips.empty:
        print("No tip data found.")
        return

    user_profiles = []

    for user_id, group in tips.groupby("user_id"):
        profile = {}
        profile["user_id"] = user_id
        profile["total_tips_sent"] = len(group)
        profile["first_tip_date"] = group["date_submitted"].min()
        profile["last_tip_date"] = group["date_submitted"].max()
        profile["active_days"] = (group["last_tip_date"].max() - group["first_tip_date"].min()).days + 1
        profile["duplicate_tip_count"] = group["is_duplicate"].sum()

        # Sentiment and emotion
        profile["average_sentiment"] = group["sentiment"].mode()[0] if not group["sentiment"].isna().all() else "Unknown"
        profile["top_emotions"] = group["emotion"].value_counts().head(2).index.tolist()

        # Tip categories
        profile["common_categories"] = group["tip_category"].value_counts().head(3).index.tolist()

        # NER Keywords
        all_entities = []
        for text in group["tip_text"]:
            ents = extract_entities(text)
            all_entities.extend([e[1] for e in ents])
        top_entities = Counter(all_entities).most_common(5)
        profile["top_keywords"] = [e[0] for e in top_entities]

        # Geo Detection
        geo_data = group.dropna(subset=["geo_latitude", "geo_longitude"])
        profile["geo_tags_detected"] = not geo_data.empty
        if not geo_data.empty:
            last_geo = geo_data.iloc[-1]
            profile["last_known_location"] = {
                "latitude": last_geo["geo_latitude"],
                "longitude": last_geo["geo_longitude"]
            }
        else:
            profile["last_known_location"] = None

        # Risk Score (Basic Heuristic)
        score = 0.5
        if profile["duplicate_tip_count"] >= 2:
            score -= 0.2
        if profile["average_sentiment"] == "Negative":
            score += 0.2
        if profile["total_tips_sent"] > 10:
            score += 0.1
        profile["reporter_risk_score"] = round(score, 2)

        user_profiles.append(profile)

    return user_profiles

# Example usage
if __name__ == "__main__":
    profiles = get_user_profiles()
    for p in profiles:
        print(json.dumps(p, indent=2))
