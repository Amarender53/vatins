import streamlit as st
import pandas as pd
import pymysql
import json
import pydeck as pdk

# Function to connect to the MySQL database
def load_user_profiles():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='5368',
        database='VATINS',
        cursorclass=pymysql.cursors.DictCursor
    )
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM UserProfileSummary")
        rows = cursor.fetchall()
    return pd.DataFrame(rows)

# Helper: decode JSON fields safely
def decode_json_field(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except:
        return []

# Helper: sentiment color badge
def get_sentiment_color(sentiment):
    sentiment = sentiment.lower()
    if sentiment == 'positive':
        return "🟢 Positive"
    elif sentiment == 'negative':
        return "🔴 Negative"
    else:
        return "🟡 Neutral"

def get_risk_label(score):
    if score is None:
        return "⚪ Unknown"
    elif score >= 0.7:
        return f"🔴 High Risk ({score:.2f})"
    elif score >= 0.5:
        return f"🟡 Risk ({score:.2f})"
    elif score >= 0.25:
        return f"🟡 Medium ({score:.2f})"
    else:
        return f"🟢 Low ({score:.2f})"

# Page setup
st.set_page_config(page_title="User Profiling Dashboard", layout="wide")
st.title("AI User Profiling Dashboard")

# Load and validate data
df = load_user_profiles()
if df.empty:
    st.warning("No profile data available.")
    st.stop()

# Sidebar filters
user_ids = df["user_id"].unique()
selected_user = st.sidebar.selectbox("Select User", user_ids)

# Get user data row
user_data = df[df["user_id"] == selected_user].iloc[0]

# Handle None risk score
if user_data["reporter_risk_score"] is None:
    user_data["reporter_risk_score"] = 0.0


# Decode JSON fields
common_categories = decode_json_field(user_data["common_categories"])
top_emotions = decode_json_field(user_data["top_emotions"])
top_keywords = decode_json_field(user_data["top_keywords"])
location = decode_json_field(user_data["last_known_location"])

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("🧾 Total Tips", user_data["total_tips_sent"])
col2.metric("📅 Active Days", user_data["active_days"])
col3.metric("🔥 Risk Level", get_risk_label(user_data["reporter_risk_score"]))

# Timeline
col4, col5 = st.columns(2)
col4.markdown(f"**First Tip:** {user_data['first_tip_date']}")
col5.markdown(f"**Last Tip:** {user_data['last_tip_date']}")

# Sentiment
st.markdown(f"**Average Sentiment:** {get_sentiment_color(user_data['average_sentiment'])}")
st.markdown(f"**Geo Tags Detected:** {'📍 Yes' if user_data['geo_tags_detected'] else '❌ No'}")

# Data categories
st.markdown("### Top Categories, Emotions & Keywords")
st.markdown(f"**Top Categories:** {', '.join(common_categories) if common_categories else 'None'}")
st.markdown(f"**Top Emotions:** {', '.join(top_emotions) if top_emotions else 'None'}")
st.markdown(f"**Top Keywords:** {', '.join(top_keywords) if top_keywords else 'None'}")

# Map if location exists
if user_data["geo_tags_detected"] and location:
    st.markdown("### 🌍 Last Known Location")
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=location["latitude"],
            longitude=location["longitude"],
            zoom=12,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=[{"lat": location["latitude"], "lon": location["longitude"]}],
                get_position='[lon, lat]',
                get_radius=200,
                get_color=[255, 0, 0],
                pickable=True,
            )
        ]
    ))
else:
    st.info("No geo-location data available.")
