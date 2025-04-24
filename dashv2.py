import streamlit as st
import pandas as pd
import pymysql
import json
import pydeck as pdk

# MySQL connection
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

def get_color(sentiment):
    if sentiment.lower() == 'negative':
        return "🔴 Negative"
    elif sentiment.lower() == 'positive':
        return "🟢 Positive"
    else:
        return "🟡 Neutral"

st.set_page_config(page_title="User Profiling Dashboard", layout="wide")
st.title("AI User Profiling Dashboard")

df = load_user_profiles()
if df.empty:
    st.warning("No profile data available.")
    st.stop()

# Sidebar
user_ids = df["user_id"].unique()
selected_user = st.sidebar.selectbox("Select User", user_ids)

# Profile data
user_data = df[df["user_id"] == selected_user].iloc[0]

# Decode JSON fields
def decode_json_field(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except:
        return []

common_categories = decode_json_field(user_data["common_categories"])
top_emotions = decode_json_field(user_data["top_emotions"])
top_keywords = decode_json_field(user_data["top_keywords"])
location = decode_json_field(user_data["last_known_location"])

# Summary
st.subheader(f"📋 User Summary: {selected_user}")
col1, col2 = st.columns(2)
col1.markdown(f"**Average Sentiment:** {get_color(user_data['average_sentiment'])}")
col2.markdown(f"**Geo Tags Detected:** {'📍 Yes' if user_data['geo_tags_detected'] else '❌ No'}")

st.markdown("---")

st.markdown(f"**Top Categories:** {', '.join(common_categories) if common_categories else 'None'}")
st.markdown(f"**Top Emotions:** {', '.join(top_emotions) if top_emotions else 'None'}")
st.markdown(f"**Top Keywords / Entities:** {', '.join(top_keywords) if top_keywords else 'None'}")

# Geo Visualization (if available)
if user_data["geo_tags_detected"] and location:
    st.subheader("📍 Last Known Location")
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
    st.info("No geo-location data available for this user.")
