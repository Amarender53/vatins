import streamlit as st
import pandas as pd
import pymysql
import json
import pydeck as pdk

# MySQL connection
def load_profiles():
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

def get_color(score):
    if score >= 0.7:
        return "🔴 High"
    elif score >= 0.4:
        return "🟡 Medium"
    else:
        return "🟢 Low"

# Streamlit UI
st.set_page_config(page_title="User Profiling Dashboard", layout="wide")
st.title("AI User Profiling Dashboard")

df = load_profiles()
if df.empty:
    st.warning("No profile data available.")
    st.stop()

# Sidebar
user_ids = df["user_id"].unique()
selected_user = st.sidebar.selectbox("Select User", user_ids)

# Filtered data
user_data = df[df["user_id"] == selected_user].iloc[0]

# Summary Cards
col1, col2, col3 = st.columns(3)
col1.metric("🧾 Total Tips", user_data["total_tips_sent"])
col2.metric("📅 Active Days", user_data["active_days"])
col3.metric("📈 Risk Score", f"{user_data['reporter_risk_score']} ({get_color(user_data['reporter_risk_score'])})")

# Main Profile
st.subheader(f"📋 Profile Summary: {selected_user}")
st.markdown(f"""
- **First Tip:** {user_data['first_tip_date']}
- **Last Tip:** {user_data['last_tip_date']}
- **Avg. Sentiment:** {user_data['average_sentiment']}
- **Top Emotions:** {user_data['top_emotions']}
- **Top Categories:** {user_data['common_categories']}
- **Top Keywords:** {user_data['top_keywords']}
""")

# Geo Visualization
if user_data["last_known_location"]:
    coords = json.loads(user_data["last_known_location"].replace("'", "\""))
    st.subheader("📍 Last Known Location")
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=coords["latitude"],
            longitude=coords["longitude"],
            zoom=12,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=[{"lat": coords["latitude"], "lon": coords["longitude"]}],
                get_position='[lon, lat]',
                get_radius=200,
                get_color=[255, 0, 0],
                pickable=True,
            )
        ]
    ))
else:
    st.info("No geo-location data available for this user.")
