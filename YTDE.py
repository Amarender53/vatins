from googleapiclient.discovery import build
from pymongo import MongoClient
import time
# 🔑 YouTube API setup
API_KEY = "AIzaSyCKRTuJPZ1xw3NGuXwUgkXuYz8ZGpcdHE8"
youtube = build("youtube", "v3", developerKey=API_KEY)

client = MongoClient("mongodb://localhost:27017/")
db = client.youtube_data

# 🧾 Channel list
channel_ids = ["UCY0uzN460vV_-f5gB8YFb8Q"]

# 🔁 Get the uploads playlist ID for a channel
def get_uploads_playlist_id(channel_id):
    response = youtube.channels().list(
        part="contentDetails,snippet,statistics",
        id=channel_id
    ).execute()
    channel = response["items"][0]
    
    # Store channel info
    db.channels.insert_one({
        "channel_id": channel_id,
        "title": channel["snippet"]["title"],
        "subscribers": int(channel["statistics"].get("subscriberCount", 0)),
        "video_count": int(channel["statistics"].get("videoCount", 0))
    })
    return channel["contentDetails"]["relatedPlaylists"]["uploads"]
# 🎞 Get ALL videos from the uploads playlist
def get_all_video_ids(playlist_id):
    video_ids = []
    next_page_token = None
    while True:
        response = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()
        for item in response["items"]:
            video_ids.append(item["snippet"]["resourceId"]["videoId"])
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return video_ids
# 📺 Get video details
def get_video_details(video_id, channel_id):
    response = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()
    if not response["items"]:
        return
    video = response["items"][0]
    
    db.videos.insert_one({
        "video_id": video_id,
        "channel_id": channel_id,
        "title": video["snippet"]["title"],
        "views": int(video["statistics"].get("viewCount", 0)),
        "likes": int(video["statistics"].get("likeCount", 0)),
        "published_at": video["snippet"]["publishedAt"]
    })
# 💬 Get ALL top-level comments
def get_all_comments(video_id):
    next_page_token = None
    comments = []
    while True:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            ).execute()
            for item in response["items"]:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "video_id": video_id,
                    "author": snippet["authorDisplayName"],
                    "text": snippet["textDisplay"]
                })
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
            time.sleep(0.1)  # slight delay to avoid hitting quota too fast
        except Exception as e:
            print(f"❌ Skipping comments for {video_id} due to error:", e)
            break
    if comments:
        db.comments.insert_many(comments)
# 🔄 Main loop: for each channel
for channel_id in channel_ids:
    print(f"📡 Channel: {channel_id}")
    uploads_playlist_id = get_uploads_playlist_id(channel_id)
    video_ids = get_all_video_ids(uploads_playlist_id)
    print(f"📽 Found {len(video_ids)} videos")
    for video_id in video_ids:
        print(f"🎥 Processing video: {video_id}")
        get_video_details(video_id, channel_id)
        get_all_comments(video_id)
print("✅ All data has been collected and saved to MongoDB.")
 