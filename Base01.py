import streamlit as st
import pandas as pd
from transformers import pipeline
from bertopic import BERTopic
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
import plotly.express as px
import folium
import streamlit_folium as sf
from datetime import datetime

# Load Models
sentiment_pipeline = pipeline("sentiment-analysis")
nlp = spacy.load("en_core_web_sm")
topic_model = BERTopic()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample Data Load (Replace with actual DB or file input)
df = pd.read_csv("sample_data.csv")
df["clean_text"] = df["text"].str.lower()

# Sentiment Analysis
df["sentiment"] = df["clean_text"].apply(lambda x: sentiment_pipeline(x)[0]['label'])

# NER
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
df["entities"] = df["clean_text"].apply(extract_entities)

# Topic Detection
topics, _ = topic_model.fit_transform(df["clean_text"])
df["topic"] = topics

# Embeddings for Semantic Search
texts = df["clean_text"].dropna().tolist()
if len(texts) > 0:
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
else:
    embeddings = None
    index = None


# Streamlit UI
st.title("AI Analysis Pipeline Dashboard")
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Overview", "Sentiment", "Topics", "Semantic Search", "Geo Analysis"])

if section == "Overview":
    st.subheader("Summary")
    st.metric("Total Records", len(df))
    st.metric("Positive Posts", (df["sentiment"] == "POSITIVE").sum())
    st.metric("Top Topic", df["topic"].mode()[0])

elif section == "Sentiment":
    st.subheader("Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    fig = px.pie(sentiment_counts, names="index", values="sentiment")
    st.plotly_chart(fig)

elif section == "Topics":
    st.subheader("Top Topics")
    topic_fig = topic_model.visualize_barchart(top_n_topics=5)
    st.components.v1.html(topic_fig.to_html(), height=600)

elif section == "Semantic Search":
    st.subheader("Semantic Search")
    if index is None or index.ntotal == 0:
        st.warning("Semantic search is unavailable. No embeddings were indexed.")
    else:
        query = st.text_input("Enter search query")
        if query:
            query_vec = embedding_model.encode([query])
            top_k = min(5, index.ntotal)
            D, I = index.search(np.array(query_vec), k=top_k)
            for idx in I[0]:
                st.markdown(f"**Match**: {df.iloc[idx]['text']}")


elif section == "Geo Analysis":
    st.subheader("Geographical Distribution")
    if "latitude" in df.columns and "longitude" in df.columns:
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        for _, row in df.iterrows():
            folium.Marker([row["latitude"], row["longitude"]], popup=row["text"]).add_to(m)
        sf.folium_static(m)
    else:
        st.warning("No geo-location data available.")
