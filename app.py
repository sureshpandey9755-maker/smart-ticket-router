import streamlit as st
import joblib
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# NLTK VADER check
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Page configuration
st.set_page_config(
    page_title="Smart Customer Support Ticket Router",
    page_icon="🎫",
    layout="centered"
)

# Load saved model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    saved_data = joblib.load("models/ticket_model.joblib")
    return saved_data["dept"], saved_data["tfidf"]

try:
    model, tfidf = load_model_and_vectorizer()
except Exception as e:
    st.error(f"Model load karne mein error aaya: {e}. Pehle terminal mein 'python train.py' run karein.")
    st.stop()

# VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# UI Design
st.title("🎫 Smart Customer Support Ticket Router")
st.write("Enter a customer support message below to automatically route it to the right department and detect its sentiment.")

user_input = st.text_area("Customer Ticket Message:", placeholder="Type your issue here, e.g., My order hasn't arrived yet and it's been 5 days...")

if st.button("Analyze Ticket"):
    if not user_input.strip():
        st.warning("Please enter a ticket message first.")
    else:
        # 1. Transform text using TF-IDF for Department Prediction
        vectorized_text = tfidf.transform([user_input])
        predicted_department = model.predict(vectorized_text)[0]
        
        # 2. Predict Sentiment using VADER
        scores = sid.polarity_scores(user_input)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = "Positive 😊"
        elif compound <= -0.05:
            sentiment = "Negative 😠"
        else:
            sentiment = "Neutral 😐"
            
        # Display Results
        st.success("Analysis Complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Assigned Department", value=predicted_department)
        with col2:
            st.metric(label="Detected Sentiment", value=sentiment)
        
        # Additional breakdown info
        with st.expander("View Detailed Sentiment Scores"):
            st.write(f"Compound Score: {compound}")
            st.json(scores)