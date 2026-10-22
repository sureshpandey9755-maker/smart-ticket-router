# 🎫 Smart Customer Support Ticket Router & Sentiment Analyzer

An end-to-end Natural Language Processing (NLP) web application built using Python, scikit-learn, NLTK VADER, and Streamlit to automatically route customer support tickets to the appropriate department and analyze their sentiment in real-time.

---

## 🚀 Features
- **Automated Ticket Routing:** Classifies customer queries into respective departments (Billing, Support, Logistics, etc.) using Machine Learning (Logistic Regression).
- **Real-Time Sentiment Analysis:** Evaluates customer emotion (Positive, Negative, Neutral) using NLTK VADER.
- **Interactive Web Interface:** Built with Streamlit for seamless user interaction and instant results.

---

## 🛠️ Tech Stack
- **Python**
- **Scikit-learn** (TF-IDF & Logistic Regression)
- **NLTK VADER** (Sentiment Analysis)
- **Streamlit** (Front-end UI)
- **Joblib** (Model Serialization)

---

## 📁 Project Structure
```text
smart-ticket-router/
│
├── data/              # Dataset files (tickets.csv)
├── models/            # Saved model and vectorizer (.joblib)
├── src/               # Preprocessing and utility scripts
├── app.py             # Streamlit web application
├── train.py           # Model training pipeline
├── generate_data.py   # Synthetic data generator
└── README.md