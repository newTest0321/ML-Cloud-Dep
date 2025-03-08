import numpy as np
import pandas as pd
import pickle
import re
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load models and vectorizers
vectorizer_fake_news = pickle.load(open('tfdif_vectorizer_fake_news.pkl', 'rb'))
fake_news_model = pickle.load(open('fake_news_model.sav', 'rb'))

vectorizer_spam_mail = pickle.load(open('tfdif_vectorizer_spam_mail.pkl', 'rb'))
spam_mail_model = pickle.load(open('spam_mail_model.sav', 'rb'))

port_stem = PorterStemmer()

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    content = [port_stem.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

def predict_fake_news(title, author):
    content = stemming(title + " " + author)
    content_vectorized = vectorizer_fake_news.transform([content])
    prediction = fake_news_model.predict(content_vectorized)
    return 'Reliable News' if prediction[0] == 0 else 'Unreliable News'

def predict_spam_mail(email_text):
    email_vectorized = vectorizer_spam_mail.transform([email_text])
    prediction = spam_mail_model.predict(email_vectorized)
    return 'Ham Mail (Not Spam)' if prediction[0] == 0 else 'Spam Mail'

# Streamlit UI
st.set_page_config(page_title="Multi-Model Cloud Deployment", layout="wide")

# Initialize session state for model selection
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "Home"

# Sidebar navigation with buttons using session state
st.sidebar.title("🔍 Choose a Model")
if st.sidebar.button("🏠 Home"):
    st.session_state.model_choice = "Home"
if st.sidebar.button("📰 Fake News Detection"):
    st.session_state.model_choice = "Fake News Detection"
if st.sidebar.button("📩 Spam Mail Detection"):
    st.session_state.model_choice = "Spam Mail Detection"

# Retrieve the current selection from session state
model_choice = st.session_state.model_choice

# Home Page
if model_choice == "Home":
    st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>🌐 Multi-Model Cloud Deployment</h1>
        <p style='text-align: center; font-size:18px;'>This application integrates multiple machine learning models to provide real-time predictions for Fake News Detection and Spam Mail Detection.</p>
        <p style='text-align: center; font-size:18px;'>Select a model from the sidebar to begin.</p>
    """, unsafe_allow_html=True)

# Fake News Detection Page
elif model_choice == "Fake News Detection":
    st.markdown("""
        <h2 style='color: #FF9800;'>📰 Fake News Detection</h2>
        <p>Enter the news details below to check if it's reliable or fake.</p>
    """, unsafe_allow_html=True)
    
    title = st.text_input("News Title:", placeholder="Enter news headline...")
    author = st.text_input("Author Name:", placeholder="Enter author's name...")
    content = st.text_area("News Content:", placeholder="Enter the news content (not used in prediction, just for reference)")
    
    if st.button("🔍 Detect Fake News", use_container_width=True):
        if title.strip() == "" or author.strip() == "":
            st.warning("Please enter both Title and Author.")
        else:
            result = predict_fake_news(title, author)
            st.success(f"Prediction: {result}")

# Spam Mail Detection Page
elif model_choice == "Spam Mail Detection":
    st.markdown("""
        <h2 style='color: #2196F3;'>📩 Spam Mail Detection</h2>
        <p>Enter an email below to check if it's spam.</p>
    """, unsafe_allow_html=True)
    
    email_text = st.text_area("Email Content:", placeholder="Paste the email content here...")
    
    if st.button("📧 Check Spam Status", use_container_width=True):
        if email_text.strip() == "":
            st.warning("Please enter an email message.")
        else:
            result = predict_spam_mail(email_text)
            st.success(f"Prediction: {result}")
