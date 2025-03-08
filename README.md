# 🌐 Multi-Model Cloud Deployment

This project is a **Streamlit Cloud Deployment** of two machine learning models:
1. **Fake News Detection** - Classifies news articles as **reliable** or **unreliable** based on their title and author.
2. **Spam Mail Detection** - Identifies whether an email is **spam** or **not spam** based on its content.

Both models are **deployed on Streamlit Cloud**, allowing real-time predictions via an interactive web interface.

---

## 🚀 Project Overview

✅ **Fake News Detection**  
- Uses **TF-IDF Vectorization** to extract features from **title & author**.
- Model: **Logistic Regression** trained on a fake news dataset.
- Input: **News title & author name**.
- Output: **Reliable or Unreliable news**.

✅ **Spam Mail Detection**  
- Uses **TF-IDF Vectorization** to convert email text into numerical form.
- Model: **Logistic Regression** trained on a spam email dataset.
- Input: **Email content**.
- Output: **Spam or Ham (Not Spam)**.

---

## 🛠 Installation & Setup (Local Execution)


### 1️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 2️⃣ **Ensure Model Files Are Properly Set**  
Before running the app, **update the model file paths** inside `app.py`. Example:

e.g:  
```python
vectorizer_fake_news = pickle.load(open('C:/Users/atkar/Desktop/Programing/AIML/Projects/Multi-Model-Cloud-Deployment/tfdif_vectorizer_fake_news.pkl', 'rb'))

fake_news_model = pickle.load(open('C:/Users/atkar/Desktop/Programing/AIML/Projects/Multi-Model-Cloud-Deployment/fake_news_model.sav', 'rb'))
```

### 3️⃣ **Run the Web App**  
```bash
streamlit run app.py
```
This will launch the app in your browser.

---

## 📡 Deployment on Streamlit Cloud
This project is **deployed on Streamlit Cloud**, making it accessible online without local setup. The deployment automatically loads the saved models and vectorizers to avoid the issue of refitting `TfidfVectorizer` on test data.

### 🔗 **Access the Live App**  
🔗 [Streamlit Cloud Deployment Link](https://your-streamlit-app-link)

---

## 🧠 How the Models Work

1. **TF-IDF Vectorization**  
   - Converts text into numerical format based on word importance.
   - **Fake News Model:** Extracts features from `title` and `author`.
   - **Spam Mail Model:** Extracts features from `email content`.

2. **Pre-Trained Models (Loaded from Saved Files)**  
   - The trained models are loaded via `pickle` to ensure **consistent feature transformation**.

3. **Prediction Execution**  
   - User input is processed through the vectorizer.
   - The transformed data is fed into the **pre-trained model**.
   - The model predicts whether the input is **fake news/spam or real news/ham**.

---
