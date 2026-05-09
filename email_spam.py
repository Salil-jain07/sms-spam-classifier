import pandas as pd
import re
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text

@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'])
    df = df.drop_duplicates()
    df = df.rename(columns={'v1':'category','v2':'message'})
    df['category'] = df['category'].replace({
        'ham':'Not Spam',
        'spam':'Spam'
    })
    df['message'] = df['message'].apply(clean_text)
    return df

df = load_data()

@st.cache_resource
def train_model():
    X = df['message']
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    return model, vectorizer

model, vectorizer = train_model()

st.title("SMS Spam Detector")

message = st.text_area("Enter your message")

if st.button("Predict"):
    msg = clean_text(message)
    pred = model.predict(vectorizer.transform([msg]))[0]

    if pred == "Spam":
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")
