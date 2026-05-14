import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep useful columns
df = df[["v1", "v2"]]

# Rename columns
df.columns = ["label", "message"]

# Convert text into numbers
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["message"])

# Labels
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Streamlit UI
st.title("📩 Spam Message Classifier")

user_message = st.text_area("Enter a message")

if st.button("Predict"):

    # Convert text
    message_vector = vectorizer.transform([user_message])

    # Predict
    prediction = model.predict(message_vector)

    if prediction[0] == "spam":
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")
