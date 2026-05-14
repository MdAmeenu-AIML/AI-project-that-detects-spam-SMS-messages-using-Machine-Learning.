import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = {
    "message": [
        "Win money now",
        "Free iPhone offer",
        "Hey how are you",
        "Let's meet tomorrow",
        "Claim your free prize"
    ],
    "label": [
        "spam",
        "spam",
        "ham",
        "ham",
        "spam"
    ]
}

# Create dataframe
df = pd.DataFrame(data)

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])

# Labels
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Test message
message = ["Congratulations! You won a free ticket"]

# Convert message
message_vector = vectorizer.transform(message)

# Prediction
prediction = model.predict(message_vector)

print("Prediction:", prediction[0])