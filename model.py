import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load CSV dataset
df = pd.read_csv("spam_data.csv")

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
