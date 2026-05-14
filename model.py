import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only useful columns
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

# Test message
message = ["I Love you"]

# Convert message
message_vector = vectorizer.transform(message)

# Prediction
prediction = model.predict(message_vector)

print("Prediction:", prediction[0])
