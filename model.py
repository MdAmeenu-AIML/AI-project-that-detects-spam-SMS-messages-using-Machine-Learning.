import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Test custom message
message = ["good morning"]

message_vector = vectorizer.transform(message)

prediction = model.predict(message_vector)

print("Prediction:", prediction[0])
