# 📩 Spam Message Classifier

A Machine Learning and NLP project that classifies SMS messages as **Spam** or **Not Spam (Ham)** using Python, Scikit-learn, and Streamlit.

---

## 🚀 Live Demo

[Open Web App](https://ai-project-that-detects-spam-sms-messages-using-machine-learni.streamlit.app/)

---

## 📌 Features

- Detects spam SMS messages
- Uses Natural Language Processing (NLP)
- Real-world Kaggle dataset
- Machine Learning classification model
- Streamlit web app interface
- Real-time prediction
- Interactive user input

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- NLP (Natural Language Processing)

---

## 📂 Project Structure

```txt
spam-message-classifier/
│
├── app.py
├── model.py
├── spam.csv
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Dataset used:

SMS Spam Collection Dataset from Kaggle.

The dataset contains thousands of real SMS messages labeled as:
- Spam
- Ham (Not Spam)

---

## 🤖 Machine Learning Workflow

1. Load SMS dataset
2. Clean and preprocess text
3. Convert text into numerical vectors using CountVectorizer
4. Train model using Multinomial Naive Bayes
5. Predict whether message is spam or ham
6. Display results using Streamlit web app

---

## 🧠 NLP Concepts Used

- Text Vectorization
- CountVectorizer
- Natural Language Processing
- Classification
- Naive Bayes Algorithm

---

## ▶️ Run Locally

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/spam-message-classifier.git
```

### Open Project Folder

```bash
cd spam-message-classifier
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 Example Predictions

### 🚨 Spam Message

```txt
Congratulations! You won free money
```

Prediction:
```txt
Spam
```

---

### ✅ Normal Message

```txt
Hey bro are we meeting tomorrow?
```

Prediction:
```txt
Not Spam
```

---

## 📈 Model Performance

The project uses:
- Train/Test Split
- Accuracy Evaluation
- Real-world SMS dataset

This helps evaluate model performance on unseen messages.

---

## 📈 Project Versions

| Version | Features |
|---|---|
| V1 | Basic spam detection |
| V2 | CSV dataset integration |
| V3 | Real Kaggle dataset |
| V4 | Accuracy evaluation |
| V5 | Streamlit web app |
| V6 | Online deployment |

---

## 📚 Learning Outcomes

This project helped learn:

- NLP basics
- Text preprocessing
- Classification algorithms
- Naive Bayes model
- Streamlit deployment
- Machine Learning workflow
- Real-world dataset handling

---

## 👨‍💻 Author

Mohammed Ameen ul Aman

AI & ML Beginner Projects 🚀
