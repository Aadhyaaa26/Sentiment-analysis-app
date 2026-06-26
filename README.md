# Moodreader — AI Sentiment Analysis App

A web app that reads a sentence and classifies it as **positive**,
**negative**, or **neutral** in real time, using a TF-IDF + Logistic
Regression NLP pipeline served through a Flask API.

![App screenshot](![Uploading screenshot.png.png…]()
)

## Features

- Real-time text classification with a live confidence score
- Custom NLP preprocessing pipeline: lowercasing, stopword removal,
  tokenization, TF-IDF feature extraction
- Visual "mood gauge" that animates to the predicted sentiment
- Lightweight Flask backend, vanilla JS frontend — no heavy frameworks

## Tech stack

| Layer        | Tools |
|--------------|-------|
| ML / NLP     | Python, scikit-learn, TF-IDF, Logistic Regression |
| Backend      | Flask, REST API |
| Frontend     | HTML, CSS, JavaScript |
| Model storage| joblib |

## How it works

1. `train_model.py` builds a labeled dataset, cleans the text, converts
   it to TF-IDF vectors, and trains a Logistic Regression classifier.
2. The trained model is saved to `model/sentiment_model.joblib`.
3. `app.py` loads that model and exposes a `/predict` endpoint.
4. The frontend sends typed text to `/predict` and animates the gauge
   based on the returned sentiment and confidence score.

## Running it locally

```
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Project structure

```
sentiment_app/
├── train_model.py     # trains and saves the model
├── app.py              # Flask server + /predict API
├── requirements.txt
├── model/               # trained model gets saved here
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## Possible improvements

- Train on a real-world dataset (e.g. IMDB reviews) for more
  realistic accuracy
- Deploy live on Render/Railway and link a live demo here
- Add batch CSV upload with a sentiment breakdown chart

## Author

Aadhya Nadar — [GitHub](https://github.com/Aadhyaaa26) · [LinkedIn](https://linkedin.com/in/aadhyanadar2605)
