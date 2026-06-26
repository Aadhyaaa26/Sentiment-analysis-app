"""
app.py
-------
This is the web server. It does two jobs:
  1. Shows the web page (the app you see in the browser)
  2. Listens for text, runs it through the trained model, and sends
     back the sentiment (positive / negative / neutral)

Run it with:
    python app.py

Then open the link it prints (usually http://127.0.0.1:5000) in your browser.
"""

import os
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "sentiment_model.joblib")
model = None  # loaded on first request, see load_model()


def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "No trained model found. Run 'python train_model.py' first."
            )
        model = joblib.load(MODEL_PATH)
    return model


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    clf = load_model()

    label = clf.predict([text])[0]
    proba = clf.predict_proba([text])[0]
    classes = clf.classes_

    scores = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    confidence = round(float(max(proba)) * 100, 1)

    return jsonify({
        "sentiment": label,
        "confidence": confidence,
        "scores": scores,
    })


if __name__ == "__main__":
    app.run(debug=True)
