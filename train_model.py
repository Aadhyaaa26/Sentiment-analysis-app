"""
train_model.py
----------------
This script teaches the computer to recognize positive, negative,
and neutral text. Run it once before starting the app:

    python train_model.py

It will:
  1. Build a dataset of labeled example sentences
  2. Clean the text (lowercase, remove stopwords/punctuation)
  3. Turn the words into numbers using TF-IDF
  4. Train a Logistic Regression model on those numbers
  5. Test it on sentences it has never seen
  6. Save the trained model to model/sentiment_model.joblib
"""

import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

random.seed(42)

# ---------------------------------------------------------------------------
# STEP 1: Build a labeled dataset
#
# In a real project you'd load this from a CSV of real reviews/tweets.
# Here we generate it from templates so the project runs instantly with
# no internet download required. Swap this for a real dataset (e.g. IMDB
# reviews, Sentiment140) any time — the rest of the pipeline stays the same.
# ---------------------------------------------------------------------------

SUBJECTS = [
    "the movie", "this product", "the service", "the food", "this app",
    "the staff", "the experience", "the book", "the game", "the hotel",
    "the flight", "the course", "this update", "the design",
    "the customer support", "the delivery", "the event", "the performance",
    "the restaurant", "the software", "this laptop", "the team",
    "the new feature", "the class", "the trip",
]

POSITIVE_PHRASES = [
    "was absolutely fantastic",
    "exceeded all my expectations",
    "made me so happy",
    "is the best I've ever experienced",
    "worked perfectly and felt great",
    "left me thoroughly impressed",
    "was a wonderful surprise",
    "is truly outstanding",
    "felt smooth and delightful",
    "is something I would recommend to everyone",
    "was a genuine pleasure to use",
    "blew me away with how good it was",
]

NEGATIVE_PHRASES = [
    "was a complete disaster",
    "left me extremely disappointed",
    "did not work at all",
    "was the worst experience ever",
    "felt broken and frustrating",
    "was a total waste of time",
    "made me regret trying it",
    "was painfully slow and annoying",
    "is something I would never recommend",
    "fell far below expectations",
    "was a frustrating mess from start to finish",
    "ruined my entire day",
]

NEUTRAL_PHRASES = [
    "was okay, nothing special",
    "worked as expected, nothing more",
    "was average overall",
    "did the job, I guess",
    "was fine but forgettable",
    "is neither good nor bad",
    "met the basic requirements",
    "was a standard experience",
    "didn't stand out in any way",
    "was acceptable for the price",
    "was about what I expected",
    "is a fairly ordinary choice",
]

PUNCTUATION_ENDINGS = [".", "!", ".", "...", "."]


def build_dataset():
    rows = []
    for subject in SUBJECTS:
        for phrase in POSITIVE_PHRASES:
            end = random.choice(PUNCTUATION_ENDINGS)
            rows.append((f"{subject.capitalize()} {phrase}{end}", "positive"))
        for phrase in NEGATIVE_PHRASES:
            end = random.choice(PUNCTUATION_ENDINGS)
            rows.append((f"{subject.capitalize()} {phrase}{end}", "negative"))
        for phrase in NEUTRAL_PHRASES:
            end = random.choice(PUNCTUATION_ENDINGS)
            rows.append((f"{subject.capitalize()} {phrase}{end}", "neutral"))
    random.shuffle(rows)
    texts = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    return texts, labels


def main():
    print("Step 1/4: Building labeled dataset...")
    texts, labels = build_dataset()
    print(f"  -> {len(texts)} labeled examples created")

    print("Step 2/4: Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"  -> {len(X_train)} for training, {len(X_test)} for testing")

    print("Step 3/4: Cleaning text + extracting TF-IDF features, then training...")
    # This single Pipeline does everything: lowercasing, stopword removal,
    # tokenization, and TF-IDF feature extraction, followed by the classifier.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    print("Step 4/4: Evaluating on unseen test sentences...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n✅ Accuracy on test data: {accuracy * 100:.1f}%\n")
    print(classification_report(y_test, predictions))

    joblib.dump(model, "model/sentiment_model.joblib")
    print("Saved trained model to model/sentiment_model.joblib")
    print("\nYou can now run the app with:  python app.py")


if __name__ == "__main__":
    main()
