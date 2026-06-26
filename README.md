# Moodreader — Sentiment Analysis App

A small web app that reads a sentence and tells you if it sounds
positive, negative, or neutral — with a little gauge that shows the
reading, like a real instrument.

## What's inside

```
sentiment_app/
├── train_model.py     <- teaches the computer (run this first)
├── app.py              <- the web server (run this second)
├── requirements.txt    <- list of tools to install
├── model/               <- where the trained model gets saved
├── templates/
│   └── index.html       <- the page you see in the browser
└── static/
    ├── style.css         <- how it looks
    └── script.js         <- how it behaves (talks to the server)
```

## How to run it (step by step)

You'll need Python installed on your computer (3.9 or newer). Open a
terminal in this folder and do these 3 things, in order:

**1. Install the tools it needs**
```
pip install -r requirements.txt
```

**2. Train the model** (this only takes a few seconds)
```
python train_model.py
```
You'll see it print its progress — building the dataset, training,
then a final accuracy score. That score is how often it gets the
right answer on sentences it has never seen before.

**3. Start the app**
```
python app.py
```
It will print a link, usually `http://127.0.0.1:5000`. Open that in
your browser. Type a sentence, hit **Analyze**, and watch the gauge
move.

## How it actually works, in plain words

1. `train_model.py` makes a big list of example sentences that are
   already labeled positive, negative, or neutral.
2. It cleans each sentence: lowercase everything, throw away
   meaningless words ("the", "is", "and"...).
3. It turns words into numbers using a method called **TF-IDF** —
   basically, "how important is this word in this sentence, compared
   to every other sentence?"
4. A **Logistic Regression** model looks at thousands of these
   number patterns and learns which patterns usually mean "positive"
   vs "negative" vs "neutral".
5. The trained model gets saved to a file (`model/sentiment_model.joblib`)
   so you don't have to retrain it every time.
6. `app.py` loads that saved model and waits for requests. When you
   type a sentence and click Analyze, the browser sends your text to
   the server, the model makes a guess, and the result comes back to
   move the gauge.

## Make it more impressive (optional next steps)

- Swap the made-up training sentences for a **real dataset** — like
  IMDB movie reviews or Twitter sentiment data — for results that
  hold up on real-world text.
- Deploy it for free on **Render** or **Railway** so you can link a
  live demo on your resume instead of just code.
- Add a page that lets people upload a CSV of many sentences at once
  and see a chart of the overall sentiment breakdown.

## If something goes wrong

- `ModuleNotFoundError` → you skipped step 1, run `pip install -r requirements.txt`
- "No trained model found" → you skipped step 2, run `python train_model.py`
- Page won't load → make sure `python app.py` is still running in
  the terminal, and that you're opening the exact link it printed
