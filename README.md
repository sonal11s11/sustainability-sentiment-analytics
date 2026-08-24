# Sustainability Sentiment Analytics

A Streamlit exploratory application based on the dissertation workflow.

## What it does
1. Loads the dissertation dataset automatically.
2. Detects the comment/text column.
3. Runs VADER sentiment analysis.
4. Shows sentiment, source, topic and time visualisations.
5. If manual sentiment labels exist, calculates accuracy, precision, recall, F1 and a confusion matrix.
6. Runs LDA topic modelling and shows topic distribution, top words and perplexity.
7. Allows analysed results to be downloaded.

## Required data
A text column is required. The app recognises names such as:
`Comment`, `Text`, `Review`, `Feedback`, `Content`.

Optional columns:
`Sentiment`, `Source`, `Topic`, `Language`, `Date`.

For the dissertation dataset, the value `US` in the Language field is treated as the English subset, matching the documented workflow.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Create a GitHub repository, upload `app.py`, `requirements.txt` and `README.md`, then deploy `app.py` using Streamlit Community Cloud.

Do not upload private datasets, passwords, API keys or personal information.

## Dissertation relationship
Power BI remains the main business dashboard. This app is an additional exploratory prototype exposing the Python NLP workflow through a web interface.
