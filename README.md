# Sustainability Sentiment Analytics

A Streamlit exploratory application based on the dissertation workflow.

## What it does
1. Upload CSV/XLSX data.
2. Detect the comment/text column.
3. Run VADER sentiment analysis.
4. Show sentiment, source, topic and time visualisations.
5. If manual sentiment labels exist, calculate accuracy, precision, recall, F1 and a confusion matrix.
6. Run LDA topic modelling and show topic distribution, top words and perplexity.
7. Download analysed results.

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
