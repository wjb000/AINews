import requests
from datetime import datetime, timedelta
from newspaper import Article
from transformers import pipeline
from textblob import TextBlob
import streamlit as st

def fetch_news(stock):
    api_key = "6552cb40d51d4d22ad84c57a3d8d5a88"
    to_date = datetime.now().isoformat()
    from_date = (datetime.now() - timedelta(days=1)).isoformat()
    url = f"https://newsapi.org/v2/everything?q={stock}&from={from_date}&to={to_date}&language=en&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to fetch news. Status code: {response.status_code}")
        return []
    data = response.json()
    return [(article['title'], article['url']) for article in data.get('articles', [])]

def extract_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return ""

def summarize_text(text):
    if not text.strip():
        return "Failed to extract article content."
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False, truncation=True)
    return summary[0]['summary_text']

def sentiment_analysis(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.2:
        return 'Positive'
    elif polarity < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

def display_results(news_data):
    for i, (title, url) in enumerate(news_data):
        st.subheader(f"Article {i+1}: {title}")
        st.write(f"URL: {url}")
        text = extract_text(url)
        summary = summarize_text(text)
        sentiment = sentiment_analysis(summary)
        st.write(f"Summary: {summary}")
        st.write(f"Sentiment: {sentiment}")

def main():
    st.title("Stock News Analyzer")
    stock = st.text_input("Enter the stock you're interested in:")

    if st.button("Analyze"):
        news_data = fetch_news(stock)
        if not news_data:
            st.warning("No news found.")
        else:
            display_results(news_data)

if __name__ == "__main__":
    main()
