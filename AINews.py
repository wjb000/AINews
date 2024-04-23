import requests
from datetime import datetime, timedelta
from newspaper import Article
from transformers import pipeline
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
    sentiment_analyzer = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", truncation=True, max_length=512)
    sentiment = sentiment_analyzer(text[:512])[0]
    return sentiment['label']

def is_relevant(text, stock):
    return stock.lower() in text.lower()

def display_results(news_data, stock):
    relevant_articles = 0
    for i, (title, url) in enumerate(news_data):
        text = extract_text(url)
        if text and is_relevant(text, stock):
            st.subheader(f"Article {relevant_articles + 1}: {title}")
            st.write(f"URL: {url}")
            summary = summarize_text(text)
            sentiment = sentiment_analysis(text)
            st.write(f"Summary: {summary}")
            st.write(f"Sentiment: {sentiment}")
            relevant_articles += 1
        elif is_relevant(title, stock):
            st.subheader(f"Article {relevant_articles + 1}: {title}")
            st.write(f"URL: {url}")
            sentiment = sentiment_analysis(title)
            st.write("Failed to extract article content.")
            st.write(f"Sentiment (based on title): {sentiment}")
            relevant_articles += 1
    
    if relevant_articles == 0:
        st.warning("No relevant articles found.")

def main():
    st.title("Stock News Analyzer")
    stock = st.text_input("Enter the stock you're interested in:")
    if st.button("Analyze"):
        news_data = fetch_news(stock)
        if not news_data:
            st.warning("No news found.")
        else:
            display_results(news_data, stock)

if __name__ == "__main__":
    main()
