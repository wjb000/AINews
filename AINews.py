import requests
from datetime import datetime, timedelta
from newspaper import Article
from transformers import pipeline
from textblob import TextBlob

def fetch_news(stock):
    api_key = "6552cb40d51d4d22ad84c57a3d8d5a88"
    to_date = datetime.now().isoformat()
    from_date = (datetime.now() - timedelta(days=1)).isoformat()
    url = f"https://newsapi.org/v2/everything?q={stock}&from={from_date}&to={to_date}&language=en&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch news. Status code: {response.status_code}")
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
        print(f"\nArticle {i+1}: {title}")
        print(f"URL: {url}")
        text = extract_text(url)
        summary = summarize_text(text)
        sentiment = sentiment_analysis(summary)
        print(f"Summary: {summary}")
        print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    stock = input("Enter the stock you're interested in: ")
    news_data = fetch_news(stock)
    if not news_data:
        print("No news found.")
        exit(0)
    display_results(news_data)
