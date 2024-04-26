import requests
import streamlit as st
from datetime import datetime, timedelta
from newspaper import Article
from transformers import pipeline

NEWSAPI_KEY = "6552cb40d51d4d22ad84c57a3d8d5a88"
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

def fetch_articles(ticker, start_date, end_date):
    query = ticker + " stock"
    params = {
        'q': query,
        'apiKey': NEWSAPI_KEY,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 100,
        'from': start_date,
        'to': end_date
    }
    
    articles = []
    while True:
        response = requests.get(NEWSAPI_ENDPOINT, params=params)
        if response.status_code != 200:
            st.error("Error fetching news: " + response.text)
            break
        
        data = response.json()
        articles.extend([(article['title'], article['url']) for article in data['articles'] if article['title'] and article['url']])
        
        if 'nextPage' not in data:
            break
        params['page'] = data['nextPage']
    
    return articles

def extract_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return ""

def is_related_to_company(text, company):
    return company.lower() in text.lower()

def sentiment_analysis(text):
    sentiment_analyzer = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", truncation=True, max_length=512)
    sentiment = sentiment_analyzer(text[:512])[0]
    return sentiment['label']

def summarize_article(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", truncation=True, max_length=1024)
    summary = summarizer(text[:1024])[0]['summary_text']
    return summary

def main():
    st.title("Stock News Search")
    
    ticker = st.text_input("Enter the stock ticker for which you want to fetch news articles:")
    
    days_back = st.slider("Select the number of days to search for articles:", min_value=1, max_value=30, value=7)
    
    if st.button("Analyze"):
        if ticker:
            current_time = datetime.utcnow()
            start_date = (current_time - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
            end_date = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            articles = fetch_articles(ticker, start_date, end_date)
            if not articles:
                st.warning("No articles found for the given ticker in the specified time range.")
            else:
                sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
                st.write(f"\nAnalyzing relevant articles...")
                relevant_articles = 0
                
                for i, (title, url) in enumerate(articles):
                    text = extract_text(url)
                    if text and is_related_to_company(text, ticker):
                        sentiment = sentiment_analysis(text)
                        sentiment_counts[sentiment] += 1
                        relevant_articles += 1
                        
                        st.write(f"Article {relevant_articles}: {title}")
                        st.write(f"URL: {url}")
                        st.write(f"Sentiment: {sentiment}")
                        
                        summary = summarize_article(text)
                        st.write(f"Summary: {summary}")
                        st.write("-" * 50)
                    
                    elif is_related_to_company(title, ticker):
                        sentiment = sentiment_analysis(title)
                        sentiment_counts[sentiment] += 1
                        relevant_articles += 1
                        
                        st.write(f"Article {relevant_articles}: {title}")
                        st.write(f"URL: {url}")
                        st.write("Failed to extract article content. Sentiment based on the title.")
                        st.write(f"Sentiment: {sentiment}")
                        st.write("-" * 50)
                
                if relevant_articles == 0:
                    st.warning("No relevant articles found for the given ticker in the specified time range.")
                else:
                    st.write(f"\nSentiment Distribution for {ticker} based on {relevant_articles} relevant articles from the past {days_back} days:")
                    for sentiment, count in sentiment_counts.items():
                        percentage = (count / relevant_articles) * 100
                        st.write(f"{sentiment.capitalize()}: {count} articles ({percentage:.2f}%)")
        else:
            st.warning("Please enter a stock ticker.")

if __name__ == "__main__":
    main()
