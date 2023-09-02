import requests
from datetime import datetime, timedelta
from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from textblob import TextBlob

def fetch_news(stock):
    api_key = "YOUR_API_KEY"
    to_date = datetime.now().isoformat()
    from_date = (datetime.now() - timedelta(days=1)).isoformat()
    
    sources = "bbc-news,the-verge,abc-news,financial-times,cnn,reuters,bloomberg,wall-street-journal"
    url = f"https://newsapi.org/v2/everything?q={stock}&from={from_date}&to={to_date}&sources={sources}&apiKey={api_key}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to fetch news. Status code: {response.status_code}")
        return []
    
    data = response.json()
    return [(article['title'], article['url']) for article in data.get('articles', [])]

def extract_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def summarize_text(text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer([text], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=30, max_length=100, early_stopping=True)
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def sentiment_analysis(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

if __name__ == "__main__":
    stock = input("Enter the stock you're interested in: ")
    news_data = fetch_news(stock)
    
    if not news_data:
        print("No news found.")
        exit(0)

    for i, (title, url) in enumerate(news_data):
        print(f"\nArticle {i+1}: {title}")
        print(f"URL: {url}")

        text = extract_text(url)
        summary = summarize_text(text)
        sentiment = sentiment_analysis(title)

        print(f"Summary: {summary}")
        print(f"Sentiment: {sentiment}")
