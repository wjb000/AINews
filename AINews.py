import os
import requests
from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def fetch_news(stock):
    api_key = "your_actual_api_key_here"  # Hardcode the API key
    if not api_key:
        print("No API Key found.")
        return []
    
    url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={api_key}"
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

    inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors='pt')
    summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=30, max_length=100, early_stopping=True)
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def main():
    stock = input("Enter the stock you're interested in: ")
    news_data = fetch_news(stock)
    
    if not news_data:
        print("No news found for the specified stock.")
        return
    
    for i, (title, url) in enumerate(news_data):
        print(f"\nArticle {i + 1}: {title}")
        print(f"URL: {url}")
        
        text = extract_text(url)
        summary = summarize_text(text)
        
        print(f"Summary: {summary}")

if __name__ == "__main__":
    main()
