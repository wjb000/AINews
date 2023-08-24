import requests
from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def fetch_news(stock):
    url = f"https://newsapi.org/v2/everything?q={stock}&apiKey=YOUR_API_KEY"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch news. Status code: {response.status_code}")
        print(response.json())
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

def get_stock_news_summary(stock):
    news_data = fetch_news(stock)
    summaries = []
    
    if not news_data:
        return []
    
    for title, url in news_data:
        text = extract_text(url)
        summary = summarize_text(text)
        
        summaries.append({
            "title": title,
            "url": url,
            "summary": summary
        })
        
    return summaries

if __name__ == "__main__":
    stock = input("Enter the stock you're interested in: ")
    summaries = get_stock_news_summary(stock)
    
    if len(summaries) == 0:
        print("No news found for the specified stock.")
    else:
        for i, summary in enumerate(summaries):
            print(f"\nArticle {i+1}: {summary['title']}")
            print(f"URL: {summary['url']}")
            print(f"Summary: {summary['summary']}")
