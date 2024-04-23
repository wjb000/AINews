# AINews: Stock News Analyzer

AINews is a Python program that fetches and analyzes news articles related to a specified stock. It utilizes the NewsAPI, BART model from the Transformers library, and TextBlob library to provide summaries and sentiment analysis of the news articles. The program offers a user-friendly Streamlit UI for easy interaction.

## Features

- Fetches news articles related to a specified stock from the past 24 hours
- Summarizes the content of each news article
- Performs sentiment analysis on the summaries
- Displays the results in a Streamlit UI


## Usage

1. Install the required dependencies.
2. Obtain an API key from NewsAPI.
3. Replace the `api_key` variable with your NewsAPI key.
4. Run `streamlit run AINews.py`.
5. Enter the stock and click "Analyze".

## Acknowledgements

- NewsAPI, Transformers, TextBlob, Streamlit

## License

MIT License
