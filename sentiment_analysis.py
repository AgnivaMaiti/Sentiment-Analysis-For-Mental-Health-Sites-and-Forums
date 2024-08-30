from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import nltk

nltk.download('vader_lexicon')
nltk.download('punkt')

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']

    if sentiment_score >= 0.05:
        return 'Positive'
    elif sentiment_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def get_text(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        if text.strip() == '':
            print("No meaningful content extracted from the URL.")
        return text
    else:
        print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
        return None

def prepare_dataset(url):
    text = get_text(url)
    if text:
        sentences = nltk.sent_tokenize(text)
        if len(sentences) == 0:
            print("No sentences found in the extracted text.")
            return

        sentiments = [analyze_sentiment(sentence) for sentence in sentences]
        df = pd.DataFrame({'Sentence': sentences, 'Sentiment': sentiments})

        if df.empty:
            print("DataFrame is empty after processing. No data to save.")
            return

        df.to_csv('website_sentiments.csv', index=False)
        print("CSV file created successfully.")
    else:
        print("No text extracted to analyze.")

def visualize_sentiments():
    try:
        df = pd.read_csv('website_sentiments.csv')
    except pd.errors.EmptyDataError:
        print("CSV file is empty. No data to visualize.")
        return

    if 'Sentence' not in df.columns or 'Sentiment' not in df.columns:
        print("Dataset columns are not as expected. Please check the dataset format.")
        return

    sentiment_counts = df['Sentiment'].value_counts()

    if sentiment_counts.empty:
        print("No sentiment data to visualize.")
        return

    plt.figure(figsize=(8, 6))
    sb.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
    plt.title('Number of Neutral, Positive, and Negative Sentences')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()



website_url = "https://en.m.wikipedia.org/wiki/Flood_myth"

prepare_dataset(website_url)
visualize_sentiments()
