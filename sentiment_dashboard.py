import pandas as pd
from dash import Dash, html, dcc
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# Load the cleaned data with original and VADER sentiments
# Ensure these files exist after running data_preprocessing.py and sentiment_analysis.py
try:
    df_cleaned = pd.read_csv('cleaned_ai_tweets.csv')
    df_vader = pd.read_csv('ai_tweets_with_vader_sentiment.csv')
    
    # Merge the VADER results back into the cleaned DataFrame if they're not already there
    # Assuming 'cleaned_tweets' is the common column or index alignment is stable
    # For simplicity, we'll just ensure we have both labels if they are in separate files initially
    # If df_cleaned already contains sentiment_label, that's fine. df_vader must contain vader_sentiment.
    df = df_cleaned.copy()
    if 'vader_sentiment' not in df.columns and 'vader_sentiment' in df_vader.columns:
        df = pd.merge(df, df_vader[['cleaned_tweets', 'vader_sentiment']], on='cleaned_tweets', how='left')

except FileNotFoundError as e:
    print(f"Error loading data files: {e}. Please ensure 'cleaned_ai_tweets.csv' and 'ai_tweets_with_vader_sentiment.csv' exist.")
    exit()

app = Dash(__name__)

# Function to generate word cloud as a base64 encoded image
def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

# Generate word clouds for each sentiment category (Original Labels)
positive_tweets_original = " ".join(df[df['sentiment_label'] == 'positive']['cleaned_tweets'].dropna().tolist())
negative_tweets_original = " ".join(df[df['sentiment_label'] == 'negative']['cleaned_tweets'].dropna().tolist())
neutral_tweets_original = " ".join(df[df['sentiment_label'] == 'neutral']['cleaned_tweets'].dropna().tolist())

wc_positive_original = plot_wordcloud(positive_tweets_original, 'Word Cloud: Positive AI Tweets (Original)')
wc_negative_original = plot_wordcloud(negative_tweets_original, 'Word Cloud: Negative AI Tweets (Original)')
wc_neutral_original = plot_wordcloud(neutral_tweets_original, 'Word Cloud: Neutral AI Tweets (Original)')

# Generate word clouds for each sentiment category (VADER Labels)
positive_tweets_vader = " ".join(df[df['vader_sentiment'] == 'positive']['cleaned_tweets'].dropna().tolist())
negative_tweets_vader = " ".join(df[df['vader_sentiment'] == 'negative']['cleaned_tweets'].dropna().tolist())
neutral_tweets_vader = " ".join(df[df['vader_sentiment'] == 'neutral']['cleaned_tweets'].dropna().tolist())

wc_positive_vader = plot_wordcloud(positive_tweets_vader, 'Word Cloud: Positive AI Tweets (VADER)')
wc_negative_vader = plot_wordcloud(negative_tweets_vader, 'Word Cloud: Negative AI Tweets (VADER)')
wc_neutral_vader = plot_wordcloud(neutral_tweets_vader, 'Word Cloud: Neutral AI Tweets (VADER)')

# Layout of the dashboard
app.layout = html.Div(children=[
    html.H1(children='AI Tweet Sentiment Analysis Dashboard'),

    html.H2(children='Overall Sentiment Distribution (Original Labels)'),
    dcc.Graph(
        id='sentiment-distribution-original',
        figure=px.bar(
            df['sentiment_label'].value_counts().reset_index(name='Count').rename(columns={'index': 'Sentiment'}),
            x='Sentiment',
            y='Count',
            labels={'Sentiment': 'Sentiment', 'Count': 'Count'},
            title='Distribution of Original Sentiment Labels',
            color='Sentiment',
            color_discrete_map={'positive':'green', 'negative':'red', 'neutral':'blue'}
        )
    ),

    html.H2(children='Overall Sentiment Distribution (VADER Labels)'),
    dcc.Graph(
        id='sentiment-distribution-vader',
        figure=px.bar(
            df['vader_sentiment'].value_counts().reset_index(name='Count').rename(columns={'index': 'Sentiment'}),
            x='Sentiment',
            y='Count',
            labels={'Sentiment': 'Sentiment', 'Count': 'Count'},
            title='Distribution of VADER Sentiment Labels',
            color='Sentiment',
            color_discrete_map={'positive':'green', 'negative':'red', 'neutral':'blue'}
        )
    ),

    html.H2(children='Word Clouds (Original Labels)'),
    html.Div(style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}, children=[
        html.Img(src=wc_positive_original, style={'margin': '10px', 'max-width': '45%'}),
        html.Img(src=wc_negative_original, style={'margin': '10px', 'max-width': '45%'}),
        html.Img(src=wc_neutral_original, style={'margin': '10px', 'max-width': '45%'}),
    ]),

    html.H2(children='Word Clouds (VADER Labels)'),
    html.Div(style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}, children=[
        html.Img(src=wc_positive_vader, style={'margin': '10px', 'max-width': '45%'}),
        html.Img(src=wc_negative_vader, style={'margin': '10px', 'max-width': '45%'}),
        html.Img(src=wc_neutral_vader, style={'margin': '10px', 'max-width': '45%'}),
    ])
])

if __name__ == '__main__':
    print("\n--- Starting Dashboard ---")
    print("Open your web browser and go to http://127.0.0.1:8050/")
    app.run_server(debug=True) 