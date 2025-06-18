import pandas as pd
from dash import Dash, html, dcc, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
import io
import base64
from collections import Counter
import numpy as np
from datetime import datetime

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
    
    # Convert date column to datetime if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Drop rows where 'date' is NaT after coercion
        df.dropna(subset=['date'], inplace=True)
        if not df.empty:
            df['month'] = df['date'].dt.to_period('M')
            df['month'] = df['month'].astype(str)
        else:
            print("Warning: 'date' column found but contains no valid dates after conversion.")

except FileNotFoundError as e:
    print(f"Error loading data files: {e}. Please ensure 'cleaned_ai_tweets.csv' and 'ai_tweets_with_vader_sentiment.csv' exist.")
    exit()

app = Dash(__name__)

# Function to generate word cloud as a base64 encoded image
def plot_wordcloud(text, title):
    if not text.strip():  # Check if text is empty
        return 'data:image/png;base64,'
    
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate(text)
        
        img = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=15, pad=20)
        plt.tight_layout(pad=0)
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        img.seek(0)
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return 'data:image/png;base64,'

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

# Calculate word frequencies
def get_word_frequencies(text, n=20):
    words = text.lower().split()
    return Counter(words).most_common(n)

# Layout of the dashboard
app.layout = html.Div([
    html.H1('X Sentiment Analysis Dashboard', 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
    
    # AI Insights Box
    html.Div([
        html.H2('AI Insights', style={'textAlign': 'center', 'color': '#8e44ad'}),
        html.Div(id='ai-insights-box', style={
            'backgroundColor': '#f3e6fa',
            'borderRadius': '10px',
            'padding': '20px',
            'marginBottom': '30px',
            'fontSize': '1.15em',
            'color': '#2c3e50',
            'boxShadow': '0 2px 8px rgba(142,68,173,0.08)'
        })
    ]),
    
    # Filters
    html.Div([
        html.Div([
            html.Label('Select Sentiment Type:'),
            dcc.Dropdown(
                id='sentiment-type',
                options=[
                    {'label': 'Original Labels', 'value': 'original'},
                    {'label': 'VADER Labels', 'value': 'vader'}
                ],
                value='original',
                style={'width': '100%'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
        
        html.Div([
            html.Label('Select Time Period:'),
            dcc.Dropdown(
                id='time-period',
                options=[
                    {'label': 'All Time', 'value': 'all'},
                    {'label': 'Last Month', 'value': 'month'},
                    {'label': 'Last Week', 'value': 'week'}
                ],
                value='all',
                style={'width': '100%'}
            )
        ], style={'width': '30%', 'display': 'inline-block'})
    ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
    
    # Main metrics
    html.Div([
        html.Div([
            html.H3('Total Tweets', style={'textAlign': 'center'}),
            html.H2(id='total-tweets', style={'textAlign': 'center', 'color': '#2c3e50'})
        ], className='metric-box'),
        html.Div([
            html.H3('Positive Tweets', style={'textAlign': 'center'}),
            html.H2(id='positive-tweets', style={'textAlign': 'center', 'color': '#27ae60'})
        ], className='metric-box'),
        html.Div([
            html.H3('Negative Tweets', style={'textAlign': 'center'}),
            html.H2(id='negative-tweets', style={'textAlign': 'center', 'color': '#c0392b'})
        ], className='metric-box'),
        html.Div([
            html.H3('Neutral Tweets', style={'textAlign': 'center'}),
            html.H2(id='neutral-tweets', style={'textAlign': 'center', 'color': '#3498db'})
        ], className='metric-box')
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'}),
    
    # Sentiment Distribution and Sentiment Over Time
    html.Div([
        html.Div([
            html.H2('Sentiment Distribution', style={'textAlign': 'center'}),
            dcc.Graph(id='sentiment-distribution')
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H2('Sentiment Over Time', style={'textAlign': 'center'}),
            dcc.Graph(id='sentiment-timeline')
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ], style={'marginBottom': '30px'}),
    
    # Word Clouds Section
    html.Div([
        html.H2('Word Clouds Analysis', style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.H3('Positive Sentiment', style={'textAlign': 'center'}),
                html.Img(id='positive-wordcloud', style={'width': '100%'})
            ], style={'width': '32%', 'display': 'inline-block'}),
            html.Div([
                html.H3('Negative Sentiment', style={'textAlign': 'center'}),
                html.Img(id='negative-wordcloud', style={'width': '100%'})
            ], style={'width': '32%', 'display': 'inline-block'}),
            html.Div([
                html.H3('Neutral Sentiment', style={'textAlign': 'center'}),
                html.Img(id='neutral-wordcloud', style={'width': '100%'})
            ], style={'width': '32%', 'display': 'inline-block'})
        ])
    ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
    
    # Word Frequency Analysis
    html.Div([
        html.H2('Top Words by Sentiment', style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.H3('Positive Words', style={'textAlign': 'center'}),
                dcc.Graph(id='positive-words')
            ], style={'width': '32%', 'display': 'inline-block'}),
            html.Div([
                html.H3('Negative Words', style={'textAlign': 'center'}),
                dcc.Graph(id='negative-words')
            ], style={'width': '32%', 'display': 'inline-block'}),
            html.Div([
                html.H3('Neutral Words', style={'textAlign': 'center'}),
                dcc.Graph(id='neutral-words')
            ], style={'width': '32%', 'display': 'inline-block'})
        ])
    ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
])

# Callback for updating metrics
@callback(
    [Output('total-tweets', 'children'),
     Output('positive-tweets', 'children'),
     Output('negative-tweets', 'children'),
     Output('neutral-tweets', 'children')],
    [Input('sentiment-type', 'value'),
     Input('time-period', 'value')]
)
def update_metrics(sentiment_type, time_period):
    sentiment_col = 'sentiment_label' if sentiment_type == 'original' else 'vader_sentiment'
    
    if time_period != 'all' and 'date' in df.columns:
        if time_period == 'month':
            filtered_df = df[df['date'] >= (pd.Timestamp.now() - pd.DateOffset(months=1))]
        else:  # week
            filtered_df = df[df['date'] >= (pd.Timestamp.now() - pd.DateOffset(weeks=1))]
    else:
        filtered_df = df
    
    total = len(filtered_df)
    positive = len(filtered_df[filtered_df[sentiment_col] == 'positive'])
    negative = len(filtered_df[filtered_df[sentiment_col] == 'negative'])
    neutral = len(filtered_df[filtered_df[sentiment_col] == 'neutral'])
    
    return total, positive, negative, neutral

# Callback for sentiment distribution
@callback(
    Output('sentiment-distribution', 'figure'),
    [Input('sentiment-type', 'value'),
     Input('time-period', 'value')]
)
def update_sentiment_distribution(sentiment_type, time_period):
    sentiment_col = 'sentiment_label' if sentiment_type == 'original' else 'vader_sentiment'
    
    if time_period != 'all' and 'date' in df.columns:
        if time_period == 'month':
            filtered_df = df[df['date'] >= (pd.Timestamp.now() - pd.DateOffset(months=1))]
        else:  # week
            filtered_df = df[df['date'] >= (pd.Timestamp.now() - pd.DateOffset(weeks=1))]
    else:
        filtered_df = df
    
    sentiment_counts = filtered_df[sentiment_col].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title=f'Sentiment Distribution ({sentiment_type.title()})',
        color=sentiment_counts.index,
        color_discrete_map={'positive': '#27ae60', 'negative': '#c0392b', 'neutral': '#3498db'}
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True)
    
    return fig

# Callback for sentiment timeline
@callback(
    Output('sentiment-timeline', 'figure'),
    [Input('sentiment-type', 'value'),
     Input('time-period', 'value')]
)
def update_sentiment_timeline(sentiment_type, time_period):
    if 'date' not in df.columns:
        return go.Figure()
    
    sentiment_col = 'sentiment_label' if sentiment_type == 'original' else 'vader_sentiment'
    
    if time_period != 'all':
        if time_period == 'month':
            filtered_df = df[df['date'] >= (pd.Timestamp.now() - pd.DateOffset(months=1))]
        else:  # week
            filtered_df = df[df['date'] >= (pd.Timestamp.now() - pd.DateOffset(weeks=1))]
    else:
        filtered_df = df
    
    timeline_data = filtered_df.groupby([filtered_df['date'].dt.to_period('M').dt.to_timestamp(), sentiment_col]).size().reset_index(name='count')
    
    fig = px.line(
        timeline_data,
        x='date',
        y='count',
        color=sentiment_col,
        title=f'Sentiment Trends Over Time ({sentiment_type.title()})',
        labels={'date': 'Date', 'count': 'Number of Tweets'},
        color_discrete_map={'positive': '#27ae60', 'negative': '#c0392b', 'neutral': '#3498db'}
    )
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Tweets',
        hovermode='x unified'
    )
    
    return fig

# Callback for word clouds
@callback(
    [Output('positive-wordcloud', 'src'),
     Output('negative-wordcloud', 'src'),
     Output('neutral-wordcloud', 'src')],
    [Input('sentiment-type', 'value'),
     Input('time-period', 'value')]
)
def update_wordclouds(sentiment_type, time_period):
    sentiment_col = 'sentiment_label' if sentiment_type == 'original' else 'vader_sentiment'
    
    if time_period != 'all' and 'date' in df.columns:
        if time_period == 'month':
            filtered_df = df[df['date'] >= (pd.Timestamp.now() - pd.DateOffset(months=1))]
        else:  # week
            filtered_df = df[df['date'] >= (pd.Timestamp.now() - pd.DateOffset(weeks=1))]
    else:
        filtered_df = df
    
    positive_text = " ".join(filtered_df[filtered_df[sentiment_col] == 'positive']['cleaned_tweets'].dropna().tolist())
    negative_text = " ".join(filtered_df[filtered_df[sentiment_col] == 'negative']['cleaned_tweets'].dropna().tolist())
    neutral_text = " ".join(filtered_df[filtered_df[sentiment_col] == 'neutral']['cleaned_tweets'].dropna().tolist())
    
    return (
        plot_wordcloud(positive_text, 'Positive Tweets'),
        plot_wordcloud(negative_text, 'Negative Tweets'),
        plot_wordcloud(neutral_text, 'Neutral Tweets')
    )

# Callback for word frequency analysis
@callback(
    [Output('positive-words', 'figure'),
     Output('negative-words', 'figure'),
     Output('neutral-words', 'figure')],
    [Input('sentiment-type', 'value'),
     Input('time-period', 'value')]
)
def update_word_frequencies(sentiment_type, time_period):
    sentiment_col = 'sentiment_label' if sentiment_type == 'original' else 'vader_sentiment'
    
    if time_period != 'all' and 'date' in df.columns:
        if time_period == 'month':
            filtered_df = df[df['date'] >= (pd.Timestamp.now() - pd.DateOffset(months=1))]
        else:  # week
            filtered_df = df[df['date'] >= (pd.Timestamp.now() - pd.DateOffset(weeks=1))]
    else:
        filtered_df = df
    
    def create_word_freq_figure(text, title, color):
        words = get_word_frequencies(text)
        words_df = pd.DataFrame(words, columns=['word', 'count'])
        
        fig = px.bar(
            words_df,
            x='count',
            y='word',
            orientation='h',
            title=title,
            color_discrete_sequence=[color]
        )
        
        fig.update_layout(
            xaxis_title='Frequency',
            yaxis_title='Word',
            showlegend=False
        )
        
        return fig
    
    positive_text = " ".join(filtered_df[filtered_df[sentiment_col] == 'positive']['cleaned_tweets'].dropna().tolist())
    negative_text = " ".join(filtered_df[filtered_df[sentiment_col] == 'negative']['cleaned_tweets'].dropna().tolist())
    neutral_text = " ".join(filtered_df[filtered_df[sentiment_col] == 'neutral']['cleaned_tweets'].dropna().tolist())
    
    return (
        create_word_freq_figure(positive_text, 'Top Positive Words', '#27ae60'),
        create_word_freq_figure(negative_text, 'Top Negative Words', '#c0392b'),
        create_word_freq_figure(neutral_text, 'Top Neutral Words', '#3498db')
    )

# Callback for AI Insights
@callback(
    Output('ai-insights-box', 'children'),
    [Input('sentiment-type', 'value'),
     Input('time-period', 'value')]
)
def update_ai_insights(sentiment_type, time_period):
    sentiment_col = 'sentiment_label' if sentiment_type == 'original' else 'vader_sentiment'
    
    if time_period != 'all' and 'date' in df.columns:
        if time_period == 'month':
            filtered_df = df[df['date'] >= (pd.Timestamp.now() - pd.DateOffset(months=1))]
        else:  # week
            filtered_df = df[df['date'] >= (pd.Timestamp.now() - pd.DateOffset(weeks=1))]
    else:
        filtered_df = df
    
    # Sentiment trends
    trend_text = ""
    if 'date' in filtered_df.columns and not filtered_df.empty:
        monthly_counts = filtered_df.groupby([filtered_df['date'].dt.to_period('M').dt.to_timestamp(), sentiment_col]).size().unstack(fill_value=0)
        if len(monthly_counts) > 1:
            last_month = monthly_counts.iloc[-1]
            prev_month = monthly_counts.iloc[-2]
            trend_parts = []
            for sent in ['positive', 'negative', 'neutral']:
                if sent in last_month and sent in prev_month:
                    diff = last_month[sent] - prev_month[sent]
                    if diff > 0:
                        trend_parts.append(f"{sent.title()} sentiment increased by {diff} compared to previous month.")
                    elif diff < 0:
                        trend_parts.append(f"{sent.title()} sentiment decreased by {abs(diff)} compared to previous month.")
            if trend_parts:
                trend_text = " ".join(trend_parts)
            else:
                trend_text = "Sentiment levels remained stable compared to the previous month."
        else:
            trend_text = "Not enough data for trend analysis."
    else:
        trend_text = "No date information available for trend analysis."
    
    # Most influential words
    def top_words(sentiment):
        text = " ".join(filtered_df[filtered_df[sentiment_col] == sentiment]['cleaned_tweets'].dropna().tolist())
        words = [w for w, _ in Counter(text.lower().split()).most_common(3)]
        return ', '.join(words) if words else 'N/A'
    top_pos = top_words('positive')
    top_neg = top_words('negative')
    top_neu = top_words('neutral')
    
    # Anomaly detection (simple spike detection)
    anomaly_text = ""
    if 'date' in filtered_df.columns and not filtered_df.empty:
        monthly_total = filtered_df.groupby(filtered_df['date'].dt.to_period('M').dt.to_timestamp()).size()
        if len(monthly_total) > 2:
            mean = monthly_total.mean()
            std = monthly_total.std()
            spikes = monthly_total[monthly_total > mean + 2*std]
            if not spikes.empty:
                anomaly_text = f"Unusual spike(s) in tweet volume detected in: {', '.join([d.strftime('%b %Y') for d in spikes.index])}. "
    
    # Compose the insight
    insight = f"{trend_text} Top positive words: {top_pos}. Top negative words: {top_neg}. Top neutral words: {top_neu}. {anomaly_text}"
    return insight

if __name__ == '__main__':
    print("\n--- Starting Dashboard ---")
    print("Open your web browser and go to http://127.0.0.1:8050/")
    app.run(debug=True) 