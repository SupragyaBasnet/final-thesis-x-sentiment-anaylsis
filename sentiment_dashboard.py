import pandas as pd
from dash import Dash, html, dcc, Input, Output, callback, dash_table
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
from bertopic import BERTopic
from transformers import pipeline
from sklearn.metrics import confusion_matrix, classification_report
import locale

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

app = Dash(__name__, suppress_callback_exceptions=True)

# Navigation bar (now a function for active link highlighting)
def navbar(pathname):
    def link(label, href):
        is_active = (pathname == href) or (href == '/' and pathname == '')
        return dcc.Link(
            label,
            href=href,
            className='nav-link active-link' if is_active else 'nav-link',
            style={'marginRight': '18px'}
        )
    return html.Nav([
        link('Overview', '/'),
        link('Tweet Explorer', '/explorer'),
        link('Topic Modeling', '/topics'),
        link('Model Comparison', '/comparison'),
    ], className='navbar')

# App layout with location
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='navbar-container'),
    html.Div(id='page-content')
])

# Page layouts (stubs for now)
def overview_layout():
    return html.Div([
        html.H1('X Sentiment Analysis Dashboard', 
                style={'textAlign': 'center', 'color': '#22304a', 'marginBottom': '30px', 'fontWeight': '700', 'letterSpacing': '0.5px'}),
        
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
        ], className='section-card', style={'marginBottom': '30px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'border': 'none'}),
        
        # Main metrics
        html.Div([
            html.Div([
                html.H3('Total Tweets', style={'textAlign': 'center'}),
                html.H2(id='total-tweets', style={'textAlign': 'center', 'color': '#22304a'})
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
        ], className='section-card'),
        
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
        ], className='section-card'),
        
        # Word Frequency Analysis
        html.Div([
            html.H2('Top Words by Sentiment', style={'textAlign': 'center', 'marginBottom': '20px', 'fontSize': '2em'}),
            html.P('These are the most frequent words in each sentiment category. The bigger the bar, the more often the word appears in tweets.', style={'textAlign': 'center', 'fontSize': '1.15em', 'color': '#6b7a90', 'marginBottom': '32px'}),
            html.Div([
                html.Div([
                    html.H3('Positive Words', style={'textAlign': 'center', 'fontSize': '1.3em'}),
                    dcc.Graph(id='positive-words', config={'displayModeBar': False})
                ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 18px'}),
                html.Div([
                    html.H3('Negative Words', style={'textAlign': 'center', 'fontSize': '1.3em'}),
                    dcc.Graph(id='negative-words', config={'displayModeBar': False})
                ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 18px'}),
                html.Div([
                    html.H3('Neutral Words', style={'textAlign': 'center', 'fontSize': '1.3em'}),
                    dcc.Graph(id='neutral-words', config={'displayModeBar': False})
                ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 18px'})
            ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'flex-start'})
        ], className='section-card'),

        # Unified AI Insights Section
        html.Div([
            html.H2('AI Insights', style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.Div(id='ai-insights-box')
        ], className='section-card', style={'maxWidth': '950px', 'margin': '0 auto 36px auto', 'background': '#f8fafd'})
    ])

def explorer_layout():
    columns = []
    if 'cleaned_tweets' in df.columns:
        columns.append({'name': 'Tweet', 'id': 'cleaned_tweets'})
    if 'date' in df.columns:
        columns.append({'name': 'Date', 'id': 'date'})
    if 'sentiment_label' in df.columns:
        columns.append({'name': 'Original Sentiment', 'id': 'sentiment_label'})
    if 'vader_sentiment' in df.columns:
        columns.append({'name': 'VADER Sentiment', 'id': 'vader_sentiment'})
    min_date = df['date'].min().date() if 'date' in df.columns else None
    max_date = df['date'].max().date() if 'date' in df.columns else None
    return html.Div([
        html.H1('Tweet Explorer', style={'textAlign': 'center'}),
        html.P('Search, filter, and explore individual tweets with their sentiment and date.', style={'textAlign': 'center', 'marginBottom': '24px'}),
        html.Div([
            html.Label('Select Date Range:'),
            dcc.DatePickerRange(
                id='explorer-date-range',
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                start_date=min_date,
                end_date=max_date,
                display_format='YYYY-MM-DD',
                style={'marginBottom': '18px'}
            ),
            html.Button('Download CSV', id='download-csv-btn', n_clicks=0, style={'marginLeft': '24px', 'padding': '8px 18px', 'fontWeight': '600', 'borderRadius': '7px', 'background': '#22304a', 'color': '#fff', 'border': 'none', 'fontSize': '1em', 'cursor': 'pointer'}),
            dcc.Download(id='download-csv')
        ], style={'textAlign': 'center', 'marginBottom': '18px'}),
        dash_table.DataTable(
            id='tweet-table',
            columns=columns,
            data=[],  # Will be filled by callback
            page_size=15,
            filter_action='native',
            sort_action='native',
            style_table={'overflowX': 'auto', 'margin': '0 auto', 'maxWidth': '1100px'},
            style_cell={
                'fontFamily': 'Inter, Segoe UI, Roboto, Arial, sans-serif',
                'fontSize': '1em',
                'padding': '10px',
                'backgroundColor': '#fff',
                'color': '#22304a',
                'border': '1px solid #e3e7ee',
                'maxWidth': '400px',
                'whiteSpace': 'normal',
            },
            style_header={
                'backgroundColor': '#f6f8fa',
                'fontWeight': 'bold',
                'color': '#22304a',
                'borderBottom': '2px solid #e3e7ee',
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'sentiment_label'},
                    'backgroundColor': '#eafaf1',
                    'color': '#27ae60',
                },
                {
                    'if': {'column_id': 'vader_sentiment'},
                    'backgroundColor': '#fbeee6',
                    'color': '#c0392b',
                },
            ],
            style_as_list_view=True,
        )
    ], style={'maxWidth': '1200px', 'margin': '0 auto', 'background': '#fff', 'borderRadius': '14px', 'boxShadow': '0 2px 10px rgba(34,48,74,0.06)', 'padding': '32px 28px', 'marginBottom': '32px'})

def topics_layout():
    sentiments = ['positive', 'negative', 'neutral']
    return html.Div([
        html.H1('Topic Modeling', style={'textAlign': 'center'}),
        html.P('Discover trending topics within each sentiment using BERTopic.', style={'textAlign': 'center', 'marginBottom': '24px'}),
        html.Div([
            html.Label('Select Sentiment:'),
            dcc.Dropdown(
                id='topic-sentiment-dropdown',
                options=[{'label': s.title(), 'value': s} for s in sentiments],
                value='positive',
                style={'width': '300px', 'margin': '0 auto'}
            )
        ], style={'textAlign': 'center', 'marginBottom': '24px'}),
        dcc.Loading(
            id='topic-loading',
            type='circle',
            color='#2d7ff9',
            children=[html.Div(id='topic-model-output')]
        )
    ], style={'maxWidth': '900px', 'margin': '0 auto', 'background': '#fff', 'borderRadius': '14px', 'boxShadow': '0 2px 10px rgba(34,48,74,0.06)', 'padding': '32px 28px', 'marginBottom': '32px'})

def comparison_layout():
    if 'sentiment_label' not in df.columns or 'vader_sentiment' not in df.columns:
        return html.Div([
            html.H1('Model Comparison', style={'textAlign': 'center'}),
            html.P('Both VADER and transformer sentiment labels are required for comparison.', style={'textAlign': 'center', 'color': '#c0392b'})
        ])
    cm = confusion_matrix(df['sentiment_label'], df['vader_sentiment'], labels=['positive', 'neutral', 'negative'])
    cm_fig = px.imshow(cm, 
        x=['positive', 'neutral', 'negative'], 
        y=['positive', 'neutral', 'negative'],
        color_continuous_scale='Blues',
        labels=dict(x='VADER Sentiment', y='Transformer Sentiment', color='Count'),
        text_auto=True,
        title='Confusion Matrix: VADER vs. Transformer'
    )
    cm_fig.update_layout(plot_bgcolor='#f8fafd', paper_bgcolor='#fff', margin=dict(l=40, r=40, t=60, b=40))
    report = classification_report(df['sentiment_label'], df['vader_sentiment'], output_dict=True, zero_division=0)
    # Prepare data for grouped bar chart
    metrics = ['precision', 'recall', 'f1-score']
    sentiments = ['positive', 'neutral', 'negative']
    metrics_data = {
        'Sentiment': [],
        'Metric': [],
        'Score': []
    }
    for sent in sentiments:
        for metric in metrics:
            metrics_data['Sentiment'].append(sent.title())
            metrics_data['Metric'].append(metric.title())
            metrics_data['Score'].append(report[sent][metric])
    metrics_df = pd.DataFrame(metrics_data)
    metrics_fig = px.bar(
        metrics_df,
        x='Sentiment',
        y='Score',
        color='Metric',
        barmode='group',
        text=metrics_df['Score'].apply(lambda x: f"{x:.2f}"),
        color_discrete_map={'Precision': '#2d7ff9', 'Recall': '#27ae60', 'F1-Score': '#c0392b'},
        title='Classification Metrics by Sentiment',
        height=400
    )
    metrics_fig.update_traces(textposition='outside', marker_line_width=0)
    metrics_fig.update_layout(
        font=dict(size=18),
        xaxis_title='Sentiment',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        plot_bgcolor='#f8fafd',
        paper_bgcolor='#fff',
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title_text='',
        bargap=0.2,
        title=dict(font=dict(size=22), x=0.5, xanchor='center')
    )
    # Use a styled HTML table for metrics
    metrics_table = html.Table([
        html.Thead(html.Tr([
            html.Th('Sentiment', style={'padding': '10px 24px', 'textAlign': 'center'}),
            html.Th('Precision', style={'padding': '10px 24px', 'textAlign': 'center'}),
            html.Th('Recall', style={'padding': '10px 24px', 'textAlign': 'center'}),
            html.Th('F1-score', style={'padding': '10px 24px', 'textAlign': 'center'})
        ])),
        html.Tbody([
            html.Tr([
                html.Td(sent.title(), style={'padding': '10px 24px', 'textAlign': 'center', 'fontWeight': '600'}),
                html.Td(f"{report[sent]['precision']:.2f}", style={'padding': '10px 24px', 'textAlign': 'center'}),
                html.Td(f"{report[sent]['recall']:.2f}", style={'padding': '10px 24px', 'textAlign': 'center'}),
                html.Td(f"{report[sent]['f1-score']:.2f}", style={'padding': '10px 24px', 'textAlign': 'center'})
            ]) for sent in sentiments
        ])
    ], style={'width': '100%', 'margin': '24px 0', 'borderCollapse': 'collapse', 'fontSize': '1.15em', 'background': '#f8fafd', 'borderRadius': '10px'})
    disagreements = df[df['sentiment_label'] != df['vader_sentiment']]
    unique_disagreements = disagreements.drop_duplicates(subset=['cleaned_tweets', 'sentiment_label', 'vader_sentiment']).head(10)
    disagreement_divs = []
    for _, row in unique_disagreements.iterrows():
        disagreement_divs.append(html.Div([
            html.Strong('Tweet: '), html.Span(row['cleaned_tweets'], style={'fontStyle': 'italic', 'color': '#6b7a90'}), html.Br(),
            html.Span(f"VADER: {row['vader_sentiment']} | Transformer: {row['sentiment_label']}", style={'color': '#c0392b', 'fontWeight': '600'})
        ], style={'marginBottom': '16px', 'padding': '10px', 'background': '#f8fafd', 'borderRadius': '8px'}))
    return html.Div([
        html.H1('Model Comparison', style={'textAlign': 'center'}),
        html.P('Compare VADER and transformer-based sentiment models on the same tweets.', style={'textAlign': 'center', 'marginBottom': '24px'}),
        dcc.Graph(figure=cm_fig),
        html.H3('Classification Metrics (Graph)', style={'marginTop': '32px'}),
        dcc.Graph(figure=metrics_fig, config={'displayModeBar': False}),
        html.H3('Classification Metrics (Table)', style={'marginTop': '32px'}),
        metrics_table,
        html.H3('Example Disagreements', style={'marginTop': '32px'}),
        html.Div(disagreement_divs)
    ], style={'maxWidth': '900px', 'margin': '0 auto', 'background': '#fff', 'borderRadius': '14px', 'boxShadow': '0 2px 10px rgba(34,48,74,0.06)', 'padding': '32px 28px', 'marginBottom': '32px'})

# Callback for page routing and navbar
@app.callback(
    [Output('page-content', 'children'), Output('navbar-container', 'children')],
    [Input('url', 'pathname')]
)
def display_page_and_navbar(pathname):
    if pathname == '/explorer':
        return explorer_layout(), navbar(pathname)
    elif pathname == '/topics':
        return topics_layout(), navbar(pathname)
    elif pathname == '/comparison':
        return comparison_layout(), navbar(pathname)
    else:
        return overview_layout(), navbar(pathname)

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
        words = get_word_frequencies(text, n=10)
        words_df = pd.DataFrame(words, columns=['word', 'count'])
        words_df['count_str'] = words_df['count'].apply(lambda x: locale.format_string('%d', x, grouping=True))
        fig = px.bar(
            words_df,
            x='count',
            y='word',
            orientation='h',
            title=title,
            color_discrete_sequence=[color],
            text='count_str',
        )
        fig.update_traces(
            textposition='outside',
            marker_line_width=0,
            marker=dict(line=dict(width=0)),
            width=0.8
        )
        fig.update_layout(
            xaxis_title='Frequency',
            yaxis_title='Word',
            showlegend=False,
            margin=dict(l=140, r=40, t=60, b=60),
            plot_bgcolor='#f8fafd',
            paper_bgcolor='#fff',
            font=dict(size=20, family='Inter, Segoe UI, Roboto, Arial, sans-serif'),
            height=600,
            bargap=0.4,
            title=dict(font=dict(size=26, family='Inter, Segoe UI, Roboto, Arial, sans-serif'), x=0.5, xanchor='center'),
            xaxis=dict(showgrid=True, gridcolor='#e3e7ee', tickfont=dict(size=18), tickformat='.2s', ticklabelposition='outside'),
            yaxis=dict(showgrid=False, tickfont=dict(size=18)),
            dragmode=False
        )
        fig.update_layout(modebar=dict(remove=['zoom', 'pan', 'select', 'lasso2d', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale']))
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
                        trend_parts.append(f"{sent.title()}: ▲ {diff} (increase)")
                    elif diff < 0:
                        trend_parts.append(f"{sent.title()}: ▼ {abs(diff)} (decrease)")
            if trend_parts:
                trend_text = html.Ul([html.Li(t) for t in trend_parts])
            else:
                trend_text = html.Div("Sentiment levels remained stable compared to the previous month.")
        else:
            trend_text = html.Div("Not enough data for trend analysis.")
    else:
        trend_text = html.Div("No date information available for trend analysis.")
    
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
                anomaly_text = html.Div([
                    html.Strong("Notable Activity: "),
                    f"Unusual spike(s) in tweet volume detected in: {', '.join([d.strftime('%b %Y') for d in spikes.index])}."
                ])
    
    # Compose the insight
    return html.Div([
        html.Div([
            html.Strong("Sentiment Change (Last Month): "),
            trend_text
        ], style={'marginBottom': '12px'}),
        html.Div([
            html.Strong("Key Drivers: "),
            html.Ul([
                html.Li(f"Top Positive Words: {top_pos}"),
                html.Li(f"Top Negative Words: {top_neg}"),
                html.Li(f"Top Neutral Words: {top_neu}")
            ])
        ], style={'marginBottom': '12px'}),
        anomaly_text if anomaly_text else None
    ])

# Callback to update tweet table based on date range
@app.callback(
    Output('tweet-table', 'data'),
    [Input('explorer-date-range', 'start_date'),
     Input('explorer-date-range', 'end_date')]
)
def update_tweet_table(start_date, end_date):
    if not start_date or not end_date:
        filtered = df
    else:
        filtered = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
    return filtered.head(1000).to_dict('records')

# Callback to download filtered data as CSV
@app.callback(
    Output('download-csv', 'data'),
    [Input('download-csv-btn', 'n_clicks'),
     Input('explorer-date-range', 'start_date'),
     Input('explorer-date-range', 'end_date')],
    prevent_initial_call=True
)
def download_csv(n_clicks, start_date, end_date):
    if not n_clicks:
        return None
    if not start_date or not end_date:
        filtered = df
    else:
        filtered = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
    return dcc.send_data_frame(filtered.to_csv, 'filtered_tweets.csv', index=False)

# Callback to run BERTopic and display topics for selected sentiment
@app.callback(
    Output('topic-model-output', 'children'),
    [Input('topic-sentiment-dropdown', 'value')]
)
def update_topic_model(selected_sentiment):
    # Filter tweets by sentiment
    tweets = df[df['sentiment_label'] == selected_sentiment]['cleaned_tweets'].dropna().astype(str).tolist()
    if not tweets or len(tweets) < 20:
        return html.Div('Not enough tweets for topic modeling.', style={'textAlign': 'center', 'color': '#c0392b'})
    # For speed, sample up to 2000 tweets
    if len(tweets) > 2000:
        tweets = np.random.choice(tweets, 2000, replace=False)
    # Fit BERTopic (cache for session if possible)
    topic_model = BERTopic(verbose=False)
    topics, probs = topic_model.fit_transform(tweets)
    topic_info = topic_model.get_topic_info().head(10)
    topic_keywords = [topic_model.get_topic(i) for i in topic_info['Topic'] if i != -1]
    topic_divs = []
    for idx, row in topic_info.iterrows():
        if row['Topic'] == -1:
            continue
        words = ', '.join([w for w, _ in topic_model.get_topic(row['Topic'])])
        example_idx = np.where(np.array(topics) == row['Topic'])[0]
        example_tweet = tweets[example_idx[0]] if len(example_idx) > 0 else ''
        topic_divs.append(html.Div([
            html.H4(f"Topic {row['Topic']+1}: {words}", style={'marginBottom': '6px', 'color': '#22304a'}),
            html.P(f"Example Tweet: {example_tweet}", style={'fontStyle': 'italic', 'color': '#6b7a90', 'marginBottom': '18px'})
        ], style={'marginBottom': '18px', 'padding': '12px', 'background': '#f8fafd', 'borderRadius': '8px'}))
    return html.Div(topic_divs)

if __name__ == '__main__':
    print("\n--- Starting Multi-Page Dashboard ---")
    print("Open your web browser and go to http://127.0.0.1:8050/")
    app.run(debug=True) 
    app.run(debug=True) 