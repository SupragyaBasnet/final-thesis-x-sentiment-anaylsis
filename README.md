# X Sentiment Analysis Dashboard

A professional, multi-page dashboard for advanced sentiment and emotion analysis of tweets, powered by state-of-the-art AI models (VADER, BERT, BERTopic, etc.).

## Features
- Multi-page navigation (Overview, Tweet Explorer, Topic Modeling, Emotion Analysis, Model Comparison, Advanced Visualizations)
- Interactive, filterable Tweet Explorer with CSV download
- Custom date range filtering
- Topic modeling (BERTopic)
- Emotion analysis (HuggingFace transformer)
- Model comparison (VADER vs. transformer)
- Word clouds, frequency analysis, and AI insights
- Modern, responsive, professional UI/UX

## How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your data files (`cleaned_ai_tweets.csv`, `ai_tweets_with_vader_sentiment.csv`) in the project directory.
3. Run the dashboard:
   ```bash
   python sentiment_dashboard.py
   ```
4. Open your browser to [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

## Adding More Data
- To enable geographical analysis, add a `location`, `country`, or `coordinates` column to your CSV.
- To enable network graphs, add columns for retweet/reply/hashtag relationships.

## Customization
- Add your logo to the `assets/` folder and update the navbar for branding.
- Tweak colors and layout in `assets/style.css`.

## Screenshots
(Add screenshots of your dashboard here)

## License
MIT 