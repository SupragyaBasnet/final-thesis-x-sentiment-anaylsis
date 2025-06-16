import pandas as pd
import torch # Import torch
from transformers import pipeline
from tqdm import tqdm

tqdm.pandas() # Enable progress_apply for pandas

if __name__ == "__main__":
    # Determine the device to use (GPU if available, else CPU)
    device = 0 if torch.cuda.is_available() else -1  # 0 for GPU index, -1 for CPU
    print(f"Device set to use: {'cuda' if device == 0 else 'cpu'}")

    input_file = 'cleaned_ai_tweets.csv'
    output_file = 'ai_tweets_with_transformer_sentiment.csv'

    print(f"Loading cleaned data from {input_file}...")
    df = pd.read_csv(input_file)

    # Initialize the sentiment analysis pipeline using a pre-trained model
    print("Loading sentiment analysis model (this may take a moment)...")
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

    print("Applying Transformer-based sentiment analysis... (This will take a while)")
    df['transformer_sentiment'] = df['cleaned_tweets'].progress_apply(lambda x: sentiment_pipeline(x)[0]['label'])

    # The model outputs 'POSITIVE' and 'NEGATIVE'. We need to map them.
    df['transformer_sentiment'] = df['transformer_sentiment'].replace({'POSITIVE': 'positive', 'NEGATIVE': 'negative'})

    print("Saving data with Transformer sentiment...")
    df.to_csv(output_file, index=False)
    print(f"Data with Transformer sentiment saved to {output_file}")

    print("\n--- Sample of Data with Transformer Sentiment ---")
    print(df.head())
    print("\n--- Transformer Sentiment Distribution ---")
    print(df['transformer_sentiment'].value_counts())
