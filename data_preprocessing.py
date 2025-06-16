import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is downloaded (already done in previous step)
# nltk.download('stopwords')
# nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # 1. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 2. Remove Twitter handles (@mentions)
    text = re.sub(r'@\w+', '', text)
    # 3. Remove hashtags (leaving the text)
    text = re.sub(r'#\w+', '', text)
    # 4. Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text) # Keep only words and spaces
    # 5. Remove numbers
    text = re.sub(r'\d+', '', text)
    # 6. Convert to lowercase
    text = text.lower()
    # 7. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization, Stop Word Removal, Lemmatization
    tokens = []
    for word in text.split():
        if word not in stop_words and len(word) > 1: # Remove single character words after cleaning
            tokens.append(lemmatizer.lemmatize(word))
            
    return " ".join(tokens)

if __name__ == "__main__":
    input_file = 'ChatGPT-Sentiment-Analysis/original data/file.csv'
    output_file = 'cleaned_ai_tweets.csv'

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    print("Preprocessing text data...")
    # Apply preprocessing to the 'tweets' column
    df['cleaned_tweets'] = df['tweets'].apply(preprocess_text)
    
    # Map labels for consistency (if needed, this dataset seems fine but good practice)
    # 'good', 'bad', 'neutral' -> 'positive', 'negative', 'neutral'
    df['sentiment_label'] = df['labels'].replace({'good': 'positive', 'bad': 'negative'})

    # Select relevant columns for the final dataset
    df_cleaned = df[['cleaned_tweets', 'sentiment_label']].copy()
    
    # Remove any rows where 'cleaned_tweets' might have become empty after cleaning
    df_cleaned.dropna(subset=['cleaned_tweets'], inplace=True)
    df_cleaned = df_cleaned[df_cleaned['cleaned_tweets'].str.strip() != '']

    df_cleaned.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

    print("\n--- Sample of Cleaned Data ---")
    print(df_cleaned.head())
    print("\n--- Cleaned Data Info ---")
    print(df_cleaned.info()) 