import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    input_file = 'ai_tweets_with_vader_sentiment.csv'

    print(f"Loading data from {input_file} for evaluation...")
    df = pd.read_csv(input_file)

    # Prepare actual and predicted labels
    # Ensure consistent ordering for metrics if needed, though report handles it
    y_true = df['sentiment_label']
    y_pred = df['vader_sentiment']

    print("\n--- VADER Performance Evaluation ---")
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=['positive', 'negative', 'neutral'])
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['positive', 'negative', 'neutral'], yticklabels=['positive', 'negative', 'neutral'])
    plt.title('VADER AI Tweet Sentiment Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('vader_ai_sentiment_confusion_matrix.png')
    print("Confusion matrix plot saved as vader_ai_sentiment_confusion_matrix.png")
    # plt.show() # Don't use plt.show() in a script that might run headless 