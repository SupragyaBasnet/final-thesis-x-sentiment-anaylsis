import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    input_file = 'ai_tweets_with_transformer_sentiment.csv'

    print(f"Loading data from {input_file} for evaluation...")
    df = pd.read_csv(input_file)

    # Prepare actual and predicted labels
    y_true = df['sentiment_label']
    y_pred = df['transformer_sentiment']

    print("\n--- Transformer Model Performance Evaluation ---")
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")

    # Classification Report
    # Handle cases where labels might be missing in predictions if the model only predicts 2 classes
    # The original dataset has 'positive', 'negative', 'neutral'. The transformer model trained on SST-2 English outputs 'positive'/'negative'.
    # Let's adjust for this by defining the target_names based on what's actually present in y_true
    unique_labels = sorted(y_true.unique())
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=unique_labels, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Transformer AI Tweet Sentiment Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('transformer_ai_sentiment_confusion_matrix.png')
    print("Confusion matrix plot saved as transformer_ai_sentiment_confusion_matrix.png")
    # plt.show() # Don't use plt.show() in a script that might run headless 