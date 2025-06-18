import pandas as pd
import numpy as np

# Load the CSV
input_file = 'cleaned_ai_tweets.csv'
df = pd.read_csv(input_file)

# Generate date range
num_rows = len(df)
date_range = pd.date_range(start='2023-01-01', end='2025-12-31', periods=num_rows)
df['date'] = date_range.strftime('%Y-%m-%d')

# Save back to CSV (overwrite original)
df.to_csv(input_file, index=False)

print(f"Added 'date' column to {input_file} spanning 2023-01-01 to 2025-12-31.") 