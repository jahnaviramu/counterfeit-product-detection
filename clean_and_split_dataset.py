import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Path to the merged CSV dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/text/product_descriptions_merged.csv')
CLEANED_PATH = os.path.join(os.path.dirname(__file__), 'data/text/product_descriptions_cleaned.csv')

# Load dataset
df = pd.read_csv(DATA_PATH)

# Basic cleaning: drop rows with missing values, strip whitespace
df = df.dropna()
df['description'] = df['description'].astype(str).str.strip()
df['label'] = df['label'].astype(str).str.strip()

# Remove duplicates
df = df.drop_duplicates()

# Save cleaned dataset
df.to_csv(CLEANED_PATH, index=False)
print(f"Cleaned data saved to {CLEANED_PATH}")

# Split into train/test (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_path = os.path.join(os.path.dirname(__file__), 'data/text/product_descriptions_train.csv')
test_path = os.path.join(os.path.dirname(__file__), 'data/text/product_descriptions_test.csv')

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"Train data saved to {train_path}")
print(f"Test data saved to {test_path}")
