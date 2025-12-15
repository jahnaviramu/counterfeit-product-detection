"""Preprocess review-level CSVs into sentence-level CSVs for training SBERT classifier.

Reads CSV files from data/text (train/test) that contain 'text' or 'description' and 'label'.
Splits each review into sentences (simple regex) and writes out *_sentences.csv with columns: text,label

Usage:
    python preprocess_reviews.py
"""
import os
import pandas as pd
import re

BASE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'text')
TRAIN_IN = os.path.join(BASE_DIR, 'product_descriptions_train.csv')
TEST_IN = os.path.join(BASE_DIR, 'product_descriptions_test.csv')
TRAIN_OUT = os.path.join(BASE_DIR, 'product_descriptions_train_sentences.csv')
TEST_OUT = os.path.join(BASE_DIR, 'product_descriptions_test_sentences.csv')

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def split_into_sentences(text):
    if not isinstance(text, str):
        return []
    # naive split; keeps punctuation
    parts = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    return parts

def process_file(in_path, out_path):
    if not os.path.exists(in_path):
        print(f"Input file not found: {in_path}")
        return
    df = pd.read_csv(in_path)
    # find text column
    if 'text' in df.columns:
        text_col = 'text'
    elif 'description' in df.columns:
        text_col = 'description'
    else:
        raise ValueError('Input CSV must contain `text` or `description` column')
    if 'label' not in df.columns:
        raise ValueError('Input CSV must contain `label` column')

    rows = []
    for _, r in df.iterrows():
        label = r['label']
        sentences = split_into_sentences(r[text_col])
        if not sentences:
            # keep at least the original text
            rows.append({'text': str(r[text_col]), 'label': label})
        else:
            for s in sentences:
                rows.append({'text': s, 'label': label})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f'Wrote {len(out_df)} sentence rows to {out_path}')

def main():
    process_file(TRAIN_IN, TRAIN_OUT)
    process_file(TEST_IN, TEST_OUT)

if __name__ == '__main__':
    main()
