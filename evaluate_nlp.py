"""evaluate_nlp.py
Evaluate SBERT classifier on a sentence-level CSV. Expects columns: text,label
Prints classification report and saves per-example predictions to CSV.
"""
import argparse
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV with columns text,label')
    parser.add_argument('--model', default='data/text/sbert_clf.joblib')
    parser.add_argument('--model-name', default='data/text/sbert_model_name.txt')
    parser.add_argument('--out', default='nlp_eval_results.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    clf = joblib.load(args.model)
    print('Loaded classifier. Classes:', getattr(clf, 'classes_', None))
    # load model name if present
    model_name = None
    if os.path.exists(args.model_name):
        with open(args.model_name, 'r', encoding='utf-8') as f:
            model_name = f.read().strip()
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(model_name or 'all-MiniLM-L6-v2')
    texts = df['text'].astype(str).tolist()
    X = embedder.encode(texts, convert_to_numpy=True)
    preds = clf.predict(X)
    proba = None
    try:
        proba = clf.predict_proba(X)
    except Exception:
        proba = None

    print(classification_report(df['label'], preds))
    cm = confusion_matrix(df['label'], preds)
    print('Confusion matrix:\n', cm)

    out = []
    for i, t in enumerate(texts):
        p = preds[i]
        score = None
        if proba is not None:
            classes = list(clf.classes_)
            idx = classes.index(p)
            score = float(proba[i][idx])
        out.append({'text': t, 'label': df['label'].iloc[i], 'pred': p, 'score': score})

    pd.DataFrame(out).to_csv(args.out, index=False)
    print('Saved detailed results to', args.out)


if __name__ == '__main__':
    main()
