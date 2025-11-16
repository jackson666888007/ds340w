import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def load_data(csv_path, text_col, label_col):
    df = pd.read_csv(csv_path)
    texts = df[text_col].astype(str).values
    labels = df[label_col].values
    return texts, labels


def build_vectorizer(name, max_features, ngram_max):
    ngram_range = (1, ngram_max)

    if name == "tfidf":
        vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )
    elif name == "count":
        vec = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )
    else:
        raise ValueError(f"Unknown vectorizer: {name}")

    return vec


def build_logreg(C):
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        solver="liblinear"
    )
    return model


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Logistic Regression Sentiment Experiment (Novelty for Week 11)"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV file, e.g. data/imdb_full.csv"
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="review",
        help="Name of the text column in the CSV"
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Name of the label column in the CSV"
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        choices=["tfidf", "count"],
        default="tfidf",
        help="Vectorizer type"
    )
    parser.add_argument(
        "--ngram_max",
        type=int,
        default=2,
        help="Max n-gram size (e.g. 1 for unigram, 2 for bigram)"
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=50000,
        help="Max number of features for the vectorizer"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test size proportion (e.g. 0.2 for 80/20 split)"
    )
    parser.add_argument(
        "--C",
        type=float,
        default=2.0,
        help="Inverse regularization strength for Logistic Regression"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for train/test split"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/logreg_results.json",
        help="Where to save metrics in JSON format"
    )

    args = parser.parse_args()

    print("Loading data...")
    texts, labels = load_data(args.data, args.text_col, args.label_col)

    print("Splitting train / test...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels
    )

    print(f"Building {args.vectorizer} vectorizer with ngram_max={args.ngram_max}...")
    vectorizer = build_vectorizer(
        args.vectorizer,
        max_features=args.max_features,
        ngram_max=args.ngram_max
    )

    print("Fitting vectorizer on training data...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Training Logistic Regression (C={args.C})...")
    model = build_logreg(C=args.C)
    model.fit(X_train_vec, y_train)

    print("Evaluating on test set...")
    y_pred = model.predict(X_test_vec)
    metrics = evaluate(y_test, y_pred)

    print("=== Logistic Regression Results ===")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "logreg",
                "vectorizer": args.vectorizer,
                "ngram_max": args.ngram_max,
                "max_features": args.max_features,
                "C": args.C,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "metrics": metrics,
            },
            f,
            indent=2
        )

    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
