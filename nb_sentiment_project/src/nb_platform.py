from dataclasses import dataclass, asdict
from typing import Tuple, Dict
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os

def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df["text"].astype(str).values, df["label"].astype(int).values

def make_vectorizer(name: str, max_features: int, ngram_max: int):
    if name == "count":
        return CountVectorizer(max_features=max_features, ngram_range=(1, ngram_max), stop_words="english")
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max), stop_words="english")

def make_nb(name: str, alpha: float):
    if name == "bnb": return BernoulliNB(alpha=alpha)
    if name == "mnb": return MultinomialNB(alpha=alpha)
    raise ValueError("model must be 'bnb' or 'mnb'")

def build_pipeline(vec_name: str, model_name: str, max_features: int, ngram_max: int, alpha: float):
    vec = make_vectorizer(vec_name, max_features, ngram_max)
    nb = make_nb(model_name, alpha)
    if isinstance(nb, BernoulliNB):
        to_dense = FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)
        return Pipeline([("vec", vec), ("to_dense", to_dense), ("clf", nb)])
    return Pipeline([("vec", vec), ("clf", nb)])

def evaluate(y_true, y_pred) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f1}

@dataclass
class ExpConfig:
    data_path: str
    vectorizer: str
    model: str
    alpha: float = 1.0
    max_features: int = 5000
    ngram_max: int = 1
    test_size: float = 0.25
    seed: int = 42

def run_experiment(cfg: ExpConfig) -> Dict:
    X, y = load_csv(cfg.data_path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y)
    pipe = build_pipeline(cfg.vectorizer, cfg.model, cfg.max_features, cfg.ngram_max, cfg.alpha)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    metrics = evaluate(y_te, y_pred)
    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "report.txt"), "w") as f:
        f.write(classification_report(y_te, y_pred, digits=4))
    return {**asdict(cfg), **metrics}
