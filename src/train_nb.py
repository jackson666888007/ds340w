import argparse, time, json
from pathlib import Path
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import numpy as np

from src.data_utils import read_text_label_csv, ensure_split_indices, make_splits

def train_eval_nb(
    Xtr, Ytr, Xva, Yva,
    ngram: Tuple[int, int], max_features: int, alpha: float, stopwords: bool
):
    vec = TfidfVectorizer(ngram_range=ngram, max_features=max_features,
                          sublinear_tf=True, lowercase=True,
                          stop_words="english" if stopwords else None)
    Xtrv = vec.fit_transform(Xtr)
    Xvav = vec.transform(Xva)
    clf = MultinomialNB(alpha=alpha)
    t0 = time.time()
    clf.fit(Xtrv, Ytr)
    train_time = time.time() - t0
    y_pred = clf.predict(Xvav)
    y_proba = clf.predict_proba(Xvav)[:, 1]
    macro_f1 = f1_score(Yva, y_pred, average="macro")
    acc = accuracy_score(Yva, y_pred)

    import timeit
    sample = Xvav[:256]
    t_inf = timeit.timeit(lambda: clf.predict(sample), number=20) / (20 * sample.shape[0])

    return {
        "vec": vec, "clf": clf, "macro_f1": macro_f1, "accuracy": acc,
        "train_time_sec": train_time, "infer_latency_sec_per_item": t_inf,
        "y_pred": y_pred, "y_proba": y_proba
    }

def main(args):
    texts, labels = read_text_label_csv(args.data_path)
    split = ensure_split_indices(len(texts), args.split_path, seed=args.seed)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = make_splits(texts, labels, split)

    grid = []
    for ngram in [(1,1), (1,2)]:
        for maxf in [20000, 50000, 100000]:
            for alpha in [0.1, 0.5, 1.0]:
                for sw in [True, False]:
                    grid.append((ngram, maxf, alpha, sw))

    best, best_cfg = None, None
    for (ngram, maxf, alpha, sw) in grid:
        r = train_eval_nb(Xtr, Ytr, Xva, Yva, ngram, maxf, alpha, sw)
        if (best is None) or (r["macro_f1"] > best["macro_f1"]):
            best, best_cfg = r, (ngram, maxf, alpha, sw)

    Xall = Xtr + Xva; Yall = Ytr + Yva
    vec = TfidfVectorizer(ngram_range=best_cfg[0], max_features=best_cfg[1],
                          sublinear_tf=True, lowercase=True,
                          stop_words="english" if best_cfg[3] else None)
    Xallv = vec.fit_transform(Xall); Xtev = vec.transform(Xte)
    clf = MultinomialNB(alpha=best_cfg[2])
    t0 = time.time(); clf.fit(Xallv, Yall); train_time = time.time() - t0

    y_pred = clf.predict(Xtev); y_proba = clf.predict_proba(Xtev)[:,1]
    macro_f1 = f1_score(Yte, y_pred, average="macro")
    acc = accuracy_score(Yte, y_pred)
    cm = confusion_matrix(Yte, y_pred).tolist()

    import timeit
    sample = Xtev[:256]
    infer_latency = timeit.timeit(lambda: clf.predict(sample), number=20) / (20 * sample.shape[0])

    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)
    import csv
    with open(Path(args.reports_dir)/"test_preds_nb.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["p_pos","y_true"])
        for p, y in zip(y_proba, Yte): w.writerow([p, y])

    feature_names = np.array(vec.get_feature_names_out())
    top_lines = []
    for cls, cls_name in enumerate(["neg","pos"]):
        top_idx = np.argsort(clf.feature_log_prob_[cls])[-20:][::-1]
        top_lines.append(f"[{cls_name}] " + ", ".join(feature_names[top_idx]))
    (Path(args.reports_dir)/"nb_top_tokens.txt").write_text("\n".join(top_lines), encoding="utf-8")

    metrics = {
        "macro_f1": macro_f1, "accuracy": acc,
        "train_time_sec": train_time,
        "infer_latency_sec_per_item": infer_latency,
        "best_cfg": {"ngram": best_cfg[0], "max_features": best_cfg[1], "alpha": best_cfg[2], "stopwords": best_cfg[3]},
        "confusion_matrix": cm
    }
    (Path(args.reports_dir)/"nb_metrics.json").write_text(json.dumps(metrics, indent=2))
    md = Path(args.out_md)
    md.write_text(
        f"# NB + TF-IDF (test)\n"
        f"- Macro-F1: **{macro_f1:.4f}**  \n- Accuracy: **{acc:.4f}**  \n"
        f"- Train time: {train_time:.2f}s  \n- Inference latency: {infer_latency*1000:.3f} ms/review  \n"
        f"- Best cfg: ngram={best_cfg[0]}, max_features={best_cfg[1]}, alpha={best_cfg[2]}, stopwords={best_cfg[3]}  \n"
        f"- Confusion matrix: {cm}\n",
        encoding="utf-8"
    )
    print("Saved:", md)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/IMDB_50k.csv")
    ap.add_argument("--split_path", default="data/split_indices.json")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--out_md", default="reports/baseline_report.md")
    ap.add_argument("--seed", type=int, default=340)
    main(ap.parse_args())
