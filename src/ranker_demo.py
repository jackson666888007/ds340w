import argparse, csv, math
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data_utils import read_text_label_csv, ensure_split_indices, make_splits

def load_probs(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r: rows.append(float(row["p_pos"]))
    return rows

def main(args):
    texts, labels = read_text_label_csv(args.data_path)
    split = ensure_split_indices(len(texts), args.split_path, seed=340)
    (_, _), (_, _), (Xte, Yte) = make_splits(texts, labels, split)

    preds_file = Path(args.reports_dir)/("test_preds_bert.csv" if Path(args.reports_dir/"test_preds_bert.csv").exists() else "test_preds_nb.csv")
    p_pos = load_probs(preds_file)

    vec = TfidfVectorizer(max_features=50000, sublinear_tf=True).fit(Xte)
    Xv = vec.transform(Xte)

    anchor = 0                    
    sims = cosine_similarity(Xv[anchor], Xv).ravel()

    def score(sim, p): return sim * (2*p - 1)  

    pairs_plain = sorted([(i, sims[i]) for i in range(len(Xte)) if i!=anchor], key=lambda x: x[1], reverse=True)[:10]
    pairs_weight = sorted([(i, score(sims[i], p_pos[i])) for i in range(len(Xte)) if i!=anchor], key=lambda x: x[1], reverse=True)[:10]

    out = Path(args.reports_dir)/"ranking_demo.md"
    with out.open("w", encoding="utf-8") as f:
        f.write("# Sentiment-weighted ranking demo\n")
        f.write(f"- Anchor review index: {anchor}\n\n")
        f.write("## Plain TF-IDF cosine top-10\n")
        for i, s in pairs_plain:
            f.write(f"- idx={i}, sim={s:.4f}, p_pos={p_pos[i]:.3f}\n")
        f.write("\n## Weighted (similarity × (2·p_pos−1)) top-10\n")
        for i, s in pairs_weight:
            f.write(f"- idx={i}, score={s:.4f}, sim≈{sims[i]:.4f}, p_pos={p_pos[i]:.3f}\n")
    print("Saved", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/IMDB_50k.csv")
    ap.add_argument("--split_path", default="data/split_indices.json")
    ap.add_argument("--reports_dir", default="reports")
    main(ap.parse_args())
