
import csv, argparse, json
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

def read_preds_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r: rows.append((float(row["p_pos"]), int(row["y_true"])))
    p, y = zip(*rows); return list(p), list(y)

def main(args):
    reports = Path(args.reports_dir)
    nb_p, y_true = read_preds_csv(reports/"test_preds_nb.csv")
    bert_p, _ = (None, None)
    bert_file = reports/"test_preds_bert.csv"
    if bert_file.exists():
        bert_p, _ = read_preds_csv(bert_file)

    from src.data_utils import read_text_label_csv, ensure_split_indices, make_splits
    texts, labels = read_text_label_csv(args.data_path)
    split = json.loads(Path(args.split_path).read_text())
    (_, _), (_, _), (Xte, Yte) = make_splits(texts, labels, split)
    analyzer = SentimentIntensityAnalyzer()
    v_p = [(analyzer.polarity_scores(t)["compound"] + 1)/2 for t in Xte]  

    out = {}

    def vote(p1, p2, pv):
        votes = (int(p1>=0.5), int(p2>=0.5) if p2 is not None else None, int(pv>=0.5))
        if votes[1] is None:
            s = votes[0] + votes[2]
            return 1 if s>=1 else 0
        s = votes[0] + votes[1] + votes[2]
        if s == 1:  
            return 1 if votes[1]==1 else 0
        if s == 2:
            return votes[1]
        return 1 if s>=2 else 0

    y_hat_vote = [vote(nb_p[i], (bert_p[i] if bert_p else None), v_p[i]) for i in range(len(Yte))]
    out["vote_macro_f1"] = f1_score(Yte, y_hat_vote, average="macro")
    out["vote_acc"]      = accuracy_score(Yte, y_hat_vote)

    feats, ys = [], []
    for i in range(len(Yte)):
        row = [nb_p[i], v_p[i]]
        if bert_p: row.append(bert_p[i])
        feats.append(row); ys.append(Yte[i])
    clf = LogisticRegression(max_iter=1000).fit(feats, ys)
    y_hat_meta = clf.predict(feats)
    out["meta_macro_f1"] = f1_score(ys, y_hat_meta, average="macro")
    out["meta_acc"]      = accuracy_score(ys, y_hat_meta)

    Path(args.reports_dir).mkdir(exist_ok=True, parents=True)
    (Path(args.reports_dir)/"vader_hybrid.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/IMDB_50k.csv")
    ap.add_argument("--split_path", default="data/split_indices.json")
    ap.add_argument("--reports_dir", default="reports")
    main(ap.parse_args())
