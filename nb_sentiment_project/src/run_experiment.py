import argparse
from nb_platform import ExpConfig, run_experiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="mnb", choices=["bnb","mnb"])
    ap.add_argument("--vectorizer", default="tfidf", choices=["count","tfidf"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max_features", type=int, default=5000)
    ap.add_argument("--ngram_max", type=int, default=1)
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res = run_experiment(ExpConfig(
        data_path=args.data, vectorizer=args.vectorizer, model=args.model,
        alpha=args.alpha, max_features=args.max_features, ngram_max=args.ngram_max,
        test_size=args.test_size, seed=args.seed
    ))
    print(res)

if __name__ == "__main__":
    main()
