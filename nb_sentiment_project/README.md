## Quickstart

### 1. Setup

```bash
git clone https://github.com/jackson666888007/ds340w.git
cd ds340w/nb_sentiment_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd nb_sentiment_project
python src/run_experiment.py \
  --data data/imdb_full.csv \
  --model mnb \
  --vectorizer tfidf \
  --ngram_max 2 \
  --alpha 1.0 \
  --max_features 50000 \
  --test_size 0.2 \
  --seed 42

python src/run_experiment.py \
  --data data/imdb_full.csv \
  --model bnb \
  --vectorizer count \
  --ngram_max 2 \
  --alpha 1.0 \
  --max_features 50000 \
  --test_size 0.2 \
  --seed 42

python src/run_experiment.py \
  --data data/imdb_full.csv \
  --model bnb \
  --vectorizer tfidf \
  --ngram_max 2 \
  --alpha 1.0 \
  --max_features 50000 \
  --test_size 0.2 \
  --seed 42

python src/run_experiment.py \
  --data data/imdb_full.csv \
  --model mnb \
  --vectorizer count \
  --ngram_max 2 \
  --alpha 1.0 \
  --max_features 50000 \
  --test_size 0.2 \
  --seed 42

python src/run_experiment.py \
  --data data/imdb_full.csv \
  --model mnb \
  --vectorizer tfidf \
  --ngram_max 2 \
  --alpha 1.0 \
  --max_features 50000 \
  --test_size 0.2 \
  --seed 42

python src/run_experiment.py \
  --data data/imdb_full.csv \
  --model mnb \
  --vectorizer count \
  --ngram_max 1 \
  --alpha 1.0 \
  --max_features 50000 \
  --test_size 0.2 \
  --seed 42

python src/run_experiment.py \
  --data data/imdb_full.csv \
  --model mnb \
  --vectorizer tfidf \
  --ngram_max 1 \
  --alpha 1.0 \
  --max_features 50000 \
  --test_size 0.2 \
  --seed 42

for a in 0.1 0.5 1.0 2.0; do
  python src/run_experiment.py \
    --data data/imdb_full.csv \
    --model mnb \
    --vectorizer tfidf \
    --ngram_max 2 \
    --alpha $a \
    --max_features 50000 \
    --test_size 0.2 \
    --seed 42
done

for k in 10000 20000 30000; do
  python src/run_experiment.py \
    --data data/imdb_full.csv \
    --model mnb \
    --vectorizer tfidf \
    --ngram_max 2 \
    --alpha 0.1 \
    --max_features $k \
    --test_size 0.2 \
    --seed 42
done

python src/run_logreg_experiment.py \
  --data data/imdb_full.csv \
  --vectorizer tfidf \
  --ngram_max 2 \
  --max_features 50000 \
  --C 2.0 \
  --test_size 0.2 \
  --seed 42

