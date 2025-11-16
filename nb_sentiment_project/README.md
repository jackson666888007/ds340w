# NB Sentiment Baseline

This project reproduces the results from **Danyal et al. (2024)**  
*Sentiment Analysis of Movie Reviews based on Naive Bayes using TF–IDF and Count Vectorizer.*

---

## Overview
This work replicates the methodology and findings of the parent paper by applying **Naive Bayes classifiers** with both **TF–IDF** and **Count Vectorizer** features on the IMDb movie review dataset.  
The code is implemented in a modular way so that each part — including feature extraction, model selection, and evaluation — can be easily modified for further testing or extensions.

---

## How to Run

```bash
pip install -r requirements.txt


python src/run_experiment.py --data data/imdb_full.csv --model mnb --vectorizer tfidf --ngram_max 1 --alpha 1.0 --max_features 50000
python src/run_experiment.py --data data/imdb_full.csv --model bnb --vectorizer count --ngram_max 1 --alpha 1.0 --max_features 50000



| Setting (Model + Vectorizer)     |  Accuracy | Precision | Recall | F1 Score |
| -------------------------------- | :-------: | :-------: | :----: | :------: |
| MultinomialNB + TF-IDF (unigram) |   0.860   |   0.866   |  0.852 |   0.859  |
| BernoulliNB + Count (unigram)    |   0.843   |   0.873   |  0.800 |   0.835  |



- The reproduced results are consistent with the parent paper’s baseline.  
- The modular structure allows easy replacement of vectorizers, models, or hyperparameters.  
- The dataset used (`data/imdb_full.csv`) contains 20,000 samples derived from IMDb for faster replication.

---

Author: Chengshun Zhao and Beiwei Niu(David-719)
Date: November 2025

