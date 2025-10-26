DS340W Project Guide — From Reviews to Recommendations

Use the big picture to condense lengthy movie reviews into a trustworthy sentiment signal, investigate the quality–speed trade-off (accuracy vs. latency/cost), and exploit that signal in a small sentiment-weighted ranking demo. All the code, data notes, and reports are reproducible and mapped to the course's midterm/final milestones.

1.Course Overview and Project Design 

Why this seminar? Platforms aggregate millions of reviews and users (and other systems) have limited time. Sentiment analysis can turn textual reviews into a single score that can be acted upon quickly by most people or other algorithms. 

Open stable domain. There are abundant and open movie reviews/public datasets that allow the project to be easy to reproduce and trace. 

Main comparison in this project. Traditional NB + TF-IDF (the traditional, transparent, fast) vs (predicted) BERT/DistilBERT has a stronger accuracy performance (but also more cost). We will report Macro-F1 (main) metrics, Accuracy, training time, and per-review inference latency.

Extension. Two lightweight extensions, (a) with VADER as a hybrid (should be particularly good as a stabilizing feature, or block, of a (related) vote, and (b) a sentiment-weighted content ranker to re-order very similar items with predicted polarity.



2.How the Accomplishments Map to DS340W Milestones


Paper triad (e.g. done above). We connected three readings to motivate:

a. To establish a solid, tunable baseline we will use classical pipelines comprised of (tokenization → TF-IDF/Count → NB).

b. We will leverage transformer advances (BERT) to obtain contextual embeddings and capture long-distance dependencies.

c. We will assess ways to incorporate sentiment signals into the recommenders to down-rank items that are consistently negative.


3.Source, and Splits

Dataset: IMDB Large Movie Review (50k). We keep the full CSV locally (size >100MB) and publish:data/IMDB_50k_sample.csv (500 rows for structure), data/*.sha256 (checksums for the original tarball and the full/sample CSV), Fixed split indices data/split_indices.json with seed=340 for 80/10/10 (train/valid/test).


4.Methods & Engineering Choices
Baseline (fast): Multinomial Naive Bayes + TF-IDF: Sweep n-grams (uni/bi-) and max_features, toggle stopwords, tune alpha, Transparent and easy to benchmark for latency and memory use. 

Deep (stronger): BERT/DistilBERT for contextual semantics:HuggingFace Trainer with early stopping; logs Macro-F1/Accuracy/time/latency, If compute is tight, default to DistilBERT and max_len=128. 

Hybrid stabilizer: VADER as an auxiliary probability for vote or meta-feature. 

Primary metrics: F1 (handles class imbalance), Accuracy, training time, per-item inference latency. 

Ablations: stopwords on/off, uni vs. bi-grams, different max_features, NB vs. VADER-only vs. tuned NB, plus BERT grid (lr, max_len, epochs). 

Error analysis: inspect negation, sarcasm, mixed polarity; publish confusion matrices and misclass examples.


5.Risks & Mitigations (what could go wrong)
Imbalance / domain shift: stratified split and Macro-F1; label distributions check.

Compute limits: DistilBERT; smaller max_len; gradient accumulation if necessary.

Noisy text: light cleaning (lowercasing, de-dup); clear label mapping {neg, pos}.


Overfitting: early stopping for BERT; cross-validation + ablations for NB.
