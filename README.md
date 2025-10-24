# Group 1 Sentiment Analysis Benchmark

"A Comprehensive Benchmark of Machine Learning Classifiers and Feature Engineering Techniques for Sentiment Analysis Across Multiple Domains" is a mini-term-project that evaluates four handwritten classifiers across three vectorization pipelines and three sentiment datasets (Amazon, IMDB, Twitter).

## Highlights
- Pure Python/Numpy implementations of Multinomial Naive Bayes, Logistic Regression, Linear SVM, and Random Forest (with a custom CART tree).
- Modular preprocessing with configurable stop-word removal and stemming.
- Pluggable Bag-of-Words, TF-IDF, and TF-IDF + bigram feature pipelines built on scikit-learn vectorizers.
- Automated experiment runner that saves metrics, confusion matrices, summaries, trained models, and plots.
- Submission helpers including PDF skeleton generator and packaging script to satisfy course requirements.

## Repository Layout
```
README.md
LICENSE
requirements.txt
src/
  config.py               # global constants, seeds, and defaults
  data_loader.py          # dataset loading, label mapping, splits
  preprocessing.py        # text cleaning helpers
  features/vectorizers.py # BoW, TF-IDF, TF-IDF + bigrams wrappers
  models/                 # custom classifier implementations
  eval.py                 # metrics, confusion matrix, plotting
  experiments.py          # experiment orchestration CLI
  run_experiment.sh       # convenience bash launcher
  utils.py                # utility functions
data/
  amazon/
  imdb/
  twitter/
```

## Setup
1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Enable stemming/stopword removal by toggling `APPLY_STEMMING` / `REMOVE_STOPWORDS` in `src/config.py`.

## Data Preparation
Place the raw CSVs inside the provided folders:
- `data/amazon/Amazon.csv` with columns `Review`, `Sentiment` (1-5). Labels are mapped to {0,1} as specified.
- `data/imdb/IMDB_Dataset.csv` containing `review`, `sentiment` (`positive`/`negative`).
- `data/twitter/Twitter_train.csv` (required) and `data/twitter/twitter_test.csv` (optional). Each requires `text` and `label` columns.

No additional preprocessing is needed; `src/data_loader.py` normalizes labels, performs stratified 80/20 splits when necessary, and applies text cleaning.

## Running Experiments
Quick start (runs every dataset × vectorizer × classifier combination):
```bash
bash src/run_experiment.sh
```

Selective runs:
```bash
python -m src.experiments --dataset imdb --vectorizer tfidf --classifier svm
python -m src.experiments --fast  # run a small smoke-test subset
```

Generated artifacts will be created in a `results` directory:
- Metrics CSV and confusion matrix per run in `results/csv/`.
- Serialized `(vectorizer, model)` bundle in `results/models/` with the `1_<dataset>_<vectorizer>_<classifier>.pkl` naming scheme.
- Summary sheet `results/summary_results.csv` aggregating all runs.
- Plotting utilities in `src/eval.py` (call `plot_metric_bar`) store figures under `results/plots/`.

## Configuration Tips
- Global defaults such as learning rates, epochs, random seeds, and feature limits live in `src/config.py`.
- Modify `FAST_MODE_COMBINATIONS` when iterating rapidly during development.
- Toggle preprocessing options (`REMOVE_STOPWORDS`, `APPLY_STEMMING`) to study their impact.

For further details, refer to inline docstrings and comments throughout the `src/` package.
