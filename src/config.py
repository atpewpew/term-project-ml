"""Global configuration for the sentiment benchmark project."""

from pathlib import Path

GROUP_NO: int = 1
RANDOM_SEED: int = 42

# Base paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_CSV_DIR = RESULTS_DIR / "csv"
RESULTS_MODELS_DIR = RESULTS_DIR / "models"
RESULTS_PLOTS_DIR = RESULTS_DIR / "plots"
SUBMISSION_PDFS_DIR = ROOT_DIR / "submission_pdfs"

# Dataset specific locations
AMAZON_DATA = DATA_DIR / "amazon" / "Amazon.csv"
IMDB_DATA = DATA_DIR / "imdb" / "IMDB_Dataset.csv"
TWITTER_TRAIN_DATA = DATA_DIR / "twitter" / "Twitter_train.csv"
TWITTER_TEST_DATA = DATA_DIR / "twitter" / "twitter_test.csv"

# Preprocessing options
REMOVE_STOPWORDS: bool = True
APPLY_STEMMING: bool = True

# Feature engineering defaults
VOCABULARY_SIZE: int | None = None  # Deprecated, use specific limits below
MAX_FEATURES_BOW_TFIDF: int | None = 2000
MAX_FEATURES_TFIDF_BIGRAMS: int | None = 2000

# Training settings
LOGISTIC_REG_ALPHA: float = 1e-4
LOGISTIC_REG_LR: float = 0.1
LOGISTIC_REG_EPOCHS: int = 50
LOGISTIC_REG_BATCH_SIZE: int = 128

SVM_LAMBDA: float = 1e-4
SVM_EPOCHS: int = 50
SVM_BATCH_SIZE: int = 128

RANDOM_FOREST_N_ESTIMATORS: int = 25
RANDOM_FOREST_MAX_DEPTH: int | None = 15
RANDOM_FOREST_MIN_SAMPLES_SPLIT: int = 5
RANDOM_FOREST_MAX_FEATURES_RATIO: float | None = None

# Experiment control
FAST_MODE_COMBINATIONS: int = 2  # number of combinations per dataset when using --fast

__all__ = [
    "GROUP_NO",
    "RANDOM_SEED",
    "ROOT_DIR",
    "DATA_DIR",
    "RESULTS_DIR",
    "RESULTS_CSV_DIR",
    "RESULTS_MODELS_DIR",
    "RESULTS_PLOTS_DIR",
    "SUBMISSION_PDFS_DIR",
    "AMAZON_DATA",
    "IMDB_DATA",
    "TWITTER_TRAIN_DATA",
    "TWITTER_TEST_DATA",
    "REMOVE_STOPWORDS",
    "APPLY_STEMMING",
    "VOCABULARY_SIZE",
    "MAX_FEATURES_BOW_TFIDF",
    "MAX_FEATURES_TFIDF_BIGRAMS",
    "LOGISTIC_REG_ALPHA",
    "LOGISTIC_REG_LR",
    "LOGISTIC_REG_EPOCHS",
    "LOGISTIC_REG_BATCH_SIZE",
    "SVM_LAMBDA",
    "SVM_EPOCHS",
    "SVM_BATCH_SIZE",
    "RANDOM_FOREST_N_ESTIMATORS",
    "RANDOM_FOREST_MAX_DEPTH",
    "RANDOM_FOREST_MIN_SAMPLES_SPLIT",
    "RANDOM_FOREST_MAX_FEATURES_RATIO",
    "FAST_MODE_COMBINATIONS",
]
