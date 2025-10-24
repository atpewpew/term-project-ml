"""Experiment orchestration script for the sentiment benchmark."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from . import config
from .data_loader import DatasetSplit, load_dataset
from .eval import compute_metrics, save_confusion_matrix, save_metrics_csv
from .features.vectorizers import BaseVectorizer, build_vectorizer
from .models.linear_svm import LinearSVMClassifier
from .models.logistic_regression import LogisticRegressionClassifier
from .models.multinomial_nb import MultinomialNaiveBayes
from .models.random_forest import RandomForestClassifier
from .utils import ensure_directory, save_pickle, set_global_seed

MODEL_FACTORY = {
    "nb": MultinomialNaiveBayes,
    "logreg": LogisticRegressionClassifier,
    "svm": LinearSVMClassifier,
    "rf": RandomForestClassifier,
}


@dataclass(slots=True)
class ExperimentResult:
    dataset: str
    vectorizer: str
    classifier: str
    metrics: dict
    confusion_path: Path
    model_path: Path


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["amazon", "imdb", "twitter"],
        help="Run a single dataset",
    )
    parser.add_argument("--vectorizer", choices=["bow", "tfidf", "tfidf_bigram"], help="Restrict to a single vectorizer")
    parser.add_argument("--classifier", choices=list(MODEL_FACTORY), help="Restrict to a single classifier")
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR, help="Base output directory")
    parser.add_argument("--fast", action="store_true", help="Run a small subset of combinations")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    set_global_seed(config.RANDOM_SEED)

    datasets = [args.dataset] if args.dataset else ["amazon", "imdb", "twitter"]
    vectorizers = [args.vectorizer] if args.vectorizer else ["bow", "tfidf", "tfidf_bigram"]
    classifiers = [args.classifier] if args.classifier else list(MODEL_FACTORY)

    combinations = list(itertools.product(datasets, vectorizers, classifiers))
    if args.fast:
        combinations = combinations[: max(1, config.FAST_MODE_COMBINATIONS)]

    summary_records: List[dict] = []
    results: List[ExperimentResult] = []

    total_runs = len(combinations)

    for idx, (dataset_name, vectorizer_name, classifier_name) in enumerate(
        combinations, start=1
    ):
        run_start = time.perf_counter()
        print(
            f"[{idx}/{total_runs}] Running {dataset_name} | {vectorizer_name} | {classifier_name}",
            flush=True,
        )
        dataset = load_dataset(dataset_name)
        vectorizer = build_vectorizer(vectorizer_name)
        model = MODEL_FACTORY[classifier_name]()
        experiment_result = run_single_experiment(
            dataset=dataset,
            vectorizer=vectorizer,
            model=model,
            output_dir=args.output,
        )
        results.append(experiment_result)
        run_duration = time.perf_counter() - run_start
        record = {
            "dataset": dataset_name,
            "vectorizer": vectorizer_name,
            "classifier": classifier_name,
        }
        record.update(experiment_result.metrics)
        summary_records.append(record)
        print(
            f"    Completed in {run_duration:5.1f}s | accuracy {experiment_result.metrics.get('accuracy', float('nan')):0.3f} | "
            f"f1 {experiment_result.metrics.get('f1', float('nan')):0.3f}",
            flush=True,
        )

    summary_df = pd.DataFrame(summary_records)
    ensure_directory(args.output)
    summary_path = Path(args.output) / "summary_results.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    return 0


def run_single_experiment(
    dataset: DatasetSplit,
    vectorizer: BaseVectorizer,
    model,
    output_dir: Path | str,
) -> ExperimentResult:
    X_train = vectorizer.fit_transform(dataset.train["clean_text"])
    X_test = vectorizer.transform(dataset.test["clean_text"])
    y_train = dataset.train["label"].to_numpy()
    y_test = dataset.test["label"].to_numpy()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = compute_metrics(y_test, predictions)

    result_base = f"{dataset.name}_{vectorizer.name}_{model.name}"
    csv_path = Path(output_dir) / "csv" / f"{result_base}.csv"
    save_metrics_csv(metrics, csv_path)

    confusion_path = Path(output_dir) / "csv" / f"{result_base}_confusion.csv"
    save_confusion_matrix(y_test, predictions, confusion_path)

    model_path = Path(output_dir) / "models" / f"{config.GROUP_NO}_{result_base}.pkl"
    payload = {"vectorizer": vectorizer, "model": model}
    save_pickle(payload, model_path)

    return ExperimentResult(
        dataset=dataset.name,
        vectorizer=vectorizer.name,
        classifier=model.name,
        metrics=metrics,
        confusion_path=confusion_path,
        model_path=model_path,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
