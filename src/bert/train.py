from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import optuna
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from ..data_io import read_pair_data
from ..paths import PROJECT_ROOT, TEST_DATA_PATH, TRAIN_DATA_PATH
from .data import ParaphraseDataset, filter_labeled

logger = logging.getLogger(__name__)


def compute_metrics(prediction: EvalPrediction) -> Dict[str, float]:
    preds = np.argmax(prediction.predictions, axis=1)
    return {"eval_accuracy": accuracy_score(prediction.label_ids, preds)}


def objective(
    trial: optuna.Trial,
    train_dataset: ParaphraseDataset,
    val_dataset: ParaphraseDataset,
    output_root: Path,
    model_name: str,
) -> float:
    learning_rate = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_train_epochs = trial.suggest_int("epochs", 2, 7)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(output_root / f"results/trial_{trial.number}"),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(output_root / f"logs/trial_{trial.number}"),
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    evaluation_result = trainer.evaluate()

    return float(evaluation_result["eval_accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT with Optuna.")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    train_data, _ = read_pair_data(TRAIN_DATA_PATH)
    labeled_train = filter_labeled(train_data)
    logger.info("Read in %s labeled training samples", len(labeled_train))

    train_split, val_split = train_test_split(
        labeled_train,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    train_dataset = ParaphraseDataset(train_split, tokenizer, max_len=args.max_len)
    val_dataset = ParaphraseDataset(val_split, tokenizer, max_len=args.max_len)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(
            trial,
            train_dataset,
            val_dataset,
            PROJECT_ROOT,
            args.model_name,
        ),
        n_trials=args.n_trials,
    )

    best_trial = study.best_trial
    logger.info("Best trial final validation accuracy: %.4f", best_trial.value)
    logger.info("Best trial hyperparameters: %s", best_trial.params)


if __name__ == "__main__":
    main()
