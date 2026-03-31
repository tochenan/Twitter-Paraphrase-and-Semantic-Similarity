# Protein Developability Prediction

This repository contains baseline and feature-based models for the SemEval-2015 Task 1 paraphrase/similarity task, along with evaluation scripts and saved outputs.

## SemEval-2015 Task 1 (overview)

The task is Paraphrase and Semantic Similarity in Twitter. Each example is a tweet pair with a label from crowd votes (e.g., "(2,3)") or an expert score (e.g., "2"). Systems are evaluated on:

- Binary paraphrase detection (true/false).
- Graded similarity via a confidence score in $[0,1]$ for ranking and max-F1/Pearson metrics.

## Structure

```
data/               # train/test data and labels
model/              # saved model artifacts
src/                # source code (modularized scripts)
systemoutputs/      # prediction outputs in SemEval format
```

Key modules in [src](src):

- [feature_engineering_svm.py](src/feature_engineering_svm.py): Linear SVM with engineered features.
- [feature_extractor.py](src/feature_extractor.py): Feature extraction for sentence pairs.
- [baseline_logisticregression.py](src/baseline_logisticregression.py): MaxEnt (logistic regression) baseline.
- [baseline_random.py](src/baseline_random.py): Random baseline.
- [pit2015_eval_single.py](src/pit2015_eval_single.py): Evaluation script.
- [pit2015_checkformat.py](src/pit2015_checkformat.py): Output format checker.
- [BERT_Finetune.ipynb](src/BERT_Finetune.ipynb): BERT fine-tuning notebook (Colab).

## Setup

Python dependencies include `nltk`, `scikit-learn`, `numpy`, and `joblib`.

NLP resources required by NLTK:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

For the MaxEnt baseline, set `MEGAM_DIR` to your Megam binary path if you plan to run it.

## Usage

Run modules from the repo root using `-m` so relative imports resolve correctly.

### Random baseline

```bash
python -m src.baseline_random
```

### Logistic regression baseline

```bash
python -m src.baseline_logisticregression
```

### Feature-based SVM

```bash
python -m src.feature_engineering_svm
```

### Evaluate outputs

```bash
python -m src.pit2015_eval_single
```

## Model notes

### BERT fine-tuning

Notebook: [src/BERT_Finetune.ipynb](src/BERT_Finetune.ipynb)

Model artifacts are hosted externally:
https://drive.google.com/drive/folders/1W9iJPkYgovqZT76XJtRm5Z6QYgm5EqOQ?usp=sharing

### Feature-based SVM

Features used:

1. TF-IDF cosine similarity
2. Jaccard similarity
3. Sentence length difference
4. Count of common words
5. POS tag similarity

Saved model: [model/best_svc_model_C_0.01.pkl](model/best_svc_model_C_0.01.pkl)

## Evaluation snapshot

Model | Code | F | P | R | P Correlation | MaxF1 | P MaxF1 | R MaxF1
---|---|---|---|---|---|---|---|---
BERT | 03BCL | 0.602 | 0.763 | 0.497 | 0.575 | 0.688 | 0.702 | 0.674
SVC | 04 | 0.646 | 0.810 | 0.537 | 0.569 | 0.653 | 0.807 | 0.549

Both models show comparable performance; the SVM slightly improves F1, while BERT can achieve higher precision with an optimal threshold.