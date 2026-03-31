from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from ..data_io import read_pair_data
from ..paths import SYSTEMOUTPUTS_DIR, TEST_DATA_PATH

logger = logging.getLogger(__name__)


def output_predictions(
    model_dir: Path,
    test_data: Iterable[Tuple[Optional[bool], str, str, str]],
    outfile: Path,
    tokenizer_name: str,
    max_len: int,
) -> None:
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = BertForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    with outfile.open("w") as outf, torch.no_grad():
        for _label, text1, text2, _trend_id in test_data:
            encoded = tokenizer.encode_plus(
                text1,
                text2,
                add_special_tokens=True,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            outputs = model(
                encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            logits = outputs[0]
            probabilities = torch.softmax(logits, dim=1)
            prob = probabilities[:, 1].item()

            if prob >= 0.5:
                outf.write("true\t" + "{0:.4f}".format(prob) + "\n")
            else:
                outf.write("false\t" + "{0:.4f}".format(prob) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BERT predictions on test data.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer-name", default="bert-base-uncased")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument(
        "--output",
        default=str(SYSTEMOUTPUTS_DIR / "PIT2015_BERT_03_BCL.output"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    test_data, _ = read_pair_data(TEST_DATA_PATH)
    output_path = Path(args.output)
    output_predictions(
        Path(args.model_dir),
        test_data,
        output_path,
        args.tokenizer_name,
        args.max_len,
    )
    logger.info("Wrote predictions to %s", output_path)


if __name__ == "__main__":
    main()
