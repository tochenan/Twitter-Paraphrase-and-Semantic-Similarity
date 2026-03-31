from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch


class ParaphraseDataset(torch.utils.data.Dataset):
    """Dataset for sentence-pair paraphrase classification."""

    def __init__(
        self,
        data: Iterable[Tuple[Optional[bool], str, str, str]],
        tokenizer,
        max_len: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sentence_pairs: List[Tuple[str, str]] = []
        self.labels: List[int] = []

        for judge, orig_sent, cand_sent, _trend_id in data:
            self.sentence_pairs.append((orig_sent, cand_sent))
            if judge is not None:
                self.labels.append(int(judge))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, item: int) -> dict:
        orig_sent, cand_sent = self.sentence_pairs[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            orig_sent,
            cand_sent,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def filter_labeled(
    data: Iterable[Tuple[Optional[bool], str, str, str]]
) -> List[Tuple[Optional[bool], str, str, str]]:
    """Keep only items with a non-None label."""
    return [item for item in data if item[0] is not None]
