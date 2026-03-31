from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FeatureExtractor:
    """Extract simple lexical and syntactic features from sentence pairs."""

    def __init__(self) -> None:
        self.sentence_pairs: List[Tuple[str, str]] = []
        self.labels: List[int] = []
        self.tfidf_vectorizer = TfidfVectorizer()
        self._stop_words = set(stopwords.words("english"))

    def load_data(
        self,
        data: Iterable[Tuple[Optional[bool], str, str, str]],
        extract_labels: bool = True,
    ) -> None:
        """Load sentence pairs and optional labels from parsed data."""
        for judge, orig_sent, cand_sent, _trend_id in data:
            self.sentence_pairs.append((orig_sent, cand_sent))
            if extract_labels and judge is not None:
                self.labels.append(int(judge))

    def _preprocess(self, text: str) -> str:
        tokens = word_tokenize(text.lower())
        filtered = [word for word in tokens if word.isalnum() and word not in self._stop_words]
        return " ".join(filtered)

    @staticmethod
    def _jaccard_similarity(left: str, right: str) -> Tuple[float, int]:
        left_set = set(left.split())
        right_set = set(right.split())
        intersection = left_set.intersection(right_set)
        union = left_set.union(right_set)
        score = len(intersection) / len(union) if union else 0
        return score, len(intersection)

    @staticmethod
    def _pos_similarity(left: str, right: str) -> float:
        pos_left = pos_tag(left.split())
        pos_right = pos_tag(right.split())
        pos_common = set(pos_left).intersection(set(pos_right))
        pos_union = set(pos_left).union(set(pos_right))
        return len(pos_common) / len(pos_union) if pos_union else 0

    def extract_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) where X is feature matrix and y are labels if present."""
        if not self.sentence_pairs:
            return np.array([]), np.array(self.labels)

        all_sentences = [self._preprocess(sent) for pair in self.sentence_pairs for sent in pair]
        self.tfidf_vectorizer.fit(all_sentences)

        cosine_similarities: List[float] = []
        jaccard_similarities: List[float] = []
        length_diffs: List[int] = []
        common_words: List[int] = []
        pos_similarities: List[float] = []

        for orig_sent, cand_sent in self.sentence_pairs:
            orig_processed = self._preprocess(orig_sent)
            cand_processed = self._preprocess(cand_sent)

            tfidf_orig = self.tfidf_vectorizer.transform([orig_processed])
            tfidf_cand = self.tfidf_vectorizer.transform([cand_processed])
            cosine_sim = cosine_similarity(tfidf_orig, tfidf_cand)[0][0]
            cosine_similarities.append(cosine_sim)

            jaccard_sim, common_count = self._jaccard_similarity(orig_processed, cand_processed)
            jaccard_similarities.append(jaccard_sim)
            common_words.append(common_count)

            length_diffs.append(abs(len(orig_sent) - len(cand_sent)))
            pos_similarities.append(self._pos_similarity(orig_processed, cand_processed))

        X = np.array(
            list(
                zip(
                    cosine_similarities,
                    jaccard_similarities,
                    length_diffs,
                    common_words,
                    pos_similarities,
                )
            )
        )
        y = np.array(self.labels)

        return X, y