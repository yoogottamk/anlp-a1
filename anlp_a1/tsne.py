import multiprocessing
import pickle
from pathlib import Path

import numpy as np
from openTSNE import TSNE

from anlp_a1.config import DATA_ROOT


class COMTSNE:
    def __init__(
        self,
        feats_path: Path = DATA_ROOT / "feat.npy",
        w2i_path: Path = DATA_ROOT / "w2i.pkl",
    ):
        self.feats = np.load(feats_path)
        with open(w2i_path, "rb") as f:
            self.word2idx = pickle.load(f)
        self.idx2word = {i: w for (w, i) in self.word2idx.items()}

        self.feat_2d = None

    def compute(self, **tsne_kwargs):
        tsne = TSNE(**tsne_kwargs)
        self.feat_2d = tsne.fit(self.feats)

    def top_n_similar(self, word: str, n: int = 10):
        assert word in self.word2idx, "Word not in vocabulary"

        if self.feat_2d is None:
            self.compute()

        word_idx = self.word2idx[word]
        word_2d = self.feat_2d[word_idx]

        distance_2d = ((self.feat_2d - word_2d) ** 2).sum(axis=1)

        n_most_similar_idx = np.argpartition(distance_2d, n + 1)

        score_word = [
            # store score, word
            (distance_2d[i], self.idx2word[i])
            # for all top n words
            for i in n_most_similar_idx[: n + 1]
            # except the one that was queried
            if i != word_idx
        ]

        score_word.sort(key=lambda x: x[0])

        return score_word


if __name__ == "__main__":
    com_tsne = COMTSNE()
    com_tsne.compute(
        perplexity=50,
        n_jobs=multiprocessing.cpu_count(),
        verbose=True,
    )
