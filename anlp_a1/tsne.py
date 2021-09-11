import multiprocessing
import pickle
from abc import ABC
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from openTSNE import TSNE

from anlp_a1.config import DATA_ROOT


class BaseTSNE(ABC):
    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.feat_2d: Union[None, np.ndarray] = None
        self.feats: np.ndarray

    def compute(self, **tsne_kwargs):
        tsne = TSNE(**tsne_kwargs)
        self.feat_2d = tsne.fit(self.feats)

    def top_n_similar(self, word: str, n: int = 10):
        assert word in self.word2idx, "Word not in vocabulary"

        word_feat = self.feats[self.word2idx[word]]
        cosine_sim = (
            (self.feats * word_feat)
            / (np.c_[np.linalg.norm(self.feats, axis=1)] * np.linalg.norm(word_feat))
        ).sum(1)

        # add extra to avoid the padding
        n_most_similar = np.argpartition(cosine_sim, -(n + 2))
        sim_word = [
            (cosine_sim[i], self.idx2word[i])
            for i in n_most_similar[-(n + 2) :]
            if i != self.word2idx[word]
        ]

        sim_word.sort(key=lambda x: x[0], reverse=True)

        return sim_word[:-1]

    def plot_tsne_neighbours(self, word: str, n: int = 10):
        nearest_neighbors = self.top_n_similar(word, n)
        neighbor_idx = [self.word2idx[w] for (_, w) in nearest_neighbors]
        word_idx = self.word2idx[word]

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 8))
        N = self.feat_2d.shape[0]
        random_idx = np.random.choice(N, size=int(N * 0.1), replace=False)
        ax0.scatter(self.feat_2d[random_idx, 0], self.feat_2d[random_idx, 1], s=8)

        for neigh in neighbor_idx:
            x, y = self.feat_2d[neigh]
            ax0.scatter(x, y, color="orange")
            ax1.scatter(x, y, color=np.random.rand(3,), label=self.idx2word[neigh])
            ax1.text(x, y, self.idx2word[neigh])

        x, y = self.feat_2d[word_idx]
        ax0.scatter(x, y, color="red")
        ax1.scatter(x, y, color="red", label=word)
        ax1.text(x, y, word)

        ax0.title.set_text(word)
        ax1.legend(loc="lower right", ncol=4, bbox_to_anchor=(1, 0), bbox_transform=fig.transFigure)

        fig.savefig(f"{self.__class__.__name__}-{word}.png")
        plt.close(fig)


class COMTSNE(BaseTSNE):
    def __init__(
        self,
        feats_path: Path = DATA_ROOT / "com-feat.npy",
        w2i_path: Path = DATA_ROOT / "w2i.pkl",
    ):
        self.feats = np.load(feats_path)
        with open(w2i_path, "rb") as f:
            self.word2idx = pickle.load(f)
        self.idx2word = {i: w for (w, i) in self.word2idx.items()}

        self.feat_2d = None


class CBOWTSNE(BaseTSNE):
    def __init__(
        self,
        model_path: Path = DATA_ROOT / "cbow.ckpt",
        wf_path: Path = DATA_ROOT / "wf.pkl",
    ):
        self.feats = torch.load(model_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]["em.weight"].numpy()
        with open(wf_path, "rb") as f:
            self.wf = pickle.load(f)

        self.word2idx = {w: i for (i, w) in enumerate(self.wf)}
        self.idx2word = {i: w for (w, i) in self.word2idx.items()}

        self.feat_2d = None


if __name__ == "__main__":
    word_list = ["mobile", "bad", "good", "working", "running"]

    com_tsne = COMTSNE()
    com_tsne.compute(perplexity=50, n_jobs=multiprocessing.cpu_count(), verbose=True)

    cbow_tsne = CBOWTSNE()
    cbow_tsne.compute(perplexity=50, n_jobs=multiprocessing.cpu_count(), verbose=True)

    for word in word_list:
        com_tsne.plot_tsne_neighbours(word)
        cbow_tsne.plot_tsne_neighbours(word)
