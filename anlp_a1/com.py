from __future__ import annotations

import code
import multiprocessing
import pickle
import random
from collections import Counter
from math import sqrt
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from tqdm.auto import tqdm

from anlp_a1.config import DATA_ROOT
from anlp_a1.dataset import Dataset
from anlp_a1.stats import generate_wf
from anlp_a1.utils import get_word_windows


def _subsample_probability(wf: dict, word: str, t: float = 1e-5) -> float:
    """
    Subsampling

    Calculates the probability to keep a word
    https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf

    Args:
        wf: word-frequency mapping
        word: word in vocab
        t: subsampling threshold
    """
    return sqrt(t / wf[word])


def _com_calculator(
    window_size: int, wf: dict, word2idx: dict, ds_start_idx: int, ds_end_idx: int
) -> Counter:
    """
    COM Calculator

    Calculates Co-Occurance Matrix for a subset of the `Dataset`

    Args:
        window_size: how many words to check on both sides for counting co-occurance
        wf: word-frequency mapping
        word2idx: word to index (integer) mapping
        ds_start_idx: starting index for subset of `Dataset`, inclusive
        ds_end_idx: ending index for subset of `Dataset`, exclusive
    """
    local_ds = Dataset()
    local_com = Counter()

    idx_iter = range(ds_start_idx, ds_end_idx)
    if ds_start_idx == 0:
        idx_iter = tqdm(idx_iter, desc="Training COMVectorizer [w0]")

    for i in idx_iter:
        item = local_ds[i]
        word_list = [w for w in item["review"].split() if w in wf]

        for surrounding_words, word in get_word_windows(word_list, window_size):
            subsampled_words = [
                w
                for w in surrounding_words
                # if _subsample_probability(wf, w) < random.uniform(0, 1)
            ]

            for com_word in subsampled_words:
                i = word2idx[word]
                j = word2idx[com_word]

                local_com[(i, j)] += 1
                local_com[(j, i)] += 1

    return local_com


class COMVectorizer:
    """
    Co-Occurance Matrix Vectorizer

    Generates a Co-Occurance Matrix and performs SVD to reduce dimension of vectors
    """

    def __init__(
        self, window_size: int = 3, vector_size: int = 64, skip_setup: bool = False
    ):
        """
        Constructor for COMVectorizer

        Args:
            window_size: how many words to check on both sides for counting co-occurance
            vector_size: size of vector for each word
            skip_setup: should wf calculation be skipped?
        """
        if not skip_setup:
            self.dataset = Dataset()
            # this will exhaust the iterator
            wf = generate_wf(self.dataset)
            self.wf = {w: f for (w, f) in wf.items() if f >= 5}
            # reset it
            self.dataset.reset()
            self.word2idx = {w: idx for (idx, w) in enumerate(self.wf.keys())}

        self.window_size = window_size
        self.vector_size = vector_size

        # the whole vocabulary won't fit in the normal way
        # need to store in a sparse matrix
        #
        # this dict maps from (i, j) -> value
        self.com = Counter()
        self.features = None

    def train(self, n_procs: int = multiprocessing.cpu_count() - 1):
        """
        Train the COMVectorizer

        Generates the Co-Occurance Matrix (sparse, CSC format)
        Applies SVD

        Args:
            n_procs: number of processes to use
        """
        n_procs = max(n_procs, 1)
        total_size = len(self.dataset)
        chunk_size = total_size // n_procs

        indices = [i * chunk_size for i in range(n_procs)]
        indices.append(total_size)

        # reverse the argument list
        # this displays the progress bar where `ds_start_idx` is 0
        # by reversing the list, the progress bar mimics actual completion closely as
        # the progress bar process will be the last one to start
        args = [
            (self.window_size, self.wf, self.word2idx, indices[i], indices[i + 1])
            for i in range(len(indices) - 1)
        ][::-1]

        with multiprocessing.Pool(n_procs) as pool:
            local_coms = pool.starmap(_com_calculator, args)

        for com in tqdm(local_coms, desc="Aggregating COM"):
            self.com += com

        # free up some memory
        del local_coms

        self.compress_com_using_svd()

        with open(DATA_ROOT / "com.pkl", "wb") as f:
            pickle.dump(dict(self.com), f)
        with open(DATA_ROOT / "w2i.pkl", "wb") as f:
            pickle.dump(dict(self.word2idx), f)
        np.save(DATA_ROOT / "com-feat.npy", self.features)

    def compress_com_using_svd(self):
        """
        Compute the svd for Co-Occurance Matrix to reduce feature dimension

        Converts internal COM representation into scipy CSC format and does SVD on the
            sparse matrix
        """
        r, c = zip(*self.com.keys())
        values = list(self.com.values())

        sparse_com = csc_matrix((values, (r, c)), dtype=float)
        self.features, _, _ = svds(
            sparse_com, k=self.vector_size, return_singular_vectors="u"
        )

    def top_n_similar(self, word: str, n: int = 10) -> List[Tuple[float, str]]:
        assert word in self.word2idx, "Word not in vocabulary"

        if self.features is None:
            self.train()

        idx2word = {i: w for (w, i) in self.word2idx.items()}

        word_feat = self.features[self.word2idx[word]]
        cosine_sim = (
            (self.features * word_feat)
            / (np.c_[np.linalg.norm(self.features, axis=1)] * np.linalg.norm(word_feat))
        ).sum(1)

        n_most_similar = np.argpartition(cosine_sim, -(n + 1))
        sim_word = [
            (cosine_sim[i], idx2word[i])
            for i in n_most_similar[-(n + 1) :]
            if i != self.word2idx[word]
        ]

        sim_word.sort(key=lambda x: x[0], reverse=True)

        return sim_word

    @classmethod
    def load_from_disk(
        cls,
        feat_path: Path = DATA_ROOT / "com-feat.npy",
        w2i_path: Path = DATA_ROOT / "w2i.pkl",
    ) -> COMVectorizer:
        """
        Load crucial inference components from file

        NOTE: the COMVectorizer obtained this way has limited functionalities

        Args:
            feat_path: path to learnt, svd-reduced Co-Occurance Matrix
            w2i_path: path to file containing mapping for word to index

        Returns:
            COMVectorizer
        """
        with open(w2i_path, "rb") as f:
            w2i = pickle.load(f)

        feats = np.load(feat_path)

        v = cls(skip_setup=True)
        v.word2idx = w2i
        v.features = feats

        return v

    def __getitem__(self, word: str) -> np.ndarray:
        """
        Gets the vector mapping for word in vocabulary

        Args:
            word: the word to get vector mapping for

        Returns:
            the vector mapping learnt by Co-Occurance Matrix method
        """
        if self.features is None:
            raise Exception(
                "COMVectorizer has not been trained yet. Either load weights from files or train."
            )

        return self.features[self.word2idx[word]]

    def __contains__(self, word: str):
        """
        Membership test: check if word is in vocabulary
        """
        return word in self.word2idx


if __name__ == "__main__":
    try:
        v = COMVectorizer.load_from_disk()
    except Exception as e:
        print("Couldn't load COMVectorizer from file. Training from scratch.", e)
        v = COMVectorizer()
        v.train()

    code.interact(
        banner="\n".join(
            [
                "v is a COMVectorizer object",
                "Try running v['camera'] to get the embeddings for that word",
                "Run v.top_n_similar('camera') to get the top 10 words related to given word",
                "For more details, run help(COMVectorizer)",
            ]
        ),
        local=dict(globals(), **locals()),
    )
