import pickle
import random
from collections import defaultdict
from math import sqrt
from pathlib import Path

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from tqdm.auto import tqdm

from anlp_a1.config import REPO_ROOT
from anlp_a1.dataset import Dataset
from anlp_a1.stats import generate_wf


class COMVectorizer:
    """
    Co-Occurance Matrix Vectorizer

    Generates a Co-Occurance Matrix and performs SVD to reduce dimension of vectors
    """

    def __init__(
        self,
        dataset: Dataset = Dataset(),
        window_size: int = 5,
        vector_size: int = 1000,
    ):
        """
        Constructor for COMVectorizer

        Args:
            dataset: the Dataset to use for training
            window_size: how many words to check on both sides for counting co-occurance
            vector_size: size of vector for each word
        """
        self.dataset = dataset

        # this will exhaust the iterator
        self.wf = generate_wf(dataset)
        # reset it
        dataset.reset()

        self.window_size = window_size
        self.vector_size = vector_size

        self.idx2word = dict(enumerate(self.wf.keys()))
        self.word2idx = {w: idx for (idx, w) in self.idx2word.items()}

        self.com = defaultdict(lambda: 0)
        self.__trained = False

        self.features = None

    def subsample_probability(self, word: str, t: float = 1e-5):
        """
        Subsampling

        Calculates the probability to keep a word
        https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf

        Args:
            word: word in vocab
            t: subsampling threshold
        """
        return sqrt(t / self.wf[word])

    def train(self):
        """
        Train the COMVectorizer

        Generates the Co-Occurance Matrix (sparse, CSC format)
        Applies SVD
        """
        if self.__trained:
            return

        for item in tqdm(self.dataset, desc="Training COMVectorizer"):
            word_list = item["review"].split()

            for idx in range(len(word_list)):
                start_idx = max(0, idx - self.window_size)
                end_idx = min(len(self.dataset), idx + self.window_size)

                word = word_list[idx]
                surrounding_words = [
                    *word_list[start_idx:idx],
                    *word_list[idx + 1 : end_idx],
                ]

                subsampled_words = [
                    w
                    for w in surrounding_words
                    if self.subsample_probability(w) < random.uniform(0, 1)
                ]

                for com_word in subsampled_words:
                    i = self.word2idx[word]
                    j = self.word2idx[com_word]

                    self.com[(i, j)] += 1
                    self.com[(j, i)] += 1

        with open(REPO_ROOT / "com.pkl", "wb") as f:
            pickle.dump(dict(self.com), f)

        self.__trained = True

    def svd(self):
        r, c = zip(*self.com.keys())
        values = list(self.com.values())

        sparse_com = csc_matrix((values, (r, c)), dtype=float)
        self.features, _, _ = svds(sparse_com, k=self.vector_size, return_singular_vectors="u")

    @classmethod
    def load_from_file(cls, path: Path = REPO_ROOT / "com.pkl"):
        with open(path, "rb") as f:
            com = pickle.load(f)

        v = cls()
        v.com = com
        v.__trained = True
        v.svd()

        return v


if __name__ == "__main__":
    try:
        v = COMVectorizer.load_from_file()
    except Exception as e:
        print(e)
        v = COMVectorizer()
        v.train()
