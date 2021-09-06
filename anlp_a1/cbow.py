from typing import Dict, List

import numpy as np

from anlp_a1.dataset import Dataset
from anlp_a1.stats import generate_wf


class CBOWVectorizer:
    def __init__(self, window_size: int = 2, vector_size: int = 64):
        """
        Constructor for CBOWVectorizer

        Args:
            window_size: how many words to check on both sides for counting co-occurance
            vector_size: size of vector for each word
        """
        self.dataset = Dataset()

        # this will exhaust the iterator
        self.wf = generate_wf(self.dataset)
        # reset it
        self.dataset.reset()

        self.word2idx: Dict[str, int] = {
            w: idx for (idx, w) in enumerate(self.wf.keys())
        }

        self.window_size = window_size
        self.vector_size = vector_size

        self.vocab_size = len(self.wf)

        self.w1 = np.random.random((self.vocab_size, self.vector_size))
        self.w2 = np.random.random((self.vector_size, self.vocab_size))

    def __forward(self, context: List[str], word: str):
        """
        CBOWVectorizer forward, as described in http://cs224d.stanford.edu/lecture_notes/notes1.pdf

        Args:
            context: surrounding words
        """
        context_idx = [self.word2idx[w] for w in context]
        word_idx = self.word2idx[word]

        context_embeddings = self.w1[context_idx]
        context_avg = np.c_[np.average(context_embeddings, axis=0)]
        score_vector = self.w2.T @ context_avg

        y_hat = CBOWVectorizer.softmax(score_vector)
        y = np.zeros_like(y_hat)
        y[word_idx] = 1

        error = CBOWVectorizer.cross_entropy_loss(y, y_hat)

    def __backward(self):
        pass

    @staticmethod
    def cross_entropy_loss(y: np.ndarray, y_hat: np.ndarray):
        return -y * np.log(y_hat)

    @staticmethod
    def softmax(y: np.ndarray) -> np.ndarray:
        e_y = np.exp(y - np.max(y))
        return e_y / e_y.sum(0)
