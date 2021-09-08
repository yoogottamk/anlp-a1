from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from anlp_a1.dataset import Dataset
from anlp_a1.stats import generate_wf
from anlp_a1.utils import get_word_windows


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
        wf = generate_wf(self.dataset)
        self.wf = {w: f for (w, f) in wf.items() if f >= 5}
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

        self.__prev_forward_data = {}

    def __forward(self, context: List[str], word: str) -> float:
        """
        CBOWVectorizer forward, as described in http://cs224d.stanford.edu/lecture_notes/notes1.pdf

        Args:
            context: surrounding words
            word: current word
        """
        context_idx = [self.word2idx[w] for w in context]
        word_idx = self.word2idx[word]

        context_embeddings = self.w1[context_idx]
        context_avg = np.c_[np.average(context_embeddings, axis=0)]
        score_vector = self.w2.T @ context_avg

        y_hat = CBOWVectorizer.softmax(score_vector)
        y: np.ndarray = np.zeros_like(y_hat)
        y[word_idx] = 1

        error = CBOWVectorizer.cross_entropy_loss(y, y_hat)

        self.__prev_forward_data["context_avg"] = context_avg
        self.__prev_forward_data["y_hat"] = y_hat
        self.__prev_forward_data["y"] = y
        self.__prev_forward_data["error"] = error

        return error.sum()

    def __backward(
        self, context: List[str], word: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        context_idx = [self.word2idx[w] for w in context]
        word_idx = self.word2idx[word]

        dw1 = (
            self.__prev_forward_data["error"]
            @ self.__prev_forward_data["context_avg"].T
        )

        _dw2 = self.w1[word_idx] * self.__prev_forward_data["error"][word_idx]
        dw2: np.ndarray = np.zeros_like(self.w2)
        for i in context_idx:
            dw2[:, i] = _dw2

        return dw1, dw2

    def train(self, n_epochs: int = 50, lr: float = 0.03):
        n = len(self.dataset)
        with tqdm(range(n_epochs), desc="Training CBOWVectorizer") as prog_bar:
            for _ in prog_bar:
                epoch_loss = 0
                with tqdm(self.dataset, leave=False) as ds_prog_bar:
                    for item in ds_prog_bar:
                        word_list = [w for w in item["review"].split() if w in self.wf]
                        for surrounding_words, word in get_word_windows(
                            word_list, self.window_size
                        ):
                            loss = self.__forward(surrounding_words, word)
                            dw1, dw2 = self.__backward(surrounding_words, word)

                            self.w1 -= lr * dw1
                            self.w2 -= lr * dw2
                            epoch_loss += loss
                            ds_prog_bar.set_description(f"Loss: {loss:.6f}")

                prog_bar.set_description(f"Loss: {(epoch_loss / n):.6f}")

    @staticmethod
    def cross_entropy_loss(y: np.ndarray, y_hat: np.ndarray):
        return -y * np.log(y_hat + 1e-9)

    @staticmethod
    def softmax(y: np.ndarray) -> np.ndarray:
        e_y: np.ndarray = np.exp(y - np.max(y))
        return e_y / e_y.sum(0)


if __name__ == "__main__":
    v = CBOWVectorizer()
    v.train()
