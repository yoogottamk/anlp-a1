import sqlite3
from pathlib import Path

from anlp_a1.config import DATA_ROOT


class Dataset:
    """
    An interface for the dataset

    Should be used as an iterator, although, indexing is also supported
    """

    def __init__(self, path: Path = DATA_ROOT / "dataset.sqlite", frac: float = 1):
        self.__conn = sqlite3.connect(path)

        # calculate and store length once
        self.len = int(
            self.__conn.cursor().execute("select count(*) from dataset").fetchone()[0]
            * frac
        )

        # store current index for iterator
        self.__idx = -1

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        self.__idx += 1

        if self.__idx < len(self):
            return self[self.__idx]

        raise StopIteration

    def reset(self):
        """
        Custom reset method for this iterator

        Resets the iterator to beginning
        """
        self.__idx = -1

    def __getitem__(self, idx: int) -> dict:
        row = (
            self.__conn.cursor()
            .execute("select * from dataset where rowid = (?)", (idx + 1,))
            .fetchone()
        )

        return dict(zip(["review"], row))

    def __del__(self):
        self.__conn.close()
