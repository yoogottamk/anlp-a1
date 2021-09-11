import sys
from collections import Counter
from typing import Dict

import pandas as pd
from tqdm.auto import tqdm

from anlp_a1.dataset import Dataset


def generate_wf(dataset: Dataset) -> Dict[str, int]:
    """
    Generates a word frequency dict

    Args:
        dataset: the Dataset object to use, defaults to the whole dataset

    Returns:
        a dictionary mapping word to frequency
    """
    wf_dict = Counter()

    for item in tqdm(dataset, desc="Calculating word frequencies"):
        for w in item["review"].split():
            wf_dict[w] += 1

    return wf_dict


def get_wf_df(dataset: Dataset) -> pd.DataFrame:
    """
    Generates a word-frequency pd.DataFrame for statistical analysis

    Args:
        dataset: the Dataset object to use, defaults to the whole dataset

    Returns:
        a pd.DataFrame containing 2 columns: `word`, `freq`
    """
    wf_dict = generate_wf(dataset)
    return pd.DataFrame(list(wf_dict.items()), columns=["word", "freq"])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        frac = float(sys.argv[1])
        ds = Dataset(frac=frac)
    else:
        ds = Dataset()

    wf = get_wf_df(ds)

    filtered_terms = wf[wf["freq"] >= 5]

    print("Summary:")
    print(filtered_terms.describe())
    print()

    print("Top 10 values:")
    print(filtered_terms.sort_values(by="freq", ascending=False).head(10))
    print()

    print("Total:", filtered_terms["freq"].sum())
