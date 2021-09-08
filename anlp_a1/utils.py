from typing import Generator, List, Tuple


def get_word_windows(
    word_list: List[str],
    window_size: int,
    return_left: bool = True,
    return_right: bool = True,
) -> Generator[Tuple[List[str], str], None, None]:
    """
    Returns word windows from a list of words

    Args:
        word_list: list of words from where windows are picked
        window_size: size of the window
        return_left: should the left window be returned?
        return_right: should the right window be returned?

    Yields:
        surrounding words and the word
    """
    n = len(word_list)
    for idx in range(n):
        start_idx = max(0, idx - window_size)
        end_idx = min(n, idx + window_size + 1)

        surrounding_words = []
        word = word_list[idx]

        if return_left:
            surrounding_words += word_list[start_idx:idx]
        if return_right:
            surrounding_words += word_list[idx + 1 : end_idx]

        yield surrounding_words, word
