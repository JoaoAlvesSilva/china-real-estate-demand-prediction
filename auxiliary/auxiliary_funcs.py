import numpy as np
from auxiliary.static_data import *


def func_aux_str_to_int(name):
    new_str = ""
    for char in name:
        if char == " ":
            new_str += "0"
        else:
            new_str += char
    return int(new_str)


def func_str_to_yr(id):
    return int(id[:4])


def func_str_to_sector(id):
    return int(func_aux_str_to_int(id[-2:]))


def func_str_to_month(id):
    return int(month_to_int[id[5:8]])


def score(y_pred: np.ndarray, y_true: np.ndarray, eps=10 ** (-12)) -> float:
    ratio = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))

    ratio_small = []
    count = 0
    n = len(ratio)
    for num in ratio:
        if num > 1:
            count += 1
        else:
            ratio_small.append(num)
    if count / n > 0.3:
        return 0
    else:
        fraction = len(ratio_small) / n
        scaled_mape = np.mean(ratio_small) / fraction
        return 1 - scaled_mape


def random_half_score(y1, y2, score_func=score, random_seed=None, n_repeats=1152):
    """
    Randomly partition two arrays in half and apply scoring function.
    Repeat n_repeats times and return the average score.

    Parameters:
    y1, y2: arrays of equal size
    score_func: scoring function to apply (default: score)
    random_seed: seed for reproducibility (optional)
    n_repeats: number of random partitions to average over (default: len(y1))

    Returns:
    Average score across all random partitions
    """
    if len(y1) != len(y2):
        raise ValueError("Arrays must be of equal size")

    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(y1)
    if n_repeats is None:
        n_repeats = n

    scores = []

    for _ in range(n_repeats):
        # Randomly select half of the indices
        indices = np.random.choice(n, size=n // 2, replace=False)

        # Get the other half
        remaining_indices = np.setdiff1d(np.arange(n), indices)

        # Extract halves
        y1_half1 = y1[indices]
        y2_half1 = y2[indices]
        y1_half2 = y1[remaining_indices]
        y2_half2 = y2[remaining_indices]

        # Apply scoring function to both halves
        score1 = score_func(y1_half1, y2_half1)
        score2 = score_func(y1_half2, y2_half2)

        # Average the two scores
        avg_score = (score1 + score2) / 2
        scores.append(avg_score)

    return np.mean(scores)
