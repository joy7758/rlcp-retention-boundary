import numpy as np


def hallucination_rate(flags):
    """Mean hallucination indicator in [0, 1]."""
    arr = np.asarray(flags, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def from_factuality_scores(scores, threshold=0.5):
    """Convert factuality scores to hallucination rate by thresholding."""
    score_arr = np.asarray(scores, dtype=float)
    return hallucination_rate(score_arr < float(threshold))
