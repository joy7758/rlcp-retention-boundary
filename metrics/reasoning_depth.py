import numpy as np


def reasoning_depth_proxy(chain_lengths):
    """Proxy for reasoning depth from chain-of-thought step lengths."""
    arr = np.asarray(chain_lengths, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())
