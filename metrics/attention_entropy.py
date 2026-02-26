import numpy as np


def _normalize_attention(attention):
    denom = attention.sum(axis=-1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return attention / denom


def _entropy(probabilities):
    p = np.clip(probabilities, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def per_layer_attention_entropy(attention_map):
    """
    Compute mean attention entropy for each layer.

    Supported shape: [layers, heads, tokens, tokens].
    """
    attn = np.asarray(attention_map, dtype=float)
    if attn.ndim != 4:
        raise ValueError("attention_map must be 4D: [layers, heads, tokens, tokens]")

    probs = _normalize_attention(attn)
    entropy = _entropy(probs)  # [layers, heads, tokens]
    per_layer = entropy.mean(axis=(1, 2))
    return per_layer


def average_attention_entropy(attention_map):
    """Return full stats including per-layer and global mean entropy."""
    per_layer = per_layer_attention_entropy(attention_map)
    return {
        "per_layer_entropy": [float(x) for x in per_layer],
        "mean_entropy": float(np.mean(per_layer)),
    }
