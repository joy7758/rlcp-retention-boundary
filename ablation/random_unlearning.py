import numpy as np


def compute_loss(task_loss, info_term, grad_reversal_term, step, total_steps, config, rng=None):
    """Random ablation: inject bounded random perturbation."""
    generator = rng if rng is not None else np.random.default_rng(2026)
    jitter = float(generator.uniform(-0.08, 0.08))
    return float(task_loss) + jitter
