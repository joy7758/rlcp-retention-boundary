def compute_loss(task_loss, info_term, grad_reversal_term, step, total_steps, config, rng=None):
    """Baseline ablation: task loss only."""
    return float(task_loss)
