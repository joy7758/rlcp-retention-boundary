from ablation.rlcp_full import resolve_beta


def compute_loss(task_loss, info_term, grad_reversal_term, step, total_steps, config, rng=None):
    """Beta-only ablation: task - beta * information term."""
    beta = resolve_beta(step=step, total_steps=total_steps, config=config)
    return float(task_loss) - beta * float(info_term)
