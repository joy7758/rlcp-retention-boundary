def compute_loss(task_loss, info_term, grad_reversal_term, step, total_steps, config, rng=None):
    """GR-only ablation: task + lambda_gr * GR term."""
    rlcp_cfg = config.get("rlcp", {})
    lambda_gr = float(rlcp_cfg.get("lambda_gr", 0.0))
    use_gr = bool(rlcp_cfg.get("grad_reversal_enabled", True))
    gr_term = float(grad_reversal_term) if use_gr else 0.0
    return float(task_loss) + lambda_gr * gr_term
