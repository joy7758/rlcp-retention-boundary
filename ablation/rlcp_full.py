import math


def rlcp_loss(task_loss, info_term, grad_reversal_term, beta, lambda_gr):
    return task_loss - beta * info_term + lambda_gr * grad_reversal_term


def _beta_value(step, total_steps, schedule, start, end):
    if total_steps <= 1:
        return float(end)

    progress = float(step) / float(total_steps - 1)
    if schedule == "exponential":
        if start <= 0 or end <= 0:
            raise ValueError("Exponential beta schedule requires start/end > 0.")
        return float(start * math.pow(end / start, progress))

    return float(start + (end - start) * progress)


def resolve_beta(step, total_steps, config):
    rlcp_cfg = config.get("rlcp", {})
    beta_cfg = rlcp_cfg.get("beta", {})

    beta_enabled = bool(beta_cfg.get("enabled", True))
    if not beta_enabled:
        return 0.0

    schedule = str(beta_cfg.get("schedule", "linear")).lower()
    start = float(beta_cfg.get("start", 0.1))
    end = float(beta_cfg.get("end", 1.0))
    return _beta_value(step=step, total_steps=total_steps, schedule=schedule, start=start, end=end)


def compute_loss(task_loss, info_term, grad_reversal_term, step, total_steps, config, rng=None):
    """Full RLCP objective with configurable beta schedule and GR switch."""
    rlcp_cfg = config.get("rlcp", {})
    if not bool(rlcp_cfg.get("enabled", True)):
        return float(task_loss)

    beta = resolve_beta(step=step, total_steps=total_steps, config=config)
    lambda_gr = float(rlcp_cfg.get("lambda_gr", 0.0))
    use_gr = bool(rlcp_cfg.get("grad_reversal_enabled", True))

    info = float(info_term) if beta != 0.0 else 0.0
    gr_term = float(grad_reversal_term) if use_gr else 0.0
    lambda_effective = lambda_gr if use_gr else 0.0

    return float(
        rlcp_loss(
            task_loss=float(task_loss),
            info_term=info,
            grad_reversal_term=gr_term,
            beta=beta,
            lambda_gr=lambda_effective,
        )
    )
