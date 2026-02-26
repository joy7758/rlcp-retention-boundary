from typing import Dict, List

RETENTION_RATES = [0.15,0.12,0.10,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]


def get_retention_rates(config: Dict) -> List[float]:
    """Return retention rates from config when present, else default sweep."""
    configured = config.get("experiment", {}).get("retention_rates", [])
    if configured:
        return [float(x) for x in configured]
    return list(RETENTION_RATES)


def retention_dirname(rate: float) -> str:
    """Stable retention directory naming, e.g. r_0.15."""
    return f"r_{rate:.2f}"
