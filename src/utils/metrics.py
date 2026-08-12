"""Skill metrics for MHW prediction evaluation."""

import numpy as np
from scipy.stats import pearsonr


def skill_by_phase(preds: np.ndarray, trues: np.ndarray,
                   doys: np.ndarray, p90_ns: np.ndarray) -> dict:
    """
    Compute Pearson r per MHW phase (all / onset / mid_event / no_mhw).

    Returns dict with keys: 'all', 'onset', 'mid_event', 'no_mhw',
    each containing {'n': int, 'r_model': float}.
    Also adds 'persist_lag7_all': float if len(trues) > 7.
    """
    from src.utils.hobday import mhw_phase_labels
    labels = mhw_phase_labels(trues, doys, p90_ns)

    results = {}
    for phase_name, mask in [
        ("all",       np.ones(len(labels), dtype=bool)),
        ("onset",     labels == 1),
        ("mid_event", labels == 2),
        ("no_mhw",    labels == 0),
    ]:
        if mask.sum() < 10:
            results[phase_name] = {"n": int(mask.sum()), "r_model": np.nan}
            continue
        r, _ = pearsonr(preds[mask], trues[mask])
        results[phase_name] = {"n": int(mask.sum()), "r_model": float(r)}

    if len(trues) > 7:
        r_p, _ = pearsonr(trues[:-7], trues[7:])
        results["persist_lag7_all"] = float(r_p)

    return results
