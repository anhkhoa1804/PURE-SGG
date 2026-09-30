"""Paired-block sample-size simulation for the preregistered E2 endpoint."""

from __future__ import annotations

import argparse
import json
import math
from statistics import NormalDist
from pathlib import Path

import numpy as np


def simulate_required_n(baseline: float, delta: float, discordance: float, alpha: float = .05,
                        power_target: float = .80, half_width: float = .05,
                        seed: int = 20260911, replicates: int = 100_000,
                        min_n: int = 25, max_n: int = 5000) -> dict:
    q01 = (discordance + delta) / 2.0
    q10 = (discordance - delta) / 2.0
    if min(q01, q10, baseline - q10, 1 - baseline - q01) < 0:
        return {"status": "invalid_parameters", "reason": "discordance/delta incompatible with baseline"}
    rng = np.random.default_rng(seed)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    rows = []
    first_power_n = None
    first_precision_n = None
    for n in range(min_n, max_n + 1, 25):
        # The paired outcome has four mutually exclusive cells.  Sampling
        # their counts directly is exactly equivalent to drawing n paired
        # blocks, while avoiding an enormous replicate-by-n boolean array.
        probs = [baseline - q10, q10, q01, 1.0 - baseline - q01]
        counts = rng.multinomial(n, probs, size=replicates)
        d_means = (counts[:, 2] - counts[:, 1]) / n
        sd = math.sqrt(max(discordance - delta * delta, 0.0) / n)
        z_stats = d_means / max(sd, 1e-12)
        power = float(np.mean(z_stats > z))
        expected_half_width = z * sd
        rows.append({"n": n, "power": power, "expected_ci_half_width": expected_half_width})
        if first_power_n is None and power >= power_target:
            first_power_n = n
        if first_precision_n is None and expected_half_width <= half_width:
            first_precision_n = n
        if first_power_n is not None and first_precision_n is not None:
            break
    required = max(x for x in (first_power_n, first_precision_n) if x is not None) if first_power_n or first_precision_n else None
    return {"status": "completed", "baseline": baseline, "delta": delta, "discordance": discordance,
            "alpha": alpha, "power_target": power_target, "half_width_target": half_width,
            "seed": seed, "replicates": replicates, "required_n": required,
            "first_power_n": first_power_n, "first_precision_n": first_precision_n, "grid": rows}


def run_grid(baselines, deltas, discordances, **kwargs):
    rows = []
    for baseline in baselines:
        for delta in deltas:
            for discordance in discordances:
                row = simulate_required_n(baseline, delta, discordance, **kwargs)
                rows.append(row)
    return {"protocol": {"seed": kwargs.get("seed", 20260911), "replicates": kwargs.get("replicates", 100_000), "rounding": "smallest multiple of 25", "paired_unit": "independent contrast block", "minimum_blocks_per_admitted_relation_pair": 100}, "attrition_scenarios": {str(rate): math.ceil(100 / rate) for rate in (.2, .4, .6, .8)}, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=100_000)
    args = parser.parse_args()
    result = run_grid([.50, .60, .70, .80], [.05], [.10, .20, .30, .40], replicates=args.replicates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "rows": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
