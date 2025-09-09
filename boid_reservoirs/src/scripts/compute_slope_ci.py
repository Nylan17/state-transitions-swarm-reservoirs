import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def fit_slope(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


def bootstrap_slope(df: pd.DataFrame, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = df["N"].to_numpy()
    y = df["total_mc"].to_numpy()
    n = len(df)
    slopes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a, _ = fit_slope(x[idx], y[idx])
        slopes[i] = a
    lo = float(np.percentile(slopes, 2.5))
    hi = float(np.percentile(slopes, 97.5))
    return lo, hi


def main():
    p = argparse.ArgumentParser("Compute slope and 95% CI for MC vs N")
    p.add_argument("--csv", required=True)
    p.add_argument("--out_md", default="boid_reservoirs/results/mc_slope_ci.md")
    p.add_argument("--boots", type=int, default=10000)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    # Restrict to N <= 2000 (headline)
    df = df[df["N"] <= 2000]
    a, b = fit_slope(df["N"].to_numpy(), df["total_mc"].to_numpy())
    lo, hi = bootstrap_slope(df, n_boot=args.boots)
    out = f"### MC vs N slope\n- Fit: MC ≈ {a:.4f}·N + {b:.2f}\n- 95% CI for slope (bootstrap): [{lo:.4f}, {hi:.4f}] (n={len(df)})\n"
    Path(args.out_md).write_text(out)
    print(out)


if __name__ == "__main__":
    main()


