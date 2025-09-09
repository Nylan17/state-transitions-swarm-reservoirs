import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def fit_slope(x: np.ndarray, y: np.ndarray):
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def main():
    p = argparse.ArgumentParser("Summarize ESN MC scaling and fit slope")
    p.add_argument("--json", required=True)
    p.add_argument("--out_md", default="boid_reservoirs/results/comparison/esn_mc_scaling_summary.md")
    args = p.parse_args()

    with open(args.json, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Map label -> reservoir size
    size_map = {"small": 600, "medium": 2400, "large": 4800}
    df["n_res"] = df["label"].map(size_map)

    # Per-size means
    g = df.groupby(["label", "n_res"]).agg(mc_mean=("total_mc", "mean"), mc_std=("total_mc", "std"), count=("total_mc", "count")).reset_index()
    g["mc_sem"] = g["mc_std"] / np.sqrt(g["count"].clip(lower=1))
    g["mc_ci95"] = 1.96 * g["mc_sem"]

    # Fit slope vs reservoir size
    x = g["n_res"].to_numpy(dtype=float)
    y = g["mc_mean"].to_numpy(dtype=float)
    a, b = fit_slope(x, y)

    # Write summary
    lines = ["### ESN MC scaling (per reservoir size)\n\n"]
    for _, row in g.iterrows():
        lines.append(f"- {row['label']} (n_res={int(row['n_res'])}): {row['mc_mean']:.2f} ± {row['mc_ci95']:.2f} (n={int(row['count'])})\n")
    lines.append(f"\n- Linear fit on means: MC ≈ {a:.5f}·n_res + {b:.2f}\n")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("".join(lines))
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()


