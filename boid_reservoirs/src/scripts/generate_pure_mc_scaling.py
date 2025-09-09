import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_group_stats(df: pd.DataFrame):
    grouped = df.groupby("N")
    stats = grouped["total_mc"].agg(["mean", "std", "count"]).reset_index()
    stats["sem"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
    stats["ci95"] = 1.96 * stats["sem"]
    return stats


def fit_linear(x: np.ndarray, y: np.ndarray):
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


def main():
    parser = argparse.ArgumentParser(description="Generate pure-MC scaling figure from CSV")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_png", default="boid_reservoirs/results/mc_scaling_pure.png")
    parser.add_argument("--out_md", default="boid_reservoirs/results/mc_scaling_pure_summary.md")
    parser.add_argument("--title", default="Pure MC: linear readout, no poly")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    stats = compute_group_stats(df)
    x = stats["N"].to_numpy()
    y = stats["mean"].to_numpy()
    a, b = fit_linear(x, y)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    ax.scatter(df["N"], df["total_mc"], s=16, alpha=0.35, color="#1f77b4", label="runs")
    ax.errorbar(stats["N"], stats["mean"], yerr=stats["ci95"], fmt="o", color="#d62728", capsize=3, label="mean ± 95% CI")
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = a * x_fit + b
    ax.plot(x_fit, y_fit, "--", color="#2ca02c", label=f"fit: MC ≈ {a:.4f}·N + {b:.2f}")
    ax.set_xlabel("Swarm size N")
    ax.set_ylabel("Total memory capacity")
    ax.set_title(args.title + " (partial)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png)
    plt.close(fig)

    lines = []
    lines.append("### Pure MC scaling (partial)\n")
    lines.append(f"- Linear fit on per-N means: MC ≈ {a:.4f}·N + {b:.2f}\n")
    for _, row in stats.iterrows():
        lines.append(
            f"- N={int(row['N'])}: mean={row['mean']:.2f}, 95% CI ±{row['ci95']:.2f} (n={int(row['count'])})\n"
        )
    Path(args.out_md).write_text("".join(lines))
    print(f"Wrote figure to {args.out_png} and summary to {args.out_md}")


if __name__ == "__main__":
    main()


