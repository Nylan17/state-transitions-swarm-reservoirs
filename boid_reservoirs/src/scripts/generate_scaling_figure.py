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
    # Simple OLS fit y = a*x + b
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


def generate_plot(df: pd.DataFrame, out_png: Path, title: str | None = None):
    stats = compute_group_stats(df)
    x = stats["N"].to_numpy()
    y = stats["mean"].to_numpy()
    a, b = fit_linear(x, y)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)

    # Scatter all runs faded
    ax.scatter(df["N"], df["total_mc"], s=14, alpha=0.25, color="#1f77b4", label="runs")
    # Means with 95% CI
    ax.errorbar(x, y, yerr=stats["ci95"], fmt="o", color="#d62728", capsize=3, label="mean ± 95% CI")

    # Fit line across shown N
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = a * x_fit + b
    ax.plot(x_fit, y_fit, "--", color="#2ca02c", label=f"fit: MC ≈ {a:.4f}·N + {b:.2f}")

    ax.set_xlabel("Swarm size N")
    ax.set_ylabel("Total memory capacity")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    return a, b, stats


def write_summary_md(out_md: Path, a: float, b: float, stats: pd.DataFrame):
    lines = []
    lines.append("### GPU scaling summary (MC vs N)\n")
    lines.append(f"- Linear fit on per-N means: MC ≈ {a:.4f}·N + {b:.2f}\n")
    for _, row in stats.iterrows():
        lines.append(
            f"- N={int(row['N'])}: mean={row['mean']:.2f}, 95% CI ±{row['ci95']:.2f} (n={int(row['count'])})\n"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate MC scaling figure from CSV")
    parser.add_argument("--csv", required=True, help="Path to gpu_scaling_complete.csv")
    parser.add_argument("--out_png", default="boid_reservoirs/results/mc_scaling.png")
    parser.add_argument("--out_md", default="boid_reservoirs/results/mc_scaling_summary.md")
    parser.add_argument("--title", default="MC scales linearly with N (GPU, noise ∈ {0,0.05,0.1})")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # Filter to N ≤ 2000 for headline figure
    df = df[df["N"] <= 2000]
    out_png = Path(args.out_png)
    a, b, stats = generate_plot(df, out_png, title=args.title)
    write_summary_md(Path(args.out_md), a, b, stats)
    print(f"Wrote figure to {out_png} and summary to {args.out_md}")


if __name__ == "__main__":
    main()


