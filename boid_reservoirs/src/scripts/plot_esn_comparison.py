import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_results(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("label")["total_mc"].agg(["mean", "std", "count"]).reset_index()
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    g["ci95"] = 1.96 * g["sem"]
    return g


def plot_summary(df: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=150)
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    ax.bar(x, df["mean"], yerr=df["ci95"], capsize=3, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total MC (mean ± 95% CI)")
    ax.set_title("ESN MC by size")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def write_md(df: pd.DataFrame, out_md: Path):
    lines = ["### ESN comparison (memory capacity)\n"]
    for _, row in df.iterrows():
        lines.append(
            f"- {row['label']}: mean={row['mean']:.2f}, 95% CI ±{row['ci95']:.2f} (n={int(row['count'])})\n"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser("Plot ESN comparison results")
    parser.add_argument("--json", required=True)
    parser.add_argument("--out_png", default="boid_reservoirs/results/comparison/esn_mc_by_size.png")
    parser.add_argument("--out_md", default="boid_reservoirs/results/comparison/esn_summary.md")
    args = parser.parse_args()

    df = load_results(args.json)
    summ = summarize(df)
    plot_summary(summ, Path(args.out_png))
    write_md(summ, Path(args.out_md))
    print(f"Wrote {args.out_png} and {args.out_md}")


if __name__ == "__main__":
    main()


