#!/usr/bin/env python3
"""
Generate a single-panel ablation figure (single-state vs two-state) at N in {800,1200,1600}
using validated means/CI from docs/claim_map.json.

Outputs:
  - boid_reservoirs/results/ablation_all_panels.png

This keeps LaTeX paths unchanged (user can upload/copy as needed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
CLAIM_MAP_PATH = REPO_ROOT / "docs" / "claim_map.json"
OUT_DIR = REPO_ROOT / "boid_reservoirs" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_BASE = OUT_DIR / "ablation_all_panels"


def load_ablation_from_claim_map(path: Path) -> Dict[int, Dict[str, Tuple[float, float, int]]]:
    data = json.loads(path.read_text())
    ab = data.get("ablation_pure_mc", {})
    if not ab:
        raise RuntimeError("ablation_pure_mc not found in claim_map.json")

    def parse_entry(entry: Dict[str, Any]) -> Tuple[float, float, int]:
        return float(entry["mean"]), float(entry["ci95"]), int(entry["n"])

    result: Dict[int, Dict[str, Tuple[float, float, int]]] = {}
    for key in ["N800", "N1200", "N1600"]:
        if key not in ab:
            continue
        N = int(key[1:])
        s1 = parse_entry(ab[key]["single_state"])
        s2 = parse_entry(ab[key]["two_state"])
        result[N] = {"single": s1, "two": s2}
    if not result:
        raise RuntimeError("No N entries found under ablation_pure_mc in claim_map.json")
    return result


def plot_grouped_ablation(ablation: Dict[int, Dict[str, Tuple[float, float, int]]]) -> plt.Figure:
    Ns: List[int] = sorted(ablation.keys())
    means_single = [ablation[N]["single"][0] for N in Ns]
    ci_single = [ablation[N]["single"][1] for N in Ns]
    means_two = [ablation[N]["two"][0] for N in Ns]
    ci_two = [ablation[N]["two"][1] for N in Ns]

    x = np.arange(len(Ns), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=150)

    bars1 = ax.bar(x - width/2, means_single, width, yerr=ci_single,
                   label="Single-state", color="#1f77b4", capsize=3)
    bars2 = ax.bar(x + width/2, means_two, width, yerr=ci_two,
                   label="Two-state", color="#d62728", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in Ns])
    ax.set_xlabel("Population N")
    ax.set_ylabel("Total MC (log scale)")

    ax.set_yscale("log")
    ymin = min([m - c for m, c in zip(means_single, ci_single)])
    if not np.isfinite(ymin) or ymin <= 0:
        ymin = 0.03
    ax.set_ylim(bottom=max(ymin, 0.03))

    ax.grid(False)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    ax.set_title("Ablation under pure MC (single vs two-state)")

    # Optional numeric annotations (only for two-state to avoid clutter)
    for rect, val in zip(bars2, means_two):
        ax.annotate(f"{val:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="#444444")

    tops = [m + c for m, c in zip(means_two, ci_two)] + [m + c for m, c in zip(means_single, ci_single)]
    y_top = max(tops) * 1.50 if len(tops) else None
    if y_top and np.isfinite(y_top):
        ax.set_ylim(top=y_top)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.28)
    return fig


def main() -> None:
    ablation = load_ablation_from_claim_map(CLAIM_MAP_PATH)
    fig = plot_grouped_ablation(ablation)
    fig.savefig(str(OUT_BASE) + ".png", dpi=300)
    plt.close(fig)
    print(f"Wrote {OUT_BASE.with_suffix('.png')}")


if __name__ == "__main__":
    main()


