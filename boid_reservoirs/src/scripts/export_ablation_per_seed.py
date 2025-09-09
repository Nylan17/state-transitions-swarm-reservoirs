#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import json


def main():
    p = argparse.ArgumentParser(description="Export per-seed ablation totals to CSV")
    p.add_argument("--claim_map", default="docs/claim_map.json")
    p.add_argument("--out", default="boid_reservoirs/results/ablation_per_seed.csv")
    args = p.parse_args()

    claim = json.loads(Path(args.claim_map).read_text())

    rows = []
    for N_key in ["N800", "N1200", "N1600"]:
        if N_key not in claim.get("ablation_pure_mc", {}):
            continue
        N = int(N_key[1:])
        entry = claim["ablation_pure_mc"][N_key]
        # We only have means/CI in claim_map; treat them as summary rows
        rows.append({"N": N, "state": "single", "mc_total": entry["single_state"]["mean"], "n": entry["single_state"]["n"], "summary": True})
        rows.append({"N": N, "state": "two",    "mc_total": entry["two_state"]["mean"],    "n": entry["two_state"]["n"],    "summary": True})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        fieldnames = ["N", "state", "mc_total", "n", "summary"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote summary rows to {out}. Add per-seed rows as needed.")


if __name__ == "__main__":
    main()
