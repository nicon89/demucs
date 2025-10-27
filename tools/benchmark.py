#!/usr/bin/env python
"""Utility to aggregate evaluation runs into a markdown benchmark table."""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

def load_metric(csv_path: Path, metric: str) -> Dict[str, float]:
    values: Dict[str, List[float]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["metric"].lower() != metric.lower():
                continue
            stem = row["stem"]
            values.setdefault(stem, []).append(float(row["value"]))
    return {stem: sum(scores) / len(scores) for stem, scores in values.items()}


def render_table(rows: List[Dict[str, float]], names: List[str], metric: str) -> str:
    stems = sorted({stem for row in rows for stem in row.keys()})
    header = ["model"] + [f"{stem} {metric.upper()}" for stem in stems] + [f"avg {metric.upper()}"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for name, row in zip(names, rows):
        values = [row.get(stem, float("nan")) for stem in stems]
        valid = [v for v in values if not math.isnan(v)]
        avg = sum(valid) / len(valid) if valid else float("nan")
        formatted = [f"{val:.3f}" if not math.isnan(val) else "nan" for val in values]
        lines.append("| " + " | ".join([name] + formatted + [f"{avg:.3f}" if not math.isnan(avg) else "nan"]) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a markdown benchmark table from eval runs")
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories containing metrics.csv")
    parser.add_argument("--names", nargs="*", help="Names to use for each run")
    parser.add_argument("--metric", default="sdr", help="Metric key to aggregate (default: sdr)")
    parser.add_argument("--output", type=Path, help="Optional path to save markdown table")
    args = parser.parse_args()

    if args.names and len(args.names) != len(args.runs):
        raise ValueError("--names must match the number of runs")

    names = args.names or [run.name for run in args.runs]
    rows = []
    for run in args.runs:
        csv_path = run / "metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing metrics.csv in {run}")
        rows.append(load_metric(csv_path, args.metric))

    table = render_table(rows, names, args.metric)
    print(table)
    if args.output:
        args.output.write_text(table + "\n")


if __name__ == "__main__":
    main()
