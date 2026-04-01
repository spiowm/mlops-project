"""
Порівняння поточних метрик з baseline для CML звіту.

Використання:
    python scripts/compare_metrics.py

Генерує markdown-таблицю (було/стало/Δ) для CML PR-коментаря.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(baseline: dict, current: dict) -> str:
    """Генерує markdown-таблицю порівняння метрик."""
    keys = sorted(set(list(baseline.keys()) + list(current.keys())) - {"_raw"})

    lines = [
        "| Metric | Baseline | Current | Δ |",
        "|--------|----------|---------|---|",
    ]

    for key in keys:
        b = baseline.get(key)
        c = current.get(key)

        if b is not None and c is not None:
            delta = float(c) - float(b)
            sign = "+" if delta > 0 else ""
            emoji = "🟢" if delta >= 0 else "🔴"
            lines.append(
                f"| {key} | {float(b):.4f} | {float(c):.4f} | {emoji} {sign}{delta:.4f} |"
            )
        elif c is not None:
            lines.append(f"| {key} | — | {float(c):.4f} | 🆕 |")
        elif b is not None:
            lines.append(f"| {key} | {float(b):.4f} | — | ❌ removed |")

    return "\n".join(lines)


def main():
    baseline_path = PROJECT_ROOT / "baseline" / "metrics.json"
    metrics_path = PROJECT_ROOT / "metrics.json"

    if not metrics_path.exists():
        print("metrics.json не знайдено — порівняння неможливе.")
        sys.exit(0)

    current = load_json(metrics_path)

    if baseline_path.exists():
        baseline = load_json(baseline_path)
        table = compare(baseline, current)
        print("### 📊 Metrics Comparison (vs. baseline)")
        print()
        print(table)
    else:
        print("### 📊 Current Metrics")
        print()
        print("| Metric | Value |")
        print("|--------|-------|")
        for k, v in sorted(current.items()):
            if k == "_raw":
                continue
            print(f"| {k} | {float(v):.4f} |")
        print()
        print("> ℹ️ No baseline found (`baseline/metrics.json`). Showing current metrics only.")


if __name__ == "__main__":
    main()
