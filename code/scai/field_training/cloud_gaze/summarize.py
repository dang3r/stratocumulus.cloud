"""Report browser counts and Wilson intervals for the release decision."""

from pathlib import Path
import math
import pandas as pd


def wilson(hits, n):
    z = 1.96
    if not n:
        return (float("nan"), float("nan"))
    p = hits / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    delta = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - delta), min(1.0, center + delta)


if __name__ == "__main__":
    results = pd.read_json(
        Path(__file__).with_name("browser-results.jsonl.gz"), lines=True
    )
    for panel, rows in results.groupby("endpoint", sort=False):
        print(panel)
        for model in ("baseline", "high_sensitivity", "positive"):
            stats = []
            for label in (1, 0):
                subset = rows[rows.label == label]
                n = len(subset)
                k = int(subset[model].sum())
                low, high = wilson(k, n)
                if not n:
                    stats.append("no examples")
                    continue
                stats.append(f"{k}/{n} ({k / n:.1%}; 95% Wilson {low:.1%}–{high:.1%})")
            print(model, "recall", stats[0], "false-positive rate", stats[1])
