"""Summarize scores using an operating point chosen on development negatives."""

from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
data = pd.read_json(root / "scores.jsonl.gz", lines=True)
neg = data.query("panel == 'development' and label == 0")
budget = sum(neg.full >= 0.25)
thresholds = {
    "full": 0.25,
    "multi": next(
        t for t in sorted(neg.multi.unique()) if sum(neg.multi >= t) <= budget
    ),
}
print("Thresholds", thresholds)
reviewed = data.query(
    "panel == 'new' and visual_label in ['likely_sc','clear','other_cloud']"
).copy()
reviewed["label"] = reviewed.visual_label.eq("likely_sc")
pairs = data.query("panel == 'reviewed-pairs'").merge(
    pd.read_csv(root / "reviewed-panels.csv")[["file", "sky_fraction"]], on="file"
)
groups = list(data.groupby("panel")) + [
    ("visual review", reviewed),
    ("paired sky <=20%", pairs[pairs.sky_fraction <= 0.2]),
]
for name, rows in groups:
    positive = rows.label == 1
    print(name, "n+", sum(positive), "n-", sum(~positive), end=" ")
    for key, threshold in thresholds.items():
        calls = rows[key] >= threshold
        print(key, "TP", sum(calls & positive), "FP", sum(calls & ~positive), end=" ")
    print()
print(
    "New photo ms median and p95",
    data.query("panel=='new'").ms.quantile([0.5, 0.95]).tolist(),
)

print("Post-hoc TP at or below each panel's full-frame false-positive count")
for name, rows in groups:
    positive = rows.label == 1
    budget = sum((rows.full >= 0.25) & ~positive)
    candidates = [float("inf"), *rows.multi.unique()]
    tp = max(
        sum((rows.multi >= t) & positive)
        for t in candidates
        if sum((rows.multi >= t) & ~positive) <= budget
    )
    print(
        name,
        "FP budget",
        budget,
        "full TP",
        sum((rows.full >= 0.25) & positive),
        "multi TP",
        tp,
    )
