"""Calibrate residual scores with nested geographic cross-validation on real urban photos."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedGroupKFold
from read_features import features
from scipy.optimize import minimize
from scipy.special import expit

u = pd.read_csv(Path(__file__).with_name("urban-sites.csv.gz"))
original = np.load("urban-head.npz")
centers = original["full"]
scale = np.float32(abs(centers).max() / 127)
q = np.rint(centers / scale) * scale
f, z = features(u)
x = normalize(f)
z = normalize(z)


def raw(t):
    return (np.exp(2 * (t @ q.T - 1)) @ original["coef"].T).ravel()


s = raw(x)
s -= np.median(s)  # constant shift absorbed by the fitted intercept
p = raw(z) - raw(x)
sky = np.array(
    [
        float(np.load(Path("gaze-cache") / (f.replace("/", "_") + ".npz"))["sky"])
        for f in u.file
    ]
)
base0 = float(np.median(raw(x)))
# Standardized scale of each feature family is fixed before fitting any fold.
xx = np.column_stack([x, z, p, np.square(sky), sky, np.ones(len(x))])
y = u.label.to_numpy()
groups = u.group.to_numpy()


def fit(idx, strength):
    X = xx[idx]
    Y = y[idx]
    start = s[idx]
    w = np.where(Y == 1, 0.5 / (Y == 1).sum(), 0.5 / (Y == 0).sum())

    def objective(v):
        pred = start + X @ v
        loss = np.sum(w * (np.logaddexp(0, pred) - Y * pred)) + 0.5 * strength * (
            v[:-1] @ v[:-1]
        )
        grad = X.T @ (w * (expit(pred) - Y))
        grad[:-1] += strength * v[:-1]
        return loss, grad

    return minimize(
        objective,
        np.zeros(X.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 90},
    ).x


def folds(ix, n, seed):
    return [
        (ix[tr], ix[te])
        for tr, te in StratifiedGroupKFold(n, shuffle=True, random_state=seed).split(
            ix, y[ix], groups[ix]
        )
    ]


def choose(t, lab):
    # Maximize the smaller of the recall and specificity margins above their requested targets.
    pos = t[lab == 1]
    neg = t[lab == 0]
    thresholds = np.unique(t)
    stat = [
        (
            min((pos >= v).mean() - 0.85, (neg < v).mean() - 0.70),
            ((pos >= v).mean() + (neg < v).mean()) / 2,
            v,
        )
        for v in thresholds
    ]
    return max(stat)


def tune(ix):
    best = None
    for c in [0.01, 0.05, 0.2, 1.0]:
        oof = np.zeros(len(y))
        for tr, va in folds(ix, 4, 701):
            h = fit(tr, c)
            oof[va] = s[va] + xx[va] @ h
        rank = choose(oof[ix], y[ix])
        if best is None or rank[:2] > best[0][:2]:
            best = (rank, c)
    rank, c = best
    return fit(ix, c), rank[-1], c


def save(name, coef, t):
    np.savez(name, coef=coef, threshold=t, base0=base0)


allids = np.arange(len(u))
out = np.zeros(len(y))
which = np.zeros(len(y), int)
for fold, (tr, te) in enumerate(folds(allids, 5, 1097)):
    h, t, c = tune(tr)
    out[te] = s[te] + xx[te] @ h - t
    which[te] = fold
    print(
        "fold",
        fold,
        "n",
        len(te),
        "strength",
        c,
        "TP/positive",
        sum((out[te] >= 0) & (y[te] == 1)),
        sum(y[te] == 1),
        "FP/negative",
        sum((out[te] >= 0) & (y[te] == 0)),
        sum(y[te] == 0),
        flush=True,
    )
    save(f"urban-fold-{fold}.npz", h, t)
u.assign(outer_fold=which, oof_margin=out).to_csv("urban-oof.csv", index=False)
print(
    "OOF",
    u.assign(call=out >= 0).groupby("label").call.agg(["sum", "size"]),
    flush=True,
)

h, t, c = tune(allids)
save("urban-release.npz", h, t)
print("all-site release strength", c, "threshold", t, flush=True)
