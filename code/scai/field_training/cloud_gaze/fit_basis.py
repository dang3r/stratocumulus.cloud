"""Rebuild the field model while reserving every geographic cell in the urban challenge."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from read_features import features

d = pd.read_csv("gaze-clean.csv")
urban = pd.read_csv(Path(__file__).with_name("urban-sites.csv.gz"))
blocked = set(urban.group)
train = d[~d.group.isin(blocked) & d.split.ne("test")].copy()
# Keep the separate second release outside fitting.
train["source"] = "CLOUD GAZE"
syn = pd.read_csv("gaze-synthetic.csv")
syn = syn[syn.url.isin(train.url)].copy()
sources = pd.concat([train, syn, pd.read_csv("commons-train.csv")], ignore_index=True)
f, _ = features(sources)
orig = pd.read_csv("dino_scores.csv")
old = np.load("dino_features.npy")[orig.split == "train"]
orig = orig[orig.split == "train"].copy()
orig["agreement"] = 1.0
orig["group"] = orig.file
sources = pd.concat([sources, orig], ignore_index=True)
x = normalize(np.vstack([f, old])).astype("f")
counts = sources.groupby(["source", "label"]).file.transform("size").to_numpy()
repeat = sources.groupby(["source", "group", "label"]).file.transform("size").to_numpy()
w = (
    counts**-0.64
    * repeat**-0.35
    * np.where(sources.source == "CLOUD GAZE", 4, 1)
    * sources.agreement.to_numpy() ** 1.5
)
w /= w.mean()
y = sources.label.to_numpy() == 1
k = np.exp(2 * (x @ x.T - 1))
h = LogisticRegression(C=0.8, max_iter=500, solver="liblinear").fit(
    k, y, sample_weight=w
)
np.savez_compressed(
    "urban-head.npz", full=x, coef=h.coef_.astype("f"), gamma=np.float32(2)
)
