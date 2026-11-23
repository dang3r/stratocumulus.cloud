from pathlib import Path
import numpy as np


def features(df):
    full = []
    patch = []
    for f in df.file:
        with np.load(Path("gaze-cache") / (f.replace("/", "_") + ".npz")) as data:
            full.append(data["full"])
            patch.append(data["patch"])
    return np.stack(full), np.stack(patch)
