"""Compute frame and region features; identical sky window geometry is used in JS."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import tensor, session
from regions import roi


def extract(im, dino, sky):
    def view(image):
        a = tensor(image)
        m = sky.run(None, {"input": a})[0].reshape(28, 28)
        pos = ((np.arange(16) + 0.5) * 28 / 16).astype(int)
        w = m[np.ix_(pos, pos)].ravel()
        t = dino.run(None, {"pixel_values": a})[0][0]
        return np.concatenate(
            [t[0], (w[:, None] * t[1:]).sum(0) / max(w.sum(), 1e-6)]
        ), m

    full, m = view(im)
    box = roi(m)
    if box and box != (0, 0, 28, 28):
        coords = tuple(v * s // 28 for v, s in zip(box, im.size * 2))
        patch, _ = view(im.crop(coords))
    else:
        patch = full
    return full, patch, float(m.mean())


if __name__ == "__main__":
    manifest = Path(sys.argv[1])
    cache = Path("gaze-cache")
    cache.mkdir(exist_ok=True)
    dino = session("dino.onnx")
    sky = session("sky.onnx")
    for filename in pd.read_csv(manifest).file:
        destination = cache / (filename.replace("/", "_") + ".npz")
        if destination.exists():
            continue
        with Image.open(filename) as image:
            full, patch, fraction = extract(image.convert("RGB"), dino, sky)
        np.savez(destination, full=full, patch=patch, sky=fraction)
