import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from common import backbone, transform
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F


def partition(name):
    n = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 100
    return "val" if n < 18 else "test" if n < 38 else "train"


class SkyFeatures(nn.Module):
    def __init__(self, detail):
        super().__init__()
        blocks = backbone().features
        self.low = blocks[:4]
        self.mid = blocks[4:7]
        self.deep = blocks[7:12] if detail else None

    def forward(self, x):
        a = self.low(x)
        middle = self.mid(a)
        b = F.interpolate(middle, size=(28, 28), mode="bilinear", align_corners=False)

        rgb = F.avg_pool2d(x, 8)
        parts = [a, b]
        if self.deep is not None:
            parts.append(
                F.interpolate(
                    self.deep(middle),
                    size=(28, 28),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return torch.cat([*parts, rgb], 1)


class Gate(nn.Module):
    def __init__(self, features, head):
        super().__init__()
        self.features = features
        self.head = head

    def forward(self, x):
        return self.head(self.features(x)).sigmoid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Train the larger sky mask used for live video",
    )
    detail = parser.parse_args().detail
    rng = np.random.default_rng(14)
    net = SkyFeatures(detail).eval()
    xs = []
    ys = []
    splits = []
    for batch in pq.ParquetFile("ade.parquet").iter_batches(batch_size=20):
        rows = batch.to_pylist()
        ims = [
            transform(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))
            for r in rows
        ]
        with torch.no_grad():
            features = net(torch.stack(ims)).numpy().transpose(0, 2, 3, 1)
        for r, f in zip(rows, features):
            m = np.array(
                Image.open(io.BytesIO(r["annotation"]["bytes"])).resize(
                    (28, 28), Image.Resampling.NEAREST
                )
            )
            valid = np.flatnonzero(m.flatten() != 0)
            if len(valid) == 0:
                continue
            ix = rng.choice(valid, min(96, len(valid)), replace=False)
            xs.append(f.reshape(784, -1)[ix])
            ys.append(m.flatten()[ix] == 3)
            splits.append(partition(r["image"]["path"]))
    train = np.array(splits) == "train"
    x = np.concatenate([v for v, t in zip(xs, train) if t])
    y = np.concatenate([v for v, t in zip(ys, train) if t])
    if detail:
        cloud = []
        for r in pd.read_csv(Path(__file__).with_name("weak-sky.csv")).itertuples():
            im = transform(Image.open(r.file).convert("RGB"))
            with torch.no_grad():
                f = net(im[None])[0].numpy().transpose(1, 2, 0)
            # Visually verified central, upper-sky rectangles. Edges and horizon excluded.
            cells = f[2:11, 5:23].reshape(-1, f.shape[-1])
            cloud.append(cells[rng.choice(len(cells), 80, replace=False)])
        cloud = np.concatenate(cloud)
        x = np.concatenate([x, cloud])
        y = np.r_[y, np.ones(len(cloud), dtype=bool)]
    clf = LogisticRegression(C=0.1, max_iter=300).fit(x, y)
    for sp in ["val", "test"]:
        x = np.concatenate([v for v, t in zip(xs, splits) if t == sp])
        y = np.concatenate([v for v, t in zip(ys, splits) if t == sp])
        p = clf.predict(x)
        print(sp, "sampled sky IoU", np.sum(p & y) / np.sum(p | y), flush=True)
    head = nn.Conv2d(clf.coef_.shape[1], 1, 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(clf.coef_[:, :, None, None]))
        head.bias.copy_(torch.tensor(clf.intercept_))
    torch.onnx.export(
        Gate(net, head).eval(),
        torch.zeros(1, 3, 224, 224),
        "sky_cloud.onnx" if detail else "sky.onnx",
        input_names=["input"],
        output_names=["sky"],
        dynamo=False,
        opset_version=17,
    )
