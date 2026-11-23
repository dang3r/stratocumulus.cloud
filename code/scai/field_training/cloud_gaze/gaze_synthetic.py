"""Render the frozen training pairs using ADE sky annotations and retained GAZE donors."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageOps

here = Path(__file__).resolve().parent
pairs = pd.read_csv(here / "gaze-synthetic.csv.gz")
donors = (
    pd.read_csv(here / "gaze-clean.csv.gz")
    .query("split=='train'")[["url", "file"]]
    .rename(columns={"file": "donor"})
)
pairs = pairs.merge(donors, on="url")
for batch in pq.ParquetFile("ade.parquet").iter_batches(batch_size=32):
    for record in batch.to_pylist():
        name = Path(record["image"]["path"]).with_suffix(".png").name
        rows = pairs[pairs.background == name]
        if rows.empty:
            continue
        scene = Image.open(io.BytesIO(record["image"]["bytes"])).convert("RGB")
        scene.thumbnail((480, 480))
        buf = io.BytesIO()
        scene.save(buf, format="JPEG")
        scene = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        mask = Image.open(io.BytesIO(record["annotation"]["bytes"])).resize(
            scene.size, Image.Resampling.NEAREST
        )
        alpha = Image.fromarray((np.asarray(mask) == 3).astype("uint8") * 255)
        for row in rows.itertuples():
            im = Image.open(row.donor).convert("RGB")
            region = im.crop(
                (im.width // 8, 0, 7 * im.width // 8, max(1, im.height // 2))
            )
            target = Path(row.file)
            target.parent.mkdir(exist_ok=True)
            Image.composite(ImageOps.fit(region, scene.size), scene, alpha).save(
                target, quality=93
            )
pairs.to_csv("gaze-synthetic.csv", index=False)
