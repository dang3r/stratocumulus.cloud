"""Render synthetic training and validation views from the frozen donor/background pairs."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageOps


def main():
    pairs = pd.read_csv(Path(__file__).with_name("facade-split.csv.gz"))
    groups = pairs.groupby("background")
    for batch in pq.ParquetFile("ade.parquet").iter_batches(batch_size=32):
        for record in batch.to_pylist():
            name = Path(record["image"]["path"]).with_suffix(".png").name
            if name not in groups.groups:
                continue
            scene = Image.open(io.BytesIO(record["image"]["bytes"])).convert("RGB")
            scene.thumbnail((480, 480))
            # Match the intermediate JPEG used to prepare the training backgrounds.
            buffer = io.BytesIO()
            scene.save(buffer, format="JPEG")
            scene = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
            mask = Image.open(io.BytesIO(record["annotation"]["bytes"])).resize(
                scene.size, Image.Resampling.NEAREST
            )
            alpha = Image.fromarray((np.asarray(mask) == 3).astype("uint8") * 255)
            for row in groups.get_group(name).itertuples():
                donor = Image.open(row.donor).convert("RGB")
                donor = donor.crop((0, 0, donor.width, int(0.65 * donor.height)))
                image = Image.composite(ImageOps.fit(donor, scene.size), scene, alpha)
                destination = Path(row.file)
                destination.parent.mkdir(exist_ok=True)
                image.save(destination, quality=90)


if __name__ == "__main__":
    main()
