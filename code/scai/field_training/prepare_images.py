"""Materialize the frozen split from the public archives and original photo URLs."""

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageOps


def save(raw, destination, orient=False):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(raw)) as original:
        image = ImageOps.exif_transpose(original) if orient else original
        image = image.convert("RGB")
        image.thumbnail((640, 640))
        image.save(destination, quality=95)


def main():
    manifest = pd.read_csv(Path(__file__).with_name("split.csv.gz"))
    for source, archive in [("CCSN", "mmichelli.zip"), ("Howard", "imbikramsaha.zip")]:
        with zipfile.ZipFile(archive) as photos:
            for row in manifest[manifest.source == source].itertuples():
                save(photos.read(row.identifier), row.file)

    lookup = manifest[manifest.source == "ADE"].set_index("identifier").file
    for batch in pq.ParquetFile("ade.parquet").iter_batches(batch_size=32):
        for record in batch.to_pylist():
            image = record["image"]
            if image["path"] in lookup:
                save(image["bytes"], lookup[image["path"]])

    for row in manifest[manifest.source == "GLOBE"].itertuples():
        if not Path(row.file).exists():
            with urlopen(row.identifier, timeout=45) as response:
                save(response.read(), row.file, orient=True)


if __name__ == "__main__":
    main()
