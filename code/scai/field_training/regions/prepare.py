"""Recover the region-reviewed GLOBE photos and paired building scenes."""

import argparse
import csv
import io
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq
import requests
from PIL import Image, ImageOps

from make_reviewed import render

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "ade", type=Path, help="ADE20K validation parquet from the parent recipe"
)
args = parser.parse_args()
labels = root / "phone-regions.csv"
pairs = root / "region-pairs.csv"
Path("region-field").mkdir(exist_ok=True)
with labels.open() as source, open("phone-panels.csv", "w") as manifest:
    writer = csv.DictWriter(manifest, ["file", "panel", "label"])
    writer.writeheader()
    for row in csv.DictReader(source):
        url = row["url"]
        file = Path("region-field") / (Path(urlparse(url).path).parent.name + ".jpg")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(response.content)))
        im.convert("RGB").save(file, quality=95)
        writer.writerow(
            dict(
                file=file,
                panel="new",
                label=int("Stratocumulus" in row["observation_genera"].split("+")),
            )
        )

with pairs.open() as source:
    masks = {Path(row["mask"]) for row in csv.DictReader(source)}
Path("facades").mkdir(exist_ok=True)
for batch in pq.ParquetFile(args.ade).iter_batches(batch_size=32):
    for record in batch.to_pylist():
        mask = Path("facades") / Path(record["image"]["path"]).with_suffix(".png").name
        if mask in masks:
            bg = Image.open(io.BytesIO(record["image"]["bytes"])).convert("RGB")
            bg.thumbnail((480, 480))
            bg.save(mask.with_suffix(".jpg"))
            annotation = Image.open(io.BytesIO(record["annotation"]["bytes"]))
            annotation.resize(bg.size, Image.Resampling.NEAREST).save(mask)
render(labels, pairs, Path("reviewed-pairs"))

Path("annotated-patches").mkdir(exist_ok=True)
with labels.open() as source, open("oracle-panels.csv", "w") as manifest:
    writer = csv.DictWriter(manifest, ["file", "panel", "label", "url"])
    writer.writeheader()
    for row in csv.DictReader(source):
        if not row["left"]:
            continue
        name = Path(urlparse(row["url"]).path).parent.name + ".jpg"
        im = Image.open(Path("region-field") / name)
        box = tuple(
            round(float(row[key]) * size)
            for key, size in zip(("left", "top", "right", "bottom"), im.size * 2)
        )
        patch = Path("annotated-patches") / name
        im.crop(box).save(patch, quality=95)
        writer.writerow(
            dict(
                file=patch,
                panel="oracle",
                label=int(row["visual_label"] == "likely_sc"),
                url=row["url"],
            )
        )
