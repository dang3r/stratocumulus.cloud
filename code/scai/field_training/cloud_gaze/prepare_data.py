"""Fetch images in a frozen manifest, matching the benchmark's resize and JPEG settings."""

import argparse
import csv
import gzip
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image, ImageOps

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest", type=Path)
args = parser.parse_args()
with gzip.open(args.manifest, "rt") as source:
    rows = list(csv.DictReader(source))


def fetch(row):
    path = Path(row["file"])
    if path.exists():
        return
    commons = bool(row.get("download")) and path.parent.name != "gallery-eval"
    response = requests.get(
        row.get("download") or row["url"],
        headers={
            "User-Agent": "CloudModelResearch/0.1 (https://github.com/dang3r/stratocumulus.cloud)"
        },
        timeout=45,
    )
    response.raise_for_status()
    with Image.open(io.BytesIO(response.content)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if path.parent.name == "commons-countries":
            image.thumbnail((500, 500))
        elif path.parent.name in ("annual-field", "annual21-field", "annual23-field"):
            image.thumbnail((720, 720))
        elif path.parent.name == "gallery-eval":
            image.thumbnail((640, 640))
        elif not commons and path.parent.name != "threshold-field":
            image.thumbnail((680, 680))
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(
            path,
            quality=94
            if commons
            else 95
            if path.parent.name == "threshold-field"
            else 93,
        )


with ThreadPoolExecutor(4) as pool:
    for _ in pool.map(fetch, rows):
        pass
