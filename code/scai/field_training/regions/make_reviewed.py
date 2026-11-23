import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from urllib.parse import urlparse


def render(annotation_file, pair_file, output):
    labels = list(csv.DictReader(open(annotation_file)))
    pairs = list(csv.DictReader(open(pair_file)))[::2]
    positives = [r for r in labels if r["visual_label"] == "likely_sc"]
    controls = [r for r in labels if r["visual_label"] in ("clear", "other_cloud")]
    rows = []
    output.mkdir(exist_ok=True)
    for index, background in enumerate(pairs):
        mask = Path(background["mask"])
        scene = Image.open(mask.with_suffix(".jpg")).convert("RGB")
        alpha = Image.fromarray(
            (np.asarray(Image.open(mask)) == 3).astype("uint8") * 255
        )
        for label, donor in enumerate(
            [controls[index % len(controls)], positives[index % len(positives)]]
        ):
            photo = Path(urlparse(donor["url"]).path).parent.name
            im = Image.open(Path("region-field") / (photo + ".jpg"))
            box = [
                round(float(donor[key]) * dimension)
                for key, dimension in zip(
                    ("left", "top", "right", "bottom"), im.size * 2
                )
            ]
            patch = ImageOps.fit(im.crop(box), scene.size)
            destination = output / f"{index}-{label}.jpg"
            Image.composite(patch, scene, alpha).save(destination, quality=94)
            rows.append(
                dict(
                    file=destination,
                    label=label,
                    panel="reviewed-pairs",
                    url=donor["url"],
                    mask=mask,
                    sky_fraction=background["sky_fraction"],
                )
            )
    with open("reviewed-panels.csv", "w") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    import sys

    render(sys.argv[1], sys.argv[2], Path("reviewed-pairs"))
