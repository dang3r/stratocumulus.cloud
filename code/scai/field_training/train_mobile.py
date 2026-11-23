import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from common import backbone, manifest, session, transform
from fit_head import fit_head
from PIL import Image
from regions import locate


def extract(model, files):
    paths = list(files)
    batches = []
    with torch.inference_mode():
        for start in range(0, len(paths), 32):
            images = [
                transform(Image.open(f).convert("RGB"))
                for f in paths[start : start + 32]
            ]
            batches.append(model(torch.stack(images)))
    return torch.cat(batches).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obstructed", action="store_true")
    obstructed = parser.parse_args().obstructed
    df = manifest()
    if obstructed:
        df = pd.concat(
            [df, pd.read_csv(Path(__file__).with_name("facade-split.csv.gz"))],
            ignore_index=True,
        )
    name = "augmented" if obstructed else "compact"
    base = backbone()
    x = extract(base, df.file)
    gate = session("sky_cloud.onnx")
    files = []
    ids = []
    usable = []
    z = x.copy()
    out = Path("compact_crops")
    out.mkdir(exist_ok=True)
    for i, r in enumerate(df.itertuples()):
        im = Image.open(r.file).convert("RGB")
        box = locate(gate, im)
        usable.append(box is not None)
        if box is None or tuple(box) == (0, 0, 28, 28):
            continue
        l, t, rr, b = box
        p = out / f"{i}.png"
        im.crop(
            (
                l * im.width // 28,
                t * im.height // 28,
                rr * im.width // 28,
                b * im.height // 28,
            )
        ).save(p)
        ids.append(i)
        files.append(p)
    z[ids] = extract(base, files)
    x = np.concatenate([x, z], axis=1)
    df["usable"] = usable
    fit_head(df, x, name, [0.005, 0.05, 0.5], 600, [0.65, 0.85, 1.01])
    torch.onnx.export(
        base,
        torch.zeros(1, 3, 224, 224),
        "compact.onnx",
        dynamo=False,
        input_names=["input"],
        output_names=["embedding"],
        opset_version=17,
    )


if __name__ == "__main__":
    main()
