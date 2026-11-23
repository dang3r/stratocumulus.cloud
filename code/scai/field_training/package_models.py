"""Assemble browser assets under the output directory after training and export."""

import argparse
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
photo = args.output / "photo"
video = args.output / "video"
photo.mkdir(parents=True, exist_ok=True)
video.mkdir(parents=True, exist_ok=True)
photo_head = json.loads(Path("dino_head.json").read_text())
settings = {k: photo_head[k] for k in ("threshold", "override")}
(photo / "settings.json").write_text(json.dumps(settings, separators=(",", ":")) + "\n")
shutil.copyfile("sky.onnx", photo / "sky.onnx")
shutil.copyfile("dino_clouds.onnx", photo / "clouds.onnx")
video_head = json.loads(Path("compact_head.json").read_text())
obstructed = json.loads(Path("augmented_head.json").read_text())
obstructed["threshold"] = 0.475
video_head.update(obstructed=obstructed, obstructedBelow=0.15)
(video / "settings.json").write_text(
    json.dumps(video_head, separators=(",", ":")) + "\n"
)
shutil.copyfile("sky_cloud.onnx", video / "sky.onnx")
shutil.copyfile("compact.onnx", video / "clouds.onnx")
