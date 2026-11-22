from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def manifest():
    return pd.read_csv(Path(__file__).with_name("split.csv.gz"))


def backbone():
    model = models.mobilenet_v3_small()
    model.load_state_dict(torch.load("pretrained.pth", map_location="cpu"))
    model.classifier = torch.nn.Identity()
    return model.eval()


def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def tensor(image):
    rgb = (
        np.asarray(
            image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        / 255
    )
    return (
        ((rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])
        .transpose(2, 0, 1)[None]
        .astype("float32")
    )
