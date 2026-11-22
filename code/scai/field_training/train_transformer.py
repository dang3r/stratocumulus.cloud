import numpy as np
from common import manifest, session, tensor
from fit_head import fit_head
from PIL import Image
from regions import roi


def embed(dino, sky, img):
    a = tensor(img)
    mask = sky.run(None, {"input": a})[0].reshape(28, 28)
    # Weight patch embeddings by their corresponding sky probabilities.
    index = ((np.arange(16) + 0.5) * 28 / 16).astype(int)
    weight = mask[np.ix_(index, index)].flatten()
    tok = dino.run(None, {"pixel_values": a})[0][0]
    mean = (tok[1:] * weight[:, None]).sum(0) / max(weight.sum(), 1e-6)
    return np.concatenate([tok[0], mean]), roi(mask) is not None


def main():
    df = manifest()
    ds = session("dino.onnx")
    sky = session("sky.onnx")
    xs = []
    usable = []
    for i, r in enumerate(df.itertuples()):
        f, u = embed(ds, sky, Image.open(r.file))
        xs.append(f)
        usable.append(u)
        if (i + 1) % 500 == 0:
            print("embedded", i + 1, flush=True)
    x = np.array(xs)
    np.save("dino_features.npy", x)
    df["usable"] = usable
    fit_head(df, x, "dino", [0.001, 0.01, 0.1], 500, [0.60, 0.85, 1.01])


if __name__ == "__main__":
    main()
