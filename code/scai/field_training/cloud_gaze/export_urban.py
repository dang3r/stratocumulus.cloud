"""Fuse the RBF and urban adaptation into a portable ONNX head."""

import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as T
from pathlib import Path
import sys


def export(basis, adaptation, out):
    b = np.load(basis)
    h = np.load(adaptation)
    centers = b["full"].T.copy()
    v = h["coef"].astype("f")
    scale = np.float32(abs(centers).max() / 127)
    q = np.rint(centers / scale).astype("int8")
    g = []
    params = []

    def const(name, data):
        params.append(N.from_array(np.asarray(data), name))
        return name

    def op(name, inputs, target, **kw):
        g.append(H.make_node(name, inputs, [target], **kw))
        return target

    centers = op(
        "DequantizeLinear",
        [const("centers_q", q), const("scale", scale), const("zero", np.int8(0))],
        "centers",
    )
    stack = op("Concat", ["full", "patch"], "views", axis=0)
    unit = op("LpNormalization", [stack], "unit", axis=1, p=2)
    dot = op("MatMul", [unit, centers], "dot")
    off = op("Sub", [dot, const("one", np.float32(1))], "cos_offset")
    rbf = op("Exp", [op("Mul", [off, const("gamma", b["gamma"])], "scaled")], "kernels")
    raw = op("MatMul", [rbf, const("raw_coef", b["coef"].T.copy())], "raw")
    pair = op(
        "Reshape", [raw, const("pair_shape", np.array([1, 2], dtype="int64"))], "pair"
    )
    z = op(
        "MatMul",
        [pair, const("alpha", np.array([[1 - v[-4]], [v[-4]]], dtype="f"))],
        "base",
    )
    flat = op(
        "Reshape",
        [unit, const("feature_shape", np.array([1, 1536], dtype="int64"))],
        "features",
    )
    residual = op("MatMul", [flat, const("linear_coef", v[:1536, None])], "linear")
    sq = op("Mul", ["sky", "sky"], "sky_sq")
    sky = op("Concat", [sq, "sky"], "sky_terms", axis=1)
    adj = op("MatMul", [sky, const("weather_coef", v[-3:-1, None])], "sky_score")
    combined = op(
        "Sum",
        [
            z,
            residual,
            adj,
            const("bias", np.float32(v[-1] - h["base0"] - h["threshold"])),
        ],
        "margin",
    )
    inputs = [
        H.make_tensor_value_info(n, T.FLOAT, [1, w])
        for n, w in [("full", 768), ("patch", 768), ("sky", 1)]
    ]
    model = H.make_model(
        H.make_graph(
            g,
            "Urban sky classifier",
            inputs,
            [H.make_tensor_value_info(combined, T.FLOAT, [1, 1])],
            params,
        ),
        opset_imports=[H.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out)


if __name__ == "__main__":
    export(*sys.argv[1:])
