import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def fuse(source, settings, output):
    m = onnx.load(source)
    g = m.graph
    h = json.loads(Path(settings).read_text())
    g.input.append(
        helper.make_tensor_value_info("sky_mask", TensorProto.FLOAT, [1, 1, 28, 28])
    )

    def const(name, value):
        g.initializer.append(numpy_helper.from_array(np.asarray(value), name))

    def op(name, inputs, target, **kw):
        g.node.append(
            helper.make_node(name, inputs, [target], name="field_" + target, **kw)
        )

    idx = ((np.arange(16) + 0.5) * 28 / 16).astype("int64")
    const("field_indices", (idx[:, None] * 28 + idx[None, :]).ravel())
    const("field_shape", [-1])
    const("weight_shape", [1, 256, 1])
    const("cls_id", np.int64(0))
    const("patch_id", np.arange(1, 257, dtype="int64"))
    const("eps", np.float32(1e-6))
    const("field_W", np.array(h["weights"], "f").T.copy())
    const("field_b", np.array(h["bias"], "f"))
    op("Reshape", ["sky_mask", "field_shape"], "flat_mask")
    op("Gather", ["flat_mask", "field_indices"], "samples", axis=0)
    op("Reshape", ["samples", "weight_shape"], "weights")
    op("Gather", ["last_hidden_state", "cls_id"], "cls", axis=1)
    op("Gather", ["last_hidden_state", "patch_id"], "patches", axis=1)
    op("Mul", ["patches", "weights"], "weighted")
    op("ReduceSum", ["weighted"], "weighted_sum", axes=[1], keepdims=0)
    op("ReduceSum", ["samples"], "raw_sum", axes=[0], keepdims=0)
    op("Max", ["raw_sum", "eps"], "denominator")
    op("Div", ["weighted_sum", "denominator"], "pooled")
    op("Concat", ["cls", "pooled"], "embedding", axis=1)
    op("MatMul", ["embedding", "field_W"], "linear")
    op("Add", ["linear", "field_b"], "logits")
    op("Softmax", ["logits"], "scores", axis=1)
    del g.output[:]
    g.output.append(helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 3]))
    g.output.append(helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 768]))
    onnx.checker.check_model(m)
    onnx.save(m, output)


if __name__ == "__main__":
    fuse("dino.onnx", "dino_head.json", "dino_clouds.onnx")
