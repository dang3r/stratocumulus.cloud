import numpy as np
from common import tensor


def roi(mask):
    m = (mask.reshape(28, 28) >= 0.5).astype(int)
    if m.mean() >= 0.65:
        return (0, 0, 28, 28)
    ii = np.pad(m, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    for w, h in [(28, 20), (24, 16), (20, 14), (16, 12), (12, 10), (10, 8)]:
        count = ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]
        y, x = np.unravel_index(count.argmax(), count.shape)
        if count[y, x] >= 0.8 * w * h:
            return (x, y, x + w, y + h)
    return None


def locate(gate, im):
    return roi(gate.run(None, {"input": tensor(im)})[0])
