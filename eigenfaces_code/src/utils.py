from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image
import imageio.v2 as imageio

ArrayF = npt.NDArray[np.float64]


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_image_gray(path: Path) -> ArrayF:
    try:
        im = Image.open(path).convert("L")
        return np.asarray(im, dtype=np.float64)
    except Exception:
        arr = imageio.imread(path)
        if arr.ndim == 3:
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        return arr.astype(np.float64)


def resize_square(arr: ArrayF, size: int) -> ArrayF:
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float64)


def gaussian_window_2d(h: int, w: int, sigma_ratio: float = 0.35) -> ArrayF:
    y = np.linspace(-(h - 1) / 2.0, (h - 1) / 2.0, h)
    x = np.linspace(-(w - 1) / 2.0, (w - 1) / 2.0, w)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    sigma = sigma_ratio * max(h, w)
    g = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return g.astype(np.float64)


def preprocess(arr: ArrayF, image_size: int, gaussian_window: bool, normalize_input: bool) -> ArrayF:
    x = resize_square(arr, image_size)
    if gaussian_window:
        x = x * gaussian_window_2d(image_size, image_size)
    v = x.reshape(-1).astype(np.float64)

    # L2 normalization per sample (normalization type is a paper ambiguity)
    if normalize_input:
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
    return v


def save_json(path: Path, obj: Any) -> None:
    import json
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
