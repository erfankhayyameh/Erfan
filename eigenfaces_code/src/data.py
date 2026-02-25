from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from .utils import ArrayF, preprocess, read_image_gray, set_seed


@dataclass(frozen=True)
class Dataset:
    x: ArrayF
    y: npt.NDArray[np.int64]
    class_names: List[str]
    paths: List[Path]


def load_orl(
    root: Path,
    image_size: int,
    gaussian_window: bool,
    normalize_input: bool,
    exts: Tuple[str, ...] = (".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"),
) -> Dataset:
    if not root.exists():
        raise FileNotFoundError(str(root))

    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]

    xs: List[ArrayF] = []
    ys: List[int] = []
    paths: List[Path] = []

    for ci, cdir in enumerate(class_dirs):
        files = sorted([fp for fp in cdir.iterdir() if fp.is_file() and fp.suffix.lower() in exts])
        for fp in files:
            arr = read_image_gray(fp)
            v = preprocess(arr, image_size=image_size, gaussian_window=gaussian_window, normalize_input=normalize_input)
            xs.append(v)
            ys.append(ci)
            paths.append(fp)

    if len(xs) == 0:
        raise FileNotFoundError(f"No images found under {root} (extensions searched: {exts})")

    x = np.stack(xs, axis=0).astype(np.float64)
    y = np.asarray(ys, dtype=np.int64)
    return Dataset(x=x, y=y, class_names=class_names, paths=paths)


@dataclass(frozen=True)
class Split:
    train_idx: npt.NDArray[np.int64]
    val_idx: npt.NDArray[np.int64]
    test_idx: npt.NDArray[np.int64]


def split_per_subject(y: npt.NDArray[np.int64], train_per_id: int, test_per_id: int, val_per_id: int, seed: int) -> Split:
    set_seed(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for c in np.unique(y):
        idx = np.where(y == c)[0].copy()
        np.random.shuffle(idx)

        need = train_per_id + test_per_id
        if len(idx) < need:
            raise ValueError(f"class {c} has {len(idx)} images, need at least {need}")

        train_block = idx[:train_per_id]
        test_block = idx[train_per_id:train_per_id + test_per_id]

        if val_per_id > 0:
            if val_per_id >= len(train_block):
                raise ValueError("val_per_id must be < train_per_id")
            val_block = train_block[:val_per_id]
            real_train = train_block[val_per_id:]
            val_idx.extend(val_block.tolist())
            train_idx.extend(real_train.tolist())
        else:
            train_idx.extend(train_block.tolist())

        test_idx.extend(test_block.tolist())

    return Split(
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
    )


def split_identities_known_unknown(class_names: List[str], n_known: int, seed: int) -> Tuple[List[int], List[int]]:
    set_seed(seed)
    ids = np.arange(len(class_names))
    np.random.shuffle(ids)
    known = sorted(ids[:n_known].tolist())
    unknown = sorted(ids[n_known:].tolist())
    return known, unknown
