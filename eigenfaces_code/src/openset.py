from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from .utils import ArrayF, set_seed


@dataclass(frozen=True)
class OpenSetSplits:
    x_train: ArrayF
    y_train: npt.NDArray[np.int64]
    x_val_known: ArrayF
    y_val_known: npt.NDArray[np.int64]
    x_test_known: ArrayF
    y_test_known: npt.NDArray[np.int64]
    x_val_unknown: ArrayF
    x_test_unknown: ArrayF
    x_val_nonface: ArrayF
    x_test_nonface: ArrayF
    known_class_names: List[str]


def make_non_faces(x_source: ArrayF, mode: str, n_samples: int, seed: int) -> ArrayF:
    set_seed(seed)
    D = x_source.shape[1]

    if mode == "noise":
        x = np.random.randn(n_samples, D).astype(np.float64)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return x / norms

    if mode == "shuffle":
        idx = np.random.randint(0, x_source.shape[0], size=(n_samples,))
        x = x_source[idx].copy()
        for i in range(n_samples):
            np.random.shuffle(x[i])
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return x / norms

    raise ValueError(f"Unknown NON_FACE_MODE: {mode}")


def build_open_set_splits(
    x: ArrayF,
    y: npt.NDArray[np.int64],
    class_names: List[str],
    known_ids: List[int],
    unknown_ids: List[int],
    train_idx: npt.NDArray[np.int64],
    val_idx: npt.NDArray[np.int64],
    test_idx: npt.NDArray[np.int64],
    n_non_face: int,
    non_face_mode: str,
    seed: int,
) -> OpenSetSplits:
    known_mask = np.isin(y, np.asarray(known_ids, dtype=np.int64))
    unknown_mask = np.isin(y, np.asarray(unknown_ids, dtype=np.int64))

    def take(idx, mask):
        sel = idx[mask[idx]]
        return x[sel], y[sel]

    x_train_k, y_train_k = take(train_idx, known_mask)
    x_val_k, y_val_k = take(val_idx, known_mask) if val_idx.size else (x[:0], y[:0])
    x_test_k, y_test_k = take(test_idx, known_mask)

    # Remap known labels to 0..Ck-1
    remap = {old: new for new, old in enumerate(known_ids)}
    y_train_k = np.asarray([remap[int(t)] for t in y_train_k], dtype=np.int64)
    y_val_k = np.asarray([remap[int(t)] for t in y_val_k], dtype=np.int64) if y_val_k.size else y_val_k.astype(np.int64)
    y_test_k = np.asarray([remap[int(t)] for t in y_test_k], dtype=np.int64)

    x_val_u = x[val_idx[unknown_mask[val_idx]]] if val_idx.size else x[:0]
    x_test_u = x[test_idx[unknown_mask[test_idx]]]

    x_non = make_non_faces(x, mode=non_face_mode, n_samples=n_non_face, seed=seed + 999)
    half = n_non_face // 2
    x_val_non = x_non[:half]
    x_test_non = x_non[half:]

    known_class_names = [class_names[i] for i in known_ids]

    return OpenSetSplits(
        x_train=x_train_k, y_train=y_train_k,
        x_val_known=x_val_k, y_val_known=y_val_k,
        x_test_known=x_test_k, y_test_known=y_test_k,
        x_val_unknown=x_val_u, x_test_unknown=x_test_u,
        x_val_nonface=x_val_non, x_test_nonface=x_test_non,
        known_class_names=known_class_names,
    )
