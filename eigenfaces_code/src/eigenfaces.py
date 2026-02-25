from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

ArrayF = npt.NDArray[np.float64]


@dataclass
class EigenfacesModel:
    mean_face: ArrayF
    eigenfaces: ArrayF        # [K, D]
    prototypes: ArrayF        # [C, K]
    class_names: List[str]

    def project(self, x: ArrayF, normalize_input: bool) -> Tuple[ArrayF, ArrayF]:
        phi = x - self.mean_face
        if normalize_input:
            n = float(np.linalg.norm(phi))
            if n > 0:
                phi = phi / n
        w = self.eigenfaces @ phi
        return phi, w

    def classify(self, w: ArrayF) -> Tuple[int, float]:
        diffs = self.prototypes - w[None, :]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        k = int(np.argmin(dists))
        return k, float(dists[k])

    def distance_to_facespace(self, phi: ArrayF, w: ArrayF) -> float:
        proj = (w @ self.eigenfaces)  # [D]
        diff = phi - proj
        return float(diff @ diff)

    def decide(self, x: ArrayF, normalize_input: bool, theta_c: float, theta_f: float) -> Dict:
        # Paper-style two-threshold decision
        phi, w = self.project(x, normalize_input=normalize_input)
        k, ek = self.classify(w)
        e = self.distance_to_facespace(phi, w)
        if (ek < theta_c) and (e < theta_f):
            return {"decision": "known", "pred": k, "pred_name": self.class_names[k], "Ek": ek, "E": e}
        if e < theta_f:
            return {"decision": "unknown", "pred": k, "pred_name": None, "Ek": ek, "E": e}
        return {"decision": "not_face", "pred": k, "pred_name": None, "Ek": ek, "E": e}


def train_eigenfaces(
    x_train: ArrayF,
    y_train: npt.NDArray[np.int64],
    class_names: List[str],
    n_components: int,
    normalize_input: bool,
) -> EigenfacesModel:
    # Mean face
    x = x_train.astype(np.float64)
    M, D = x.shape
    mean_face = np.mean(x, axis=0)

    # A = centered (optionally normalized) training set
    A = x - mean_face[None, :]
    if normalize_input:
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        A = A / norms

    # Small MxM trick: L = A A^T
    L = A @ A.T
    evals, evecs = np.linalg.eigh(L)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]

    K = min(int(n_components), M - 1)
    V = evecs[:, :K]              # [M, K]

    # Eigenfaces (paper Eq. 6): u_k = sum_i v_{k,i} phi_i  -> U = V^T A
    U = (V.T @ A)                 # [K, D]
    Un = np.linalg.norm(U, axis=1, keepdims=True)
    Un = np.where(Un > 0, Un, 1.0)
    U = U / Un

    # Weights for training samples
    W = (U @ (x - mean_face[None, :]).T).T  # [M, K]

    # Prototypes: mean weight vector per class
    C = len(class_names)
    prototypes = np.zeros((C, K), dtype=np.float64)
    for c in range(C):
        idx = np.where(y_train == c)[0]
        if len(idx) > 0:
            prototypes[c] = np.mean(W[idx], axis=0)

    return EigenfacesModel(mean_face=mean_face, eigenfaces=U, prototypes=prototypes, class_names=class_names)
