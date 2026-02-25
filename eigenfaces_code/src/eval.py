from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix

from .eigenfaces import EigenfacesModel

ArrayF = npt.NDArray[np.float64]


def closed_set_accuracy(model: EigenfacesModel, x: ArrayF, y: npt.NDArray[np.int64], normalize_input: bool) -> Dict:
    preds: List[int] = []
    for i in range(x.shape[0]):
        _, w = model.project(x[i], normalize_input=normalize_input)
        k, _ = model.classify(w)
        preds.append(k)
    preds_a = np.asarray(preds, dtype=np.int64)
    acc = float(np.mean(preds_a == y))
    cm = confusion_matrix(y, preds_a, labels=list(range(len(model.class_names))))
    return {"accuracy": acc, "confusion_matrix": cm, "preds": preds_a}


def compute_distances(model: EigenfacesModel, x: ArrayF, normalize_input: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ek = np.zeros((x.shape[0],), dtype=np.float64)
    E = np.zeros((x.shape[0],), dtype=np.float64)
    pred = np.zeros((x.shape[0],), dtype=np.int64)
    for i in range(x.shape[0]):
        phi, w = model.project(x[i], normalize_input=normalize_input)
        k, ek = model.classify(w)
        e = model.distance_to_facespace(phi, w)
        Ek[i] = ek
        E[i] = e
        pred[i] = k
    return Ek, E, pred


def choose_thresholds_percentile(Ek_val: np.ndarray, E_val: np.ndarray, theta_c_percentile: float, theta_f_percentile: float) -> Tuple[float, float]:
    theta_c = float(np.percentile(Ek_val, theta_c_percentile))
    theta_f = float(np.percentile(E_val, theta_f_percentile))
    return theta_c, theta_f


def choose_thresholds_grid_search(
    Ek_known: np.ndarray,
    E_known: np.ndarray,
    y_known: np.ndarray,
    pred_known: np.ndarray,
    Ek_unk: np.ndarray,
    E_unk: np.ndarray,
    E_non: np.ndarray,
    theta_c_grid: List[float],
    theta_f_grid: List[float],
    alpha_false_known: float,
) -> Tuple[float, float, Dict]:
    best_score = -1e18
    best_tc, best_tf = float(theta_c_grid[0]), float(theta_f_grid[0])
    best_diag: Dict = {}

    for tc in theta_c_grid:
        for tf in theta_f_grid:
            known_is_known = (Ek_known < tc) & (E_known < tf)
            known_correct = known_is_known & (pred_known == y_known)
            known_correct_rate = float(np.mean(known_correct)) if known_correct.size else 0.0

            unk_false_known_rate = float(np.mean((Ek_unk < tc) & (E_unk < tf))) if Ek_unk.size else 0.0
            non_false_face_rate = float(np.mean(E_non < tf)) if E_non.size else 0.0

            score = known_correct_rate - alpha_false_known * unk_false_known_rate - alpha_false_known * non_false_face_rate
            if score > best_score:
                best_score = score
                best_tc, best_tf = float(tc), float(tf)
                best_diag = {
                    "objective": float(best_score),
                    "known_correct_rate": known_correct_rate,
                    "unknown_false_known_rate": unk_false_known_rate,
                    "nonface_false_face_rate": non_false_face_rate,
                }

    return best_tc, best_tf, best_diag


def open_set_metrics(
    model: EigenfacesModel,
    x_known_test: ArrayF,
    y_known_test: np.ndarray,
    x_unknown_test: ArrayF,
    x_nonface_test: ArrayF,
    normalize_input: bool,
    theta_c: float,
    theta_f: float,
) -> Dict:
    Ek_k, E_k, pred_k = compute_distances(model, x_known_test, normalize_input=normalize_input)
    known_is_known = (Ek_k < theta_c) & (E_k < theta_f)
    known_correct = known_is_known & (pred_k == y_known_test)
    known_correct_rate = float(np.mean(known_correct))
    known_reject_rate = float(1.0 - np.mean(known_is_known))

    if x_unknown_test.size:
        Ek_u, E_u, _ = compute_distances(model, x_unknown_test, normalize_input=normalize_input)
        unk_false_known_rate = float(np.mean((Ek_u < theta_c) & (E_u < theta_f)))
    else:
        unk_false_known_rate = 0.0
    unk_reject_rate = float(1.0 - unk_false_known_rate)

    if x_nonface_test.size:
        _, E_n, _ = compute_distances(model, x_nonface_test, normalize_input=normalize_input)
        nonface_reject_rate = float(np.mean(E_n >= theta_f))
    else:
        nonface_reject_rate = 0.0
    nonface_false_face_rate = float(1.0 - nonface_reject_rate)

    return {
        "theta_c": float(theta_c),
        "theta_f": float(theta_f),
        "known_correct_rate": known_correct_rate,
        "known_reject_rate": known_reject_rate,
        "unknown_reject_rate": unk_reject_rate,
        "unknown_false_known_rate": float(unk_false_known_rate),
        "nonface_reject_rate": nonface_reject_rate,
        "nonface_false_face_rate": nonface_false_face_rate,
    }
