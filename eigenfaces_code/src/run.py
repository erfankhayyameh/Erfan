from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from . import config as cfg
from .data import load_orl, split_identities_known_unknown, split_per_subject
from .eigenfaces import train_eigenfaces
from .eval import (
    choose_thresholds_grid_search,
    choose_thresholds_percentile,
    closed_set_accuracy,
    compute_distances,
    open_set_metrics,
)
from .openset import build_open_set_splits
from .utils import ensure_dir, save_json, set_seed


def plot_accuracy_vs_k(outdir: Path, ks: List[int], accs: List[float]) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(ks, accs, marker="o")
    plt.title("Closed-set accuracy vs #eigenfaces (K)")
    plt.xlabel("K (n_components)")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "accuracy_vs_k.png", dpi=200)
    plt.close()


def main() -> None:
    set_seed(cfg.SEED)
    ensure_dir(cfg.OUTDIR)

    ds = load_orl(
        root=cfg.DATA_ROOT,
        image_size=cfg.IMAGE_SIZE,
        gaussian_window=cfg.GAUSSIAN_WINDOW,
        normalize_input=cfg.NORMALIZE_INPUT,
    )

    # ORL per-subject split
    split = split_per_subject(
        y=ds.y,
        train_per_id=cfg.TRAIN_PER_ID,
        test_per_id=cfg.TEST_PER_ID,
        val_per_id=cfg.VAL_PER_ID if cfg.OPEN_SET else 0,
        seed=cfg.SEED,
    )

    x_train = ds.x[split.train_idx]
    y_train = ds.y[split.train_idx]
    x_test = ds.x[split.test_idx]
    y_test = ds.y[split.test_idx]

    # A) Sweep K -> closed-set accuracy
    sweep_rows: List[Dict] = []
    ks: List[int] = []
    accs: List[float] = []
    for k in cfg.K_SWEEP:
        model_k = train_eigenfaces(
            x_train=x_train,
            y_train=y_train,
            class_names=ds.class_names,
            n_components=int(k),
            normalize_input=cfg.NORMALIZE_INPUT,
        )
        m = closed_set_accuracy(model_k, x_test, y_test, normalize_input=cfg.NORMALIZE_INPUT)
        ks.append(int(k))
        accs.append(float(m["accuracy"]))
        sweep_rows.append({"K": int(k), "test_accuracy": float(m["accuracy"])})
    save_json(cfg.OUTDIR / "sweep_accuracy_vs_k.json", {"results": sweep_rows})
    plot_accuracy_vs_k(cfg.OUTDIR, ks, accs)

    # B) Main closed-set run (K = N_COMPONENTS)
    model = train_eigenfaces(
        x_train=x_train,
        y_train=y_train,
        class_names=ds.class_names,
        n_components=int(cfg.N_COMPONENTS),
        normalize_input=cfg.NORMALIZE_INPUT,
    )
    closed = closed_set_accuracy(model, x_test, y_test, normalize_input=cfg.NORMALIZE_INPUT)

    cm = closed["confusion_matrix"]
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix (closed-set test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(cfg.OUTDIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # C) Open-set evaluation (paper-style), automatic thresholds
    openset_report: Dict = {"enabled": bool(cfg.OPEN_SET)}
    if cfg.OPEN_SET:
        known_ids, unknown_ids = split_identities_known_unknown(ds.class_names, n_known=int(cfg.N_KNOWN_IDENTITIES), seed=cfg.SEED)

        osplits = build_open_set_splits(
            x=ds.x,
            y=ds.y,
            class_names=ds.class_names,
            known_ids=known_ids,
            unknown_ids=unknown_ids,
            train_idx=split.train_idx,
            val_idx=split.val_idx,
            test_idx=split.test_idx,
            n_non_face=int(cfg.N_NON_FACE),
            non_face_mode=str(cfg.NON_FACE_MODE),
            seed=cfg.SEED,
        )

        model_open = train_eigenfaces(
            x_train=osplits.x_train,
            y_train=osplits.y_train,
            class_names=osplits.known_class_names,
            n_components=int(cfg.N_COMPONENTS),
            normalize_input=cfg.NORMALIZE_INPUT,
        )

        # Validation distances (for auto threshold selection)
        if osplits.x_val_known.size:
            Ek_k, E_k, pred_k = compute_distances(model_open, osplits.x_val_known, normalize_input=cfg.NORMALIZE_INPUT)
        else:
            Ek_k, E_k, pred_k = np.array([]), np.array([]), np.array([])

        if osplits.x_val_unknown.size:
            Ek_u, E_u, _ = compute_distances(model_open, osplits.x_val_unknown, normalize_input=cfg.NORMALIZE_INPUT)
        else:
            Ek_u, E_u = np.array([]), np.array([])

        if osplits.x_val_nonface.size:
            _, E_n, _ = compute_distances(model_open, osplits.x_val_nonface, normalize_input=cfg.NORMALIZE_INPUT)
        else:
            E_n = np.array([])

        tuning: Dict = {"method": str(cfg.THRESHOLD_METHOD)}
        if cfg.THRESHOLD_METHOD == "percentile":
            theta_c, theta_f = choose_thresholds_percentile(
                Ek_val=Ek_k,
                E_val=E_k,
                theta_c_percentile=float(cfg.THETA_C_PERCENTILE),
                theta_f_percentile=float(cfg.THETA_F_PERCENTILE),
            )
            tuning.update({"theta_c_percentile": float(cfg.THETA_C_PERCENTILE), "theta_f_percentile": float(cfg.THETA_F_PERCENTILE)})
        elif cfg.THRESHOLD_METHOD == "grid_search":
            theta_c, theta_f, diag = choose_thresholds_grid_search(
                Ek_known=Ek_k, E_known=E_k, y_known=osplits.y_val_known, pred_known=pred_k,
                Ek_unk=Ek_u, E_unk=E_u, E_non=E_n,
                theta_c_grid=list(cfg.THETA_C_GRID),
                theta_f_grid=list(cfg.THETA_F_GRID),
                alpha_false_known=float(cfg.GRID_ALPHA_FALSE_KNOWN),
            )
            tuning.update(diag)
            tuning.update({"alpha_false_known": float(cfg.GRID_ALPHA_FALSE_KNOWN)})
        else:
            raise ValueError(f"Unknown THRESHOLD_METHOD: {cfg.THRESHOLD_METHOD}")

        metrics = open_set_metrics(
            model=model_open,
            x_known_test=osplits.x_test_known,
            y_known_test=osplits.y_test_known,
            x_unknown_test=osplits.x_test_unknown,
            x_nonface_test=osplits.x_test_nonface,
            normalize_input=cfg.NORMALIZE_INPUT,
            theta_c=float(theta_c),
            theta_f=float(theta_f),
        )

        openset_report = {
            "enabled": True,
            "known_identities": [ds.class_names[i] for i in known_ids],
            "unknown_identities": [ds.class_names[i] for i in unknown_ids],
            "non_face_mode": str(cfg.NON_FACE_MODE),
            "n_non_face": int(cfg.N_NON_FACE),
            "thresholds": {"theta_c": float(theta_c), "theta_f": float(theta_f)},
            "tuning": tuning,
            "metrics": metrics,
        }

    report = {
        "dataset": str(cfg.DATA_ROOT),
        "n_total": int(ds.x.shape[0]),
        "n_classes": int(len(ds.class_names)),
        "preprocess": {
            "image_size": int(cfg.IMAGE_SIZE),
            "gaussian_window": bool(cfg.GAUSSIAN_WINDOW),
            "normalize_input": bool(cfg.NORMALIZE_INPUT),
        },
        "split": {
            "train": int(len(split.train_idx)),
            "val": int(len(split.val_idx)),
            "test": int(len(split.test_idx)),
            "train_per_id": int(cfg.TRAIN_PER_ID),
            "val_per_id": int(cfg.VAL_PER_ID if cfg.OPEN_SET else 0),
            "test_per_id": int(cfg.TEST_PER_ID),
        },
        "model": {"n_components": int(cfg.N_COMPONENTS)},
        "closed_set_test_accuracy": float(closed["accuracy"]),
        "openset": openset_report,
        "artifacts": {
            "sweep_accuracy_vs_k": "sweep_accuracy_vs_k.json",
            "accuracy_vs_k_plot": "accuracy_vs_k.png",
            "confusion_matrix_plot": "confusion_matrix.png",
            "report_json": "report.json",
        },
    }

    save_json(cfg.OUTDIR / "report.json", report)
    print("Done. Outputs in:", cfg.OUTDIR.resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
