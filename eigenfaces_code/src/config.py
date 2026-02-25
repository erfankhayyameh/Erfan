from __future__ import annotations

from pathlib import Path
from typing import List, Literal

# =========================
# Paths
# =========================
DATA_ROOT: Path = Path(r"C:\Users\Daria\Desktop\eigenfaces\orl")  # ORL root containing s1..s40
OUTDIR: Path = Path("runs/final")

# =========================
# Reproducibility
# =========================
SEED: int = 123

# =========================
# Preprocess (paper: "centered and normalized"; exact normalization is an ambiguity)
# =========================
IMAGE_SIZE: int = 112
GAUSSIAN_WINDOW: bool = False
NORMALIZE_INPUT: bool = True

# =========================
# ORL split protocol (closed-set)
# =========================
TRAIN_PER_ID: int = 5
TEST_PER_ID: int = 5
VAL_PER_ID: int = 1
# used for threshold selection in open-set mode


# =========================
# PCA / Eigenfaces
# =========================
N_COMPONENTS: int =40
K_SWEEP: List[int] = [5, 10, 20, 30, 40, 60, 80, 100]

# =========================
# Open-set evaluation (paper-style known/unknown/not-face)
# =========================
OPEN_SET: bool = True
N_KNOWN_IDENTITIES: int = 30

# Non-face generation (ORL has no explicit non-face set) — evaluation aid only
NON_FACE_MODE: Literal["noise", "shuffle"] = "shuffle"
N_NON_FACE: int = 200

# =========================
# Threshold selection (NO manual thresholds)
# =========================
THRESHOLD_METHOD: Literal["grid_search", "percentile"] = "grid_search"

# Grid search option
THETA_C_GRID: List[float] = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
THETA_F_GRID: List[float] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
GRID_ALPHA_FALSE_KNOWN: float = 2.0

# Percentile option
THETA_F_PERCENTILE: float = 95.0
THETA_C_PERCENTILE: float = 95.0
