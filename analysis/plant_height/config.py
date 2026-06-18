"""Shared configuration for plant-height analysis scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

# Data root produced by utils/unzip_data.py
DEFAULT_DATA_ROOT = Path("D:/staged")

# Output layout under analysis/plant_height
# All step scripts write into these folders by default.
PLANT_HEIGHT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PLANT_HEIGHT_DIR / "processed"
FIGURES_DIR = PLANT_HEIGHT_DIR / "figures"
MODELS_DIR = PLANT_HEIGHT_DIR / "models"

PLY_DIR = PROCESSED_DIR / "ply"
METRICS_CSV = PROCESSED_DIR / "depth_analysis_metrics.csv"
PLOT_VALUES_CSV = PROCESSED_DIR / "depth_results_w1_w2_values.csv"
PLOT_FIGURE_PNG = FIGURES_DIR / "depth_results_w1_w2.png"
MODEL_INPUT_CSV = MODELS_DIR / "mixedlm_model_input.csv"
MODEL_SUMMARY_TXT = MODELS_DIR / "mixedlm_height_estimate_summary.txt"
MODEL_COEFFICIENTS_CSV = MODELS_DIR / "mixedlm_height_estimate_coefficients.csv"
MODEL_TRAJECTORY_PNG = MODELS_DIR / "mixedlm_height_estimate_trajectories.png"

# Scope: W1/W2 only (control vs drought)
ALLOWED_BEDS: Tuple[str, str] = ("W1", "W2")

# Default processing parameters
# These are defaults only; step/orchestrator CLI flags can override them per run.
DEFAULT_FRAME_INDEX = 7
DEFAULT_HOUR = 15
DEFAULT_EXG_THRESHOLD = 5.0
DEFAULT_SENSOR_HEIGHT_M = 0.60
DEFAULT_VEGETATION_PT_THRESHOLD = 1000
DEFAULT_START_DATE = "2025_09_01"
DEFAULT_END_DATE = "2025_09_14"
# Mixed-model stage often starts at intervention onset, later than extraction window.
DEFAULT_MODEL_START_DATE = "2025_09_03"


# ---------------------------------------------------------------------------
# Camera parameters and extrinsics copied from align_test.py
# ---------------------------------------------------------------------------
# Keep these synchronized with the validated alignment setup.

K_DEPTH = np.array(
    [
        [213.07864182105573, 0.0, 106.30867219370424],
        [0.0, 213.4773675713787, 87.48242329736486],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DEPTH_DISTORTION = np.array(
    [
        0.06563883079346727,
        -2.1462902300917204,
        0.0004996931251144205,
        -0.003320348250271367,
        3.229347799887664,
    ],
    dtype=np.float64,
)

K_RGB = np.array(
    [
        [1397.1693759311743, 0.0, 932.7568965495196],
        [0.0, 1391.8123785653943, 566.6211724978231],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

RGB_DISTORTION = np.array(
    [
        0.14612012441410802,
        -0.3064798846037176,
        0.0018885253236924308,
        -0.006218715146850731,
        0.2488477543929147,
    ],
    dtype=np.float64,
)

RVEC_DEPTH_TO_RGB = (-0.010361366746608086, 0.057755279059599554, 3.1366704147707054)
TVEC_DEPTH_TO_RGB = (0.0537813722846987, 0.04292643179470712, -0.0019552023110125907)

ROBOT_TO_DEPTH_YAW_DEG = 90.0
