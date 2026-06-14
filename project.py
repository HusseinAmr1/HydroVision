from __future__ import annotations

import argparse  # CLI parser (train/predict/import-labels commands and flags).
import copy  # Cheap deep copies for cached metadata payloads.
import csv  # Read/write simple CSV manifests and queue files.
import hashlib  # Build content hashes to detect duplicate feedback samples.
import json  # Read/write JSON configs, reports, and prediction payloads.
import os  # OS-level helpers (env vars and path utilities).
import re  # Parse datetime tokens from filenames and metadata text.
import shutil  # Copy/restore model artifacts during promotion rollback/backup.
import subprocess  # Launch detached live-monitor window for training commands.
from collections import (
    defaultdict,
)  # Group discovered pairs by sensor/filename efficiently.
from dataclasses import dataclass  # Lightweight typed records for dataset entities.
from datetime import datetime, timedelta, timezone  # UTC timestamps and ETA windows.
from functools import lru_cache  # Cache expensive temporal lookup helpers.
from pathlib import Path  # Cross-platform path handling.
from typing import Any  # Flexible typing for mixed JSON/dict payloads.
from uuid import uuid4  # Generate unique IDs for predictions/samples.

import joblib  # Persist/load sklearn models.
import matplotlib.pyplot as plt  # Save visual previews (image/probability/mask).
import numpy as np  # Core numeric array operations.
import pandas as pd  # CSV aggregation and tabular processing.
import tifffile  # TIFF read/write for satellite images and masks.
from matplotlib.ticker import MaxNLocator, PercentFormatter
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
)  # Temporal risk model for short weather sequences.
from sklearn.linear_model import (
    LogisticRegression,
)  # Risk classifier + pixel classifier.
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    jaccard_score,
    precision_recall_fscore_support,
    roc_auc_score,
)  # Validation metrics.
from sklearn.model_selection import (
    StratifiedKFold,
)  # Stratified CV splits for risk model evaluation.
from sklearn.pipeline import Pipeline  # Reusable sklearn pipelines (scaler + model).
from sklearn.preprocessing import StandardScaler  # Scale tabular risk features.
from sklearn.utils.class_weight import (
    compute_sample_weight,
)  # Balance temporal model classes via sample weights.

try:
    from scipy import ndimage
    from scipy.spatial import ConvexHull, QhullError
except Exception:
    ndimage = None
    ConvexHull = None
    QhullError = Exception

try:
    from env_utils import (
        resolve_env_path,
        load_dotenv,
    )  # Portable env/file path resolution across machines.
except Exception:

    def resolve_env_path(
        env_var: str, *, base_dir: Path, default_relative: str | Path
    ) -> Path:
        raw = os.getenv(env_var, "").strip()
        rel = Path(default_relative)
        preferred = (base_dir / rel).resolve()
        cwd_candidate = (Path.cwd() / rel).resolve()
        if raw:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            if candidate.exists():
                return candidate
            if preferred.exists():
                return preferred
            if cwd_candidate.exists():
                return cwd_candidate
            return candidate
        if preferred.exists():
            return preferred
        if cwd_candidate.exists():
            return cwd_candidate
        return preferred

    def load_dotenv(path: Path, *, override: bool = False) -> dict[str, str]:
        _ = (path, override)
        return {}


from model_security import safe_joblib_load  # Safer joblib loading from trusted paths.

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None

# File map (high-level):
# 1) Data discovery/pairing
# 2) Segmentation/risk training helpers
# 3) Pipeline V3 runtime routing + inference helpers
# 4) Feedback + active learning utilities
# 5) Predict flow + reporting/audit

# ==============================
# Global Configuration
# ==============================
# All defaults used by CLI and GUI live here.
# Changing values in this section changes the default behavior everywhere.
PROJECT_BASE_DIR = Path(__file__).resolve().parent
TRAIN_LIVE_MONITOR_ACTIVE_ENV = "FLOOD_TRAIN_MONITOR_ACTIVE"
TRAIN_LIVE_MONITOR_DISABLE_ENV = "FLOOD_DISABLE_AUTO_TRAIN_MONITOR"
# Prefer project-local .env values for reproducible defaults in this workspace.
load_dotenv(PROJECT_BASE_DIR / ".env", override=True)


def _default_dataset_dir() -> Path:
    for name in ("dataset", "Dataset"):
        candidate = (PROJECT_BASE_DIR / name).resolve()
        if candidate.exists():
            return candidate
    return (PROJECT_BASE_DIR / "dataset").resolve()


DEFAULT_DATASET_DIR = _default_dataset_dir()


def _default_data_roots_from_env() -> list[str]:
    raw = os.getenv("FLOOD_DATA_ROOTS", "").strip()
    default_roots = [r"S1, France", r"S2, France", r"Sentinel1", r"Sentinel2"]
    if DEFAULT_DATASET_DIR.exists():
        default_roots = [
            str((DEFAULT_DATASET_DIR / "S1, France").resolve()),
            str((DEFAULT_DATASET_DIR / "S2, France").resolve()),
            str((DEFAULT_DATASET_DIR / "Sentinel1").resolve()),
            str((DEFAULT_DATASET_DIR / "Sentinel2").resolve()),
        ]
    if not raw:
        return default_roots
    tokens = [x.strip() for x in re.split(r"[;\r\n]+", raw) if x.strip()]
    return tokens if tokens else default_roots


def _normalize_data_roots(raw_roots: list[str]) -> list[str]:
    def _strip_outer_quotes(text: str) -> str:
        s = str(text).strip()
        if len(s) >= 2 and (
            (s.startswith('"') and s.endswith('"'))
            or (s.startswith("'") and s.endswith("'"))
        ):
            s = s[1:-1].strip()
        return s

    def _root_exists(text: str) -> bool:
        s = _strip_outer_quotes(text)
        if not s:
            return False
        p = Path(s)
        if p.exists():
            return True
        if not p.is_absolute():
            return (PROJECT_BASE_DIR / p).exists()
        return False

    cleaned = [_strip_outer_quotes(x) for x in (raw_roots or []) if str(x).strip()]
    if len(cleaned) < 2:
        return cleaned

    merged: list[str] = []
    i = 0
    while i < len(cleaned):
        token = cleaned[i]
        if i + 1 < len(cleaned):
            nxt = cleaned[i + 1]
            joined = f"{token} {nxt}".strip()
            if (not _root_exists(token)) and _root_exists(joined):
                token = joined
                i += 1
        merged.append(token)
        i += 1
    return merged


DEFAULT_DATA_ROOTS = _default_data_roots_from_env()
DEFAULT_WEATHER_CSV_RELATIVE: Path = (
    (DEFAULT_DATASET_DIR / "Final_Full_Data_Matched.csv")
    if DEFAULT_DATASET_DIR.exists()
    else Path("Final_Full_Data_Matched.csv")
)
DEFAULT_TEMPORAL_CSV_RELATIVE: Path = (
    (DEFAULT_DATASET_DIR / "ERA5_Final1_Combined.csv")
    if DEFAULT_DATASET_DIR.exists()
    else Path("ERA5_Final1_Combined.csv")
)
DEFAULT_CSV_PATH = str(
    resolve_env_path(
        "WEATHER_CSV_PATH",
        base_dir=PROJECT_BASE_DIR,
        default_relative=DEFAULT_WEATHER_CSV_RELATIVE,
    )
)
DEFAULT_TEST_IMAGES: list[str] = []
SENSOR_CHANNELS = {"S1": 2, "S2": 9}
DISCOVERY_ERROR_TYPES = {
    "missing_image_for_mask",
    "ambiguous_image_for_mask",
    "root_missing",
    "unknown_sensor_root",
}
WEATHER_FEATURE_NAMES = [
    "Temperature_mean",
    "Temperature_min",
    "Temperature_max",
    "tp_mean",
    "tp_max",
    "tp_sum",
    "runoff_mean",
    "runoff_max",
    "runoff_sum",
    "lat_grid_mean",
    "lon_grid_mean",
]
IMAGE_FEATURE_NAMES = ["pred_flood_ratio", "pred_prob_mean", "pred_prob_p90"]
DEFAULT_AUTO_TILING_PIXELS = 2_000_000
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_OVERLAP = 64
DEFAULT_PREDICT_BATCH_ROWS = 300_000
DEFAULT_RISK_THRESHOLD_PROFILE = "balanced"
RISK_THRESHOLD_PROFILES: dict[str, float] = {
    "early_warning": 0.35,
    "balanced": 0.50,
    "high_precision": 0.70,
}


def _env_float_clamped(
    name: str, default: float, *, min_value: float = 0.0, max_value: float = 1.0
) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return float(default)
    try:
        val = float(raw)
    except Exception:
        return float(default)
    return float(np.clip(val, min_value, max_value))


# Sensor-aware flood-presence policy:
# S1 (radar) is more trusted for direct flood presence cues.
# S2 (optical) uses a stricter gate to reduce false positives from visual artifacts.
SENSOR_POLICY_CONFIG: dict[str, dict[str, Any]] = {
    "S1": {
        # Calibrated on training tables to reduce over-triggered "detect=1".
        "presence_ratio_threshold": _env_float_clamped(
            "S1_PRESENCE_RATIO_THRESHOLD", 0.045
        ),
        "risk_threshold_offset": _env_float_clamped(
            "S1_RISK_THRESHOLD_OFFSET", 0.00, min_value=-1.0, max_value=1.0
        ),
        "combine_rule": "or",
    },
    "S2": {
        # Calibrated on training tables to reduce over-triggered "detect=1".
        "presence_ratio_threshold": _env_float_clamped(
            "S2_PRESENCE_RATIO_THRESHOLD", 0.05
        ),
        "risk_threshold_offset": _env_float_clamped(
            "S2_RISK_THRESHOLD_OFFSET", 0.05, min_value=-1.0, max_value=1.0
        ),
        "combine_rule": "and",
    },
}
DEFAULT_DRIFT_ZSCORE_THRESHOLD = 4.0
DEFAULT_CHANNEL_EPS = 1e-6
DEFAULT_AUDIT_ROTATE_MB = 20
MASK_FLOOD_POLICY_CHOICES = [
    "auto",
    "gt0",
    "class1",
    "class2",
    "class5",
    "class2_or_5",
]
DEFAULT_MASK_FLOOD_POLICY = (
    str(os.getenv("MASK_FLOOD_POLICY", "auto")).strip().lower() or "auto"
)
if DEFAULT_MASK_FLOOD_POLICY not in MASK_FLOOD_POLICY_CHOICES:
    DEFAULT_MASK_FLOOD_POLICY = "auto"
ACTIVE_MASK_FLOOD_POLICY = DEFAULT_MASK_FLOOD_POLICY
SEGMENTATION_MASK_SYNC_POLICY_CHOICES = ["strict", "event-window", "all"]
DEFAULT_SEGMENTATION_MASK_SYNC_POLICY = (
    str(os.getenv("SEGMENTATION_MASK_SYNC_POLICY", "strict"))
    .strip()
    .lower()
    .replace("_", "-")
    or "strict"
)
if DEFAULT_SEGMENTATION_MASK_SYNC_POLICY not in SEGMENTATION_MASK_SYNC_POLICY_CHOICES:
    DEFAULT_SEGMENTATION_MASK_SYNC_POLICY = "strict"
SEGMENTATION_IMAGE_BALANCE_POLICY_CHOICES = ["none", "equal-flood-non-flood"]
DEFAULT_SEGMENTATION_IMAGE_BALANCE_POLICY = (
    str(os.getenv("SEGMENTATION_IMAGE_BALANCE_POLICY", "equal-flood-non-flood"))
    .strip()
    .lower()
    .replace("_", "-")
    or "equal-flood-non-flood"
)
if DEFAULT_SEGMENTATION_IMAGE_BALANCE_POLICY not in SEGMENTATION_IMAGE_BALANCE_POLICY_CHOICES:
    DEFAULT_SEGMENTATION_IMAGE_BALANCE_POLICY = "equal-flood-non-flood"
DEFAULT_SEGMENTATION_BALANCE_MIN_FLOOD_RATIO = _env_float_clamped(
    "SEGMENTATION_BALANCE_MIN_FLOOD_RATIO",
    0.02,
    min_value=0.0,
    max_value=1.0,
)
PIPELINE_V3_BACKEND_ID = "pipeline_v3"
LEGACY_PIPELINE_BACKEND_IDS = {"unet"}
BACKEND_CHOICES = ["auto", PIPELINE_V3_BACKEND_ID]
PIPELINE_DISPLAY_NAMES = {
    "auto": "Auto",
    "legacy": "Pipeline V2",
    "unet": "Pipeline V3",
    PIPELINE_V3_BACKEND_ID: "Pipeline V3",
}


def normalize_pipeline_backend_id(value: str | None) -> str:
    key = str(value or "").strip().lower()
    if key in LEGACY_PIPELINE_BACKEND_IDS:
        return PIPELINE_V3_BACKEND_ID
    if key in BACKEND_CHOICES:
        return key
    return "auto"


def pipeline_display_name(value: str | None, *, default: str = "Pipeline") -> str:
    key = normalize_pipeline_backend_id(value)
    if not key:
        return default
    return PIPELINE_DISPLAY_NAMES.get(key, key.replace("_", " ").title())
# Supported segmentation model kinds for training/compare CLI.
DL_MODEL_CHOICES = [
    "small_unet",
    "fcn_resnet50",
    "deeplabv3_resnet50",
    "maskrcnn_resnet50",
    "segformer_b0",
    "segformer_b2",
    "smp_deeplabv3plus_resnet18",
    "smp_deeplabv3plus_resnet34",
    "smp_deeplabv3plus_resnet50",
    "smp_unet_efficientnet-b0",
    "smp_unet_efficientnet-b2",
    "smp_unet_inceptionresnetv2",
    "smp_unet_resnet18",
    "smp_unet_resnet34",
    "smp_unet_resnet50",
    "smp_unet_resnet101",
    "smp_unet_vgg16",
]
PIPELINE_MODEL_S1_NAME = "unet_model_s1.pth"
PIPELINE_MODEL_S2_NAME = "unet_model_s2.pth"
RISK_WITH_WEATHER_PIPELINE_NAME = "risk_model_with_weather_s1_unet.joblib"
RISK_NO_WEATHER_PIPELINE_NAME = "risk_model_no_weather_global_unet.joblib"
RISK_TEMPORAL_PIPELINE_NAME = "risk_model_temporal_gb_s1_unet.joblib"
RISK_TEMPORAL_METRICS_PIPELINE_NAME = "risk_temporal_cv_metrics_unet.json"
RISK_TEMPORAL_TABLE_PIPELINE_NAME = "risk_temporal_training_table_unet.csv"
ACTIVE_BACKEND_NAME = "active_backend.json"
PIPELINE_BEST_PROFILE_NAME = "unet_best_profile.json"
DATASET_METADATA_CSV_NAME = "dataset_image_metadata.csv"
DATASET_METADATA_SUMMARY_NAME = "dataset_image_metadata_summary.json"

TEMPORAL_WEATHER_FEATURE_NAMES = [
    "temp_last",
    "temp_mean",
    "temp_std",
    "temp_min",
    "temp_max",
    "temp_delta_last_first",
    "temp_trend_slope",
    "temp_recent3_mean",
    "temp_recent6_mean",
    "temp_recent12_mean",
    "tp_last",
    "tp_mean",
    "tp_std",
    "tp_max",
    "tp_sum",
    "tp_delta_last_first",
    "tp_trend_slope",
    "tp_recent3_sum",
    "tp_recent6_sum",
    "tp_recent12_sum",
    "tp_recent24_sum",
    "tp_recent3_mean",
    "tp_recent12_mean",
    "tp_recent3_to_12_ratio",
    "runoff_last",
    "runoff_mean",
    "runoff_std",
    "runoff_max",
    "runoff_sum",
    "runoff_delta_last_first",
    "runoff_trend_slope",
    "runoff_recent3_sum",
    "runoff_recent6_sum",
    "runoff_recent12_sum",
    "runoff_recent24_sum",
    "runoff_recent3_mean",
    "runoff_recent12_mean",
    "runoff_recent3_to_12_ratio",
    "seq_len",
    "date_span_hours",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "lat_grid_mean",
    "lon_grid_mean",
]
TEMPORAL_FEATURE_NAMES = list(TEMPORAL_WEATHER_FEATURE_NAMES) + list(
    IMAGE_FEATURE_NAMES
)
TEMPORAL_MODEL_TYPE_CHOICES = ["adaboost", "gradient_boosting", "lstm"]
TEMPORAL_COMPARE_MODEL_CHOICES = [
    "gradient_boosting",
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
    "adaboost",
    "logistic_regression",
    "mlp",
    "knn",
    "gaussian_nb",
    "svm_rbf",
    "lstm",
    "gru",
    "bilstm",
    "tcn",
    "xgboost",
    "lightgbm",
    "catboost",
]
TEMPORAL_DEEP_MODEL_CHOICES = ["lstm", "gru", "bilstm", "tcn"]
TEMPORAL_SEQUENCE_FEATURE_NAMES = [
    "temperature",
    "tp",
    "runoff",
    "lat_grid",
    "lon_grid",
]
TEMPORAL_LSTM_FEATURE_NAMES = list(TEMPORAL_SEQUENCE_FEATURE_NAMES) + list(
    IMAGE_FEATURE_NAMES
)
TEMPORAL_LSTM_MAX_SEQ_LEN = 32
TEMPORAL_LSTM_HIDDEN_SIZE = 32
TEMPORAL_LSTM_LAYERS = 1
TEMPORAL_LSTM_DROPOUT = 0.1
TEMPORAL_LSTM_EPOCHS = 140
TEMPORAL_LSTM_BATCH_SIZE = 16
TEMPORAL_LSTM_LR = 1e-3
TEMPORAL_LSTM_WEIGHT_DECAY = 1e-4
TEMPORAL_LSTM_PATIENCE = 16


def _resolve_temporal_n_jobs(default: int = -1) -> int:
    raw = os.getenv("TEMPORAL_N_JOBS", "").strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except Exception:
        return int(default)
    if value == 0:
        return int(default)
    return int(value)


# ==============================
# Core Dataset Structures
# ==============================
# PairRecord: one image + one ground-truth mask
# DiscoveryResult: full output of recursive data discovery
@dataclass(frozen=True)
class PairRecord:
    sensor: str
    root: Path
    image_path: Path
    mask_path: Path
    filename: str


@dataclass
class DiscoveryResult:
    pairs: list[PairRecord]
    issues: list[dict[str, Any]]
    root_summaries: list[dict[str, Any]]
    image_index: dict[str, list[Path]]
    pair_by_image: dict[Path, PairRecord]
    pairs_by_sensor: dict[str, list[PairRecord]]
    pairs_by_sensor_filename: dict[tuple[str, str], list[PairRecord]]


# ==============================
# CLI Wiring
# ==============================
# train-pipeline: builds segmentation + risk models
# predict: runs inference on one image
def add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-s1-path", type=str, default=None)
    parser.add_argument("--model-s2-path", type=str, default=None)
    parser.add_argument("--risk-model-with-weather-path", type=str, default=None)
    parser.add_argument("--risk-model-no-weather-path", type=str, default=None)
    parser.add_argument("--risk-model-temporal-path", type=str, default=None)
    parser.add_argument("--temporal-csv-path", type=str, default=None)
    parser.add_argument("--temporal-bridge-csv-path", type=str, default=None)
    parser.add_argument(
        "--allow-untrusted-model-paths",
        action="store_true",
        help="Allow loading external .joblib/.pkl model files outside trusted project roots.",
    )


def add_shared_mask_policy_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mask-flood-policy",
        type=str,
        choices=MASK_FLOOD_POLICY_CHOICES,
        default=DEFAULT_MASK_FLOOD_POLICY,
        help=(
            "How to convert mask classes into binary flood labels. "
            "'auto' uses class-aware mapping per dataset."
        ),
    )


def parse_args() -> argparse.Namespace:
    # Central CLI definition for all project workflows.
    # Keeping it in one place makes GUI/API/CLI behavior consistent.
    parser = argparse.ArgumentParser(
        description="Any-area flood prediction: segmentation + risk score with optional weather features."
    )
    subparsers = parser.add_subparsers(dest="command")

    benchmark_model_choices: list[str]
    try:
        from benchmark_models import IMAGE_BENCHMARK_MODEL_CHOICES

        benchmark_model_choices = [
            str(x).strip().lower()
            for x in IMAGE_BENCHMARK_MODEL_CHOICES
            if str(x).strip()
        ]
        if not benchmark_model_choices:
            benchmark_model_choices = list(DL_MODEL_CHOICES)
    except Exception:
        benchmark_model_choices = list(DL_MODEL_CHOICES)

    train_pipeline_parser = subparsers.add_parser(
        "train-pipeline",
        help="Train Pipeline V3 segmentation + risk models.",
    )
    train_pipeline_parser.add_argument(
        "--data-roots", nargs="+", default=DEFAULT_DATA_ROOTS
    )
    train_pipeline_parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV_PATH)
    train_pipeline_parser.add_argument("--output-dir", type=str, default="outputs")
    train_pipeline_parser.add_argument(
        "--test-images", nargs="*", default=DEFAULT_TEST_IMAGES
    )
    train_pipeline_parser.add_argument("--no-flood-roots", nargs="*", default=[])
    train_pipeline_parser.add_argument("--val-ratio", type=float, default=0.15)
    train_pipeline_parser.add_argument("--seed", type=int, default=42)
    train_pipeline_parser.add_argument("--threshold", type=float, default=0.5)
    train_pipeline_parser.add_argument("--epochs", type=int, default=20)
    train_pipeline_parser.add_argument("--early-stopping-patience", type=int, default=4)
    train_pipeline_parser.add_argument("--patch-size", type=int, default=384)
    train_pipeline_parser.add_argument("--stride", type=int, default=256)
    train_pipeline_parser.add_argument("--batch-size-s1", type=int, default=8)
    train_pipeline_parser.add_argument("--batch-size-s2", type=int, default=4)
    train_pipeline_parser.add_argument("--lr", type=float, default=1e-3)
    train_pipeline_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_pipeline_parser.add_argument("--max-patches-per-image", type=int, default=8)
    train_pipeline_parser.add_argument("--infer-batch-size", type=int, default=8)
    train_pipeline_parser.add_argument(
        "--model-kind", type=str, choices=DL_MODEL_CHOICES, default="small_unet"
    )
    train_pipeline_parser.add_argument(
        "--temporal-model-type",
        type=str,
        choices=TEMPORAL_MODEL_TYPE_CHOICES,
        default="adaboost",
    )
    train_pipeline_parser.add_argument(
        "--segmentation-mask-sync-policy",
        type=str,
        choices=SEGMENTATION_MASK_SYNC_POLICY_CHOICES,
        default=DEFAULT_SEGMENTATION_MASK_SYNC_POLICY,
        help=(
            "Which organized-dataset mask groups are trusted for segmentation. "
            "'strict' uses original image/mask pairs only; 'event-window' also uses "
            "near-event downloaded images; 'all' keeps legacy behavior."
        ),
    )
    train_pipeline_parser.add_argument(
        "--segmentation-source-groups",
        nargs="*",
        default=None,
        help=(
            "Optional explicit source_group allow-list for segmentation training. "
            "Overrides --segmentation-mask-sync-policy when provided."
        ),
    )
    train_pipeline_parser.add_argument(
        "--segmentation-balance-policy",
        type=str,
        choices=SEGMENTATION_IMAGE_BALANCE_POLICY_CHOICES,
        default=DEFAULT_SEGMENTATION_IMAGE_BALANCE_POLICY,
        help=(
            "Image-level balancing for segmentation training. "
            "'equal-flood-non-flood' downsamples the majority class per sensor."
        ),
    )
    train_pipeline_parser.add_argument(
        "--segmentation-balance-min-flood-ratio",
        type=float,
        default=DEFAULT_SEGMENTATION_BALANCE_MIN_FLOOD_RATIO,
        help=(
            "Minimum binary-mask flood ratio used to classify a training image "
            "as flood for image-level balancing."
        ),
    )
    train_pipeline_parser.add_argument("--temporal-csv-path", type=str, default=None)
    train_pipeline_parser.add_argument("--temporal-bridge-csv-path", type=str, default=None)
    train_pipeline_parser.add_argument(
        "--disable-auto-best-profile",
        action="store_true",
        help="Disable auto-loading the saved Pipeline V3 best-profile configuration.",
    )
    train_pipeline_parser.add_argument(
        "--no-live-monitor",
        action="store_true",
        help="Do not auto-open the live training monitor window.",
    )
    add_shared_mask_policy_arg(train_pipeline_parser)

    compare_parser = subparsers.add_parser(
        "compare-algorithms",
        help="Train multiple deep segmentation algorithms and generate a comparison report.",
    )
    compare_parser.add_argument("--data-roots", nargs="+", default=DEFAULT_DATA_ROOTS)
    compare_parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV_PATH)
    compare_parser.add_argument("--output-dir", type=str, default="outputs")
    compare_parser.add_argument("--test-images", nargs="*", default=DEFAULT_TEST_IMAGES)
    compare_parser.add_argument("--no-flood-roots", nargs="*", default=[])
    compare_parser.add_argument("--val-ratio", type=float, default=0.2)
    compare_parser.add_argument("--seed", type=int, default=42)
    compare_parser.add_argument("--threshold", type=float, default=0.5)
    compare_parser.add_argument("--epochs", type=int, default=15)
    compare_parser.add_argument("--epochs-list", nargs="+", type=int, default=None)
    compare_parser.add_argument("--early-stopping-patience", type=int, default=4)
    compare_parser.add_argument("--patch-size", type=int, default=384)
    compare_parser.add_argument("--stride", type=int, default=256)
    compare_parser.add_argument("--val-ratios", nargs="+", type=float, default=None)
    compare_parser.add_argument("--batch-size-s1", type=int, default=8)
    compare_parser.add_argument("--batch-size-s2", type=int, default=4)
    compare_parser.add_argument("--lr", type=float, default=1e-3)
    compare_parser.add_argument("--weight-decay", type=float, default=1e-4)
    compare_parser.add_argument("--max-patches-per-image", type=int, default=8)
    compare_parser.add_argument("--infer-batch-size", type=int, default=8)
    compare_parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=benchmark_model_choices,
        default=list(benchmark_model_choices),
    )
    compare_parser.add_argument(
        "--temporal-model-type",
        type=str,
        choices=TEMPORAL_MODEL_TYPE_CHOICES,
        default="adaboost",
    )
    compare_parser.add_argument("--temporal-csv-path", type=str, default=None)
    compare_parser.add_argument("--temporal-bridge-csv-path", type=str, default=None)
    compare_parser.add_argument(
        "--promote-best-by",
        type=str,
        choices=["none", "iou", "accuracy"],
        default="iou",
        help="Optionally copy best algorithm artifacts to the root output directory for dashboard usage.",
    )
    add_shared_mask_policy_arg(compare_parser)

    benchmark_parser = subparsers.add_parser(
        "train-benchmark",
        help="One-command full sweep across algorithms + epochs + train/validation ratios.",
    )
    benchmark_parser.add_argument("--data-roots", nargs="+", default=DEFAULT_DATA_ROOTS)
    benchmark_parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV_PATH)
    benchmark_parser.add_argument("--output-dir", type=str, default="outputs")
    benchmark_parser.add_argument(
        "--test-images", nargs="*", default=DEFAULT_TEST_IMAGES
    )
    benchmark_parser.add_argument("--no-flood-roots", nargs="*", default=[])
    benchmark_parser.add_argument("--seed", type=int, default=42)
    benchmark_parser.add_argument("--threshold", type=float, default=0.5)
    benchmark_parser.add_argument(
        "--epochs-list", nargs="+", type=int, default=[5, 10, 15]
    )
    benchmark_parser.add_argument("--early-stopping-patience", type=int, default=4)
    benchmark_parser.add_argument("--patch-size", type=int, default=384)
    benchmark_parser.add_argument("--stride", type=int, default=256)
    benchmark_parser.add_argument(
        "--val-ratios", nargs="+", type=float, default=[0.15, 0.20]
    )
    benchmark_parser.add_argument("--batch-size-s1", type=int, default=8)
    benchmark_parser.add_argument("--batch-size-s2", type=int, default=4)
    benchmark_parser.add_argument("--lr", type=float, default=1e-3)
    benchmark_parser.add_argument("--weight-decay", type=float, default=1e-4)
    benchmark_parser.add_argument("--max-patches-per-image", type=int, default=8)
    benchmark_parser.add_argument("--infer-batch-size", type=int, default=8)
    benchmark_parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=benchmark_model_choices,
        default=list(benchmark_model_choices),
    )
    benchmark_parser.add_argument(
        "--temporal-model-type",
        type=str,
        choices=TEMPORAL_MODEL_TYPE_CHOICES,
        default="adaboost",
    )
    benchmark_parser.add_argument("--temporal-csv-path", type=str, default=None)
    benchmark_parser.add_argument("--temporal-bridge-csv-path", type=str, default=None)
    benchmark_parser.add_argument(
        "--promote-best-by",
        type=str,
        choices=["none", "iou", "accuracy"],
        default="iou",
    )
    add_shared_mask_policy_arg(benchmark_parser)

    temporal_compare_parser = subparsers.add_parser(
        "compare-temporal-models",
        help="Benchmark multiple temporal forecasting algorithms and export comparison report/charts.",
    )
    temporal_compare_parser.add_argument("--output-dir", type=str, default="outputs")
    temporal_compare_parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV_PATH)
    temporal_compare_parser.add_argument("--base-table-path", type=str, default=None)
    temporal_compare_parser.add_argument("--temporal-csv-path", type=str, default=None)
    temporal_compare_parser.add_argument("--temporal-bridge-csv-path", type=str, default=None)
    temporal_compare_parser.add_argument(
        "--models",
        nargs="+",
        default=list(TEMPORAL_COMPARE_MODEL_CHOICES),
    )
    temporal_compare_parser.add_argument(
        "--epochs-list", nargs="+", type=int, default=[10, 20]
    )
    temporal_compare_parser.add_argument(
        "--val-ratios", nargs="+", type=float, default=[0.15, 0.20]
    )
    temporal_compare_parser.add_argument("--seed", type=int, default=42)
    temporal_compare_parser.add_argument("--batch-size", type=int, default=16)
    temporal_compare_parser.add_argument("--lr", type=float, default=1e-3)
    temporal_compare_parser.add_argument("--weight-decay", type=float, default=1e-4)
    temporal_compare_parser.add_argument("--hidden-size", type=int, default=48)
    temporal_compare_parser.add_argument("--dropout", type=float, default=0.10)
    temporal_compare_parser.add_argument("--patience", type=int, default=8)
    temporal_compare_parser.add_argument(
        "--max-seq-len", type=int, default=TEMPORAL_LSTM_MAX_SEQ_LEN
    )
    temporal_compare_parser.add_argument(
        "--score-metric",
        type=str,
        choices=["roc_auc", "f1", "accuracy", "iou"],
        default="roc_auc",
    )
    temporal_compare_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    import_labels_parser = subparsers.add_parser(
        "import-labels",
        help="Import labeled masks for feedback samples and mark them eligible for retraining.",
    )
    import_labels_parser.add_argument(
        "--csv", type=str, required=True, help="CSV with sample_id and label_mask_path."
    )
    import_labels_parser.add_argument("--output-dir", type=str, default="outputs")
    import_labels_parser.add_argument(
        "--strict", action="store_true", help="Fail if any row is invalid."
    )

    predict_parser = subparsers.add_parser(
        "predict", help="Predict flood mask + risk score for one image."
    )
    predict_parser.add_argument("--image", type=str, required=True)
    predict_parser.add_argument(
        "--sensor", type=str, choices=["S1", "S2"], default=None
    )
    predict_parser.add_argument("--data-roots", nargs="+", default=DEFAULT_DATA_ROOTS)
    predict_parser.add_argument("--output-dir", type=str, default="outputs")
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    predict_parser.add_argument(
        "--risk-threshold-profile",
        type=str,
        choices=sorted(RISK_THRESHOLD_PROFILES.keys()),
        default=DEFAULT_RISK_THRESHOLD_PROFILE,
    )
    predict_parser.add_argument("--risk-threshold", type=float, default=None)
    predict_parser.add_argument(
        "--auto-tiling-pixels", type=int, default=DEFAULT_AUTO_TILING_PIXELS
    )
    predict_parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    predict_parser.add_argument(
        "--tile-overlap", type=int, default=DEFAULT_TILE_OVERLAP
    )
    predict_parser.add_argument(
        "--predict-batch-rows", type=int, default=DEFAULT_PREDICT_BATCH_ROWS
    )
    predict_parser.add_argument("--strict-geospatial-checks", action="store_true")
    predict_parser.add_argument(
        "--drift-zscore-threshold", type=float, default=DEFAULT_DRIFT_ZSCORE_THRESHOLD
    )
    predict_parser.add_argument("--weather-json", type=str, default=None)
    predict_parser.add_argument("--weather-kv", nargs="*", default=None)
    predict_parser.add_argument(
        "--prediction-json-name", type=str, default="prediction.json"
    )
    predict_parser.add_argument("--disable-feedback-collection", action="store_true")
    predict_parser.add_argument("--feedback-output-dir", type=str, default=None)
    predict_parser.add_argument(
        "--backend", type=str, choices=BACKEND_CHOICES, default="auto"
    )
    predict_parser.add_argument(
        "--pipeline-patch-size",
        "--unet-patch-size",
        dest="pipeline_patch_size",
        type=int,
        default=None,
        help="Pipeline V3 inference patch size. The --unet-* alias is accepted for old scripts.",
    )
    predict_parser.add_argument(
        "--pipeline-stride",
        "--unet-stride",
        dest="pipeline_stride",
        type=int,
        default=None,
        help="Pipeline V3 inference stride. The --unet-* alias is accepted for old scripts.",
    )
    predict_parser.add_argument(
        "--pipeline-batch-size",
        "--unet-batch-size",
        dest="pipeline_batch_size",
        type=int,
        default=8,
        help="Pipeline V3 inference batch size. The --unet-* alias is accepted for old scripts.",
    )
    add_shared_mask_policy_arg(predict_parser)
    add_shared_model_args(predict_parser)

    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["train"])
    if hasattr(args, "data_roots") and isinstance(args.data_roots, list):
        args.data_roots = _normalize_data_roots(args.data_roots)
    if hasattr(args, "no_flood_roots") and isinstance(args.no_flood_roots, list):
        args.no_flood_roots = _normalize_data_roots(args.no_flood_roots)
    return args


def import_segmentation_pipeline() -> Any:
    try:
        import segmentation_pipeline
    except Exception as ex:
        raise RuntimeError(
            "segmentation_pipeline.py is required for Pipeline V3 train/predict flows. "
            "Keep segmentation_pipeline.py in the project folder and install dependencies via "
            "`pip install -r requirements-pipeline.txt`."
        ) from ex
    return segmentation_pipeline


# ==============================
# Path/Sensor/Discovery Helpers
# ==============================
# These utilities normalize TIFF layout, infer sensor type, and build
# deterministic issue reports during dataset scan.
def _to_channels_last(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image dimensions: {arr.shape}")
    if arr.shape[0] <= 16 and arr.shape[1] == arr.shape[2]:
        return np.moveaxis(arr, 0, -1)
    return arr


# ==============================
# Dataset Discovery
# ==============================
# These helpers scan arbitrary dataset roots, infer sensor identity from folder
# names, and build a deduplicated list of image/mask pairs that all downstream
# stages can reuse.
def infer_sensor_from_root(root: Path) -> str | None:
    text = str(root).upper()
    # Support common global-folder naming used in this project.
    compact = re.sub(r"[^A-Z0-9]+", "", text)
    if "SENTINEL1" in compact or "SENTINAL1" in compact:
        return "S1"
    if "SENTINEL2" in compact or "SENTINAL2" in compact:
        return "S2"
    if "S1" in text:
        return "S1"
    if "S2" in text:
        return "S2"
    return None


def make_issue(
    stage: str,
    issue_type: str,
    *,
    sensor: str | None = None,
    root: Path | None = None,
    filename: str | None = None,
    image_path: Path | None = None,
    mask_path: Path | None = None,
    details: str | None = None,
    candidate_count: int | None = None,
    candidates: list[Path] | None = None,
) -> dict[str, Any]:
    issue: dict[str, Any] = {"stage": stage, "issue_type": issue_type}
    if sensor is not None:
        issue["sensor"] = sensor
    if root is not None:
        issue["root"] = str(root)
    if filename is not None:
        issue["filename"] = filename
    if image_path is not None:
        issue["image_path"] = str(image_path)
    if mask_path is not None:
        issue["mask_path"] = str(mask_path)
    if details is not None:
        issue["details"] = details
    if candidate_count is not None:
        issue["candidate_count"] = candidate_count
    if candidates is not None:
        issue["candidates"] = "|".join(str(x) for x in candidates)
    return issue


def summarize_issue_types(issues: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for issue in issues or []:
        stage = str(issue.get("stage", "")).strip() or "unknown"
        issue_type = str(issue.get("issue_type", "")).strip() or "unknown"
        counts[f"{stage}:{issue_type}"] += 1
    return {k: int(v) for k, v in sorted(counts.items())}


def filter_issues_for_filenames(
    issues: list[dict[str, Any]], target_filenames: list[str] | None
) -> list[dict[str, Any]]:
    if not issues:
        return []
    if not target_filenames:
        return [dict(x) for x in issues]
    keep = {str(x).strip() for x in target_filenames if str(x).strip()}
    if not keep:
        return [dict(x) for x in issues]
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for raw in issues:
        issue = dict(raw)
        filename = str(issue.get("filename", "")).strip()
        if filename and filename not in keep:
            continue
        key = (
            str(issue.get("stage", "")),
            str(issue.get("issue_type", "")),
            filename,
            str(issue.get("details", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(issue)
    return out


def pack_issue_report(
    issues: list[dict[str, Any]], *, max_examples: int = 200
) -> dict[str, Any]:
    items = [dict(x) for x in (issues or [])]
    return {
        "issues_count": int(len(items)),
        "issue_type_counts": summarize_issue_types(items),
        "issues_truncated": bool(len(items) > int(max_examples)),
        "issues": items[: max(0, int(max_examples))],
    }


def has_part(path: Path, part_name: str) -> bool:
    target = part_name.lower()
    return any(p.lower() == target for p in path.parts)


def _norm_dir_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def is_mask_path(path: Path) -> bool:
    mask_tokens = {
        "mask",
        "masks",
        "flood",
        "floodmap",
        "floodmaps",
        "floodmask",
        "floodmasks",
        "label",
        "labels",
        "labelhand",
    }
    mask_prefixes = ("floodmap", "floodmask", "labelhand")
    for part in path.parts:
        norm = _norm_dir_name(part)
        if norm in mask_tokens or any(norm.startswith(x) for x in mask_prefixes):
            return True
    return False


def find_sensor_subroots(root: Path) -> list[Path]:
    out: list[Path] = []
    try:
        children = list(root.iterdir())
    except Exception:
        return out
    for child in children:
        if not child.is_dir():
            continue
        if has_part(child, "outputs"):
            continue
        if infer_sensor_from_root(child) is not None:
            out.append(child.resolve())
    out = sorted(set(out), key=lambda x: str(x).lower())
    return out


# Scan one root and resolve image<->mask pairs by filename.
def discover_pairs_for_root(
    root: Path,
) -> tuple[
    list[PairRecord], list[dict[str, Any]], dict[str, Any], dict[str, list[Path]]
]:
    issues: list[dict[str, Any]] = []
    root = root.resolve()
    sensor = infer_sensor_from_root(root)
    if sensor is None:
        issues.append(
            make_issue(
                "discovery",
                "unknown_sensor_root",
                root=root,
                details="root path does not contain S1/S2",
            )
        )
        summary = {
            "root": str(root),
            "sensor": None,
            "images_indexed": 0,
            "masks_found": 0,
            "pairs_valid": 0,
        }
        return [], issues, summary, {}

    all_tifs = list(root.rglob("*.tif"))
    image_index: dict[str, list[Path]] = defaultdict(list)
    mask_paths: list[Path] = []

    # Images and masks can live in different subfolders under the same root. Build
    # a filename index for images first, then match masks back against it.
    for p in all_tifs:
        if has_part(p, "outputs"):
            continue
        if is_mask_path(p):
            mask_paths.append(p)
            continue
        image_index[p.name].append(p.resolve())

    pairs: list[PairRecord] = []
    for mask_path in sorted(mask_paths, key=lambda x: str(x).lower()):
        candidates = sorted(
            image_index.get(mask_path.name, []), key=lambda x: str(x).lower()
        )
        if len(candidates) == 1:
            pairs.append(
                PairRecord(
                    sensor=sensor,
                    root=root,
                    image_path=candidates[0],
                    mask_path=mask_path.resolve(),
                    filename=mask_path.name,
                )
            )
        elif len(candidates) == 0:
            issues.append(
                make_issue(
                    "discovery",
                    "missing_image_for_mask",
                    sensor=sensor,
                    root=root,
                    filename=mask_path.name,
                    mask_path=mask_path,
                    candidate_count=0,
                )
            )
        else:
            issues.append(
                make_issue(
                    "discovery",
                    "ambiguous_image_for_mask",
                    sensor=sensor,
                    root=root,
                    filename=mask_path.name,
                    mask_path=mask_path,
                    candidate_count=len(candidates),
                    candidates=candidates,
                )
            )

    summary = {
        "root": str(root),
        "sensor": sensor,
        "images_indexed": int(sum(len(v) for v in image_index.values())),
        "unique_image_filenames": int(len(image_index)),
        "masks_found": int(len(mask_paths)),
        "pairs_valid": int(len(pairs)),
        "issues": int(len(issues)),
    }
    return pairs, issues, summary, image_index


# Merge all roots, deduplicate pairs, and build fast lookup indexes.
def discover_dataset(data_roots: list[Path]) -> DiscoveryResult:
    all_pairs: list[PairRecord] = []
    issues: list[dict[str, Any]] = []
    root_summaries: list[dict[str, Any]] = []
    merged_image_index: dict[str, list[Path]] = defaultdict(list)

    for raw_root in data_roots:
        root = raw_root.resolve()
        if not root.exists():
            issues.append(make_issue("discovery", "root_missing", root=root))
            root_summaries.append(
                {
                    "root": str(root),
                    "sensor": infer_sensor_from_root(root),
                    "images_indexed": 0,
                    "masks_found": 0,
                    "pairs_valid": 0,
                }
            )
            continue

        if infer_sensor_from_root(root) is None:
            # Some top-level roots are generic folders that contain separate S1/S2
            # subfolders. Expand those automatically instead of rejecting them.
            sensor_subroots = find_sensor_subroots(root)
            if sensor_subroots:
                expanded_root_summary = {
                    "root": str(root),
                    "sensor": None,
                    "images_indexed": 0,
                    "masks_found": 0,
                    "pairs_valid": 0,
                    "expanded_to": [str(x) for x in sensor_subroots],
                }
                root_summaries.append(expanded_root_summary)
                for subroot in sensor_subroots:
                    pairs, root_issues, root_summary, image_index = (
                        discover_pairs_for_root(subroot)
                    )
                    all_pairs.extend(pairs)
                    issues.extend(root_issues)
                    root_summaries.append(root_summary)
                    for name, paths in image_index.items():
                        merged_image_index[name].extend(paths)
                continue

        pairs, root_issues, root_summary, image_index = discover_pairs_for_root(root)
        all_pairs.extend(pairs)
        issues.extend(root_issues)
        root_summaries.append(root_summary)
        for name, paths in image_index.items():
            merged_image_index[name].extend(paths)

    deduped_pairs: list[PairRecord] = []
    seen = set()
    for pair in all_pairs:
        key = (pair.sensor, str(pair.image_path), str(pair.mask_path))
        if key in seen:
            continue
        seen.add(key)
        deduped_pairs.append(pair)

    image_index = {
        k: sorted(set(v), key=lambda x: str(x).lower())
        for k, v in merged_image_index.items()
    }
    pair_by_image: dict[Path, PairRecord] = {}
    pairs_by_sensor: dict[str, list[PairRecord]] = {"S1": [], "S2": []}
    pairs_by_sensor_filename: dict[tuple[str, str], list[PairRecord]] = defaultdict(
        list
    )

    for pair in sorted(
        deduped_pairs, key=lambda p: (p.sensor, p.filename, str(p.image_path))
    ):
        pair_by_image[pair.image_path] = pair
        pairs_by_sensor.setdefault(pair.sensor, []).append(pair)
        pairs_by_sensor_filename[(pair.sensor, pair.filename)].append(pair)

    return DiscoveryResult(
        pairs=deduped_pairs,
        issues=issues,
        root_summaries=root_summaries,
        image_index=image_index,
        pair_by_image=pair_by_image,
        pairs_by_sensor=pairs_by_sensor,
        pairs_by_sensor_filename=pairs_by_sensor_filename,
    )


# Discover optional no-flood image pools (images only, no masks) grouped by sensor.
# These are used to add negative samples to tile-level risk training.
def discover_no_flood_images(
    no_flood_roots: list[Path],
) -> tuple[dict[str, list[Path]], list[dict[str, Any]], list[dict[str, Any]]]:
    images_by_sensor: dict[str, list[Path]] = {"S1": [], "S2": []}
    issues: list[dict[str, Any]] = []
    root_summaries: list[dict[str, Any]] = []

    def collect_for_root(root: Path) -> None:
        sensor = infer_sensor_from_root(root)
        if sensor is None:
            issues.append(
                make_issue(
                    "no_flood_discovery",
                    "unknown_sensor_root",
                    root=root,
                    details="no-flood root path does not contain S1/S2",
                )
            )
            root_summaries.append(
                {
                    "root": str(root),
                    "sensor": None,
                    "images_found": 0,
                    "issues": 1,
                }
            )
            return
        count = 0
        for p in root.rglob("*.tif"):
            if has_part(p, "outputs"):
                continue
            if is_mask_path(p):
                continue
            images_by_sensor.setdefault(sensor, []).append(p.resolve())
            count += 1
        root_summaries.append(
            {
                "root": str(root),
                "sensor": sensor,
                "images_found": int(count),
                "issues": 0,
            }
        )

    for raw_root in no_flood_roots:
        root = Path(raw_root).resolve()
        if not root.exists():
            issues.append(make_issue("no_flood_discovery", "root_missing", root=root))
            root_summaries.append(
                {"root": str(root), "sensor": None, "images_found": 0, "issues": 1}
            )
            continue

        if infer_sensor_from_root(root) is None:
            sensor_subroots = find_sensor_subroots(root)
            if sensor_subroots:
                root_summaries.append(
                    {
                        "root": str(root),
                        "sensor": None,
                        "images_found": 0,
                        "issues": 0,
                        "expanded_to": [str(x) for x in sensor_subroots],
                    }
                )
                for subroot in sensor_subroots:
                    collect_for_root(subroot)
                continue

        collect_for_root(root)

    for sensor in ("S1", "S2"):
        deduped = sorted(
            set(images_by_sensor.get(sensor, [])), key=lambda x: str(x).lower()
        )
        images_by_sensor[sensor] = deduped

    return images_by_sensor, issues, root_summaries


# ==============================
# Dataset Metadata Export
# ==============================
# Metadata export builds a run-local inventory of every image seen during
# training. That inventory is later used by audits, UI status panels, and quick
# dataset statistics without rescanning raw TIFFs.
def _infer_hwc_from_tiff_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) == 2:
        return int(shape[0]), int(shape[1]), 1
    if len(shape) == 3:
        # Most satellite stacks are either HWC or CHW.
        if int(shape[0]) <= 16 and int(shape[1]) > 16 and int(shape[2]) > 16:
            return int(shape[1]), int(shape[2]), int(shape[0])
        if int(shape[2]) <= 16 and int(shape[0]) > 16 and int(shape[1]) > 16:
            return int(shape[0]), int(shape[1]), int(shape[2])
        return int(shape[0]), int(shape[1]), int(shape[2])
    # Fallback for uncommon TIFF layouts.
    if len(shape) >= 2:
        return int(shape[-2]), int(shape[-1]), 1
    return 0, 0, 0


def _safe_tiff_profile(path: Path) -> tuple[int, int, int, str, tuple[int, ...]]:
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        shape = tuple(int(x) for x in series.shape)
        dtype = str(series.dtype)
    h, w, c = _infer_hwc_from_tiff_shape(shape)
    return h, w, c, dtype, shape


def collect_dataset_image_metadata(
    *,
    discovery: DiscoveryResult,
    no_flood_roots: list[Path] | None = None,
    progress_callback: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen_images: set[Path] = set()
    no_flood_images: dict[str, list[Path]] = {}
    if no_flood_roots:
        no_flood_images, nf_issues, _ = discover_no_flood_images(no_flood_roots)
        issues.extend(nf_issues)

    total_candidates = int(len(discovery.pairs)) + int(
        sum(len(images) for images in no_flood_images.values())
    )
    processed_candidates = 0

    def emit_progress(*, sensor: str | None = None, filename: str | None = None) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback(
                done=int(processed_candidates),
                total=int(total_candidates),
                sensor=(str(sensor) if sensor else None),
                filename=(str(filename) if filename else None),
            )
        except Exception:
            pass

    def add_row(
        *,
        sensor: str,
        image_path: Path,
        mask_path: Path | None,
        sample_source: str,
    ) -> None:
        # Collect lightweight image/GeoTIFF profile data without loading full image
        # arrays unless a mask-derived flood ratio is also needed.
        resolved_image = image_path.resolve()
        if resolved_image in seen_images:
            return
        seen_images.add(resolved_image)

        try:
            height, width, channels, dtype, tiff_shape = _safe_tiff_profile(
                resolved_image
            )
            geo = inspect_geospatial_metadata(resolved_image)
            pixel_scale = geo.get("pixel_scale")
            tiepoint = geo.get("tiepoint")
            true_flood_ratio: float | None = None
            if mask_path is not None and mask_path.exists():
                try:
                    y_true = to_binary_mask(load_mask(mask_path), mask_path=mask_path)
                    true_flood_ratio = float(np.mean(y_true))
                except Exception as ex:
                    issues.append(
                        make_issue(
                            "metadata_export",
                            "mask_profile_failed",
                            sensor=sensor,
                            filename=resolved_image.name,
                            image_path=resolved_image,
                            mask_path=mask_path,
                            details=str(ex),
                        )
                    )

            rows.append(
                {
                    "sensor": sensor,
                    "filename": resolved_image.name,
                    "image_path": str(resolved_image),
                    "sample_source": sample_source,
                    "has_mask": int(mask_path is not None),
                    "mask_path": (
                        str(mask_path.resolve()) if mask_path is not None else ""
                    ),
                    "height": int(height),
                    "width": int(width),
                    "channels": int(channels),
                    "dtype": dtype,
                    "tiff_shape": json.dumps(list(tiff_shape)),
                    "has_geotiff_metadata": int(bool(geo.get("has_geotiff_metadata"))),
                    "epsg": str(geo.get("epsg") or ""),
                    "pixel_scale": (
                        json.dumps(pixel_scale)
                        if isinstance(pixel_scale, (list, tuple))
                        else ""
                    ),
                    "tiepoint": (
                        json.dumps(tiepoint)
                        if isinstance(tiepoint, (list, tuple))
                        else ""
                    ),
                    "geospatial_status": str(geo.get("status", "unknown")),
                    "geospatial_warning_count": int(len(geo.get("warnings", []) or [])),
                    "geospatial_warnings": "|".join(
                        str(x) for x in (geo.get("warnings", []) or [])
                    ),
                    "true_flood_ratio": true_flood_ratio,
                    "recorded_at_utc": utc_now_iso(),
                }
            )
        except Exception as ex:
            issues.append(
                make_issue(
                    "metadata_export",
                    "image_profile_failed",
                    sensor=sensor,
                    filename=resolved_image.name,
                    image_path=resolved_image,
                    mask_path=mask_path,
                    details=str(ex),
                )
            )

    emit_progress()
    for pair in sorted(
        discovery.pairs, key=lambda p: (p.sensor, p.filename, str(p.image_path))
    ):
        add_row(
            sensor=str(pair.sensor),
            image_path=pair.image_path,
            mask_path=pair.mask_path,
            sample_source="flood_pair",
        )
        processed_candidates += 1
        if (
            processed_candidates <= 1
            or processed_candidates >= total_candidates
            or processed_candidates % 10 == 0
        ):
            emit_progress(sensor=str(pair.sensor), filename=str(pair.filename))

    if no_flood_images:
        for sensor, images in no_flood_images.items():
            for image_path in images:
                add_row(
                    sensor=str(sensor),
                    image_path=image_path,
                    mask_path=None,
                    sample_source="no_flood_root",
                )
                processed_candidates += 1
                if (
                    processed_candidates <= 1
                    or processed_candidates >= total_candidates
                    or processed_candidates % 10 == 0
                ):
                    emit_progress(sensor=str(sensor), filename=str(image_path.name))

    by_sensor: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    with_mask = 0
    for row in rows:
        by_sensor[str(row["sensor"])] += 1
        by_source[str(row["sample_source"])] += 1
        with_mask += int(row.get("has_mask", 0))

    summary: dict[str, Any] = {
        "total_images": int(len(rows)),
        "with_mask_count": int(with_mask),
        "without_mask_count": int(len(rows) - with_mask),
        "by_sensor": {k: int(v) for k, v in sorted(by_sensor.items())},
        "by_source": {k: int(v) for k, v in sorted(by_source.items())},
        "issues_count": int(len(issues)),
    }
    return rows, issues, summary


def export_dataset_image_metadata(
    *,
    output_dir: Path,
    discovery: DiscoveryResult,
    no_flood_roots: list[Path] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    rows, issues, summary = collect_dataset_image_metadata(
        discovery=discovery,
        no_flood_roots=no_flood_roots,
        progress_callback=progress_callback,
    )
    # Export both row-level CSV and a compact JSON summary so callers can choose
    # between detailed analysis and cheap status reads.
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv_path = output_dir / DATASET_METADATA_CSV_NAME
    metadata_summary_path = output_dir / DATASET_METADATA_SUMMARY_NAME
    metadata_issues_path = output_dir / "dataset_metadata_issues.csv"

    write_csv(
        metadata_csv_path,
        rows,
        fieldnames=[
            "sensor",
            "filename",
            "image_path",
            "sample_source",
            "has_mask",
            "mask_path",
            "height",
            "width",
            "channels",
            "dtype",
            "tiff_shape",
            "has_geotiff_metadata",
            "epsg",
            "pixel_scale",
            "tiepoint",
            "geospatial_status",
            "geospatial_warning_count",
            "geospatial_warnings",
            "true_flood_ratio",
            "recorded_at_utc",
        ],
    )

    if issues:
        write_csv(
            metadata_issues_path,
            issues,
            fieldnames=[
                "stage",
                "issue_type",
                "sensor",
                "root",
                "filename",
                "image_path",
                "mask_path",
                "details",
                "candidate_count",
                "candidates",
            ],
        )
    else:
        with metadata_issues_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "stage",
                    "issue_type",
                    "sensor",
                    "root",
                    "filename",
                    "image_path",
                    "mask_path",
                    "details",
                    "candidate_count",
                    "candidates",
                ],
            )
            writer.writeheader()

    payload = {
        **summary,
        "metadata_csv_path": str(metadata_csv_path.resolve()),
        "metadata_summary_path": str(metadata_summary_path.resolve()),
        "metadata_issues_path": str(metadata_issues_path.resolve()),
    }
    save_json(metadata_summary_path, payload)
    return payload


# ==============================
# Data Loading / Split / Sampling
# ==============================
# This block reads TIFFs, binarizes masks, splits train/val, and samples
# representative pixels to keep memory bounded.
def load_image(path: Path, required_channels: int) -> np.ndarray:
    # Normalize TIFF layout to HWC and enforce the exact channel count expected by
    # the selected sensor/model path.
    arr = np.asarray(tifffile.imread(path))
    arr = _to_channels_last(arr)
    if arr.shape[-1] < required_channels:
        if required_channels == 2 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 2, axis=-1)
        else:
            raise ValueError(
                f"{path} has {arr.shape[-1]} channels but needs {required_channels}"
            )
    arr = arr[..., :required_channels].astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def load_mask(path: Path) -> np.ndarray:
    # Masks are forced to a simple 2D plane because all downstream segmentation
    # logic assumes one binary target per pixel.
    arr = np.asarray(tifffile.imread(path))
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D: {path}, got {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _normalize_mask_flood_policy(policy: str | None) -> str:
    candidate = (
        str(policy).strip().lower()
        if policy is not None
        else str(ACTIVE_MASK_FLOOD_POLICY).strip().lower()
    )
    if candidate not in MASK_FLOOD_POLICY_CHOICES:
        return DEFAULT_MASK_FLOOD_POLICY
    return candidate


def set_active_mask_flood_policy(policy: str | None) -> str:
    global ACTIVE_MASK_FLOOD_POLICY
    ACTIVE_MASK_FLOOD_POLICY = _normalize_mask_flood_policy(policy)
    return ACTIVE_MASK_FLOOD_POLICY


def _infer_auto_mask_flood_policy(mask: np.ndarray, mask_path: Path | None = None) -> str:
    if mask_path is not None:
        lower_path = str(mask_path).replace("\\", "/").lower()
        if "sen1floods" in lower_path:
            return "class1"
        # Current project roots come from STURM/EMS-style multi-class flood maps.
        # In those rasters, class 1 is the flooded-area target while classes 2-5
        # describe other water-related features (rivers/open water/reservoirs/lakes).
        if any(
            token in lower_path
            for token in (
                "dataset/s1, france/",
                "dataset/s2, france/",
                "dataset/sentinel1/",
                "dataset/sentinel2/",
                "sturm-flood",
                "sturm_flood",
            )
        ):
            return "class1"

    arr = np.asarray(mask)
    if arr.size == 0:
        return "gt0"
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.unique(np.rint(arr).astype(np.int64))
    if vals.size == 0:
        return "gt0"
    if np.any(vals < 0):
        return "class1" if int(np.any(vals == 1)) == 1 else "gt0"
    if np.all(np.isin(vals, [0, 1])):
        return "class1"
    positive_vals = {int(v) for v in vals.tolist() if int(v) > 0}
    if not positive_vals:
        return "gt0"
    if positive_vals == {2}:
        return "class2"
    if positive_vals == {5}:
        return "class5"
    if positive_vals == {2, 5}:
        return "class2_or_5"
    if positive_vals == {1}:
        return "class1"
    # Multi-class positive masks are safer to collapse to "any positive pixel"
    # than to arbitrarily pick one class id and silently drop the rest.
    if positive_vals & {3, 4}:
        return "gt0"
    return "gt0"


def to_binary_mask(
    mask: np.ndarray,
    *,
    policy: str | None = None,
    mask_path: Path | None = None,
) -> np.ndarray:
    # Convert project-specific multiclass mask conventions into one binary flood
    # target according to the active mask policy.
    arr = np.asarray(mask).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    effective_policy = _normalize_mask_flood_policy(policy)
    if effective_policy == "auto":
        effective_policy = _infer_auto_mask_flood_policy(arr, mask_path=mask_path)

    if effective_policy == "gt0":
        out = arr > 0.0
    elif effective_policy == "class1":
        out = np.isclose(arr, 1.0)
    elif effective_policy == "class2":
        out = np.isclose(arr, 2.0)
    elif effective_policy == "class5":
        out = np.isclose(arr, 5.0)
    elif effective_policy == "class2_or_5":
        out = np.logical_or(np.isclose(arr, 2.0), np.isclose(arr, 5.0))
    else:
        out = arr > 0.0
    return out.astype(np.uint8)


def split_pairs_for_sensor(
    pairs: list[PairRecord],
    test_filenames: set[str],
    val_ratio: float,
    seed: int,
) -> tuple[list[PairRecord], list[PairRecord]]:
    # Split at image level, not patch level, to avoid the same scene leaking into
    # both train and validation through different tiles.
    candidates = [p for p in pairs if p.filename not in test_filenames]
    if len(candidates) < 1:
        raise RuntimeError(
            f"Not enough files to split for sensor {pairs[0].sensor if pairs else 'unknown'}"
        )
    if val_ratio <= 0:
        train_pairs = sorted(candidates, key=lambda p: p.filename)
        return train_pairs, []
    if len(candidates) < 2:
        raise RuntimeError(
            f"Not enough files to create validation split for sensor {pairs[0].sensor if pairs else 'unknown'}"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    val_count = max(1, int(round(len(candidates) * val_ratio)))
    if val_count >= len(candidates):
        val_count = len(candidates) - 1
    val_pairs = sorted(candidates[:val_count], key=lambda p: p.filename)
    train_pairs = sorted(candidates[val_count:], key=lambda p: p.filename)
    return train_pairs, val_pairs


def stratified_sample_indices(
    y: np.ndarray, max_samples: int, rng: np.random.Generator
) -> np.ndarray:
    n_total = y.size
    if max_samples <= 0 or max_samples >= n_total:
        return np.arange(n_total)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return rng.choice(n_total, size=max_samples, replace=False)

    n_pos = min(pos_idx.size, max_samples // 2)
    n_neg = min(neg_idx.size, max_samples - n_pos)
    sel_pos = (
        rng.choice(pos_idx, size=n_pos, replace=False)
        if n_pos > 0
        else np.empty(0, dtype=int)
    )
    sel_neg = (
        rng.choice(neg_idx, size=n_neg, replace=False)
        if n_neg > 0
        else np.empty(0, dtype=int)
    )
    selected = np.concatenate([sel_pos, sel_neg])

    remain = max_samples - selected.size
    if remain > 0:
        all_idx = np.arange(n_total)
        left = np.setdiff1d(all_idx, selected, assume_unique=False)
        if left.size > 0:
            fill = rng.choice(left, size=min(remain, left.size), replace=False)
            selected = np.concatenate([selected, fill])
    rng.shuffle(selected)
    return selected


def predict_mask(
    model: Pipeline, x_img: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    h, w, c = x_img.shape
    probs = (
        model.predict_proba(x_img.reshape(-1, c))[:, 1].astype(np.float32).reshape(h, w)
    )
    preds = (probs >= threshold).astype(np.uint8)
    return preds, probs


def _predict_probs_in_batches(
    model: Pipeline, x_flat: np.ndarray, batch_rows: int
) -> np.ndarray:
    n = x_flat.shape[0]
    if batch_rows <= 0 or n <= batch_rows:
        return model.predict_proba(x_flat)[:, 1].astype(np.float32)
    out = np.empty((n,), dtype=np.float32)
    for start in range(0, n, batch_rows):
        end = min(start + batch_rows, n)
        out[start:end] = model.predict_proba(x_flat[start:end])[:, 1].astype(np.float32)
    return out


# Compute tile start positions with overlap so the full scene is covered.
def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size >= length:
        return [0]
    stride = max(1, tile_size - max(0, overlap))
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = max(0, length - tile_size)
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


# Memory-safe inference for very large scenes: predict tile by tile, then blend.
def predict_mask_tiled(
    model: Pipeline,
    x_img: np.ndarray,
    threshold: float,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    batch_rows: int = DEFAULT_PREDICT_BATCH_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    h, w, c = x_img.shape
    tile_size = max(64, int(tile_size))
    tile_overlap = max(0, int(tile_overlap))
    if tile_size <= tile_overlap:
        tile_overlap = max(0, tile_size // 4)

    y_starts = _tile_starts(h, tile_size, tile_overlap)
    x_starts = _tile_starts(w, tile_size, tile_overlap)

    prob_sum = np.zeros((h, w), dtype=np.float32)
    prob_count = np.zeros((h, w), dtype=np.float32)

    for y0 in y_starts:
        y1 = min(h, y0 + tile_size)
        for x0 in x_starts:
            x1 = min(w, x0 + tile_size)
            tile = x_img[y0:y1, x0:x1, :]
            flat = tile.reshape(-1, c)
            probs_flat = _predict_probs_in_batches(model, flat, batch_rows=batch_rows)
            probs_tile = probs_flat.reshape(y1 - y0, x1 - x0)
            prob_sum[y0:y1, x0:x1] += probs_tile
            prob_count[y0:y1, x0:x1] += 1.0

    probs = prob_sum / np.clip(prob_count, 1.0, None)
    preds = (probs >= threshold).astype(np.uint8)
    return preds, probs


# Auto-switch between direct inference and tiled inference by image size.
def predict_mask_auto(
    model: Pipeline,
    x_img: np.ndarray,
    threshold: float,
    *,
    auto_tiling_pixels: int = DEFAULT_AUTO_TILING_PIXELS,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    batch_rows: int = DEFAULT_PREDICT_BATCH_ROWS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    h, w, _ = x_img.shape
    n_pixels = int(h * w)
    use_tiling = bool(auto_tiling_pixels > 0 and n_pixels > auto_tiling_pixels)
    if use_tiling:
        pred_mask, pred_prob = predict_mask_tiled(
            model=model,
            x_img=x_img,
            threshold=threshold,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            batch_rows=batch_rows,
        )
        meta = {
            "mode": "tiled",
            "image_pixels": n_pixels,
            "auto_tiling_pixels": int(auto_tiling_pixels),
            "tile_size": int(tile_size),
            "tile_overlap": int(tile_overlap),
            "batch_rows": int(batch_rows),
        }
        return pred_mask, pred_prob, meta

    pred_mask, pred_prob = predict_mask(model, x_img, threshold)
    meta = {
        "mode": "direct",
        "image_pixels": n_pixels,
        "auto_tiling_pixels": int(auto_tiling_pixels),
        "tile_size": int(tile_size),
        "tile_overlap": int(tile_overlap),
        "batch_rows": int(batch_rows),
    }
    return pred_mask, pred_prob, meta


# ==============================
# Metrics, Validation, and File Outputs
# ==============================
# Centralized metric/report helpers used by both train and predict flows.
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(jaccard_score(y_true, y_pred, average="binary", zero_division=0)),
        "flood_ratio_true": float(np.mean(y_true)),
        "flood_ratio_pred": float(np.mean(y_pred)),
    }


def evaluate_pairs(
    model: Pipeline,
    pairs: list[PairRecord],
    required_channels: int,
    threshold: float,
) -> tuple[
    dict[str, float],
    list[dict[str, Any]],
    int,
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for pair in pairs:
        try:
            x_img = load_image(pair.image_path, required_channels)
            y_true = to_binary_mask(
                load_mask(pair.mask_path), mask_path=pair.mask_path
            )
            y_pred, _ = predict_mask(model, x_img, threshold)
        except Exception as ex:
            issues.append(
                make_issue(
                    "validation",
                    "pair_eval_failed",
                    sensor=pair.sensor,
                    root=pair.root,
                    filename=pair.filename,
                    image_path=pair.image_path,
                    mask_path=pair.mask_path,
                    details=str(ex),
                )
            )
            continue

        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        row = {
            "sensor": pair.sensor,
            "filename": pair.filename,
            "image_path": str(pair.image_path),
            "mask_path": str(pair.mask_path),
            "pixels": int(y_true_flat.size),
        }
        row.update(compute_metrics(y_true_flat, y_pred_flat))
        rows.append(row)
        y_true_all.append(y_true_flat)
        y_pred_all.append(y_pred_flat)

    if not y_true_all:
        return (
            {},
            rows,
            0,
            issues,
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.uint8),
        )

    y_true_concat = np.concatenate(y_true_all)
    y_pred_concat = np.concatenate(y_pred_all)
    metrics = compute_metrics(y_true_concat, y_pred_concat)
    return metrics, rows, int(y_true_concat.size), issues, y_true_concat, y_pred_concat


def save_json(path: Path, payload: dict[str, Any], *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as f:
            if compact:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            else:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(temp_path, path)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


def write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None
) -> None:
    cols: list[str] = list(fieldnames or [])
    if not cols:
        for row in rows:
            for key in row.keys():
                if key not in cols:
                    cols.append(key)
    if not cols:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def save_preview(
    path: Path, image: np.ndarray, prob: np.ndarray, pred: np.ndarray
) -> None:
    def norm(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-6)

    def norm_prob_vis(x: np.ndarray) -> np.ndarray:
        # Contrast-stretch probability maps for human-friendly preview.
        x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        lo = float(np.quantile(x, 0.02))
        hi = float(np.quantile(x, 0.98))
        if hi - lo < 1e-6:
            lo = float(np.min(x))
            hi = float(np.max(x))
        if hi - lo < 1e-6:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    c0 = norm(image[..., 0])
    c1 = norm(image[..., 1]) if image.shape[-1] > 1 else c0
    prob_vis = norm_prob_vis(prob)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(c0, cmap="gray")
    axes[0].set_title("Channel 0")
    axes[0].axis("off")
    axes[1].imshow(c1, cmap="gray")
    axes[1].set_title("Channel 1")
    axes[1].axis("off")
    axes[2].imshow(prob_vis, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("Flood Probability (enhanced)")
    axes[2].axis("off")
    axes[3].imshow(pred, cmap="Blues", vmin=0, vmax=1)
    axes[3].set_title("Detected Flood Mask")
    axes[3].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==============================
# Inference Input Resolution
# ==============================
# Helpers that map requested test names to actual files and detect sensor.
def resolve_test_images(
    test_images: list[str],
    image_index: dict[str, list[Path]],
) -> tuple[list[Path], list[dict[str, Any]]]:
    resolved: list[Path] = []
    issues: list[dict[str, Any]] = []
    for item in test_images:
        p = Path(item)
        if p.exists():
            resolved.append(p.resolve())
            continue

        key = p.name
        candidates = sorted(image_index.get(key, []), key=lambda x: str(x).lower())
        if len(candidates) == 1:
            resolved.append(candidates[0])
        elif len(candidates) > 1:
            preferred = sorted(
                candidates,
                key=lambda x: (0 if "S1" in str(x).upper() else 1, str(x).lower()),
            )[0]
            resolved.append(preferred)
            issues.append(
                make_issue(
                    "test_resolve",
                    "ambiguous_test_name",
                    filename=key,
                    candidate_count=len(candidates),
                    candidates=candidates,
                    details=f"picked {preferred}",
                )
            )
        else:
            issues.append(
                make_issue("test_resolve", "missing_test_image", filename=item)
            )

    unique_paths: list[Path] = []
    seen = set()
    for p in resolved:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        unique_paths.append(p)
    return unique_paths, issues


# Sensor detection prefers channels, then falls back to path roots.
def detect_sensor_for_image(
    path: Path, roots_with_sensor: list[tuple[Path, str]]
) -> str | None:
    rp = path.resolve()
    matched_roots: list[tuple[int, str]] = []
    for root, sensor in roots_with_sensor:
        try:
            if rp.is_relative_to(root.resolve()):
                matched_roots.append((len(root.resolve().parts), sensor))
        except Exception:
            continue
    root_sensor = None
    if matched_roots:
        matched_roots.sort(key=lambda x: x[0], reverse=True)
        root_sensor = matched_roots[0][1]

    arr = np.asarray(tifffile.imread(path))
    arr = _to_channels_last(arr)
    channels = arr.shape[-1]
    if channels == SENSOR_CHANNELS["S1"]:
        channel_sensor = "S1"
    elif channels == SENSOR_CHANNELS["S2"]:
        channel_sensor = "S2"
    else:
        channel_sensor = None

    # Channels-based detection is a safer fallback and resolves root naming ambiguities.
    if channel_sensor is not None:
        return channel_sensor
    if root_sensor is not None:
        return root_sensor
    return None


# ==============================
# CSV Aggregation & Hybrid Feature Preparation
# ==============================
# Build one row per filename from weather/time-series CSV input.
def aggregate_csv_features(csv_path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    # Collapse raw weather CSV rows into one compact per-image row used by the
    # "with weather" risk model and by CSV-based lookup fallbacks.
    issues: list[dict[str, Any]] = []
    if not csv_path.exists():
        issues.append(make_issue("csv", "csv_missing", details=str(csv_path)))
        return pd.DataFrame(), issues

    df = pd.read_csv(csv_path)
    target_cols = ["filename", "temperature", "tp", "runoff", "lat_grid", "lon_grid"]
    col_map: dict[str, str] = {}
    for col in df.columns:
        key = str(col).lower()
        if key in target_cols and key not in col_map:
            col_map[key] = col
    missing = [c for c in target_cols if c not in col_map]
    if missing:
        issues.append(
            make_issue("csv", "csv_missing_columns", details=",".join(missing))
        )
        return pd.DataFrame(), issues

    df = df.rename(columns={col_map[k]: k for k in target_cols})
    agg = (
        df.groupby("filename", as_index=False)
        .agg(
            Temperature_mean=("temperature", "mean"),
            Temperature_min=("temperature", "min"),
            Temperature_max=("temperature", "max"),
            tp_mean=("tp", "mean"),
            tp_max=("tp", "max"),
            tp_sum=("tp", "sum"),
            runoff_mean=("runoff", "mean"),
            runoff_max=("runoff", "max"),
            runoff_sum=("runoff", "sum"),
            lat_grid_mean=("lat_grid", "mean"),
            lon_grid_mean=("lon_grid", "mean"),
        )
        .sort_values("filename")
        .reset_index(drop=True)
    )
    return agg, issues


def find_weather_feature_record(
    df: pd.DataFrame, filename: str
) -> tuple[dict[str, Any] | None, str | None]:
    # Filename matching is intentionally tolerant because project CSVs and TIFFs
    # may disagree on extension or path prefix. Ambiguous matches are rejected.
    query_keys = filename_match_keys(filename)
    if not query_keys:
        return None, "missing_filename"
    filename_exact = str(filename).strip()
    if filename_exact:
        exact = df[df["filename"].astype(str).str.strip() == filename_exact]
        if len(exact) == 1:
            row = exact.iloc[0]
            return {
                "filename": str(row.get("filename", "")),
                **{name: row.get(name, "") for name in WEATHER_FEATURE_NAMES},
            }, None

    matched: dict[str, dict[str, Any]] = {}
    query_set = set(query_keys)
    for row in df.itertuples(index=False):
        row_name = str(getattr(row, "filename", "")).strip()
        if not row_name:
            continue
        row_keys = filename_match_keys(row_name)
        if any(k in query_set for k in row_keys):
            record = {"filename": row_name}
            for name in WEATHER_FEATURE_NAMES:
                record[name] = getattr(row, name, "")
            matched[row_name] = record

    if not matched:
        return None, "weather_csv_missing_filename"
    if len(matched) == 1:
        return next(iter(matched.values())), None

    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for row_name, record in matched.items():
        row_keys = set(filename_match_keys(row_name))
        score = 0
        if query_keys[0] in row_keys:
            score += 2
        if len(query_keys) > 1 and query_keys[1] in row_keys:
            score += 1
        ranked.append((score, row_name, record))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None, "weather_csv_ambiguous_filename"
    return ranked[0][2], None


def lookup_weather_features_for_filename(
    csv_path: Path, filename: str
) -> tuple[dict[str, float], str | None]:
    agg, issues = aggregate_csv_features(csv_path)
    if issues:
        return {}, "weather_csv_invalid"
    if agg.empty:
        return {}, "weather_csv_empty"
    rec, status = find_weather_feature_record(agg, filename)
    if rec is None:
        return {}, status or "weather_csv_missing_filename"
    out: dict[str, float] = {}
    for name in WEATHER_FEATURE_NAMES:
        try:
            out[name] = float(rec.get(name, 0.0))
        except Exception:
            out[name] = 0.0
    return out, None


def filename_match_keys(raw_name: str | None) -> tuple[str, ...]:
    if raw_name is None:
        return ()
    text = str(raw_name).strip()
    if not text:
        return ()
    basename = text.replace("\\", "/").split("/")[-1].strip().lower()
    if not basename:
        return ()
    keys: list[str] = [basename]
    stem = Path(basename).stem
    if stem and stem not in keys:
        keys.append(stem)
    if "__" in stem:
        source_tile_stem = stem.split("__", 1)[0].strip()
        if source_tile_stem:
            keys.extend(
                [
                    source_tile_stem,
                    f"{source_tile_stem}.tif",
                    f"{source_tile_stem}.tiff",
                ]
            )
    if "." not in basename and stem:
        keys.extend([f"{stem}.tif", f"{stem}.tiff"])
    elif basename.endswith(".tif") and stem:
        keys.append(f"{stem}.tiff")
    elif basename.endswith(".tiff") and stem:
        keys.append(f"{stem}.tif")
    return tuple(dict.fromkeys(keys))


def _normalize_csv_token(text: str) -> str:
    return "".join(ch for ch in str(text).strip().lower() if ch.isalnum())


@lru_cache(maxsize=16)
def _csv_header_columns(csv_path_str: str) -> tuple[str, ...]:
    p = Path(csv_path_str)
    if not p.exists():
        return ()
    try:
        cols = pd.read_csv(p, nrows=0).columns.tolist()
    except Exception:
        return ()
    return tuple(str(c) for c in cols)


def _csv_has_any_column_tokens(csv_path: Path, tokens: set[str]) -> bool:
    normalized_tokens = {_normalize_csv_token(t) for t in tokens}
    for col in _csv_header_columns(str(csv_path.resolve())):
        if _normalize_csv_token(col) in normalized_tokens:
            return True
    return False


def _find_csv_column(df: pd.DataFrame, tokens: set[str]) -> str | None:
    normalized_tokens = {_normalize_csv_token(t) for t in tokens}
    for col in df.columns:
        if _normalize_csv_token(str(col)) in normalized_tokens:
            return str(col)
    return None


def _find_csv_column_prefer(df: pd.DataFrame, ordered_tokens: list[str]) -> str | None:
    norm_to_col = {_normalize_csv_token(str(col)): str(col) for col in df.columns}
    for token in ordered_tokens:
        key = _normalize_csv_token(token)
        if key in norm_to_col:
            return norm_to_col[key]
    return None


def _safe_numeric_array(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)
    return arr[np.isfinite(arr)]


def _trend_slope(arr: np.ndarray) -> float:
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=np.float32)
    x_centered = x - float(np.mean(x))
    y_centered = arr.astype(np.float32) - float(np.mean(arr))
    denom = float(np.sum(x_centered * x_centered))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def _tail_window(arr: np.ndarray, window: int) -> np.ndarray:
    if arr.size <= 0:
        return np.zeros((0,), dtype=np.float32)
    use_n = max(1, min(int(window), int(arr.size)))
    return np.asarray(arr[-use_n:], dtype=np.float32)


def _tail_window_mean(arr: np.ndarray, window: int) -> float:
    tail = _tail_window(arr, window)
    if tail.size <= 0:
        return 0.0
    return float(np.mean(tail))


def _tail_window_sum(arr: np.ndarray, window: int) -> float:
    tail = _tail_window(arr, window)
    if tail.size <= 0:
        return 0.0
    return float(np.sum(tail))


def _safe_ratio(num: float, den: float, eps: float = 1e-6) -> float:
    if abs(float(den)) <= float(eps):
        return 0.0
    return float(float(num) / float(den))


def _sort_temporal_group_rows(group: pd.DataFrame) -> pd.DataFrame:
    if int(group["_date"].notna().sum()) >= 2:
        return group.sort_values(["_date", "_row_id"])
    return group.sort_values("_row_id")


def _build_temporal_weather_feature_row(group: pd.DataFrame) -> dict[str, float]:
    g = _sort_temporal_group_rows(group.copy())
    temp_vals = _safe_numeric_array(g["temperature"])
    tp_vals = _safe_numeric_array(g["tp"])
    runoff_vals = _safe_numeric_array(g["runoff"])
    lat_vals = _safe_numeric_array(g["lat_grid"])
    lon_vals = _safe_numeric_array(g["lon_grid"])
    date_vals = g["_date"].dropna() if "_date" in g.columns else pd.Series(dtype="datetime64[ns, UTC]")

    row = {name: 0.0 for name in TEMPORAL_WEATHER_FEATURE_NAMES}
    row["seq_len"] = float(len(g))
    if int(g["_date"].notna().sum()) >= 2:
        dt_min = g["_date"].min()
        dt_max = g["_date"].max()
        if pd.notna(dt_min) and pd.notna(dt_max):
            row["date_span_hours"] = float((dt_max - dt_min).total_seconds() / 3600.0)

    if lat_vals.size > 0:
        row["lat_grid_mean"] = float(np.mean(lat_vals))
    if lon_vals.size > 0:
        row["lon_grid_mean"] = float(np.mean(lon_vals))

    if temp_vals.size > 0:
        row["temp_last"] = float(temp_vals[-1])
        row["temp_mean"] = float(np.mean(temp_vals))
        row["temp_std"] = float(np.std(temp_vals))
        row["temp_min"] = float(np.min(temp_vals))
        row["temp_max"] = float(np.max(temp_vals))
        row["temp_delta_last_first"] = float(temp_vals[-1] - temp_vals[0])
        row["temp_trend_slope"] = _trend_slope(temp_vals)
        row["temp_recent3_mean"] = _tail_window_mean(temp_vals, 3)
        row["temp_recent6_mean"] = _tail_window_mean(temp_vals, 6)
        row["temp_recent12_mean"] = _tail_window_mean(temp_vals, 12)

    if tp_vals.size > 0:
        row["tp_last"] = float(tp_vals[-1])
        row["tp_mean"] = float(np.mean(tp_vals))
        row["tp_std"] = float(np.std(tp_vals))
        row["tp_max"] = float(np.max(tp_vals))
        row["tp_sum"] = float(np.sum(tp_vals))
        row["tp_delta_last_first"] = float(tp_vals[-1] - tp_vals[0])
        row["tp_trend_slope"] = _trend_slope(tp_vals)
        row["tp_recent3_sum"] = _tail_window_sum(tp_vals, 3)
        row["tp_recent6_sum"] = float(np.sum(tp_vals[-6:]))
        row["tp_recent12_sum"] = _tail_window_sum(tp_vals, 12)
        row["tp_recent24_sum"] = _tail_window_sum(tp_vals, 24)
        row["tp_recent3_mean"] = _tail_window_mean(tp_vals, 3)
        row["tp_recent12_mean"] = _tail_window_mean(tp_vals, 12)
        row["tp_recent3_to_12_ratio"] = _safe_ratio(
            row["tp_recent3_sum"], row["tp_recent12_sum"]
        )

    if runoff_vals.size > 0:
        row["runoff_last"] = float(runoff_vals[-1])
        row["runoff_mean"] = float(np.mean(runoff_vals))
        row["runoff_std"] = float(np.std(runoff_vals))
        row["runoff_max"] = float(np.max(runoff_vals))
        row["runoff_sum"] = float(np.sum(runoff_vals))
        row["runoff_delta_last_first"] = float(runoff_vals[-1] - runoff_vals[0])
        row["runoff_trend_slope"] = _trend_slope(runoff_vals)
        row["runoff_recent3_sum"] = _tail_window_sum(runoff_vals, 3)
        row["runoff_recent6_sum"] = float(np.sum(runoff_vals[-6:]))
        row["runoff_recent12_sum"] = _tail_window_sum(runoff_vals, 12)
        row["runoff_recent24_sum"] = _tail_window_sum(runoff_vals, 24)
        row["runoff_recent3_mean"] = _tail_window_mean(runoff_vals, 3)
        row["runoff_recent12_mean"] = _tail_window_mean(runoff_vals, 12)
        row["runoff_recent3_to_12_ratio"] = _safe_ratio(
            row["runoff_recent3_sum"], row["runoff_recent12_sum"]
        )

    if not date_vals.empty:
        dt_last = pd.Timestamp(date_vals.iloc[-1])
        month_angle = 2.0 * np.pi * float(dt_last.month - 1) / 12.0
        doy_angle = 2.0 * np.pi * float(dt_last.dayofyear - 1) / 366.0
        row["month_sin"] = float(np.sin(month_angle))
        row["month_cos"] = float(np.cos(month_angle))
        row["doy_sin"] = float(np.sin(doy_angle))
        row["doy_cos"] = float(np.cos(doy_angle))

    return row


def _load_temporal_raw_frame(
    csv_path: Path,
    *,
    require_filename: bool = True,
    allow_missing_runoff: bool = False,
    issue_stage: str = "temporal_csv",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    if not csv_path.exists():
        issues.append(make_issue(issue_stage, "csv_missing", details=str(csv_path)))
        return pd.DataFrame(), issues

    try:
        header_df = pd.read_csv(csv_path, nrows=0)
    except Exception as ex:
        issues.append(make_issue(issue_stage, "csv_read_failed", details=str(ex)))
        return pd.DataFrame(), issues

    filename_col = _find_csv_column_prefer(
        header_df, ["filename", "file", "image", "imagename"]
    )
    if filename_col is None:
        filename_col = _find_csv_column_prefer(header_df, ["tile_id", "floodmap_id"])
    temp_col = _find_csv_column(header_df, {"temperature", "temp"})
    tp_col = _find_csv_column(header_df, {"tp", "precipitation", "rainfall"})
    runoff_col = _find_csv_column(header_df, {"runoff"})
    lat_col = _find_csv_column(header_df, {"lat_grid", "latgrid", "latitude", "lat"})
    lon_col = _find_csv_column(
        header_df, {"lon_grid", "longrid", "longitude", "lon", "lng"}
    )
    date_col = _find_csv_column(
        header_df, {"date", "datetime", "timestamp", "time", "date_match"}
    )

    required = {
        "temperature": temp_col,
        "tp": tp_col,
        "lat_grid": lat_col,
        "lon_grid": lon_col,
    }
    if require_filename:
        required["filename"] = filename_col
    if not allow_missing_runoff:
        required["runoff"] = runoff_col
    missing = [key for key, value in required.items() if value is None]
    if missing:
        issues.append(
            make_issue(issue_stage, "csv_missing_columns", details=",".join(missing))
        )
        return pd.DataFrame(), issues

    usecols = [temp_col, tp_col, lat_col, lon_col]
    if runoff_col is not None:
        usecols.append(runoff_col)
    if filename_col is not None:
        usecols.append(filename_col)
    if date_col is not None:
        usecols.append(date_col)
    usecols = list(dict.fromkeys([str(c) for c in usecols if c is not None]))
    try:
        raw_df = pd.read_csv(csv_path, usecols=usecols)
    except Exception as ex:
        issues.append(make_issue(issue_stage, "csv_read_failed", details=str(ex)))
        return pd.DataFrame(), issues

    payload: dict[str, Any] = {
        "temperature": raw_df[temp_col],
        "tp": raw_df[tp_col],
        "lat_grid": raw_df[lat_col],
        "lon_grid": raw_df[lon_col],
    }
    if runoff_col is None:
        payload["runoff"] = np.nan
    else:
        payload["runoff"] = raw_df[runoff_col]
    if filename_col is not None:
        payload["filename"] = raw_df[filename_col]
    else:
        payload["filename"] = ""

    df = pd.DataFrame(payload)
    df["filename"] = df["filename"].astype(str).str.strip()
    if require_filename:
        df = df[df["filename"] != ""].copy()
        if df.empty:
            issues.append(make_issue(issue_stage, "csv_empty_after_filename_filter"))
            return pd.DataFrame(), issues

    for col in TEMPORAL_SEQUENCE_FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["_row_id"] = np.arange(len(df), dtype=np.int32)
    if date_col is not None:
        df["_date"] = pd.to_datetime(raw_df[date_col], errors="coerce", utc=True)
    else:
        df["_date"] = pd.NaT
    df["lat_grid"] = pd.to_numeric(df["lat_grid"], errors="coerce")
    df["lon_grid"] = pd.to_numeric(df["lon_grid"], errors="coerce")
    return df, issues


def build_temporal_csv_features(
    csv_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    # Summarize a variable-length temporal sequence into one fixed feature row per
    # filename so tree-based models can consume it directly.
    df, issues = _load_temporal_raw_frame(
        csv_path,
        require_filename=True,
        allow_missing_runoff=True,
        issue_stage="temporal_csv",
    )
    if df.empty:
        return pd.DataFrame(), issues

    records: list[dict[str, Any]] = []
    for filename, grp in df.groupby("filename", sort=False):
        row = _build_temporal_weather_feature_row(grp)
        row["filename"] = str(filename)
        records.append(row)

    if not records:
        issues.append(make_issue("temporal_csv", "no_rows_after_grouping"))
        return pd.DataFrame(), issues

    out = pd.DataFrame(records)
    out = out[["filename"] + TEMPORAL_WEATHER_FEATURE_NAMES].copy()
    for col in TEMPORAL_WEATHER_FEATURE_NAMES:
        out[col] = (
            pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(np.float32)
        )
    out = out.sort_values("filename").reset_index(drop=True)
    return out, issues


def build_temporal_sequence_lookup(
    csv_path: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    # Preserve the full per-filename sequence for LSTM-style temporal models.
    df, issues = _load_temporal_raw_frame(
        csv_path,
        require_filename=True,
        allow_missing_runoff=True,
        issue_stage="temporal_csv",
    )
    if df.empty:
        return {}, issues

    sequences: dict[str, np.ndarray] = {}
    for filename, grp in df.groupby("filename", sort=False):
        g = _sort_temporal_group_rows(grp.copy())
        seq = g[TEMPORAL_SEQUENCE_FEATURE_NAMES].to_numpy(dtype=np.float32)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        if seq.ndim == 2 and seq.shape[0] > 0:
            sequences[str(filename)] = seq

    if not sequences:
        issues.append(make_issue("temporal_csv", "no_sequences_after_grouping"))
    return sequences, issues


def find_temporal_sequence_record(
    sequence_lookup: dict[str, np.ndarray],
    filename: str,
) -> tuple[np.ndarray | None, str | None]:
    exact_key = str(filename).strip()
    if exact_key and exact_key in sequence_lookup:
        seq = sequence_lookup.get(exact_key)
        return (seq, None) if seq is not None else (None, "temporal_csv_missing_filename")

    query_keys = filename_match_keys(filename)
    if not query_keys:
        return None, "missing_filename"

    matched: dict[str, np.ndarray] = {}
    for row_name, seq in sequence_lookup.items():
        row_keys = set(filename_match_keys(row_name))
        if row_keys.intersection(query_keys):
            matched[row_name] = seq

    if not matched:
        return None, "temporal_csv_missing_filename"
    if len(matched) == 1:
        return next(iter(matched.values())), None
    return None, "temporal_csv_ambiguous_filename"


def build_temporal_anchor_map(
    bridge_csv_path: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    if not bridge_csv_path.exists():
        issues.append(
            make_issue("temporal_bridge", "csv_missing", details=str(bridge_csv_path))
        )
        return {}, issues

    try:
        df = pd.read_csv(bridge_csv_path)
    except Exception as ex:
        issues.append(make_issue("temporal_bridge", "csv_read_failed", details=str(ex)))
        return {}, issues

    filename_col = _find_csv_column_prefer(
        df, ["filename", "tile_id", "file", "image", "imagename", "floodmap_id"]
    )
    date_col = _find_csv_column(
        df,
        {
            "date",
            "datetime",
            "timestamp",
            "time",
            "date_match",
            "sentinel_date",
            "floodmap_date",
        },
    )
    lat_col = _find_csv_column(
        df, {"lat_grid", "latgrid", "latitude", "lat", "latitude_deg"}
    )
    lon_col = _find_csv_column(
        df, {"lon_grid", "longrid", "longitude", "lon", "lng", "longitude_deg"}
    )

    required = {
        "filename": filename_col,
        "date": date_col,
        "lat_grid": lat_col,
        "lon_grid": lon_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        issues.append(
            make_issue(
                "temporal_bridge", "csv_missing_columns", details=",".join(missing)
            )
        )
        return {}, issues

    d = pd.DataFrame(
        {
            "filename": df[filename_col],
            "date": pd.to_datetime(df[date_col], errors="coerce", utc=True),
            "lat_grid": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon_grid": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    )
    d["filename"] = d["filename"].astype(str).str.strip()
    d = d[(d["filename"] != "") & d["lat_grid"].notna() & d["lon_grid"].notna()].copy()
    if d.empty:
        issues.append(make_issue("temporal_bridge", "csv_empty_after_filter"))
        return {}, issues

    anchor_map: dict[str, dict[str, Any]] = {}
    for filename, grp in d.groupby("filename", sort=False):
        g = grp.sort_values("date")
        anchor = {
            "filename": str(filename),
            "lat_grid": float(g["lat_grid"].mean()),
            "lon_grid": float(g["lon_grid"].mean()),
            "anchor_date": g["date"].dropna().max(),
        }
        anchor_map[str(filename)] = anchor

    return anchor_map, issues


@lru_cache(maxsize=4)
def _cached_temporal_anchor_map(
    bridge_csv_path_str: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    return build_temporal_anchor_map(Path(bridge_csv_path_str))


def find_temporal_anchor_record(
    anchor_map: dict[str, dict[str, Any]],
    filename: str,
) -> tuple[dict[str, Any] | None, str | None]:
    exact_key = str(filename).strip()
    if exact_key and exact_key in anchor_map:
        rec = anchor_map.get(exact_key)
        return (rec, None) if rec is not None else (None, "temporal_bridge_missing_filename")

    query_keys = filename_match_keys(filename)
    if not query_keys:
        return None, "missing_filename"

    matched: dict[str, dict[str, Any]] = {}
    for row_name, rec in anchor_map.items():
        row_keys = set(filename_match_keys(row_name))
        if row_keys.intersection(query_keys):
            matched[row_name] = rec

    if not matched:
        return None, "temporal_bridge_missing_filename"
    if len(matched) == 1:
        return next(iter(matched.values())), None
    return None, "temporal_bridge_ambiguous_filename"


def _merge_bridge_anchor_context(
    anchor_ctx: dict[str, Any],
    *,
    bridge_csv_path: Path | None,
    image_filename: str | None,
    image_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    meta: dict[str, Any] = {
        "bridge_status": None,
        "bridge_csv_path": str(bridge_csv_path.resolve()) if bridge_csv_path else None,
        "bridge_filename": None,
    }
    if bridge_csv_path is None:
        meta["bridge_status"] = "not_configured"
        return anchor_ctx, meta
    if not bridge_csv_path.exists():
        meta["bridge_status"] = "csv_missing"
        return anchor_ctx, meta

    lookup_name = str(image_filename or "").strip()
    if not lookup_name and image_path is not None:
        lookup_name = image_path.name
    if not lookup_name:
        meta["bridge_status"] = "missing_filename"
        return anchor_ctx, meta

    anchor_map, issues = _cached_temporal_anchor_map(str(bridge_csv_path.resolve()))
    if not anchor_map:
        meta["bridge_status"] = "csv_invalid" if issues else "csv_empty"
        return anchor_ctx, meta

    bridge_rec, status = find_temporal_anchor_record(anchor_map, lookup_name)
    if bridge_rec is None:
        meta["bridge_status"] = status or "temporal_bridge_missing_filename"
        return anchor_ctx, meta

    out = dict(anchor_ctx)
    meta["bridge_status"] = "matched"
    meta["bridge_filename"] = str(bridge_rec.get("filename") or "")

    if out.get("lat") is None:
        lat = _to_finite_float(bridge_rec.get("lat_grid"))
        if lat is not None:
            out["lat"] = lat
            out["coord_source"] = "temporal_bridge:lat_grid"
    if out.get("lon") is None:
        lon = _to_finite_float(bridge_rec.get("lon_grid"))
        if lon is not None:
            out["lon"] = lon
            out["coord_source"] = "temporal_bridge:lat_lon_grid"

    # Prefer the downloaded satellite acquisition date parsed from the filename.
    # Use the bridge date only when the image itself has no date token.
    if out.get("anchor_date") is None:
        bridge_date = bridge_rec.get("anchor_date")
        if isinstance(bridge_date, pd.Timestamp) and pd.notna(bridge_date):
            out["anchor_date"] = bridge_date
            out["anchor_date_iso"] = bridge_date.isoformat()
            out["time_source"] = "temporal_bridge:date"

    source_parts = [x for x in [out.get("coord_source"), out.get("time_source")] if x]
    out["anchor_source"] = "+".join(str(x) for x in source_parts) if source_parts else "unavailable"
    if out.get("anchor_date") is not None and not out.get("anchor_date_iso"):
        anchor_date = out.get("anchor_date")
        if isinstance(anchor_date, pd.Timestamp) and pd.notna(anchor_date):
            out["anchor_date_iso"] = anchor_date.isoformat()
    return out, meta


def _nearest_grid_value(values: np.ndarray, target: float) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    idx = int(np.argmin(np.abs(arr - float(target))))
    return float(arr[idx])


def build_temporal_sequence_lookup_from_datetime_coords(
    *,
    temporal_csv_path: Path,
    bridge_csv_path: Path,
    target_filenames: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    weather_df, weather_issues = _load_temporal_raw_frame(
        temporal_csv_path,
        require_filename=False,
        allow_missing_runoff=True,
        issue_stage="temporal_era5",
    )
    issues.extend(weather_issues)
    if weather_df.empty:
        return {}, issues

    anchor_map, anchor_issues = build_temporal_anchor_map(bridge_csv_path)
    issues.extend(anchor_issues)
    if not anchor_map:
        return {}, issues

    target_list = (
        list(target_filenames) if target_filenames else sorted(anchor_map.keys())
    )
    all_lat = weather_df["lat_grid"].to_numpy(dtype=np.float64)
    all_lon = weather_df["lon_grid"].to_numpy(dtype=np.float64)

    out: dict[str, np.ndarray] = {}
    for filename in target_list:
        anchor, status = find_temporal_anchor_record(anchor_map, str(filename))
        if anchor is None:
            issues.append(
                make_issue(
                    "temporal_datetime_coords",
                    status or "anchor_missing",
                    filename=str(filename),
                )
            )
            continue

        lat_sel = _nearest_grid_value(all_lat, float(anchor["lat_grid"]))
        lon_sel = _nearest_grid_value(all_lon, float(anchor["lon_grid"]))
        if lat_sel is None or lon_sel is None:
            issues.append(
                make_issue(
                    "temporal_datetime_coords",
                    "era5_grid_not_available",
                    filename=str(filename),
                )
            )
            continue

        g = weather_df[
            (np.isclose(weather_df["lat_grid"].to_numpy(dtype=np.float64), lat_sel))
            & (np.isclose(weather_df["lon_grid"].to_numpy(dtype=np.float64), lon_sel))
        ].copy()
        if g.empty:
            issues.append(
                make_issue(
                    "temporal_datetime_coords",
                    "era5_grid_slice_empty",
                    filename=str(filename),
                )
            )
            continue
        g = _sort_temporal_group_rows(g)

        anchor_date = anchor.get("anchor_date")
        if pd.notna(anchor_date):
            g_before = g[g["_date"] <= anchor_date]
            if not g_before.empty:
                g = g_before

        keep_n = max(int(TEMPORAL_LSTM_MAX_SEQ_LEN) * 3, 48)
        if len(g) > keep_n:
            g = g.tail(keep_n)

        seq = g[TEMPORAL_SEQUENCE_FEATURE_NAMES].to_numpy(dtype=np.float32)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        if seq.ndim == 2 and seq.shape[0] > 0:
            out[str(filename)] = seq
        else:
            issues.append(
                make_issue(
                    "temporal_datetime_coords",
                    "era5_sequence_empty",
                    filename=str(filename),
                )
            )

    return out, issues


@lru_cache(maxsize=4)
def _cached_datetime_coords_sequence_lookup(
    temporal_csv_path_str: str,
    bridge_csv_path_str: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    return build_temporal_sequence_lookup_from_datetime_coords(
        temporal_csv_path=Path(temporal_csv_path_str),
        bridge_csv_path=Path(bridge_csv_path_str),
        target_filenames=None,
    )


def build_temporal_sequence_lookup_hybrid(
    *,
    temporal_csv_path: Path,
    bridge_csv_path: Path | None,
    target_filenames: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], str]:
    temporal_resolved = temporal_csv_path.resolve()
    bridge = (
        bridge_csv_path.resolve() if bridge_csv_path is not None else temporal_resolved
    )
    normalized_target_filenames = list(
        dict.fromkeys(
            str(x).strip() for x in (target_filenames or []) if str(x).strip()
        )
    )

    issues_filename: list[dict[str, Any]] = []
    seq_by_filename: dict[str, np.ndarray] = {}
    has_filename_like_col = _csv_has_any_column_tokens(
        temporal_resolved,
        {"filename", "file", "image", "imagename", "tile_id", "floodmap_id"},
    )
    if has_filename_like_col:
        seq_by_filename, issues_filename = build_temporal_sequence_lookup(
            temporal_resolved
        )
        if seq_by_filename:
            if normalized_target_filenames:
                filtered: dict[str, np.ndarray] = {}
                for name in normalized_target_filenames:
                    seq, status = find_temporal_sequence_record(
                        seq_by_filename, str(name)
                    )
                    if seq is not None:
                        filtered[str(name)] = seq
                    elif status is not None:
                        issues_filename.append(
                            make_issue(
                                "temporal_sequence_lookup",
                                status,
                                filename=str(name),
                            )
                        )
                if filtered:
                    return filtered, issues_filename, "filename"
            else:
                return seq_by_filename, issues_filename, "filename"

    use_targeted_lookup = bool(normalized_target_filenames) and len(
        normalized_target_filenames
    ) <= 4
    if use_targeted_lookup:
        seq_by_dt, issues_dt = build_temporal_sequence_lookup_from_datetime_coords(
            temporal_csv_path=temporal_resolved,
            bridge_csv_path=bridge,
            target_filenames=normalized_target_filenames,
        )
    else:
        seq_by_dt_all, issues_dt_cached = _cached_datetime_coords_sequence_lookup(
            str(temporal_resolved),
            str(bridge),
        )
        issues_dt = filter_issues_for_filenames(
            issues_dt_cached, normalized_target_filenames
        )
        if normalized_target_filenames:
            filtered_dt: dict[str, np.ndarray] = {}
            for name in normalized_target_filenames:
                seq, status = find_temporal_sequence_record(seq_by_dt_all, str(name))
                if seq is not None:
                    filtered_dt[str(name)] = seq
                elif status is not None:
                    issues_dt.append(
                        make_issue(
                            "temporal_datetime_coords",
                            status,
                            filename=str(name),
                        )
                    )
            seq_by_dt = filtered_dt
        else:
            seq_by_dt = seq_by_dt_all

    cleaned_filename_issues: list[dict[str, Any]] = []
    for issue in issues_filename:
        issue_type = str(issue.get("issue_type", ""))
        details = str(issue.get("details", ""))
        if issue_type == "csv_missing_columns" and "filename" in details:
            continue
        if issue_type in {
            "csv_empty_after_filename_filter",
            "no_sequences_after_grouping",
        }:
            continue
        cleaned_filename_issues.append(issue)
    return seq_by_dt, cleaned_filename_issues + issues_dt, "datetime_coords"


def find_temporal_feature_record(
    df: pd.DataFrame, filename: str
) -> tuple[dict[str, Any] | None, str | None]:
    query_keys = filename_match_keys(filename)
    if not query_keys:
        return None, "missing_filename"
    filename_exact = str(filename).strip()
    if filename_exact:
        exact = df[df["filename"].astype(str).str.strip() == filename_exact]
        if len(exact) == 1:
            row = exact.iloc[0]
            return {
                "filename": str(row.get("filename", "")),
                **{name: row.get(name, "") for name in TEMPORAL_WEATHER_FEATURE_NAMES},
            }, None

    matched: dict[str, dict[str, Any]] = {}
    query_set = set(query_keys)
    for row in df.itertuples(index=False):
        row_name = str(getattr(row, "filename", "")).strip()
        if not row_name:
            continue
        row_keys = filename_match_keys(row_name)
        if any(k in query_set for k in row_keys):
            record = {"filename": row_name}
            for name in TEMPORAL_WEATHER_FEATURE_NAMES:
                record[name] = getattr(row, name, "")
            matched[row_name] = record

    if not matched:
        return None, "temporal_csv_missing_filename"
    if len(matched) == 1:
        return next(iter(matched.values())), None
    return None, "temporal_csv_ambiguous_filename"


def lookup_temporal_features_for_filename(
    csv_path: Path, filename: str
) -> tuple[dict[str, float], str | None]:
    temporal_df, issues = build_temporal_csv_features(csv_path)
    if not temporal_df.empty:
        rec, status = find_temporal_feature_record(temporal_df, filename)
        if rec is not None:
            return {
                name: float(rec.get(name, 0.0))
                for name in TEMPORAL_WEATHER_FEATURE_NAMES
            }, None
        return {}, status
    if issues:
        return {}, "temporal_csv_invalid"
    return {}, "temporal_csv_empty"


def _parse_utc_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    raw = value
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    try:
        ts = pd.to_datetime(raw, errors="coerce", utc=True)
    except Exception:
        return None
    if isinstance(ts, pd.Timestamp) and pd.notna(ts):
        return ts
    return None


def _parse_datetime_from_filename(filename: str | None) -> pd.Timestamp | None:
    if filename is None:
        return None
    stem = Path(str(filename)).stem
    if not stem:
        return None

    full_match = re.search(
        r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)[Tt _-]?([0-2]\d)([0-5]\d)([0-5]\d)",
        stem,
    )
    if full_match is not None:
        y, m, d, hh, mm, ss = full_match.groups()
        return _parse_utc_timestamp(f"{y}-{m}-{d} {hh}:{mm}:{ss}+00:00")

    date_only = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", stem)
    if date_only is not None:
        y, m, d = date_only.groups()
        return _parse_utc_timestamp(f"{y}-{m}-{d} 00:00:00+00:00")
    return None


def _to_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _extract_lat_lon_from_geo_meta(
    geo_meta: dict[str, Any] | None,
) -> tuple[float | None, float | None, str | None]:
    if not isinstance(geo_meta, dict):
        return None, None, None

    direct_pairs = [
        ("center_lat", "center_lon", "geo_center"),
        ("lat", "lon", "geo_lat_lon"),
        ("latitude", "longitude", "geo_lat_lon"),
    ]
    for lat_key, lon_key, source in direct_pairs:
        lat = _to_finite_float(geo_meta.get(lat_key))
        lon = _to_finite_float(geo_meta.get(lon_key))
        if (
            lat is not None
            and lon is not None
            and -90.0 <= lat <= 90.0
            and -180.0 <= lon <= 180.0
        ):
            return lat, lon, source

    tie = geo_meta.get("tiepoint")
    scale = geo_meta.get("pixel_scale")
    shape = geo_meta.get("image_shape")
    tie_x: float | None = None
    tie_y: float | None = None
    if isinstance(tie, list):
        if len(tie) >= 6:
            tie_x = _to_finite_float(tie[3])
            tie_y = _to_finite_float(tie[4])
        elif len(tie) >= 2:
            tie_x = _to_finite_float(tie[0])
            tie_y = _to_finite_float(tie[1])
    sx = (
        _to_finite_float(scale[0])
        if isinstance(scale, list) and len(scale) >= 1
        else None
    )
    sy = (
        _to_finite_float(scale[1])
        if isinstance(scale, list) and len(scale) >= 2
        else None
    )
    height = None
    width = None
    if isinstance(shape, list) and len(shape) >= 2:
        try:
            height = int(shape[0])
            width = int(shape[1])
        except Exception:
            height = None
            width = None

    if tie_x is not None and tie_y is not None:
        if (
            width
            and height
            and sx is not None
            and sy is not None
            and sx > 0.0
            and sy > 0.0
        ):
            center_x = tie_x + sx * (float(width) / 2.0)
            center_y = tie_y - sy * (float(height) / 2.0)
            center_lon = center_x
            center_lat = center_y
            if -90.0 <= center_lat <= 90.0 and -180.0 <= center_lon <= 180.0:
                return center_lat, center_lon, "geo_tiepoint_center"
            epsg_raw = geo_meta.get("epsg")
            epsg = int(epsg_raw) if isinstance(epsg_raw, (int, np.integer)) else None
            if epsg is not None:
                try:
                    # Optional conversion for projected CRS (for example EPSG:32631 -> WGS84).
                    from pyproj import Transformer  # type: ignore

                    transformer = Transformer.from_crs(
                        f"EPSG:{epsg}", "EPSG:4326", always_xy=True
                    )
                    lon_wgs84, lat_wgs84 = transformer.transform(center_x, center_y)
                    lon_wgs84 = float(lon_wgs84)
                    lat_wgs84 = float(lat_wgs84)
                    if -90.0 <= lat_wgs84 <= 90.0 and -180.0 <= lon_wgs84 <= 180.0:
                        return lat_wgs84, lon_wgs84, "geo_projected_to_wgs84"
                except Exception:
                    pass
        if -90.0 <= tie_y <= 90.0 and -180.0 <= tie_x <= 180.0:
            return tie_y, tie_x, "geo_tiepoint"

    bbox = geo_meta.get("projected_bbox")
    if isinstance(bbox, dict):
        x_min = _to_finite_float(bbox.get("x_min"))
        x_max = _to_finite_float(bbox.get("x_max"))
        y_min = _to_finite_float(bbox.get("y_min"))
        y_max = _to_finite_float(bbox.get("y_max"))
        if None not in (x_min, x_max, y_min, y_max):
            center_lon = float((x_min + x_max) / 2.0)
            center_lat = float((y_min + y_max) / 2.0)
            if -90.0 <= center_lat <= 90.0 and -180.0 <= center_lon <= 180.0:
                return center_lat, center_lon, "geo_bbox_center"

    return None, None, None


def _extract_temporal_anchor_datetime(
    image_filename: str | None,
    image_path: Path | None,
    geo_meta: dict[str, Any] | None,
) -> tuple[pd.Timestamp | None, str | None]:
    if isinstance(geo_meta, dict):
        for key in [
            "image_datetime_utc",
            "datetime_utc",
            "acquisition_time_utc",
            "timestamp_utc",
            "capture_time_utc",
        ]:
            ts = _parse_utc_timestamp(geo_meta.get(key))
            if ts is not None:
                return ts, f"geo_meta:{key}"

    ts = _parse_datetime_from_filename(image_filename)
    if ts is not None:
        return ts, "filename"

    if image_path is not None:
        ts = _parse_datetime_from_filename(image_path.name)
        if ts is not None:
            return ts, "image_path"

    return None, None


def infer_temporal_anchor_context(
    *,
    image_filename: str | None,
    image_path: Path | None = None,
    geo_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lat, lon, coord_source = _extract_lat_lon_from_geo_meta(geo_meta)
    anchor_date, time_source = _extract_temporal_anchor_datetime(
        image_filename, image_path, geo_meta
    )
    source_parts = [x for x in [coord_source, time_source] if x]
    return {
        "lat": lat,
        "lon": lon,
        "anchor_date": anchor_date,
        "anchor_date_iso": anchor_date.isoformat() if anchor_date is not None else None,
        "coord_source": coord_source,
        "time_source": time_source,
        "anchor_source": "+".join(source_parts) if source_parts else "unavailable",
    }


@lru_cache(maxsize=2)
def _cached_temporal_weather_frame_for_anchor(
    csv_path_str: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    return _load_temporal_raw_frame(
        Path(csv_path_str),
        require_filename=False,
        allow_missing_runoff=True,
        issue_stage="temporal_era5",
    )


def _slice_temporal_weather_group_for_anchor(
    *,
    csv_path: Path,
    lat: float | None,
    lon: float | None,
    anchor_date: pd.Timestamp | None,
) -> tuple[pd.DataFrame, str | None]:
    weather_df, issues = _cached_temporal_weather_frame_for_anchor(
        str(csv_path.resolve())
    )
    if weather_df.empty:
        if issues:
            return pd.DataFrame(), "temporal_csv_invalid"
        return pd.DataFrame(), "temporal_csv_empty"

    g: pd.DataFrame
    # Preferred path: match nearest ERA5 grid by coordinates.
    if lat is not None and lon is not None:
        all_lat = weather_df["lat_grid"].to_numpy(dtype=np.float64)
        all_lon = weather_df["lon_grid"].to_numpy(dtype=np.float64)
        lat_sel = _nearest_grid_value(all_lat, float(lat))
        lon_sel = _nearest_grid_value(all_lon, float(lon))
        if lat_sel is None or lon_sel is None:
            return pd.DataFrame(), "temporal_era5_grid_not_available"

        g = weather_df[
            (np.isclose(weather_df["lat_grid"].to_numpy(dtype=np.float64), lat_sel))
            & (np.isclose(weather_df["lon_grid"].to_numpy(dtype=np.float64), lon_sel))
        ].copy()
        if g.empty:
            return pd.DataFrame(), "temporal_era5_grid_slice_empty"
    else:
        # Fallback path: time-only matching when coordinates are missing.
        if anchor_date is None or pd.isna(anchor_date):
            return pd.DataFrame(), "temporal_anchor_missing_coordinates"
        weather_sorted = _sort_temporal_group_rows(weather_df.copy())
        recent = weather_sorted[weather_sorted["_date"] <= anchor_date].copy()
        if recent.empty:
            recent = weather_sorted.copy()
        if recent.empty:
            return pd.DataFrame(), "temporal_era5_sequence_empty"

        pool_tail = max(int(TEMPORAL_LSTM_MAX_SEQ_LEN) * 8, 128)
        pool = recent.tail(pool_tail).copy()
        top_grid = (
            pool.groupby(["lat_grid", "lon_grid"], dropna=False)
            .size()
            .sort_values(ascending=False)
        )
        if top_grid.empty:
            return pd.DataFrame(), "temporal_era5_grid_not_available"
        lat_sel = float(top_grid.index[0][0])
        lon_sel = float(top_grid.index[0][1])
        g = recent[
            (np.isclose(recent["lat_grid"].to_numpy(dtype=np.float64), lat_sel))
            & (np.isclose(recent["lon_grid"].to_numpy(dtype=np.float64), lon_sel))
        ].copy()
        if g.empty:
            return pd.DataFrame(), "temporal_era5_grid_slice_empty"

    g = _sort_temporal_group_rows(g)
    if anchor_date is not None and pd.notna(anchor_date):
        g_before = g[g["_date"] <= anchor_date]
        if not g_before.empty:
            g = g_before

    keep_n = max(int(TEMPORAL_LSTM_MAX_SEQ_LEN) * 3, 48)
    if len(g) > keep_n:
        g = g.tail(keep_n).copy()

    if g.empty:
        return pd.DataFrame(), "temporal_era5_sequence_empty"
    return g, None


def lookup_temporal_sequence_for_anchor(
    *,
    csv_path: Path,
    lat: float | None,
    lon: float | None,
    anchor_date: pd.Timestamp | None,
) -> tuple[np.ndarray | None, str | None]:
    group_df, status = _slice_temporal_weather_group_for_anchor(
        csv_path=csv_path,
        lat=lat,
        lon=lon,
        anchor_date=anchor_date,
    )
    if status is not None or group_df.empty:
        return None, status or "temporal_era5_sequence_empty"

    seq = group_df[TEMPORAL_SEQUENCE_FEATURE_NAMES].to_numpy(dtype=np.float32)
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    if seq.ndim != 2 or seq.shape[0] <= 0:
        return None, "temporal_era5_sequence_empty"
    return seq, None


def build_temporal_sequence_lookup_from_base_rows(
    *,
    base_df: pd.DataFrame,
    temporal_csv_path: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], str]:
    issues: list[dict[str, Any]] = []
    if base_df.empty:
        return {}, issues, "base_rows_anchor"

    out: dict[str, np.ndarray] = {}
    geo_meta_cache: dict[str, dict[str, Any] | None] = {}
    sequence_cache: dict[
        tuple[float | None, float | None, str | None], tuple[np.ndarray | None, str | None]
    ] = {}
    dedup_df = base_df.drop_duplicates(subset=["filename"], keep="first").copy()

    for row in dedup_df.itertuples(index=False):
        filename = str(getattr(row, "filename", "")).strip()
        if not filename:
            continue

        lat = _to_finite_float(getattr(row, "lat_grid_mean", None))
        lon = _to_finite_float(getattr(row, "lon_grid_mean", None))
        image_path_raw = str(getattr(row, "image_path", "") or "").strip()
        image_path = Path(image_path_raw) if image_path_raw else None
        geo_meta: dict[str, Any] | None = None

        if (lat is None or lon is None) and image_path is not None:
            cache_key = (
                str(image_path.resolve()) if image_path.exists() else str(image_path)
            )
            if cache_key in geo_meta_cache:
                geo_meta = geo_meta_cache[cache_key]
            else:
                try:
                    geo_meta = inspect_geospatial_metadata(image_path)
                except Exception:
                    geo_meta = None
                geo_meta_cache[cache_key] = geo_meta

        anchor_ctx = infer_temporal_anchor_context(
            image_filename=filename,
            image_path=image_path,
            geo_meta=geo_meta,
        )
        if lat is None:
            lat = _to_finite_float(anchor_ctx.get("lat"))
        if lon is None:
            lon = _to_finite_float(anchor_ctx.get("lon"))
        anchor_date = anchor_ctx.get("anchor_date")
        cache_key = (
            round(float(lat), 6) if lat is not None else None,
            round(float(lon), 6) if lon is not None else None,
            anchor_date.isoformat() if anchor_date is not None else None,
        )
        if cache_key in sequence_cache:
            seq, status = sequence_cache[cache_key]
        else:
            seq, status = lookup_temporal_sequence_for_anchor(
                csv_path=temporal_csv_path.resolve(),
                lat=lat,
                lon=lon,
                anchor_date=anchor_date,
            )
            sequence_cache[cache_key] = (seq, status)
        if seq is not None:
            out[filename] = seq
            continue

        issues.append(
            make_issue(
                "temporal_base_rows_anchor",
                status or "temporal_sequence_missing_for_base_row",
                filename=filename,
                image_path=image_path,
            )
        )

    return out, issues, "base_rows_anchor"


def lookup_temporal_features_for_anchor(
    *,
    csv_path: Path,
    lat: float | None,
    lon: float | None,
    anchor_date: pd.Timestamp | None,
) -> tuple[dict[str, float], str | None]:
    group_df, status = _slice_temporal_weather_group_for_anchor(
        csv_path=csv_path,
        lat=lat,
        lon=lon,
        anchor_date=anchor_date,
    )
    if status is not None or group_df.empty:
        return {}, status or "temporal_era5_sequence_empty"

    row = _build_temporal_weather_feature_row(group_df)
    return {
        name: float(row.get(name, 0.0)) for name in TEMPORAL_WEATHER_FEATURE_NAMES
    }, None


def _build_weather_feature_row_for_risk(group: pd.DataFrame) -> dict[str, float]:
    g = _sort_temporal_group_rows(group.copy())
    temp_vals = _safe_numeric_array(g["temperature"])
    tp_vals = _safe_numeric_array(g["tp"])
    runoff_vals = _safe_numeric_array(g["runoff"])
    lat_vals = _safe_numeric_array(g["lat_grid"])
    lon_vals = _safe_numeric_array(g["lon_grid"])

    row = {name: 0.0 for name in WEATHER_FEATURE_NAMES}
    if temp_vals.size > 0:
        row["Temperature_mean"] = float(np.mean(temp_vals))
        row["Temperature_min"] = float(np.min(temp_vals))
        row["Temperature_max"] = float(np.max(temp_vals))
    if tp_vals.size > 0:
        row["tp_mean"] = float(np.mean(tp_vals))
        row["tp_max"] = float(np.max(tp_vals))
        row["tp_sum"] = float(np.sum(tp_vals))
    if runoff_vals.size > 0:
        row["runoff_mean"] = float(np.mean(runoff_vals))
        row["runoff_max"] = float(np.max(runoff_vals))
        row["runoff_sum"] = float(np.sum(runoff_vals))
    if lat_vals.size > 0:
        row["lat_grid_mean"] = float(np.mean(lat_vals))
    if lon_vals.size > 0:
        row["lon_grid_mean"] = float(np.mean(lon_vals))
    return row


def lookup_weather_features_for_anchor(
    *,
    csv_path: Path,
    lat: float | None,
    lon: float | None,
    anchor_date: pd.Timestamp | None,
) -> tuple[dict[str, float], str | None]:
    group_df, status = _slice_temporal_weather_group_for_anchor(
        csv_path=csv_path,
        lat=lat,
        lon=lon,
        anchor_date=anchor_date,
    )
    if status is not None or group_df.empty:
        return {}, status or "temporal_era5_sequence_empty"
    row = _build_weather_feature_row_for_risk(group_df)
    return {name: float(row.get(name, 0.0)) for name in WEATHER_FEATURE_NAMES}, None


def lookup_weather_features_for_image_from_temporal(
    *,
    csv_path: Path,
    image_filename: str | None,
    image_path: Path | None = None,
    geo_meta: dict[str, Any] | None = None,
    bridge_csv_path: Path | None = None,
) -> tuple[dict[str, float], str | None, dict[str, Any]]:
    # Convert whatever the caller knows about the image (name, path, coordinates,
    # acquisition date) into one normalized temporal anchor for ERA5 lookup.
    image_name = str(image_filename).strip() if image_filename is not None else ""
    anchor_ctx = infer_temporal_anchor_context(
        image_filename=(
            image_name
            if image_name
            else (image_path.name if image_path is not None else None)
        ),
        image_path=image_path,
        geo_meta=geo_meta,
    )
    anchor_ctx, bridge_meta = _merge_bridge_anchor_context(
        anchor_ctx,
        bridge_csv_path=bridge_csv_path,
        image_filename=(
            image_name
            if image_name
            else (image_path.name if image_path is not None else None)
        ),
        image_path=image_path,
    )
    out_meta: dict[str, Any] = {
        "lookup_mode": (
            "coords_datetime"
            if anchor_ctx.get("lat") is not None and anchor_ctx.get("lon") is not None
            else "time_only"
        ),
        "anchor_source": anchor_ctx.get("anchor_source"),
        "anchor_lat": anchor_ctx.get("lat"),
        "anchor_lon": anchor_ctx.get("lon"),
        "anchor_time_utc": anchor_ctx.get("anchor_date_iso"),
        "coord_source": anchor_ctx.get("coord_source"),
        "time_source": anchor_ctx.get("time_source"),
        "bridge_status": bridge_meta.get("bridge_status"),
        "bridge_csv_path": bridge_meta.get("bridge_csv_path"),
        "bridge_filename": bridge_meta.get("bridge_filename"),
    }
    # Weather lookup itself stays centralized; this wrapper mainly standardizes the
    # anchor and returns enough metadata to explain how values were resolved.
    values, status = lookup_weather_features_for_anchor(
        csv_path=csv_path.resolve(),
        lat=anchor_ctx.get("lat"),
        lon=anchor_ctx.get("lon"),
        anchor_date=anchor_ctx.get("anchor_date"),
    )
    if status is not None:
        return {}, status, out_meta
    return values, None, out_meta


def build_meta_model(seed: int = 42) -> Pipeline:
    # Risk heads use only a few tabular features, so a calibrated linear model is
    # easier to audit and less likely to overfit than a heavier estimator.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=int(seed),
                ),
            ),
        ]
    )


def build_temporal_risk_model(seed: int, model_type: str = "adaboost") -> Any:
    m = str(model_type or "adaboost").strip().lower()
    if m == "gradient_boosting":
        return GradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            n_estimators=300,
            subsample=0.9,
            random_state=seed,
        )
    # Default temporal tabular model after benchmark comparisons.
    return AdaBoostClassifier(
        n_estimators=500,
        learning_rate=0.05,
        random_state=seed,
    )


class TemporalLSTMClassifier(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int = TEMPORAL_LSTM_HIDDEN_SIZE,
        num_layers: int = TEMPORAL_LSTM_LAYERS,
        dropout: float = TEMPORAL_LSTM_DROPOUT,
    ) -> None:
        if nn is None:
            raise RuntimeError("torch is required for temporal LSTM model")
        super().__init__()
        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), 1),
        )

    def forward(self, x: Any, lengths: Any) -> Any:
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]
        logits = self.head(h_last).squeeze(1)
        return logits


def _pad_temporal_sequences(
    sequences: list[np.ndarray],
    *,
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not sequences:
        return (
            np.zeros(
                (0, int(max_seq_len), len(TEMPORAL_LSTM_FEATURE_NAMES)),
                dtype=np.float32,
            ),
            np.zeros((0,), dtype=np.int64),
        )
    feature_dim = int(sequences[0].shape[1])
    x = np.zeros((len(sequences), int(max_seq_len), feature_dim), dtype=np.float32)
    lengths = np.zeros((len(sequences),), dtype=np.int64)
    for idx, seq in enumerate(sequences):
        seq = np.asarray(seq, dtype=np.float32)
        if seq.ndim != 2:
            continue
        use_len = int(min(int(max_seq_len), int(seq.shape[0])))
        if use_len <= 0:
            continue
        x[idx, :use_len, :] = seq[:use_len, :]
        lengths[idx] = int(use_len)
    lengths[lengths <= 0] = 1
    return x, lengths


def _serialize_state_dict_numpy(state_dict: Any) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in state_dict.items():
        arr = value.detach().cpu().numpy()
        out[str(key)] = np.array(arr)
    return out


def _restore_state_dict_from_numpy(state_dict_np: dict[str, Any]) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required to restore LSTM model state")
    out: dict[str, Any] = {}
    for key, value in state_dict_np.items():
        arr = np.asarray(value)
        out[str(key)] = torch.from_numpy(arr)
    return out


def _train_temporal_lstm_bundle(
    *,
    x_train: np.ndarray,
    len_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    if torch is None or nn is None:
        raise RuntimeError("torch is required for temporal LSTM training")

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalLSTMClassifier(
        input_size=int(x_train.shape[-1]),
        hidden_size=TEMPORAL_LSTM_HIDDEN_SIZE,
        num_layers=TEMPORAL_LSTM_LAYERS,
        dropout=TEMPORAL_LSTM_DROPOUT,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(TEMPORAL_LSTM_LR),
        weight_decay=float(TEMPORAL_LSTM_WEIGHT_DECAY),
    )

    y_float = y_train.astype(np.float32)
    pos_count = float(np.sum(y_float > 0.5))
    neg_count = float(np.sum(y_float <= 0.5))
    if pos_count <= 0.0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = max(1.0, neg_count / max(pos_count, 1.0))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    )

    batch_size = max(1, int(TEMPORAL_LSTM_BATCH_SIZE))
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(y_train), dtype=np.int64)

    best_loss = float("inf")
    best_state: dict[str, np.ndarray] | None = None
    best_epoch = 0
    patience = 0
    loss_curve: list[float] = []

    for epoch_idx in range(1, int(TEMPORAL_LSTM_EPOCHS) + 1):
        rng.shuffle(indices)
        batch_losses: list[float] = []
        model.train()
        for start in range(0, len(indices), batch_size):
            take = indices[start : start + batch_size]
            xb = torch.from_numpy(x_train[take]).to(device=device, dtype=torch.float32)
            lb = torch.from_numpy(len_train[take]).to(device=device, dtype=torch.int64)
            yb = torch.from_numpy(y_float[take]).to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        if not batch_losses:
            continue
        epoch_loss = float(np.mean(batch_losses))
        loss_curve.append(epoch_loss)
        if epoch_loss < (best_loss - 1e-6):
            best_loss = epoch_loss
            best_epoch = int(epoch_idx)
            patience = 0
            best_state = _serialize_state_dict_numpy(model.state_dict())
        else:
            patience += 1
            if patience >= int(TEMPORAL_LSTM_PATIENCE):
                break

    if best_state is None:
        best_state = _serialize_state_dict_numpy(model.state_dict())
        best_epoch = int(max(1, len(loss_curve)))
        best_loss = float(loss_curve[-1]) if loss_curve else 0.0

    return {
        "model_name": "lstm",
        "state_dict": best_state,
        "input_size": int(x_train.shape[-1]),
        "hidden_size": int(TEMPORAL_LSTM_HIDDEN_SIZE),
        "num_layers": int(TEMPORAL_LSTM_LAYERS),
        "dropout": float(TEMPORAL_LSTM_DROPOUT),
        "max_seq_len": int(x_train.shape[1]),
        "epochs_trained": int(len(loss_curve)),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_loss),
        "loss_curve": [float(v) for v in loss_curve],
        "device_used": str(device),
    }


def _predict_temporal_lstm_proba(
    *,
    model_bundle: dict[str, Any],
    x_input: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    if torch is None or nn is None:
        raise RuntimeError("torch is required for temporal LSTM inference")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalLSTMClassifier(
        input_size=int(model_bundle.get("input_size", x_input.shape[-1])),
        hidden_size=int(model_bundle.get("hidden_size", TEMPORAL_LSTM_HIDDEN_SIZE)),
        num_layers=int(model_bundle.get("num_layers", TEMPORAL_LSTM_LAYERS)),
        dropout=float(model_bundle.get("dropout", TEMPORAL_LSTM_DROPOUT)),
    ).to(device)
    state_dict_np = model_bundle.get("state_dict")
    if not isinstance(state_dict_np, dict):
        raise ValueError("LSTM model bundle is missing state_dict")
    state_dict = _restore_state_dict_from_numpy(state_dict_np)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    xb = torch.from_numpy(np.asarray(x_input, dtype=np.float32)).to(device=device)
    lb = torch.from_numpy(np.asarray(lengths, dtype=np.int64)).to(device=device)
    with torch.no_grad():
        logits = model(xb, lb)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    return probs


def _compose_temporal_lstm_sequence(
    *,
    weather_sequence: np.ndarray,
    pred_feats: dict[str, float],
) -> np.ndarray:
    seq = np.asarray(weather_sequence, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError("weather_sequence must be 2D")
    image_vec = np.array(
        [float(pred_feats.get(name, 0.0)) for name in IMAGE_FEATURE_NAMES],
        dtype=np.float32,
    )
    image_seq = np.repeat(image_vec[None, :], seq.shape[0], axis=0)
    combined = np.concatenate([seq, image_seq], axis=1)
    return np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _summarize_temporal_sequence_features(
    weather_sequence: np.ndarray,
) -> dict[str, float]:
    seq = np.asarray(weather_sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[0] <= 0:
        return {}
    seq_df = pd.DataFrame(seq, columns=TEMPORAL_SEQUENCE_FEATURE_NAMES)
    seq_df["_row_id"] = np.arange(len(seq_df), dtype=np.int32)
    seq_df["_date"] = pd.NaT
    row = _build_temporal_weather_feature_row(seq_df)
    return {
        name: float(row.get(name, 0.0))
        for name in TEMPORAL_WEATHER_FEATURE_NAMES
        if row.get(name) is not None
    }


def _prepare_temporal_tabular_frame(
    *,
    base_df: pd.DataFrame,
    temporal_csv_path: Path,
    temporal_bridge_csv_path: Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    temporal_df, temporal_issues = build_temporal_csv_features(temporal_csv_path)
    if not temporal_df.empty:
        issues.extend(temporal_issues)
        merged = pd.merge(
            base_df[["filename", "y_tile", *IMAGE_FEATURE_NAMES]],
            temporal_df,
            on="filename",
            how="inner",
        )
        if merged.empty:
            issues.append(
                make_issue(
                    "temporal_compare",
                    "no_overlap_between_base_rows_and_temporal_csv",
                    details=str(temporal_csv_path),
                )
            )
            return pd.DataFrame(), issues
        ordered_cols = [
            "filename",
            "y_tile",
            *TEMPORAL_WEATHER_FEATURE_NAMES,
            *IMAGE_FEATURE_NAMES,
        ]
        merged = merged[ordered_cols].copy()
        merged["y_tile"] = merged["y_tile"].astype(np.uint8)
        for col in TEMPORAL_WEATHER_FEATURE_NAMES + IMAGE_FEATURE_NAMES:
            merged[col] = (
                pd.to_numeric(merged[col], errors="coerce")
                .fillna(0.0)
                .astype(np.float32)
            )
        return merged, issues

    # Fallback for ERA5-like CSV files that have no filename column:
    # build per-filename temporal weather features via bridge coords/datetime lookup.
    target_names = [
        str(x).strip() for x in base_df["filename"].tolist() if str(x).strip()
    ]
    direct_lookup, direct_issues, _direct_lookup_mode = (
        build_temporal_sequence_lookup_from_base_rows(
            base_df=base_df,
            temporal_csv_path=temporal_csv_path,
        )
    )
    missing_for_hybrid = [name for name in target_names if name not in direct_lookup]
    sequence_lookup: dict[str, np.ndarray] = dict(direct_lookup)
    seq_issues: list[dict[str, Any]] = []
    if missing_for_hybrid:
        hybrid_lookup, seq_issues, _lookup_mode = build_temporal_sequence_lookup_hybrid(
            temporal_csv_path=temporal_csv_path,
            bridge_csv_path=temporal_bridge_csv_path,
            target_filenames=missing_for_hybrid,
        )
        sequence_lookup.update(hybrid_lookup)
    unresolved_after_all = [name for name in target_names if name not in sequence_lookup]
    issues.extend(filter_issues_for_filenames(direct_issues, unresolved_after_all))
    benign_temporal_issues: list[dict[str, Any]] = []
    for issue in temporal_issues:
        issue_type = str(issue.get("issue_type", ""))
        details = str(issue.get("details", ""))
        if issue_type == "csv_missing_columns" and "filename" in details:
            continue
        if issue_type in {"csv_empty_after_filename_filter", "no_rows_after_grouping"}:
            continue
        benign_temporal_issues.append(issue)
    issues.extend(benign_temporal_issues)
    issues.extend(seq_issues)
    if not sequence_lookup:
        return pd.DataFrame(), issues

    missing_lookup_filenames = {
        str(issue.get("filename", "")).strip()
        for issue in seq_issues
        if str(issue.get("filename", "")).strip()
    }
    rows: list[dict[str, Any]] = []
    for row in base_df.itertuples(index=False):
        filename = str(getattr(row, "filename", "")).strip()
        if not filename:
            continue
        seq_weather, status = find_temporal_sequence_record(sequence_lookup, filename)
        if seq_weather is None:
            if filename in missing_lookup_filenames:
                continue
            issues.append(
                make_issue(
                    "temporal_compare",
                    status or "temporal_sequence_missing_for_tabular_frame",
                    filename=filename,
                )
            )
            continue
        seq_arr = np.asarray(seq_weather, dtype=np.float32)
        if seq_arr.ndim != 2 or seq_arr.shape[0] <= 0:
            continue
        seq_df = pd.DataFrame(seq_arr, columns=TEMPORAL_SEQUENCE_FEATURE_NAMES)
        seq_df["_row_id"] = np.arange(len(seq_df), dtype=np.int32)
        seq_df["_date"] = pd.NaT
        weather_row = _build_temporal_weather_feature_row(seq_df)
        out_row: dict[str, Any] = {
            "filename": filename,
            "y_tile": int(getattr(row, "y_tile")),
        }
        out_row.update(weather_row)
        for name in IMAGE_FEATURE_NAMES:
            out_row[name] = float(getattr(row, name))
        rows.append(out_row)

    if not rows:
        return pd.DataFrame(), issues
    merged = pd.DataFrame(rows)
    ordered_cols = [
        "filename",
        "y_tile",
        *TEMPORAL_WEATHER_FEATURE_NAMES,
        *IMAGE_FEATURE_NAMES,
    ]
    merged = merged[ordered_cols].copy()
    merged["y_tile"] = merged["y_tile"].astype(np.uint8)
    for col in TEMPORAL_WEATHER_FEATURE_NAMES + IMAGE_FEATURE_NAMES:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).astype(
            np.float32
        )
    return merged, issues


def train_temporal_risk_model_from_rows(
    *,
    csv_path: Path,
    base_rows_df: pd.DataFrame,
    model_path: Path,
    metrics_path: Path,
    training_table_path: Path,
    seed: int,
    backend_tag: str,
    model_type: str = "gradient_boosting",
    bridge_csv_path: Path | None = None,
) -> dict[str, Any]:
    # Build a temporal risk model on top of segmentation-derived image rows.
    # base_rows_df already contains image-level prediction features; this function
    # merges weather history, fits the requested temporal model, and exports the
    # model bundle + metrics/report artifacts used later by prediction/runtime.
    issues: list[dict[str, Any]] = []
    required_cols = {"filename", "y_tile", *IMAGE_FEATURE_NAMES}
    missing_cols = sorted(required_cols - set(base_rows_df.columns))
    if missing_cols:
        summary = {
            "status": "skipped",
            "reason": "base_rows_missing_columns",
            "missing_columns": missing_cols,
            "backend_tag": backend_tag,
            "requested_model_type": str(model_type),
        }
        save_json(metrics_path, summary)
        return summary

    requested_model_type = str(model_type or "adaboost").strip().lower()
    if requested_model_type not in TEMPORAL_MODEL_TYPE_CHOICES:
        requested_model_type = "adaboost"
    bridge_csv_resolved = (
        bridge_csv_path.resolve() if bridge_csv_path is not None else csv_path.resolve()
    )

    # Keep only the columns needed to train temporal models. Optional spatial
    # columns are preserved because they improve temporal anchoring when filenames
    # alone are not enough to join weather history.
    base_keep_cols = ["filename", "y_tile", *IMAGE_FEATURE_NAMES]
    for optional_col in ["image_path", "lat_grid_mean", "lon_grid_mean"]:
        if optional_col in base_rows_df.columns and optional_col not in base_keep_cols:
            base_keep_cols.append(optional_col)
    if "true_flood_ratio" in base_rows_df.columns:
        base_keep_cols.append("true_flood_ratio")
    base_df = base_rows_df[base_keep_cols].copy()
    base_df["filename"] = base_df["filename"].astype(str).str.strip()
    base_df = base_df[base_df["filename"] != ""].copy()
    for col in IMAGE_FEATURE_NAMES:
        base_df[col] = (
            pd.to_numeric(base_df[col], errors="coerce").fillna(0.0).astype(np.float32)
        )
    base_df["y_tile"] = (
        pd.to_numeric(base_df["y_tile"], errors="coerce").fillna(0).astype(np.uint8)
    )
    if "true_flood_ratio" in base_df.columns:
        base_df["true_flood_ratio"] = pd.to_numeric(
            base_df["true_flood_ratio"], errors="coerce"
        ).fillna(0.0)

    if base_df.empty:
        summary = {
            "status": "skipped",
            "reason": "base_rows_empty_after_filter",
            "backend_tag": backend_tag,
            "requested_model_type": requested_model_type,
            "issues_count": int(len(issues)),
        }
        save_json(metrics_path, summary)
        return summary
    base_row_count = int(len(base_df))
    base_unique_filenames = int(base_df["filename"].nunique())
    temporal_rows_after_merge = 0
    temporal_unique_filenames = 0

    effective_model_type = requested_model_type
    if requested_model_type == "lstm" and (torch is None or nn is None):
        effective_model_type = "adaboost"
        issues.append(
            make_issue(
                "temporal_lstm",
                "torch_missing_fallback_to_adaboost",
                details="torch/nn is not available in current environment",
            )
        )

    feature_cols = list(TEMPORAL_WEATHER_FEATURE_NAMES) + list(IMAGE_FEATURE_NAMES)
    training_df: pd.DataFrame
    if effective_model_type == "lstm":
        # LSTM consumes variable-length sequences later, so at this stage it only
        # needs the cleaned base rows plus optional anchor metadata.
        output_cols = ["filename", "y_tile", *IMAGE_FEATURE_NAMES]
        for optional_col in ["image_path", "lat_grid_mean", "lon_grid_mean"]:
            if optional_col in base_df.columns and optional_col not in output_cols:
                output_cols.append(optional_col)
        if "true_flood_ratio" in base_df.columns:
            output_cols.append("true_flood_ratio")
        training_df = base_df[output_cols].copy()
        temporal_rows_after_merge = int(len(training_df))
        temporal_unique_filenames = int(training_df["filename"].nunique())
    else:
        # Tabular temporal models need one fully merged row per filename where
        # weather history has already been summarized into fixed numeric features.
        merged_tabular, temporal_issues = _prepare_temporal_tabular_frame(
            base_df=base_df,
            temporal_csv_path=csv_path.resolve(),
            temporal_bridge_csv_path=bridge_csv_resolved,
        )
        issues.extend(temporal_issues)
        if merged_tabular.empty:
            summary = {
                "status": "skipped",
                "reason": "temporal_csv_unavailable_or_invalid",
                "backend_tag": backend_tag,
                "requested_model_type": requested_model_type,
                "effective_model_type": effective_model_type,
                "issues_count": int(len(issues)),
            }
            save_json(metrics_path, summary)
            return summary
        temporal_rows_after_merge = int(len(merged_tabular))
        temporal_unique_filenames = int(merged_tabular["filename"].nunique())
        output_cols = ["filename", *feature_cols, "y_tile"]
        if "true_flood_ratio" in base_df.columns:
            output_cols.append("true_flood_ratio")
            merge_true = base_df[["filename", "true_flood_ratio"]].copy()
            merged_tabular = pd.merge(
                merged_tabular,
                merge_true,
                on="filename",
                how="left",
            )
        training_df = merged_tabular[output_cols].copy()

    training_df.to_csv(training_table_path, index=False)

    y_all = training_df["y_tile"].to_numpy(dtype=np.uint8)
    class_counts = np.bincount(y_all, minlength=2).tolist()
    if len(np.unique(y_all)) < 2:
        summary = {
            "status": "skipped",
            "reason": "single_class_target",
            "backend_tag": backend_tag,
            "requested_model_type": requested_model_type,
            "effective_model_type": effective_model_type,
            "n_samples": int(len(y_all)),
            "class_counts": class_counts,
            "feature_columns": feature_cols,
            "issues_count": int(len(issues)),
        }
        save_json(metrics_path, summary)
        return summary

    fold_metrics: list[dict[str, float]] = []
    n_splits = min(5, int(min(class_counts)))
    oof_raw_probabilities = np.full((len(y_all),), np.nan, dtype=np.float32)
    if effective_model_type == "lstm":
        # Sequence lookup happens in two passes:
        # 1) direct anchoring from the base rows
        # 2) hybrid fallback through the bridge CSV for unresolved filenames
        direct_lookup, direct_issues, _direct_lookup_mode = (
            build_temporal_sequence_lookup_from_base_rows(
                base_df=training_df,
                temporal_csv_path=csv_path.resolve(),
            )
        )
        requested_filenames = [str(x) for x in training_df["filename"].tolist()]
        missing_for_hybrid = [name for name in requested_filenames if name not in direct_lookup]
        sequence_lookup = dict(direct_lookup)
        seq_issues: list[dict[str, Any]] = []
        sequence_lookup_mode = "base_rows_anchor"
        if missing_for_hybrid:
            hybrid_lookup, seq_issues, sequence_lookup_mode = (
                build_temporal_sequence_lookup_hybrid(
                    temporal_csv_path=csv_path.resolve(),
                    bridge_csv_path=bridge_csv_resolved,
                    target_filenames=missing_for_hybrid,
                )
            )
            sequence_lookup.update(hybrid_lookup)
        unresolved_after_all = [
            name for name in requested_filenames if name not in sequence_lookup
        ]
        issues.extend(filter_issues_for_filenames(direct_issues, unresolved_after_all))
        issues.extend(seq_issues)

        sequences: list[np.ndarray] = []
        labels: list[int] = []
        sequence_lookup_cached_for_model: dict[str, np.ndarray] = {}
        for row in training_df.itertuples(index=False):
            filename = str(getattr(row, "filename", "")).strip()
            if not filename:
                continue
            seq_weather, seq_status = find_temporal_sequence_record(
                sequence_lookup, filename
            )
            if seq_status is not None or seq_weather is None:
                issues.append(
                    make_issue(
                        "temporal_lstm",
                        "sequence_missing_for_filename",
                        filename=filename,
                        details=str(seq_status),
                    )
                )
                continue
            try:
                pred_row = {
                    name: float(getattr(row, name)) for name in IMAGE_FEATURE_NAMES
                }
            except Exception:
                pred_row = {name: 0.0 for name in IMAGE_FEATURE_NAMES}
            sequence_lookup_cached_for_model[filename] = np.asarray(
                seq_weather, dtype=np.float32
            )
            seq_full = _compose_temporal_lstm_sequence(
                weather_sequence=seq_weather, pred_feats=pred_row
            )
            if seq_full.shape[0] <= 0:
                continue
            sequences.append(seq_full)
            labels.append(int(getattr(row, "y_tile")))

        if not sequences:
            summary = {
                "status": "skipped",
                "reason": "temporal_lstm_no_sequences_after_merge",
                "backend_tag": backend_tag,
                "requested_model_type": requested_model_type,
                "effective_model_type": effective_model_type,
                "sequence_lookup_mode": sequence_lookup_mode,
                "issues_count": int(len(issues)),
            }
            save_json(metrics_path, summary)
            return summary

        y_seq = np.asarray(labels, dtype=np.uint8)
        seq_class_counts = np.bincount(y_seq, minlength=2).tolist()
        if len(np.unique(y_seq)) < 2:
            summary = {
                "status": "skipped",
                "reason": "single_class_target_after_sequence_merge",
                "backend_tag": backend_tag,
                "requested_model_type": requested_model_type,
                "effective_model_type": effective_model_type,
                "n_samples": int(len(y_seq)),
                "class_counts": seq_class_counts,
                "issues_count": int(len(issues)),
            }
            save_json(metrics_path, summary)
            return summary

        max_seq_len = int(
            min(TEMPORAL_LSTM_MAX_SEQ_LEN, max(int(seq.shape[0]) for seq in sequences))
        )
        sequence_lookup_cached_for_model = {
            str(name): np.asarray(seq[-max_seq_len:], dtype=np.float32)
            for name, seq in sequence_lookup_cached_for_model.items()
            if np.asarray(seq).ndim == 2 and int(np.asarray(seq).shape[0]) > 0
        }
        x_all_seq, len_all_seq = _pad_temporal_sequences(
            sequences, max_seq_len=max_seq_len
        )

        oof_raw_probabilities_seq = np.full((len(y_seq),), np.nan, dtype=np.float32)
        seq_n_splits = min(5, int(min(seq_class_counts)))
        if seq_n_splits >= 2:
            skf = StratifiedKFold(
                n_splits=seq_n_splits, shuffle=True, random_state=seed
            )
            for fold_idx, (tr, te) in enumerate(skf.split(x_all_seq, y_seq), start=1):
                fold_bundle = _train_temporal_lstm_bundle(
                    x_train=x_all_seq[tr],
                    len_train=len_all_seq[tr],
                    y_train=y_seq[tr],
                    seed=int(seed + fold_idx),
                )
                prob = _predict_temporal_lstm_proba(
                    model_bundle=fold_bundle,
                    x_input=x_all_seq[te],
                    lengths=len_all_seq[te],
                )
                oof_raw_probabilities_seq[te] = prob.astype(np.float32)
                pred = (prob >= 0.5).astype(np.uint8)
                m = compute_metrics(y_seq[te], pred)
                if len(np.unique(y_seq[te])) == 2:
                    m["roc_auc"] = float(roc_auc_score(y_seq[te], prob))
                    m["brier_score"] = float(brier_score_loss(y_seq[te], prob))
                m["fold"] = fold_idx
                fold_metrics.append(m)

        probability_calibrator = None
        calibrated_oof_prob = np.asarray(oof_raw_probabilities_seq, dtype=np.float32)
        valid_oof_mask = np.isfinite(oof_raw_probabilities_seq)
        probability_calibration = {
            "used": False,
            "method": None,
            "source": "oof_raw_probabilities",
            "reason": "oof_predictions_unavailable",
        }
        threshold_tuning = {
            "used": False,
            "metric": "f1",
            "best_threshold": 0.5,
            "default_threshold": 0.5,
            "reason": "oof_predictions_unavailable",
        }
        oof_evaluation: dict[str, Any] = {}
        if int(np.sum(valid_oof_mask)) >= 2:
            probability_calibrator, calibrated_oof_prob_valid, probability_calibration = (
                _fit_probability_calibrator_from_oof(
                    y_seq[valid_oof_mask],
                    oof_raw_probabilities_seq[valid_oof_mask],
                    seed=int(seed),
                )
            )
            calibrated_oof_prob = np.asarray(oof_raw_probabilities_seq, dtype=np.float32)
            calibrated_oof_prob[valid_oof_mask] = calibrated_oof_prob_valid
            threshold_tuning = _tune_probability_threshold(
                y_seq[valid_oof_mask],
                calibrated_oof_prob[valid_oof_mask],
                metric_name="f1",
            )
            threshold_tuning["source"] = (
                "oof_calibrated_probabilities"
                if bool(probability_calibration.get("used"))
                else "oof_raw_probabilities"
            )
            raw_default_metrics = compute_metrics(
                y_seq[valid_oof_mask],
                (oof_raw_probabilities_seq[valid_oof_mask] >= 0.5).astype(np.uint8),
            )
            raw_default_metrics["roc_auc"] = float(
                roc_auc_score(
                    y_seq[valid_oof_mask], oof_raw_probabilities_seq[valid_oof_mask]
                )
            )
            raw_default_metrics["brier_score"] = float(
                brier_score_loss(
                    y_seq[valid_oof_mask], oof_raw_probabilities_seq[valid_oof_mask]
                )
            )
            oof_evaluation = {
                "n_samples": int(np.sum(valid_oof_mask)),
                "raw_default_threshold": raw_default_metrics,
                "calibrated_default_threshold": threshold_tuning.get("default_metrics"),
                "threshold_tuned": {
                    "threshold": float(threshold_tuning.get("best_threshold", 0.5)),
                    **(
                        threshold_tuning.get("best_metrics")
                        if isinstance(threshold_tuning.get("best_metrics"), dict)
                        else {}
                    ),
                },
            }
        decision_threshold = float(
            np.clip(threshold_tuning.get("best_threshold", 0.5), 0.0, 1.0)
        )

        final_bundle = _train_temporal_lstm_bundle(
            x_train=x_all_seq,
            len_train=len_all_seq,
            y_train=y_seq,
            seed=int(seed),
        )
        save_bundle: dict[str, Any] = {
            "model_name": "lstm",
            "backend_tag": backend_tag,
            "trained_at_utc": utc_now_iso(),
            "feature_columns": feature_cols,
            "sequence_feature_names": list(TEMPORAL_SEQUENCE_FEATURE_NAMES),
            "image_feature_names": list(IMAGE_FEATURE_NAMES),
            "lstm_feature_names": list(TEMPORAL_LSTM_FEATURE_NAMES),
            "target_definition": "y_tile = int(true_flood_ratio >= 0.02)",
            "sensor": "S1",
            "max_seq_len": int(max_seq_len),
            "sequence_lookup_mode": sequence_lookup_mode,
            "temporal_csv_path": str(csv_path.resolve()),
            "bridge_csv_path": str(bridge_csv_resolved),
            "n_samples": int(len(y_seq)),
            "class_counts": seq_class_counts,
            "base_rows_count": int(base_row_count),
            "base_unique_filenames": int(base_unique_filenames),
            "temporal_rows_after_merge": int(len(y_seq)),
            "temporal_unique_filenames": int(len(sequences)),
            "temporal_coverage_ratio": (
                float(len(y_seq) / max(1, base_row_count))
            ),
            "sequence_lookup_cached": sequence_lookup_cached_for_model,
            "probability_calibrator": probability_calibrator,
            "probability_calibration": probability_calibration,
            "decision_threshold": decision_threshold,
            "threshold_tuning": threshold_tuning,
            **final_bundle,
        }
        joblib.dump(save_bundle, model_path)
        issue_report = pack_issue_report(issues)

        summary = {
            "status": "ok",
            "backend_tag": backend_tag,
            "model_name": "lstm",
            "requested_model_type": requested_model_type,
            "effective_model_type": effective_model_type,
            "n_samples": int(len(y_seq)),
            "class_counts": seq_class_counts,
            "n_splits": int(seq_n_splits),
            "base_rows_count": int(base_row_count),
            "base_unique_filenames": int(base_unique_filenames),
            "temporal_rows_after_merge": int(len(y_seq)),
            "temporal_unique_filenames": int(len(sequences)),
            "temporal_coverage_ratio": (
                float(len(y_seq) / max(1, base_row_count))
            ),
            "feature_columns": feature_cols,
            "sequence_feature_names": list(TEMPORAL_SEQUENCE_FEATURE_NAMES),
            "lstm_feature_names": list(TEMPORAL_LSTM_FEATURE_NAMES),
            "max_seq_len": int(max_seq_len),
            "sequence_lookup_mode": sequence_lookup_mode,
            "temporal_csv_path": str(csv_path.resolve()),
            "bridge_csv_path": str(bridge_csv_resolved),
            "issues_count": int(issue_report["issues_count"]),
            "issue_type_counts": issue_report["issue_type_counts"],
            "issues_truncated": bool(issue_report["issues_truncated"]),
            "issues": issue_report["issues"],
            "model_path": str(model_path.resolve()),
            "training_table_path": str(training_table_path.resolve()),
            "fold_metrics": fold_metrics,
            "probability_calibration": probability_calibration,
            "decision_threshold": decision_threshold,
            "threshold_tuning": threshold_tuning,
            "oof_evaluation": oof_evaluation,
            "train_meta": {
                "epochs_trained": int(final_bundle.get("epochs_trained", 0)),
                "best_epoch": int(final_bundle.get("best_epoch", 0)),
                "best_train_loss": float(final_bundle.get("best_train_loss", 0.0)),
                "device_used": str(final_bundle.get("device_used", "cpu")),
            },
        }
        if fold_metrics:
            keys = [k for k in fold_metrics[0].keys() if k != "fold"]
            summary["cv_mean"] = {
                k: float(np.mean([m[k] for m in fold_metrics])) for k in keys
            }
            summary["cv_std"] = {
                k: float(np.std([m[k] for m in fold_metrics])) for k in keys
            }
        save_json(metrics_path, summary)
        return summary

    # For tabular temporal models, learn calibration/thresholds from out-of-fold
    # probabilities instead of training-set probabilities to reduce optimistic bias.
    x_all = training_df[feature_cols].to_numpy(dtype=np.float32)
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (tr, te) in enumerate(skf.split(x_all, y_all), start=1):
            fold_model = build_temporal_risk_model(
                seed + fold_idx, model_type=effective_model_type
            )
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_all[tr])
            _fit_estimator_with_optional_sample_weight(
                fold_model,
                x_all[tr],
                y_all[tr],
                sample_weight=sample_weight,
            )
            prob = fold_model.predict_proba(x_all[te])[:, 1].astype(np.float32)
            oof_raw_probabilities[te] = prob
            pred = (prob >= 0.5).astype(np.uint8)
            m = compute_metrics(y_all[te], pred)
            if len(np.unique(y_all[te])) == 2:
                m["roc_auc"] = float(roc_auc_score(y_all[te], prob))
                m["brier_score"] = float(brier_score_loss(y_all[te], prob))
            m["fold"] = fold_idx
            fold_metrics.append(m)

    probability_calibrator = None
    calibrated_oof_prob = np.asarray(oof_raw_probabilities, dtype=np.float32)
    valid_oof_mask = np.isfinite(oof_raw_probabilities)
    probability_calibration = {
        "used": False,
        "method": None,
        "source": "oof_raw_probabilities",
        "reason": "oof_predictions_unavailable",
    }
    threshold_tuning = {
        "used": False,
        "metric": "f1",
        "best_threshold": 0.5,
        "default_threshold": 0.5,
        "reason": "oof_predictions_unavailable",
    }
    oof_evaluation: dict[str, Any] = {}
    if int(np.sum(valid_oof_mask)) >= 2:
        probability_calibrator, calibrated_oof_prob_valid, probability_calibration = (
            _fit_probability_calibrator_from_oof(
                y_all[valid_oof_mask],
                oof_raw_probabilities[valid_oof_mask],
                seed=int(seed),
            )
        )
        calibrated_oof_prob = np.asarray(oof_raw_probabilities, dtype=np.float32)
        calibrated_oof_prob[valid_oof_mask] = calibrated_oof_prob_valid
        threshold_tuning = _tune_probability_threshold(
            y_all[valid_oof_mask],
            calibrated_oof_prob[valid_oof_mask],
            metric_name="f1",
        )
        threshold_tuning["source"] = (
            "oof_calibrated_probabilities"
            if bool(probability_calibration.get("used"))
            else "oof_raw_probabilities"
        )
        raw_default_metrics = compute_metrics(
            y_all[valid_oof_mask],
            (oof_raw_probabilities[valid_oof_mask] >= 0.5).astype(np.uint8),
        )
        raw_default_metrics["roc_auc"] = float(
            roc_auc_score(y_all[valid_oof_mask], oof_raw_probabilities[valid_oof_mask])
        )
        raw_default_metrics["brier_score"] = float(
            brier_score_loss(y_all[valid_oof_mask], oof_raw_probabilities[valid_oof_mask])
        )
        oof_evaluation = {
            "n_samples": int(np.sum(valid_oof_mask)),
            "raw_default_threshold": raw_default_metrics,
            "calibrated_default_threshold": threshold_tuning.get("default_metrics"),
            "threshold_tuned": {
                "threshold": float(threshold_tuning.get("best_threshold", 0.5)),
                **(
                    threshold_tuning.get("best_metrics")
                    if isinstance(threshold_tuning.get("best_metrics"), dict)
                    else {}
                ),
            },
        }
    decision_threshold = float(
        np.clip(threshold_tuning.get("best_threshold", 0.5), 0.0, 1.0)
    )

    final_model = build_temporal_risk_model(seed, model_type=effective_model_type)
    final_weight = compute_sample_weight(class_weight="balanced", y=y_all)
    _fit_estimator_with_optional_sample_weight(
        final_model,
        x_all,
        y_all,
        sample_weight=final_weight,
    )
    bundle = {
        "model": final_model,
        "model_name": str(effective_model_type),
        "backend_tag": backend_tag,
        "trained_at_utc": utc_now_iso(),
        "feature_columns": feature_cols,
        "target_definition": "y_tile = int(true_flood_ratio >= 0.02)",
        "sensor": "S1",
        "probability_calibrator": probability_calibrator,
        "probability_calibration": probability_calibration,
        "decision_threshold": decision_threshold,
        "threshold_tuning": threshold_tuning,
    }
    joblib.dump(bundle, model_path)
    issue_report = pack_issue_report(issues)

    summary = {
        "status": "ok",
        "backend_tag": backend_tag,
        "model_name": str(effective_model_type),
        "requested_model_type": requested_model_type,
        "effective_model_type": effective_model_type,
        "n_samples": int(len(y_all)),
        "class_counts": class_counts,
        "n_splits": int(n_splits),
        "base_rows_count": int(base_row_count),
        "base_unique_filenames": int(base_unique_filenames),
        "temporal_rows_after_merge": int(temporal_rows_after_merge),
        "temporal_unique_filenames": int(temporal_unique_filenames),
        "temporal_coverage_ratio": (
            float(temporal_rows_after_merge / max(1, base_row_count))
        ),
        "feature_columns": feature_cols,
        "issues_count": int(issue_report["issues_count"]),
        "issue_type_counts": issue_report["issue_type_counts"],
        "issues_truncated": bool(issue_report["issues_truncated"]),
        "issues": issue_report["issues"],
        "model_path": str(model_path.resolve()),
        "training_table_path": str(training_table_path.resolve()),
        "fold_metrics": fold_metrics,
        "probability_calibration": probability_calibration,
        "decision_threshold": decision_threshold,
        "threshold_tuning": threshold_tuning,
        "oof_evaluation": oof_evaluation,
    }
    if fold_metrics:
        keys = [k for k in fold_metrics[0].keys() if k != "fold"]
        summary["cv_mean"] = {
            k: float(np.mean([m[k] for m in fold_metrics])) for k in keys
        }
        summary["cv_std"] = {
            k: float(np.std([m[k] for m in fold_metrics])) for k in keys
        }
    save_json(metrics_path, summary)
    return summary


def get_temporal_model_path(*, output_dir: Path, backend: str = PIPELINE_V3_BACKEND_ID) -> Path:
    _ = backend
    return (output_dir / RISK_TEMPORAL_PIPELINE_NAME).resolve()


def resolve_temporal_paths(
    args: argparse.Namespace, default_csv_path: Path
) -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent
    temporal_csv = (
        Path(args.temporal_csv_path).resolve()
        if getattr(args, "temporal_csv_path", None)
        else resolve_env_path(
            "TEMPORAL_CSV_PATH",
            base_dir=base_dir,
            default_relative=DEFAULT_TEMPORAL_CSV_RELATIVE,
        ).resolve()
    )
    temporal_bridge_csv = (
        Path(args.temporal_bridge_csv_path).resolve()
        if getattr(args, "temporal_bridge_csv_path", None)
        else resolve_env_path(
            "TEMPORAL_BRIDGE_CSV_PATH",
            base_dir=base_dir,
            default_relative=str(default_csv_path),
        ).resolve()
    )
    return temporal_csv, temporal_bridge_csv


def _resolve_temporal_model_bundle(
    model_or_bundle: Any,
) -> tuple[Any | None, list[str], str, dict[str, Any]]:
    if model_or_bundle is None:
        return None, list(TEMPORAL_FEATURE_NAMES), "gradient_boosting", {}
    if isinstance(model_or_bundle, dict):
        model_name = (
            str(model_or_bundle.get("model_name", "gradient_boosting")).strip().lower()
        )
        feature_cols = model_or_bundle.get("feature_columns")
        if not isinstance(feature_cols, list) or not feature_cols:
            feature_cols = list(TEMPORAL_FEATURE_NAMES)
        if model_name == "lstm":
            return model_or_bundle, [str(x) for x in feature_cols], "lstm", model_or_bundle
        model = model_or_bundle.get("model")
        return model, [str(x) for x in feature_cols], model_name, model_or_bundle
    return model_or_bundle, list(TEMPORAL_FEATURE_NAMES), "gradient_boosting", {}


def predict_temporal_risk(
    *,
    image_filename: str | None,
    sensor: str | None,
    pred_feats: dict[str, float],
    csv_path: Path,
    temporal_model: Any | None,
    risk_threshold: float = 0.5,
    bridge_csv_path: Path | None = None,
    image_path: Path | None = None,
    geo_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "temporal_status": "unavailable",
        "temporal_risk_score": None,
        "temporal_risk_score_percent": None,
        "temporal_risk_label": None,
        "temporal_risk_text": "unavailable",
        "temporal_model_used": None,
        "temporal_weather_match_status": None,
        "temporal_horizon": "short_term_sequence_window",
        "temporal_sensor": str(sensor).upper() if sensor is not None else None,
        "temporal_sensor_policy": "shared_model_cross_sensor",
        "temporal_risk_threshold": None,
        "temporal_risk_threshold_source": None,
        "temporal_probability_calibrated": False,
        "temporal_probability_calibration_method": None,
    }

    model, feature_cols, model_name, temporal_bundle = _resolve_temporal_model_bundle(
        temporal_model
    )
    if model is None:
        out["temporal_status"] = "temporal_model_missing"
        return out
    probability_calibrator = (
        temporal_bundle.get("probability_calibrator")
        if isinstance(temporal_bundle, dict)
        else None
    )
    probability_calibration_meta = (
        temporal_bundle.get("probability_calibration")
        if isinstance(temporal_bundle, dict)
        else None
    )
    effective_threshold = float(risk_threshold)
    threshold_source = "runtime_risk_threshold"
    if isinstance(temporal_bundle, dict):
        stored_threshold = temporal_bundle.get("decision_threshold")
        try:
            if stored_threshold not in (None, ""):
                effective_threshold = float(np.clip(float(stored_threshold), 0.0, 1.0))
                threshold_source = "model_bundle"
        except Exception:
            pass

    image_name = str(image_filename).strip() if image_filename is not None else ""
    anchor_ctx = infer_temporal_anchor_context(
        image_filename=(
            image_name
            if image_name
            else (image_path.name if image_path is not None else None)
        ),
        image_path=image_path,
        geo_meta=geo_meta,
    )
    if anchor_ctx.get("lat") is not None and anchor_ctx.get("lon") is not None:
        out["temporal_anchor_lat"] = float(anchor_ctx["lat"])
        out["temporal_anchor_lon"] = float(anchor_ctx["lon"])
    if anchor_ctx.get("anchor_date_iso"):
        out["temporal_anchor_time_utc"] = str(anchor_ctx["anchor_date_iso"])
    if anchor_ctx.get("anchor_source"):
        out["temporal_anchor_source"] = str(anchor_ctx["anchor_source"])

    if model_name == "lstm":

        def _predict_from_sequence(
            seq_weather: np.ndarray, lookup_mode: str
        ) -> dict[str, Any]:
            temporal_snapshot = _summarize_temporal_sequence_features(seq_weather)
            try:
                seq_full = _compose_temporal_lstm_sequence(
                    weather_sequence=seq_weather, pred_feats=pred_feats
                )
                max_seq_len = (
                    int(model.get("max_seq_len", TEMPORAL_LSTM_MAX_SEQ_LEN))
                    if isinstance(model, dict)
                    else int(TEMPORAL_LSTM_MAX_SEQ_LEN)
                )
                x_row, len_row = _pad_temporal_sequences(
                    [seq_full], max_seq_len=max_seq_len
                )
                prob = float(
                    _predict_temporal_lstm_proba(
                        model_bundle=model, x_input=x_row, lengths=len_row
                    )[0]
                )
                prob = float(
                    _apply_probability_calibrator(probability_calibrator, [prob])[0]
                )
            except Exception as ex:
                out["temporal_status"] = "prediction_failed"
                out["temporal_error"] = str(ex)
                return out

            label = int(prob >= float(effective_threshold))
            out.update(
                {
                    "temporal_status": "ok",
                    "temporal_risk_score": prob,
                    "temporal_risk_score_percent": to_percent(prob),
                    "temporal_risk_label": label,
                    "temporal_risk_text": (
                        "flood_risk_predicted"
                        if label == 1
                        else "no_flood_risk_predicted"
                    ),
                    "temporal_model_used": "lstm",
                    "temporal_weather_match_status": (
                        "matched_by_coords"
                        if lookup_mode == "coords_datetime"
                        else (
                            "matched_by_time_only"
                            if lookup_mode == "time_only"
                            else "matched"
                        )
                    ),
                    "temporal_lookup_mode": lookup_mode,
                    "temporal_risk_threshold": float(effective_threshold),
                    "temporal_risk_threshold_source": threshold_source,
                    "temporal_probability_calibrated": bool(
                        isinstance(probability_calibration_meta, dict)
                        and probability_calibration_meta.get("used")
                    ),
                    "temporal_probability_calibration_method": (
                        str(probability_calibration_meta.get("method"))
                        if isinstance(probability_calibration_meta, dict)
                        and probability_calibration_meta.get("method")
                        else None
                    ),
                }
            )
            if temporal_snapshot:
                out["temporal_feature_snapshot"] = temporal_snapshot
            return out

        model_sequence_lookup: dict[str, np.ndarray] = {}
        if isinstance(model, dict):
            cached_lookup_raw = model.get("sequence_lookup_cached")
            if isinstance(cached_lookup_raw, dict):
                for key, value in cached_lookup_raw.items():
                    try:
                        arr = np.asarray(value, dtype=np.float32)
                    except Exception:
                        continue
                    if arr.ndim == 2 and arr.shape[0] > 0:
                        model_sequence_lookup[str(key)] = arr

        model_temporal_csv = None
        if isinstance(model, dict):
            temporal_raw = model.get("temporal_csv_path")
            if isinstance(temporal_raw, str) and temporal_raw.strip():
                candidate = Path(temporal_raw)
                if candidate.exists():
                    model_temporal_csv = candidate
        model_bridge = None
        if isinstance(model, dict):
            bridge_raw = model.get("bridge_csv_path")
            if isinstance(bridge_raw, str) and bridge_raw.strip():
                bridge_candidate = Path(bridge_raw)
                if bridge_candidate.exists():
                    model_bridge = bridge_candidate
        temporal_source_csv = (
            model_temporal_csv if model_temporal_csv is not None else csv_path
        )
        bridge = bridge_csv_path if bridge_csv_path is not None else model_bridge

        last_status: str | None = None
        seq_weather: np.ndarray | None = None

        if seq_weather is None:
            anchor_lookup_mode = "coords_datetime"
            if anchor_ctx.get("lat") is None or anchor_ctx.get("lon") is None:
                anchor_lookup_mode = "time_only"
            seq_weather, anchor_status = lookup_temporal_sequence_for_anchor(
                csv_path=temporal_source_csv.resolve(),
                lat=anchor_ctx.get("lat"),
                lon=anchor_ctx.get("lon"),
                anchor_date=anchor_ctx.get("anchor_date"),
            )
            if seq_weather is not None and anchor_status is None:
                return _predict_from_sequence(seq_weather, anchor_lookup_mode)
            if anchor_status is not None:
                last_status = anchor_status

        if seq_weather is None and model_sequence_lookup and image_name:
            seq_weather, weather_status = find_temporal_sequence_record(
                model_sequence_lookup, image_name
            )
            if seq_weather is not None and weather_status is None:
                return _predict_from_sequence(seq_weather, "model_cached")
            last_status = weather_status or "temporal_csv_missing_filename"

        if seq_weather is None and image_name:
            sequence_lookup, issues, sequence_lookup_mode = (
                build_temporal_sequence_lookup_hybrid(
                    temporal_csv_path=temporal_source_csv.resolve(),
                    bridge_csv_path=bridge,
                    target_filenames=[image_name],
                )
            )
            if sequence_lookup:
                seq_weather, weather_status = find_temporal_sequence_record(
                    sequence_lookup, image_name
                )
                if seq_weather is not None and weather_status is None:
                    return _predict_from_sequence(seq_weather, sequence_lookup_mode)
                last_status = weather_status or "temporal_csv_missing_filename"
            else:
                last_status = "temporal_csv_invalid" if issues else "temporal_csv_empty"

        out["temporal_status"] = last_status or (
            "missing_filename" if not image_name else "temporal_csv_missing_filename"
        )
        out["temporal_weather_match_status"] = out["temporal_status"]
        return out

    temporal_weather: dict[str, float] = {}
    weather_status: str | None = None
    temporal_lookup_mode = "coords_datetime"
    if anchor_ctx.get("lat") is None or anchor_ctx.get("lon") is None:
        temporal_lookup_mode = "time_only"
    temporal_weather_anchor, anchor_status = lookup_temporal_features_for_anchor(
        csv_path=csv_path,
        lat=anchor_ctx.get("lat"),
        lon=anchor_ctx.get("lon"),
        anchor_date=anchor_ctx.get("anchor_date"),
    )
    if anchor_status is None and temporal_weather_anchor:
        temporal_weather = temporal_weather_anchor
        weather_status = None
    else:
        weather_status = anchor_status
        if image_name:
            temporal_weather_name, name_status = lookup_temporal_features_for_filename(
                csv_path, image_name
            )
            if name_status is None and temporal_weather_name:
                temporal_weather = temporal_weather_name
                weather_status = None
                temporal_lookup_mode = "filename"
            else:
                weather_status = weather_status or name_status
        elif weather_status is None:
            weather_status = "missing_filename"

    if weather_status is not None:
        out["temporal_status"] = weather_status
        out["temporal_weather_match_status"] = weather_status
        out["temporal_lookup_mode"] = temporal_lookup_mode
        return out

    feature_map: dict[str, float] = {**temporal_weather}
    for name in IMAGE_FEATURE_NAMES:
        try:
            feature_map[name] = float(pred_feats.get(name, 0.0))
        except Exception:
            feature_map[name] = 0.0
    missing = [name for name in feature_cols if name not in feature_map]
    if missing:
        out["temporal_status"] = "feature_schema_mismatch"
        out["temporal_missing_features"] = missing
        return out

    row = np.array(
        [feature_map[name] for name in feature_cols], dtype=np.float32
    ).reshape(1, -1)
    try:
        prob = float(model.predict_proba(row)[0, 1])
    except Exception as ex:
        out["temporal_status"] = "prediction_failed"
        out["temporal_error"] = str(ex)
        return out

    prob = float(_apply_probability_calibrator(probability_calibrator, [prob])[0])
    label = int(prob >= float(effective_threshold))
    out.update(
        {
            "temporal_status": "ok",
            "temporal_risk_score": prob,
            "temporal_risk_score_percent": to_percent(prob),
            "temporal_risk_label": label,
            "temporal_risk_text": (
                "flood_risk_predicted" if label == 1 else "no_flood_risk_predicted"
            ),
            "temporal_model_used": model_name,
            "temporal_weather_match_status": (
                "matched_by_coords"
                if temporal_lookup_mode == "coords_datetime"
                else (
                    "matched_by_time_only"
                    if temporal_lookup_mode == "time_only"
                    else "matched"
                )
            ),
            "temporal_lookup_mode": temporal_lookup_mode,
            "temporal_risk_threshold": float(effective_threshold),
            "temporal_risk_threshold_source": threshold_source,
            "temporal_probability_calibrated": bool(
                isinstance(probability_calibration_meta, dict)
                and probability_calibration_meta.get("used")
            ),
            "temporal_probability_calibration_method": (
                str(probability_calibration_meta.get("method"))
                if isinstance(probability_calibration_meta, dict)
                and probability_calibration_meta.get("method")
                else None
            ),
        }
    )
    if temporal_weather:
        out["temporal_feature_snapshot"] = {
            name: float(temporal_weather.get(name, 0.0))
            for name in TEMPORAL_WEATHER_FEATURE_NAMES
            if temporal_weather.get(name) is not None
        }
    return out


def build_prediction_eta(
    *,
    detection_label: int | None,
    prediction_label: int | None,
    temporal_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    out: dict[str, Any] = {
        "prediction_applicable": False,
        "prediction_status": "unavailable",
        "prediction_eta_text": "unavailable",
        "prediction_eta_start_utc": None,
        "prediction_eta_end_utc": None,
        "prediction_eta_source": None,
        "prediction_eta_note": None,
        "prediction_eta_horizon": None,
        "prediction_eta_days_min": None,
        "prediction_eta_days_max": None,
        "prediction_eta_hours_min": None,
        "prediction_eta_hours_max": None,
        "prediction_eta_confidence_percent": None,
        "prediction_eta_confidence_level": None,
    }

    if detection_label == 1:
        out.update(
            {
                "prediction_applicable": False,
                "prediction_status": "suppressed_due_to_detection",
                "prediction_eta_text": "not_applicable_flood_already_detected",
                "prediction_eta_note": "Prediction is disabled because flood is currently detected.",
            }
        )
        return out

    if prediction_label is None:
        out.update(
            {
                "prediction_applicable": False,
                "prediction_status": "no_prediction_label",
                "prediction_eta_text": "prediction_unavailable",
            }
        )
        return out

    out["prediction_applicable"] = True
    out["prediction_status"] = "active"

    temporal = temporal_payload if isinstance(temporal_payload, dict) else {}
    temporal_status = str(temporal.get("temporal_status", "unavailable"))
    temporal_score = temporal.get("temporal_risk_score")

    if temporal_status == "ok" and temporal_score is not None:
        try:
            score = float(temporal_score)
        except Exception:
            score = None
        if score is None:
            out.update(
                {
                    "prediction_eta_text": "time_window_unavailable",
                    "prediction_eta_source": "temporal",
                }
            )
            return out

        temporal_thr = 0.5
        try:
            temporal_thr = float(temporal.get("temporal_risk_threshold", 0.5))
        except Exception:
            temporal_thr = 0.5
        if score >= temporal_thr:
            denom = max(1e-6, 1.0 - temporal_thr)
            margin = (score - temporal_thr) / denom
        else:
            denom = max(1e-6, temporal_thr)
            margin = (temporal_thr - score) / denom
        margin = float(np.clip(margin, 0.0, 1.0))
        conf_pct = 50.0 + 50.0 * margin
        if conf_pct >= 85.0:
            conf_level = "high"
        elif conf_pct >= 70.0:
            conf_level = "medium"
        else:
            conf_level = "low"
        out["prediction_eta_confidence_percent"] = float(conf_pct)
        out["prediction_eta_confidence_level"] = conf_level

        if score >= 0.90:
            h_start, h_end = 6, 12
        elif score >= 0.80:
            h_start, h_end = 12, 24
        elif score >= 0.75:
            d_start, d_end = 1, 3
        elif score >= 0.55:
            d_start, d_end = 3, 7
        elif score >= 0.35:
            d_start, d_end = 7, 14
        elif score >= 0.20:
            d_start, d_end = 14, 30
        elif score >= 0.14:
            d_start, d_end = 30, 90
        elif score >= 0.10:
            d_start, d_end = 90, 180
        elif score >= 0.07:
            d_start, d_end = 180, 365
        elif score >= 0.04:
            d_start, d_end = 365, 730
        elif score >= 0.02:
            d_start, d_end = 730, 1825
        else:
            out.update(
                {
                    "prediction_eta_text": "no_elevated_signal_next_5_years",
                    "prediction_eta_source": "temporal",
                    "prediction_eta_note": "Very low temporal risk score; no strong signal in the next 5 years.",
                    "prediction_eta_horizon": "long_term",
                }
            )
            return out

        if score >= 0.80:
            start_dt = now_utc + timedelta(hours=int(h_start))
            end_dt = now_utc + timedelta(hours=int(h_end))
            out.update(
                {
                    "prediction_eta_text": f"possible_flood_window_{h_start}_to_{h_end}_hours",
                    "prediction_eta_start_utc": start_dt.isoformat(),
                    "prediction_eta_end_utc": end_dt.isoformat(),
                    "prediction_eta_source": "temporal_immediate",
                    "prediction_eta_note": "Heuristic immediate window from temporal risk score, not an exact event time.",
                    "prediction_eta_hours_min": int(h_start),
                    "prediction_eta_hours_max": int(h_end),
                    "prediction_eta_horizon": "immediate_term",
                }
            )
            return out

        start_dt = now_utc + timedelta(days=int(d_start))
        end_dt = now_utc + timedelta(days=int(d_end))
        if d_end <= 30:
            horizon = "short_term"
        elif d_end <= 365:
            horizon = "mid_term"
        else:
            horizon = "long_term"
        source_tag = "temporal" if horizon != "long_term" else "temporal_long_range"
        out.update(
            {
                "prediction_eta_text": f"possible_flood_window_{d_start}_to_{d_end}_days",
                "prediction_eta_start_utc": start_dt.isoformat(),
                "prediction_eta_end_utc": end_dt.isoformat(),
                "prediction_eta_source": source_tag,
                "prediction_eta_note": "Heuristic window from temporal risk score, not an exact event time.",
                "prediction_eta_days_min": int(d_start),
                "prediction_eta_days_max": int(d_end),
                "prediction_eta_horizon": horizon,
            }
        )
        return out

    out.update(
        {
            "prediction_eta_text": "time_window_unavailable_temporal_data_required",
            "prediction_eta_source": "risk_only",
            "prediction_eta_note": "Temporal CSV match/model is needed for time-window estimation.",
        }
    )
    return out


def summarize_prediction_features(
    pred_mask: np.ndarray, pred_prob: np.ndarray
) -> dict[str, float]:
    return {
        "pred_flood_ratio": float(np.mean(pred_mask)),
        "pred_prob_mean": float(np.mean(pred_prob)),
        "pred_prob_p90": float(np.quantile(pred_prob, 0.9)),
    }


def to_percent(value: float | None, ndigits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100.0, ndigits)


# ==============================
# Calibration / Threshold / Monitoring Utilities
# ==============================
# This section controls decision thresholding, calibrated probability models,
# input drift checks, geospatial validation, and prediction audit logs.
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_risk_threshold(profile_name: str, override: float | None) -> float:
    if override is not None:
        value = float(override)
    else:
        value = float(
            RISK_THRESHOLD_PROFILES.get(
                profile_name, RISK_THRESHOLD_PROFILES[DEFAULT_RISK_THRESHOLD_PROFILE]
            )
        )
    return float(np.clip(value, 0.0, 1.0))


def resolve_segmentation_threshold(
    requested_threshold: float | None,
    *,
    bundle_threshold: Any = None,
    default_threshold: float = 0.5,
) -> tuple[float, str]:
    runtime_value = float(
        requested_threshold if requested_threshold is not None else default_threshold
    )
    runtime_value = float(np.clip(runtime_value, 0.0, 1.0))
    bundle_value = _to_float_or_none(bundle_threshold)
    if bundle_value is not None:
        bundle_value = float(np.clip(bundle_value, 0.0, 1.0))
    # If the caller did not move off the default threshold, prefer the tuned bundle value.
    if bundle_value is not None and abs(runtime_value - float(default_threshold)) <= 1e-8:
        return float(bundle_value), "model_bundle"
    return float(runtime_value), "runtime_argument"


def _fit_estimator_with_optional_sample_weight(
    estimator: Any,
    x_data: np.ndarray,
    y_data: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> Any:
    if sample_weight is not None:
        try:
            estimator.fit(x_data, y_data, sample_weight=sample_weight)
            return estimator
        except TypeError:
            pass
    estimator.fit(x_data, y_data)
    return estimator


def _apply_probability_calibrator(
    calibrator: Any | None,
    raw_probabilities: np.ndarray | list[float] | float,
) -> np.ndarray:
    raw = np.asarray(raw_probabilities, dtype=np.float32).reshape(-1)
    raw = np.clip(raw, 1e-6, 1.0 - 1e-6)
    if calibrator is None:
        return raw.astype(np.float32)
    try:
        calibrated = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.clip(
            np.asarray(calibrated, dtype=np.float32), 0.0, 1.0
        ).astype(np.float32)
    except Exception:
        return raw.astype(np.float32)


def _fit_probability_calibrator_from_oof(
    y_true: np.ndarray,
    raw_probabilities: np.ndarray,
    *,
    seed: int,
) -> tuple[Any | None, np.ndarray, dict[str, Any]]:
    summary: dict[str, Any] = {
        "used": False,
        "method": None,
        "source": "oof_raw_probabilities",
        "n_samples": 0,
        "class_counts": None,
        "reason": None,
        "brier_score_before": None,
        "brier_score_after": None,
        "roc_auc_before": None,
        "roc_auc_after": None,
    }
    y_arr = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    prob_arr = np.asarray(raw_probabilities, dtype=np.float32).reshape(-1)
    valid_mask = np.isfinite(prob_arr)
    y_arr = y_arr[valid_mask]
    prob_arr = np.clip(prob_arr[valid_mask], 1e-6, 1.0 - 1e-6)
    summary["n_samples"] = int(y_arr.size)
    if y_arr.size <= 0:
        summary["reason"] = "empty_oof_probabilities"
        return None, prob_arr.astype(np.float32), summary

    class_counts = np.bincount(y_arr, minlength=2).tolist()
    summary["class_counts"] = class_counts
    if len(np.unique(y_arr)) < 2:
        summary["reason"] = "single_class_target"
        return None, prob_arr.astype(np.float32), summary
    if int(min(class_counts)) < 2:
        summary["reason"] = "insufficient_minority_support"
        return None, prob_arr.astype(np.float32), summary

    calibrator = LogisticRegression(max_iter=2000, random_state=seed)
    try:
        calibrator.fit(prob_arr.reshape(-1, 1), y_arr)
    except Exception as ex:
        summary["reason"] = f"calibrator_fit_failed: {ex}"
        return None, prob_arr.astype(np.float32), summary
    calibrated_prob = _apply_probability_calibrator(calibrator, prob_arr)
    summary.update(
        {
            "used": True,
            "method": "platt_scaling_oof",
            "brier_score_before": float(brier_score_loss(y_arr, prob_arr)),
            "brier_score_after": float(brier_score_loss(y_arr, calibrated_prob)),
        }
    )
    if len(np.unique(y_arr)) == 2:
        summary["roc_auc_before"] = float(roc_auc_score(y_arr, prob_arr))
        summary["roc_auc_after"] = float(roc_auc_score(y_arr, calibrated_prob))
    return calibrator, calibrated_prob.astype(np.float32), summary


def _tune_probability_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    metric_name: str = "f1",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "used": False,
        "metric": metric_name,
        "best_threshold": 0.5,
        "default_threshold": 0.5,
        "thresholds_evaluated": 0,
        "default_metrics": None,
        "best_metrics": None,
        "improvement": None,
        "reason": None,
    }
    y_arr = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    prob_arr = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    valid_mask = np.isfinite(prob_arr)
    y_arr = y_arr[valid_mask]
    prob_arr = np.clip(prob_arr[valid_mask], 0.0, 1.0)
    if y_arr.size <= 0:
        out["reason"] = "empty_probability_vector"
        return out
    if len(np.unique(y_arr)) < 2:
        out["reason"] = "single_class_target"
        return out

    grid = np.linspace(0.05, 0.95, 37, dtype=np.float32)
    quantiles = np.quantile(prob_arr, np.linspace(0.05, 0.95, 19))
    thresholds = np.unique(np.round(np.concatenate([grid, quantiles]), 4))
    default_pred = (prob_arr >= 0.5).astype(np.uint8)
    default_metrics = compute_metrics(y_arr, default_pred)
    out["default_metrics"] = default_metrics

    best_payload: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float, float] | None = None
    for threshold in thresholds:
        pred = (prob_arr >= float(threshold)).astype(np.uint8)
        metrics = compute_metrics(y_arr, pred)
        rank_key = (
            float(metrics.get(metric_name, 0.0)),
            float(metrics.get("iou", 0.0)),
            float(metrics.get("precision", 0.0)),
            float(metrics.get("recall", 0.0)),
            float(-abs(float(threshold) - 0.5)),
        )
        if best_key is None or rank_key > best_key:
            best_key = rank_key
            best_payload = {
                "threshold": float(threshold),
                **metrics,
            }

    out["thresholds_evaluated"] = int(len(thresholds))
    if best_payload is None:
        out["reason"] = "threshold_search_failed"
        return out

    best_threshold = float(best_payload["threshold"])
    best_metrics = {k: v for k, v in best_payload.items() if k != "threshold"}
    out.update(
        {
            "used": True,
            "best_threshold": best_threshold,
            "best_metrics": best_metrics,
            "improvement": {
                metric_name: float(
                    best_metrics.get(metric_name, 0.0)
                    - default_metrics.get(metric_name, 0.0)
                ),
                "iou": float(
                    best_metrics.get("iou", 0.0) - default_metrics.get("iou", 0.0)
                ),
                "precision": float(
                    best_metrics.get("precision", 0.0)
                    - default_metrics.get("precision", 0.0)
                ),
                "recall": float(
                    best_metrics.get("recall", 0.0)
                    - default_metrics.get("recall", 0.0)
                ),
            },
        }
    )
    return out


def load_training_input_profile(output_dir: Path) -> dict[str, Any] | None:
    p = output_dir / "input_profile.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def detect_input_drift(
    x_img: np.ndarray,
    sensor: str,
    profile: dict[str, Any] | None,
    *,
    zscore_threshold: float = DEFAULT_DRIFT_ZSCORE_THRESHOLD,
) -> dict[str, Any]:
    # Simple runtime drift check: compare mean channel intensity of the incoming
    # image against the training profile stored with the promoted run.
    result: dict[str, Any] = {"status": "not_checked", "warnings": []}
    if profile is None:
        result["status"] = "profile_missing"
        return result
    sensor_profile = (
        profile.get("sensors", {}).get(sensor)
        if isinstance(profile.get("sensors"), dict)
        else None
    )
    if not isinstance(sensor_profile, dict) or sensor_profile.get("status") != "ok":
        result["status"] = "sensor_profile_missing"
        return result

    mean_ref = sensor_profile.get("mean")
    std_ref = sensor_profile.get("std")
    if not isinstance(mean_ref, list) or not isinstance(std_ref, list):
        result["status"] = "invalid_profile"
        return result

    c = x_img.shape[-1]
    if len(mean_ref) != c or len(std_ref) != c:
        result["status"] = "channel_mismatch"
        result["warnings"].append("channel_count_mismatch_with_training_profile")
        return result

    img_mean = np.mean(x_img.reshape(-1, c), axis=0)
    zscores: list[float] = []
    warnings: list[str] = []
    for i in range(c):
        denom = max(float(std_ref[i]), DEFAULT_CHANNEL_EPS)
        z = abs((float(img_mean[i]) - float(mean_ref[i])) / denom)
        zscores.append(float(z))
        if z > zscore_threshold:
            warnings.append(f"channel_{i}_mean_zscore_{z:.2f}_above_{zscore_threshold}")

    result["status"] = "ok"
    result["zscore_threshold"] = float(zscore_threshold)
    result["channel_mean_zscores"] = zscores
    result["warnings"] = warnings
    return result


@lru_cache(maxsize=512)
def _inspect_geospatial_metadata_cached(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    payload: dict[str, Any] = {
        "status": "ok",
        "path": str(path),
        "warnings": [],
        "epsg": None,
        "pixel_scale": None,
        "tiepoint": None,
        "center_lat": None,
        "center_lon": None,
        "image_datetime_utc": None,
        "has_model_tiepoint": False,
        "has_geotiff_metadata": False,
    }
    try:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            tags = {tag.name: tag.value for tag in page.tags.values()}
            geo = tf.geotiff_metadata if isinstance(tf.geotiff_metadata, dict) else {}
            payload["has_geotiff_metadata"] = bool(geo)
            payload["has_model_tiepoint"] = "ModelTiepointTag" in tags
            if "ModelTiepointTag" in tags:
                raw_tie = tags["ModelTiepointTag"]
                if isinstance(raw_tie, (tuple, list, np.ndarray)) and len(raw_tie) >= 6:
                    payload["tiepoint"] = [float(raw_tie[3]), float(raw_tie[4])]
            if "ModelPixelScaleTag" in tags:
                raw_scale = tags["ModelPixelScaleTag"]
                if (
                    isinstance(raw_scale, (tuple, list, np.ndarray))
                    and len(raw_scale) >= 2
                ):
                    payload["pixel_scale"] = [float(raw_scale[0]), float(raw_scale[1])]
            epsg = None
            if isinstance(geo, dict):
                epsg = geo.get("ProjectedCSTypeGeoKey") or geo.get(
                    "GeographicTypeGeoKey"
                )
            payload["epsg"] = int(epsg) if isinstance(epsg, (int, np.integer)) else None
            payload["image_shape"] = [
                int(page.imagelength),
                int(page.imagewidth),
                int(page.samplesperpixel or 1),
            ]
            raw_dt = (
                tags.get("DateTime")
                or tags.get("DateTimeOriginal")
                or tags.get("DateTimeDigitized")
            )
            parsed_dt = _parse_utc_timestamp(raw_dt)
            if parsed_dt is not None:
                payload["image_datetime_utc"] = parsed_dt.isoformat()
    except Exception as ex:
        payload["status"] = "error"
        payload["warnings"].append(f"geospatial_read_failed: {ex}")
        return payload

    if not payload["has_geotiff_metadata"]:
        payload["warnings"].append("missing_geotiff_metadata")
    if payload["epsg"] is None:
        payload["warnings"].append("missing_epsg")
    if payload["pixel_scale"] is None:
        payload["warnings"].append("missing_model_pixel_scale")
    else:
        sx, sy = payload["pixel_scale"]
        if sx <= 0 or sy <= 0:
            payload["warnings"].append("invalid_pixel_scale_non_positive")
    if not payload["has_model_tiepoint"]:
        payload["warnings"].append("missing_model_tiepoint")

    center_lat, center_lon, _ = _extract_lat_lon_from_geo_meta(payload)
    if center_lat is not None and center_lon is not None:
        payload["center_lat"] = float(center_lat)
        payload["center_lon"] = float(center_lon)
    return payload


def inspect_geospatial_metadata(path: Path) -> dict[str, Any]:
    # GeoTIFF tags are immutable per file. Cache the parsed payload to avoid
    # reopening the same TIFF across metadata export, training, and prediction.
    return copy.deepcopy(_inspect_geospatial_metadata_cached(str(path.resolve())))


def enforce_geospatial_checks(geo_meta: dict[str, Any], strict: bool) -> None:
    if geo_meta.get("status") == "error":
        raise ValueError("Failed to read GeoTIFF metadata.")
    warnings = geo_meta.get("warnings", [])
    if strict and warnings:
        raise ValueError(f"Geospatial checks failed: {', '.join(warnings)}")


def estimate_flood_area_km2(
    pred_mask: np.ndarray, geo_meta: dict[str, Any]
) -> float | None:
    pixel_scale = geo_meta.get("pixel_scale")
    if not isinstance(pixel_scale, list) or len(pixel_scale) < 2:
        return None
    try:
        sx = float(pixel_scale[0])
        sy = float(pixel_scale[1])
    except Exception:
        return None
    if sx <= 0 or sy <= 0:
        return None
    flood_pixels = int(np.sum(pred_mask > 0))
    area_m2 = float(flood_pixels) * sx * sy
    return round(area_m2 / 1_000_000.0, 6)


def build_geo_summary(
    pred_mask: np.ndarray, geo_meta: dict[str, Any]
) -> dict[str, Any]:
    # Convert raw geospatial tags + predicted mask footprint into a UI/API-friendly
    # summary that can be shown without reopening the source TIFF.
    flood_pixels = int(np.sum(pred_mask > 0))
    flood_area_km2 = estimate_flood_area_km2(pred_mask, geo_meta)
    shape = geo_meta.get("image_shape")
    width = None
    height = None
    if isinstance(shape, list) and len(shape) >= 2:
        try:
            height = int(shape[0])
            width = int(shape[1])
        except Exception:
            height = None
            width = None

    summary: dict[str, Any] = {
        "status": "ok",
        "flood_pixels": flood_pixels,
        "flood_area_km2": flood_area_km2,
        "epsg": geo_meta.get("epsg"),
        "pixel_scale": geo_meta.get("pixel_scale"),
        "image_shape": shape,
        # UI map overlay uses simple CRS in pixel coordinates.
        "map_overlay": {
            "crs": "simple",
            "width": width,
            "height": height,
            "bounds": [[0, 0], [height, width]] if (height and width) else None,
            "center": [height / 2.0, width / 2.0] if (height and width) else None,
        },
    }

    tiepoint = geo_meta.get("tiepoint")
    pixel_scale = geo_meta.get("pixel_scale")
    if (
        isinstance(tiepoint, list)
        and len(tiepoint) >= 2
        and isinstance(pixel_scale, list)
        and len(pixel_scale) >= 2
        and width
        and height
    ):
        try:
            x0 = float(tiepoint[0])
            y0 = float(tiepoint[1])
            sx = float(pixel_scale[0])
            sy = float(pixel_scale[1])
            bbox = {
                "x_min": x0,
                "x_max": x0 + sx * width,
                "y_max": y0,
                "y_min": y0 - sy * height,
            }
            summary["projected_bbox"] = bbox
            summary["projected_bbox_polygon"] = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [bbox["x_min"], bbox["y_min"]],
                        [bbox["x_max"], bbox["y_min"]],
                        [bbox["x_max"], bbox["y_max"]],
                        [bbox["x_min"], bbox["y_max"]],
                        [bbox["x_min"], bbox["y_min"]],
                    ]
                ],
            }
        except Exception:
            pass

    return summary


def _confidence_label_from_percent(value: float | None) -> str | None:
    if value is None:
        return None
    pct = float(value)
    if pct >= 85.0:
        return "high"
    if pct >= 70.0:
        return "medium"
    return "low"


def _confidence_from_score_margin(
    score: Any, threshold: Any = 0.5, predicted_label: Any = None
) -> tuple[float | None, str | None]:
    try:
        s = float(score)
        t = float(threshold)
    except Exception:
        return None, None
    if not (0.0 <= s <= 1.0 and 0.0 < t < 1.0):
        return None, None
    if predicted_label in (0, 1):
        label = int(predicted_label)
    else:
        label = int(s >= t)
    if label == 1:
        denom = max(1e-6, 1.0 - t)
        margin = (s - t) / denom
    else:
        denom = max(1e-6, t)
        margin = (t - s) / denom
    margin = float(np.clip(margin, 0.0, 1.0))
    pct = 50.0 + 50.0 * margin
    return float(pct), _confidence_label_from_percent(pct)


def _keep_top_components(
    mask: np.ndarray,
    *,
    min_pixels: int = 1,
    max_regions: int | None = None,
) -> np.ndarray:
    zone = np.asarray(mask).astype(bool)
    if zone.size <= 0 or not bool(np.any(zone)) or ndimage is None:
        return zone
    labeled, num = ndimage.label(zone)
    if int(num) <= 0:
        return zone
    keep_ids: list[tuple[int, int]] = []
    for comp_id in range(1, int(num) + 1):
        size = int(np.sum(labeled == comp_id))
        if size >= int(max(1, min_pixels)):
            keep_ids.append((size, comp_id))
    if not keep_ids:
        return np.zeros_like(zone, dtype=bool)
    keep_ids.sort(reverse=True)
    if max_regions is not None and int(max_regions) > 0:
        keep_ids = keep_ids[: int(max_regions)]
    keep = np.zeros_like(zone, dtype=bool)
    for _, comp_id in keep_ids:
        keep |= labeled == int(comp_id)
    return keep


def postprocess_segmentation_mask(
    mask: np.ndarray,
    *,
    enabled: bool = True,
    min_region_pixels: int | None = None,
    min_region_scene_ratio: float = 0.0005,
    max_regions: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    original = np.asarray(mask).astype(bool)
    meta: dict[str, Any] = {
        "enabled": bool(enabled),
        "method": "connected_component_min_area",
        "min_region_pixels": None,
        "min_region_scene_ratio": float(min_region_scene_ratio),
        "max_regions": int(max_regions) if max_regions is not None else None,
        "input_positive_pixels": int(np.sum(original)),
        "output_positive_pixels": int(np.sum(original)),
        "removed_positive_pixels": 0,
        "applied": False,
    }
    if not bool(enabled) or original.ndim != 2 or original.size <= 0 or ndimage is None:
        if ndimage is None:
            meta["reason"] = "scipy_ndimage_unavailable"
        return original.astype(np.uint8), meta

    if min_region_pixels is None:
        min_pixels = max(8, int(round(float(original.size) * float(min_region_scene_ratio))))
    else:
        min_pixels = max(1, int(min_region_pixels))
    meta["min_region_pixels"] = int(min_pixels)
    if not bool(np.any(original)):
        return original.astype(np.uint8), meta

    cleaned = _keep_top_components(
        original,
        min_pixels=int(min_pixels),
        max_regions=max_regions,
    )
    input_pixels = int(np.sum(original))
    output_pixels = int(np.sum(cleaned))
    meta.update(
        {
            "output_positive_pixels": output_pixels,
            "removed_positive_pixels": max(0, input_pixels - output_pixels),
            "applied": bool(output_pixels != input_pixels),
        }
    )
    return cleaned.astype(np.uint8), meta


def summarize_zone_components(
    mask: np.ndarray,
    *,
    min_pixels: int = 1,
    max_regions: int = 3,
) -> list[dict[str, Any]]:
    zone = np.asarray(mask).astype(bool)
    if zone.size <= 0 or not bool(np.any(zone)) or ndimage is None:
        return []
    labeled, num = ndimage.label(zone)
    if int(num) <= 0:
        return []
    objects = ndimage.find_objects(labeled)
    rows: list[dict[str, Any]] = []
    total_pixels = max(1, int(zone.size))
    for comp_id, obj in enumerate(objects, start=1):
        if obj is None:
            continue
        comp_mask = labeled[obj] == int(comp_id)
        area_pixels = int(np.sum(comp_mask))
        if area_pixels < int(max(1, min_pixels)):
            continue
        y0 = int(obj[0].start)
        y1 = int(obj[0].stop)
        x0 = int(obj[1].start)
        x1 = int(obj[1].stop)
        ys, xs = np.where(labeled == int(comp_id))
        rows.append(
            {
                "component_id": int(comp_id),
                "area_pixels": area_pixels,
                "scene_pct": float(area_pixels / total_pixels * 100.0),
                "x_min": x0,
                "x_max": x1,
                "y_min": y0,
                "y_max": y1,
                "x_center": float(np.mean(xs))
                if xs.size > 0
                else float((x0 + x1) / 2.0),
                "y_center": float(np.mean(ys))
                if ys.size > 0
                else float((y0 + y1) / 2.0),
            }
        )
    rows.sort(key=lambda x: float(x.get("area_pixels", 0)), reverse=True)
    return rows[: max(1, int(max_regions))]


def extract_prediction_zone_mask(
    *,
    pred_mask: np.ndarray,
    pred_prob: np.ndarray,
    detection_label: int | None,
    prediction_label: int | None,
    prediction_status: str | None,
    seg_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    current_mask = (np.asarray(pred_mask) > 0).astype(np.uint8)
    zone_mask = current_mask.copy()
    zone_kind = "detected_current_scene"
    zone_title = "Detected Flood Mask"
    zone_source = "pred_mask"
    zone_threshold: float | None = None
    notes: list[str] = []
    if int(detection_label or 0) == 0 and str(prediction_status or "active") == "active":
        zone_threshold = float(np.clip(float(seg_threshold) - 0.15, 0.20, 0.45))
        zone_mask = (np.asarray(pred_prob, dtype=np.float32) >= zone_threshold).astype(
            np.uint8
        )
        zone_kind = (
            "predicted_expected_zones"
            if int(prediction_label or 0) == 1
            else "predicted_low_confidence_hints"
        )
        zone_title = (
            "Predicted Flood Zones"
            if int(prediction_label or 0) == 1
            else "Prediction Hint Zones"
        )
        zone_source = "pred_prob_threshold"
        if int(np.sum(zone_mask)) == 0:
            q90 = float(np.quantile(np.asarray(pred_prob, dtype=np.float32), 0.90))
            zone_mask = (np.asarray(pred_prob, dtype=np.float32) >= q90).astype(np.uint8)
            zone_source = "pred_prob_top_q90"
            notes.append(
                f"No pixels above prediction-zone threshold {zone_threshold:.2f}; top 10% probability area was used."
            )
        if int(prediction_label or 0) != 1:
            min_blob = max(12, int(zone_mask.size * 0.0015))
            zone_mask = _keep_top_components(
                zone_mask > 0,
                min_pixels=min_blob,
                max_regions=2,
            ).astype(np.uint8)
            notes.append(
                "Low-confidence prediction hints were cleaned by removing tiny scattered blobs."
            )
    coverage_pct = float(np.mean(zone_mask > 0) * 100.0) if zone_mask.size > 0 else 0.0
    min_zone_pixels = max(6, int(zone_mask.size * 0.0008)) if zone_mask.size > 0 else 1
    components = summarize_zone_components(
        zone_mask > 0,
        min_pixels=min_zone_pixels,
        max_regions=3,
    )
    meta = {
        "zone_kind": zone_kind,
        "zone_title": zone_title,
        "zone_source": zone_source,
        "zone_threshold": zone_threshold,
        "zone_coverage_percent": coverage_pct,
        "zone_components": components,
        "notes": notes,
    }
    return zone_mask.astype(np.uint8), meta


def _pixel_to_map_xy(
    x_pixel: float,
    y_pixel: float,
    geo_meta: dict[str, Any] | None,
) -> tuple[float, float] | None:
    if not isinstance(geo_meta, dict):
        return None
    tiepoint = geo_meta.get("tiepoint")
    pixel_scale = geo_meta.get("pixel_scale")
    if (
        not isinstance(tiepoint, list)
        or len(tiepoint) < 2
        or not isinstance(pixel_scale, list)
        or len(pixel_scale) < 2
    ):
        return None
    try:
        x0 = float(tiepoint[0])
        y0 = float(tiepoint[1])
        sx = float(pixel_scale[0])
        sy = float(pixel_scale[1])
    except Exception:
        return None
    return (x0 + float(x_pixel) * sx, y0 - float(y_pixel) * sy)


def _component_polygon_xy(
    xs: np.ndarray,
    ys: np.ndarray,
) -> list[list[float]]:
    if xs.size <= 0 or ys.size <= 0:
        return []
    x_min = float(np.min(xs))
    x_max = float(np.max(xs) + 1.0)
    y_min = float(np.min(ys))
    y_max = float(np.max(ys) + 1.0)
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    if points.shape[0] > 2000:
        idx = np.linspace(0, points.shape[0] - 1, 2000, dtype=np.int64)
        points = points[idx]
    if points.shape[0] >= 3 and ConvexHull is not None:
        try:
            hull = ConvexHull(points)
            poly = [[float(points[i, 0]), float(points[i, 1])] for i in hull.vertices]
            if poly and poly[0] != poly[-1]:
                poly.append(poly[0])
            if len(poly) >= 4:
                return poly
        except QhullError:
            pass
        except Exception:
            pass
    return [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
        [x_min, y_min],
    ]


def build_prediction_zone_geojson(
    *,
    zone_mask: np.ndarray,
    zone_meta: dict[str, Any],
    geo_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    # Convert the selected prediction zone mask into polygons in either pixel space
    # or projected CRS coordinates when GeoTIFF metadata is available.
    zone = np.asarray(zone_mask).astype(bool)
    features: list[dict[str, Any]] = []
    geometry_crs = "pixel"
    epsg_value = None
    if isinstance(geo_meta, dict) and geo_meta.get("epsg") is not None:
        try:
            epsg_value = int(geo_meta.get("epsg"))
            geometry_crs = f"EPSG:{epsg_value}"
        except Exception:
            epsg_value = None
    if zone.size <= 0 or not bool(np.any(zone)) or ndimage is None:
        return {
            "type": "FeatureCollection",
            "features": [],
            "properties": {
                "zone_kind": zone_meta.get("zone_kind"),
                "geometry_crs": geometry_crs,
                "epsg": epsg_value,
            },
        }
    labeled, num = ndimage.label(zone)
    total_pixels = max(1, int(zone.size))
    min_pixels = max(6, int(zone.size * 0.0008))
    for comp_id in range(1, int(num) + 1):
        ys, xs = np.where(labeled == int(comp_id))
        area_pixels = int(xs.size)
        if area_pixels < int(min_pixels):
            continue
        polygon_xy = _component_polygon_xy(xs, ys)
        if not polygon_xy:
            continue
        map_coords: list[list[float]] = []
        if geometry_crs != "pixel":
            for x_val, y_val in polygon_xy:
                mapped = _pixel_to_map_xy(x_val, y_val, geo_meta)
                if mapped is None:
                    map_coords = []
                    geometry_crs = "pixel"
                    break
                map_coords.append([float(mapped[0]), float(mapped[1])])
        coords = map_coords if map_coords else polygon_xy
        x_min = int(np.min(xs))
        x_max = int(np.max(xs) + 1)
        y_min = int(np.min(ys))
        y_max = int(np.max(ys) + 1)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "component_id": int(comp_id),
                    "zone_kind": str(zone_meta.get("zone_kind", "zone")),
                    "zone_title": str(zone_meta.get("zone_title", "Zone")),
                    "zone_source": str(zone_meta.get("zone_source", "unknown")),
                    "zone_threshold": zone_meta.get("zone_threshold"),
                    "area_pixels": area_pixels,
                    "scene_pct": float(area_pixels / total_pixels * 100.0),
                    "pixel_bbox": {
                        "x_min": x_min,
                        "x_max": x_max,
                        "y_min": y_min,
                        "y_max": y_max,
                    },
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "zone_kind": zone_meta.get("zone_kind"),
            "zone_title": zone_meta.get("zone_title"),
            "zone_source": zone_meta.get("zone_source"),
            "zone_threshold": zone_meta.get("zone_threshold"),
            "zone_coverage_percent": zone_meta.get("zone_coverage_percent"),
            "geometry_crs": geometry_crs,
            "epsg": epsg_value,
            "notes": zone_meta.get("notes", []),
        },
    }


def _build_geotiff_extratags(geo_meta: dict[str, Any] | None) -> list[tuple[Any, ...]]:
    if not isinstance(geo_meta, dict):
        return []
    extratags: list[tuple[Any, ...]] = []
    tiepoint = geo_meta.get("tiepoint")
    pixel_scale = geo_meta.get("pixel_scale")
    if isinstance(pixel_scale, list) and len(pixel_scale) >= 2:
        try:
            sx = float(pixel_scale[0])
            sy = float(pixel_scale[1])
            extratags.append((33550, "d", 3, (sx, sy, 0.0), False))
        except Exception:
            pass
    if isinstance(tiepoint, list) and len(tiepoint) >= 2:
        try:
            x0 = float(tiepoint[0])
            y0 = float(tiepoint[1])
            extratags.append((33922, "d", 6, (0.0, 0.0, 0.0, x0, y0, 0.0), False))
        except Exception:
            pass
    epsg_raw = geo_meta.get("epsg")
    try:
        epsg = int(epsg_raw) if epsg_raw is not None else None
    except Exception:
        epsg = None
    if epsg is not None:
        is_geographic = int(epsg) == 4326
        model_type = 2 if is_geographic else 1
        epsg_key = 2048 if is_geographic else 3072
        geo_key_dir = [
            1,
            1,
            0,
            3,
            1024,
            0,
            1,
            model_type,
            1025,
            0,
            1,
            1,
            epsg_key,
            0,
            1,
            int(epsg),
        ]
        extratags.append((34735, "H", len(geo_key_dir), tuple(geo_key_dir), False))
    return extratags


def write_geo_tiff(
    path: Path,
    array: np.ndarray,
    *,
    geo_meta: dict[str, Any] | None = None,
    dtype: np.dtype | None = None,
) -> Path:
    data = np.asarray(array)
    if dtype is not None:
        data = data.astype(dtype)
    tifffile.imwrite(
        path,
        data,
        photometric="minisblack" if data.ndim == 2 else None,
        metadata=None,
        extratags=_build_geotiff_extratags(geo_meta) or None,
    )
    return path


def export_prediction_geo_artifacts(
    *,
    output_dir: Path,
    stem: str,
    zone_mask: np.ndarray,
    pred_prob: np.ndarray,
    geo_meta: dict[str, Any] | None,
    zone_meta: dict[str, Any],
) -> dict[str, Any]:
    # Bundle all map-ready outputs together so CLI, GUI, and API can export the
    # same geo artifacts without duplicating logic.
    output_dir.mkdir(parents=True, exist_ok=True)
    zone_geojson = build_prediction_zone_geojson(
        zone_mask=zone_mask,
        zone_meta=zone_meta,
        geo_meta=geo_meta,
    )
    geojson_path = output_dir / f"{stem}_zones.geojson"
    with geojson_path.open("w", encoding="utf-8") as f:
        json.dump(zone_geojson, f, indent=2, ensure_ascii=False)
    zone_mask_tif_path = output_dir / f"{stem}_zones_mask_georef.tif"
    prob_tif_path = output_dir / f"{stem}_pred_prob_georef.tif"
    write_geo_tiff(
        zone_mask_tif_path,
        np.asarray(zone_mask, dtype=np.uint8),
        geo_meta=geo_meta,
        dtype=np.uint8,
    )
    write_geo_tiff(
        prob_tif_path,
        np.asarray(pred_prob, dtype=np.float32),
        geo_meta=geo_meta,
        dtype=np.float32,
    )
    return {
        "prediction_zone_geojson": str(geojson_path.resolve()),
        "prediction_zone_mask_geotiff": str(zone_mask_tif_path.resolve()),
        "prediction_probability_geotiff": str(prob_tif_path.resolve()),
        "prediction_zone_geojson_data": zone_geojson,
    }


def build_prediction_confidence(
    *,
    pred_feats: dict[str, Any],
    risk_payload: dict[str, Any],
    temporal_payload: dict[str, Any] | None,
    prediction_eta: dict[str, Any] | None,
    drift_meta: dict[str, Any] | None,
    geo_meta: dict[str, Any] | None,
    weather_statuses: list[str] | None = None,
) -> dict[str, Any]:
    # Confidence is not a single model score. It combines risk margin, temporal
    # availability, drift/geospatial warnings, and weather data quality penalties.
    risk = risk_payload if isinstance(risk_payload, dict) else {}
    temporal = temporal_payload if isinstance(temporal_payload, dict) else {}
    eta = prediction_eta if isinstance(prediction_eta, dict) else {}
    drift = drift_meta if isinstance(drift_meta, dict) else {}
    geo = geo_meta if isinstance(geo_meta, dict) else {}
    issues: list[str] = []
    geo_warnings = [str(x) for x in (geo.get("warnings") or []) if str(x).strip()]
    drift_warnings = [str(x) for x in (drift.get("warnings") or []) if str(x).strip()]
    for item in weather_statuses or []:
        text = str(item).strip()
        if text:
            issues.append(text)
    if risk.get("risk_warning") == "incomplete_weather_features_fallback_used":
        issues.append("weather_features_incomplete_fallback_used")
    if geo_warnings:
        issues.extend([f"geo:{x}" for x in geo_warnings])
    if drift_warnings:
        issues.extend([f"drift:{x}" for x in drift_warnings])
    temporal_status = str(temporal.get("temporal_status", "") or "")
    if temporal_status and temporal_status not in {
        "ok",
        "skipped_due_to_detection",
        "unavailable",
    }:
        issues.append(f"temporal:{temporal_status}")

    if int(risk.get("detection_label", 0) or 0) == 1:
        flood_ratio = float(pred_feats.get("pred_flood_ratio", 0.0) or 0.0)
        p90 = float(pred_feats.get("pred_prob_p90", 0.0) or 0.0)
        base_pct = float(
            np.clip(
                55.0 + 45.0 * min(1.0, max(p90, min(1.0, flood_ratio / 0.10))),
                5.0,
                99.0,
            )
        )
        basis = "segmentation_detection"
    elif temporal_status == "ok" and temporal.get("temporal_risk_score") is not None:
        base_pct, _ = _confidence_from_score_margin(
            temporal.get("temporal_risk_score"),
            temporal.get("temporal_risk_threshold", 0.5),
            temporal.get("temporal_risk_label"),
        )
        basis = "temporal_risk"
    else:
        base_pct, _ = _confidence_from_score_margin(
            risk.get("risk_score"),
            risk.get("risk_threshold", 0.5),
            risk.get("prediction_label"),
        )
        basis = "scene_risk"
    if base_pct is None:
        base_pct = 55.0

    penalty = 0.0
    penalty += 8.0 * float(len(drift_warnings))
    penalty += 5.0 * float(len(geo_warnings))
    penalty += 4.0 * float(len(weather_statuses or []))
    if risk.get("risk_warning") == "incomplete_weather_features_fallback_used":
        penalty += 8.0
    if temporal_status not in {"ok", "skipped_due_to_detection", "unavailable"}:
        penalty += 8.0
    eta_conf = _to_float_or_none(eta.get("prediction_eta_confidence_percent"))
    if eta_conf is not None:
        base_pct = float((float(base_pct) * 0.7) + (float(eta_conf) * 0.3))
    overall_pct = float(np.clip(float(base_pct) - penalty, 5.0, 99.0))
    return {
        "status": "ok",
        "basis": basis,
        "base_confidence_percent": float(base_pct),
        "overall_confidence_percent": overall_pct,
        "overall_confidence_level": _confidence_label_from_percent(overall_pct),
        "data_quality_level": (
            "high"
            if len(issues) == 0
            else ("medium" if len(issues) <= 2 else "low")
        ),
        "data_quality_warnings": issues,
        "geospatial_warnings": geo_warnings,
        "drift_warnings": drift_warnings,
        "out_of_distribution_warning": bool(drift_warnings),
        "temporal_available": temporal_status == "ok",
    }


def build_forecast_timeline(
    *,
    risk_payload: dict[str, Any],
    temporal_payload: dict[str, Any] | None,
    prediction_eta: dict[str, Any] | None,
) -> dict[str, Any]:
    # Present the current scene result and heuristic short/mid/long horizon views
    # in one timeline structure that UI/API can render directly.
    risk = risk_payload if isinstance(risk_payload, dict) else {}
    temporal = temporal_payload if isinstance(temporal_payload, dict) else {}
    eta = prediction_eta if isinstance(prediction_eta, dict) else {}
    now_utc = datetime.now(timezone.utc)
    detection_label = int(risk.get("detection_label", 0) or 0)
    prediction_label = risk.get("prediction_label")
    eta_start = _parse_utc_timestamp(eta.get("prediction_eta_start_utc"))
    eta_end = _parse_utc_timestamp(eta.get("prediction_eta_end_utc"))
    eta_conf = _to_float_or_none(eta.get("prediction_eta_confidence_percent"))
    eta_conf_level = str(eta.get("prediction_eta_confidence_level", "") or "")

    def _window_state(horizon_days: int) -> dict[str, Any]:
        window_end = now_utc + timedelta(days=int(horizon_days))
        if detection_label == 1:
            return {
                "horizon_days": int(horizon_days),
                "status": "current_flood_detected",
                "label": "detected_now",
                "confidence_percent": None,
                "confidence_level": None,
                "note": "Current flood is already detected; future prediction window is suppressed.",
            }
        if temporal.get("temporal_status") != "ok":
            return {
                "horizon_days": int(horizon_days),
                "status": "unavailable",
                "label": "unavailable",
                "confidence_percent": eta_conf,
                "confidence_level": eta_conf_level or None,
                "note": "Temporal forecast is unavailable for this scene.",
            }
        if eta_start is not None and eta_end is not None:
            return {
                "horizon_days": int(horizon_days),
                "status": "ok",
                "label": "elevated" if eta_start <= window_end else "watch",
                "confidence_percent": eta_conf,
                "confidence_level": eta_conf_level or None,
                "eta_start_utc": eta_start.isoformat(),
                "eta_end_utc": eta_end.isoformat(),
                "note": (
                    f"Estimated risk window intersects the next {horizon_days} days."
                    if eta_start <= window_end
                    else f"Estimated risk window starts after the next {horizon_days} days."
                ),
            }
        return {
            "horizon_days": int(horizon_days),
            "status": "ok",
            "label": "elevated" if int(prediction_label or 0) == 1 else "low",
            "confidence_percent": eta_conf,
            "confidence_level": eta_conf_level or None,
            "note": "Heuristic horizon view inferred from temporal risk score.",
        }

    return {
        "status": "ok",
        "label_support": "heuristic_horizon_view_without_future_ground_truth_labels",
        "items": [
            {
                "horizon_days": 0,
                "status": "ok",
                "label": "detected_now" if detection_label == 1 else "not_detected_now",
                "confidence_percent": None,
                "confidence_level": None,
                "note": (
                    "Current-scene segmentation result."
                    if detection_label == 1
                    else "No current flood detected in this scene."
                ),
            },
            _window_state(7),
            _window_state(30),
            _window_state(90),
        ],
    }


def build_prediction_explanation(
    *,
    pred_feats: dict[str, Any],
    risk_payload: dict[str, Any],
    temporal_payload: dict[str, Any] | None,
    decision_support: dict[str, Any] | None,
    confidence_payload: dict[str, Any] | None,
    zone_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    # Turn raw numeric outputs into a compact narrative: one summary sentence, key
    # drivers, caveats, and recommended actions.
    risk = risk_payload if isinstance(risk_payload, dict) else {}
    temporal = temporal_payload if isinstance(temporal_payload, dict) else {}
    support = decision_support if isinstance(decision_support, dict) else {}
    confidence = confidence_payload if isinstance(confidence_payload, dict) else {}
    zone = zone_meta if isinstance(zone_meta, dict) else {}
    temporal_snapshot = temporal.get("temporal_feature_snapshot")
    if not isinstance(temporal_snapshot, dict):
        temporal_snapshot = {}

    if int(risk.get("detection_label", 0) or 0) == 1:
        summary = "Flood is currently detected in the scene, so future prediction is suppressed."
    elif int(risk.get("prediction_label", 0) or 0) == 1:
        summary = "No flood is detected now, but the risk path indicates elevated future flood likelihood."
    else:
        summary = "No flood is detected now and the current prediction path does not show a strong future flood signal."

    drivers: list[str] = []
    flood_ratio_pct = to_percent(pred_feats.get("pred_flood_ratio"))
    prob_mean_pct = to_percent(pred_feats.get("pred_prob_mean"))
    prob_p90_pct = to_percent(pred_feats.get("pred_prob_p90"))
    if flood_ratio_pct is not None:
        drivers.append(f"Detected/estimated flood footprint: {float(flood_ratio_pct):.2f}% of scene pixels.")
    if prob_mean_pct is not None and prob_p90_pct is not None:
        drivers.append(
            f"Segmentation probability profile: mean={float(prob_mean_pct):.2f}%, p90={float(prob_p90_pct):.2f}%."
        )
    tp_recent3 = _to_float_or_none(temporal_snapshot.get("tp_recent3_sum"))
    tp_recent12 = _to_float_or_none(temporal_snapshot.get("tp_recent12_sum"))
    runoff_recent3 = _to_float_or_none(temporal_snapshot.get("runoff_recent3_sum"))
    runoff_recent12 = _to_float_or_none(temporal_snapshot.get("runoff_recent12_sum"))
    if tp_recent3 is not None and tp_recent12 is not None:
        drivers.append(
            f"Recent precipitation context: 3-step sum={tp_recent3:.3f}, 12-step sum={tp_recent12:.3f}."
        )
    if runoff_recent3 is not None and runoff_recent12 is not None:
        drivers.append(
            f"Recent runoff context: 3-step sum={runoff_recent3:.3f}, 12-step sum={runoff_recent12:.3f}."
        )
    temporal_score_pct = temporal.get("temporal_risk_score_percent")
    temporal_thr = _to_float_or_none(temporal.get("temporal_risk_threshold"))
    if temporal_score_pct is not None:
        if temporal_thr is not None:
            drivers.append(
                f"Temporal score={float(temporal_score_pct):.2f}% against threshold={float(temporal_thr) * 100.0:.2f}%."
            )
        else:
            drivers.append(f"Temporal score={float(temporal_score_pct):.2f}%.")
    zone_pct = _to_float_or_none(zone.get("zone_coverage_percent"))
    if zone_pct is not None:
        drivers.append(f"Highlighted zone coverage: {zone_pct:.2f}% of the scene.")
    for item in support.get("primary_drivers") or []:
        text = str(item).strip()
        if text and text not in drivers:
            drivers.append(text)

    caveats: list[str] = []
    for item in confidence.get("data_quality_warnings") or []:
        text = str(item).strip()
        if text:
            caveats.append(text)
    for item in support.get("warnings") or []:
        text = str(item).strip()
        if text and text not in caveats:
            caveats.append(text)

    return {
        "status": "ok",
        "summary": summary,
        "key_drivers": drivers[:6],
        "caveats": caveats[:6],
        "recommended_actions": support.get("recommended_actions") or [],
    }


def build_prediction_analysis(
    *,
    pred_mask: np.ndarray,
    pred_prob: np.ndarray,
    pred_feats: dict[str, Any],
    risk_payload: dict[str, Any],
    temporal_payload: dict[str, Any] | None,
    prediction_eta: dict[str, Any] | None,
    decision_support: dict[str, Any] | None,
    drift_meta: dict[str, Any] | None,
    geo_meta: dict[str, Any] | None,
    weather_statuses: list[str] | None,
    seg_threshold: float,
) -> dict[str, Any]:
    # Compose all higher-level interpretation layers around a prediction:
    # cleaned zone mask, confidence, explanation, and horizon timeline.
    zone_mask, zone_meta = extract_prediction_zone_mask(
        pred_mask=pred_mask,
        pred_prob=pred_prob,
        detection_label=risk_payload.get("detection_label"),
        prediction_label=risk_payload.get("prediction_label"),
        prediction_status=risk_payload.get("prediction_status"),
        seg_threshold=float(seg_threshold),
    )
    confidence = build_prediction_confidence(
        pred_feats=pred_feats,
        risk_payload=risk_payload,
        temporal_payload=temporal_payload,
        prediction_eta=prediction_eta,
        drift_meta=drift_meta,
        geo_meta=geo_meta,
        weather_statuses=weather_statuses,
    )
    timeline = build_forecast_timeline(
        risk_payload=risk_payload,
        temporal_payload=temporal_payload,
        prediction_eta=prediction_eta,
    )
    explanation = build_prediction_explanation(
        pred_feats=pred_feats,
        risk_payload=risk_payload,
        temporal_payload=temporal_payload,
        decision_support=decision_support,
        confidence_payload=confidence,
        zone_meta=zone_meta,
    )
    return {
        "zone_mask": zone_mask,
        "zone_meta": zone_meta,
        "confidence": confidence,
        "timeline": timeline,
        "explanation": explanation,
    }


def _pick_metrics_block(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    block = payload.get("cv_mean")
    if isinstance(block, dict):
        return block
    return payload


def collect_experiment_run_summaries(
    base_dir: Path, *, limit: int = 12
) -> list[dict[str, Any]]:
    root = Path(base_dir).resolve()
    candidate_dirs: list[Path] = []
    if root.is_dir():
        if root.name.lower().startswith("outputs"):
            candidate_dirs.append(root)
        try:
            candidate_dirs.extend(
                path.resolve()
                for path in root.iterdir()
                if path.is_dir() and path.name.lower().startswith("outputs")
            )
        except Exception:
            pass
    candidates = sorted(
        {
            (artifact_dir / "unet_train_report.json").resolve()
            for artifact_dir in candidate_dirs
            if (artifact_dir / "unet_train_report.json").exists()
        },
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    rows: list[dict[str, Any]] = []
    seen_dirs: set[Path] = set()
    for report_path in candidates:
        artifact_dir = report_path.parent.resolve()
        if artifact_dir in seen_dirs:
            continue
        seen_dirs.add(artifact_dir)

        report = load_json_file(report_path)
        if not isinstance(report, dict):
            continue
        run_config = load_json_file(artifact_dir / "run_config.json") or {}
        config = report.get("config") if isinstance(report.get("config"), dict) else {}
        if not config and isinstance(run_config, dict):
            config = (
                run_config.get("train_args")
                if isinstance(run_config.get("train_args"), dict)
                else run_config
            )
        sensors = report.get("sensors") if isinstance(report.get("sensors"), dict) else {}
        s1 = sensors.get("S1") if isinstance(sensors.get("S1"), dict) else {}
        s2 = sensors.get("S2") if isinstance(sensors.get("S2"), dict) else {}
        global_metrics = load_json_file(artifact_dir / "unet_val_metrics_global.json")
        if not isinstance(global_metrics, dict) and isinstance(
            report.get("global_val_metrics"), dict
        ):
            global_metrics = report.get("global_val_metrics")

        risk_fallback = load_json_file(artifact_dir / "risk_no_weather_cv_metrics_unet.json")
        risk_temporal = load_json_file(artifact_dir / RISK_TEMPORAL_METRICS_PIPELINE_NAME)

        fallback_block = _pick_metrics_block(risk_fallback)
        temporal_block = _pick_metrics_block(risk_temporal)
        modified_at_utc = datetime.fromtimestamp(
            report_path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        rows.append(
            {
                "run_name": artifact_dir.name,
                "run_type": PIPELINE_V3_BACKEND_ID,
                "artifact_dir": str(artifact_dir),
                "report_path": str(report_path),
                "updated_at_utc": modified_at_utc,
                "status": str(report.get("status", "unknown")),
                "device": str(report.get("device", "") or ""),
                "model_kind": str(config.get("model_kind", "unknown")),
                "patch_size": config.get("patch_size"),
                "stride": config.get("stride"),
                "epochs": config.get("epochs"),
                "threshold": config.get("threshold"),
                "val_ratio": config.get("val_ratio"),
                "seg_f1": _to_float_or_none((global_metrics or {}).get("f1")),
                "seg_iou": _to_float_or_none((global_metrics or {}).get("iou")),
                "seg_accuracy": _to_float_or_none((global_metrics or {}).get("accuracy")),
                "risk_auc_fallback": _to_float_or_none(fallback_block.get("roc_auc")),
                "temporal_auc": _to_float_or_none(temporal_block.get("roc_auc")),
                "best_epoch_s1": s1.get("best_epoch"),
                "best_epoch_s2": s2.get("best_epoch"),
                "s1_f1": _to_float_or_none(
                    (
                        s1.get("val_metrics_image_level")
                        if isinstance(s1.get("val_metrics_image_level"), dict)
                        else {}
                    ).get("f1")
                ),
                "s2_f1": _to_float_or_none(
                    (
                        s2.get("val_metrics_image_level")
                        if isinstance(s2.get("val_metrics_image_level"), dict)
                        else {}
                    ).get("f1")
                ),
            }
        )
        if len(rows) >= int(limit):
            break
    return rows


def _failure_examples_from_report(
    df: pd.DataFrame,
    *,
    mode: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    for col in ["flood_ratio_true", "flood_ratio_pred", "f1", "iou", "recall", "precision"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    if mode == "false_positive":
        work = work[
            (work.get("flood_ratio_true", 0.0) <= 0.001)
            & (work.get("flood_ratio_pred", 0.0) >= 0.01)
        ].copy()
        if work.empty:
            return []
        work["rank_score"] = work["flood_ratio_pred"]
        work = work.sort_values(["rank_score", "f1"], ascending=[False, True])
    elif mode == "false_negative":
        work = work[
            (work.get("flood_ratio_true", 0.0) >= 0.01)
            & (work.get("recall", 0.0) <= 0.25)
        ].copy()
        if work.empty:
            return []
        work["rank_score"] = work["flood_ratio_true"] - work.get("flood_ratio_pred", 0.0)
        work = work.sort_values(["recall", "rank_score"], ascending=[True, False])
    else:
        work = work.sort_values("iou", ascending=True)
    rows: list[dict[str, Any]] = []
    for row in work.head(int(limit)).to_dict(orient="records"):
        rows.append(
            {
                "filename": str(row.get("filename", "")),
                "f1": _to_float_or_none(row.get("f1")),
                "iou": _to_float_or_none(row.get("iou")),
                "recall": _to_float_or_none(row.get("recall")),
                "precision": _to_float_or_none(row.get("precision")),
                "true_flood_ratio_percent": to_percent(row.get("flood_ratio_true")),
                "pred_flood_ratio_percent": to_percent(row.get("flood_ratio_pred")),
            }
        )
    return rows


def build_failure_analysis_report(artifact_dir: Path) -> dict[str, Any]:
    root = Path(artifact_dir).resolve()
    sensors_out: dict[str, Any] = {}
    best_sensor: str | None = None
    best_sensor_f1 = -1.0

    for sensor in ("S1", "S2"):
        report_path = root / f"unet_val_report_{sensor.lower()}.csv"
        if not report_path.exists():
            sensors_out[sensor] = {
                "status": "missing",
                "report_path": str(report_path),
            }
            continue
        try:
            df = pd.read_csv(report_path)
        except Exception as ex:
            sensors_out[sensor] = {
                "status": "read_failed",
                "report_path": str(report_path),
                "error": str(ex),
            }
            continue
        if df.empty:
            sensors_out[sensor] = {
                "status": "empty",
                "report_path": str(report_path),
            }
            continue

        for col in ["f1", "iou", "flood_ratio_true", "flood_ratio_pred"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        mean_f1 = float(df["f1"].mean()) if "f1" in df.columns else 0.0
        mean_iou = float(df["iou"].mean()) if "iou" in df.columns else 0.0
        mean_true = (
            float(df["flood_ratio_true"].mean()) if "flood_ratio_true" in df.columns else 0.0
        )
        mean_pred = (
            float(df["flood_ratio_pred"].mean()) if "flood_ratio_pred" in df.columns else 0.0
        )
        if mean_f1 > best_sensor_f1:
            best_sensor_f1 = mean_f1
            best_sensor = sensor
        if mean_pred > max(mean_true * 1.35, mean_true + 0.01):
            issue_summary = "overpredicting_flood_extent"
        elif mean_true > max(mean_pred * 1.35, mean_pred + 0.01):
            issue_summary = "underpredicting_flood_extent"
        else:
            issue_summary = "balanced_or_mixed_errors"
        sensors_out[sensor] = {
            "status": "ok",
            "report_path": str(report_path),
            "images": int(len(df)),
            "mean_f1": mean_f1,
            "mean_iou": mean_iou,
            "mean_true_ratio_percent": to_percent(mean_true),
            "mean_pred_ratio_percent": to_percent(mean_pred),
            "issue_summary": issue_summary,
            "false_positive_examples": _failure_examples_from_report(
                df, mode="false_positive"
            ),
            "false_negative_examples": _failure_examples_from_report(
                df, mode="false_negative"
            ),
            "worst_iou_examples": _failure_examples_from_report(
                df, mode="worst_iou"
            ),
        }

    available = {
        key: value
        for key, value in sensors_out.items()
        if isinstance(value, dict) and value.get("status") == "ok"
    }
    comparison_note = "Validation CSVs are not available."
    if len(available) == 2:
        s1_f1 = float(available["S1"].get("mean_f1", 0.0))
        s2_f1 = float(available["S2"].get("mean_f1", 0.0))
        if s1_f1 > s2_f1 + 1e-6:
            comparison_note = "S1 is stronger than S2 on validation F1."
        elif s2_f1 > s1_f1 + 1e-6:
            comparison_note = "S2 is stronger than S1 on validation F1."
        else:
            comparison_note = "S1 and S2 are roughly tied on validation F1."
    elif len(available) == 1:
        comparison_note = f"Only {next(iter(available.keys()))} validation report is available."

    return {
        "status": "ok" if available else "unavailable",
        "artifact_dir": str(root),
        "best_sensor_by_f1": best_sensor,
        "comparison_note": comparison_note,
        "sensors": sensors_out,
    }


def build_submission_model_report(artifact_dir: Path) -> dict[str, Any]:
    root = Path(artifact_dir).resolve()
    train_report = load_json_file(root / "unet_train_report.json") or {}
    registry = load_model_registry(root) or {}
    global_metrics = load_json_file(root / "unet_val_metrics_global.json") or {}
    s1_metrics = load_json_file(root / "unet_val_metrics_s1.json") or {}
    s2_metrics = load_json_file(root / "unet_val_metrics_s2.json") or {}
    risk_weather = _pick_metrics_block(
        load_json_file(root / "risk_with_weather_cv_metrics_unet.json")
    )
    risk_no_weather = _pick_metrics_block(
        load_json_file(root / "risk_no_weather_cv_metrics_unet.json")
    )
    risk_temporal = _pick_metrics_block(
        load_json_file(root / RISK_TEMPORAL_METRICS_PIPELINE_NAME)
    )
    sensors = train_report.get("sensors") if isinstance(train_report.get("sensors"), dict) else {}
    config = train_report.get("config") if isinstance(train_report.get("config"), dict) else {}
    failure_report = build_failure_analysis_report(root)

    sensor_thresholds: dict[str, Any] = {}
    for sensor in ("S1", "S2"):
        payload = sensors.get(sensor) if isinstance(sensors.get(sensor), dict) else {}
        sensor_thresholds[sensor] = {
            "decision_threshold": payload.get("decision_threshold"),
            "threshold_tuning": payload.get("threshold_tuning"),
            "validation_metrics": s1_metrics if sensor == "S1" else s2_metrics,
        }

    split_note = (
        "Current artifacts report a validation split, but not a strict event-held-out "
        "external test set. Treat this as validation evidence unless an event-level "
        "test set is added."
    )
    if registry.get("split_strategy"):
        split_note = str(registry.get("split_strategy"))

    return {
        "status": "ok",
        "artifact_dir": str(root),
        "generated_at_utc": utc_now_iso(),
        "official_project_name": "Flood Intelligence Platform",
        "model_family": "Pipeline V3",
        "model_kind": config.get("model_kind") or registry.get("model_kind"),
        "validation_summary": {
            "global_segmentation": global_metrics,
            "S1_segmentation": s1_metrics,
            "S2_segmentation": s2_metrics,
            "risk_with_weather": risk_weather,
            "risk_no_weather": risk_no_weather,
            "temporal_risk": risk_temporal,
        },
        "sensor_thresholds": sensor_thresholds,
        "postprocessing": {
            "runtime_mask_postprocessing": "connected_component_min_area",
            "default_min_region_scene_ratio": 0.0005,
            "purpose": "remove tiny isolated flood blobs after probability thresholding",
        },
        "split_assessment": {
            "val_ratio": config.get("val_ratio"),
            "seed": config.get("seed"),
            "note": split_note,
            "recommended_next_step": "Add a strict event-held-out test set for stronger external evidence.",
        },
        "failure_analysis": failure_report,
        "presentation_guidance": [
            "Lead with F1, IoU, recall, and ROC AUC rather than accuracy alone.",
            "Mention that S2 validation is stronger than S1 in the current artifacts.",
            "Describe post-processing as a conservative cleanup step, not a new model.",
            "Be clear that the current evidence is validation-based unless event-held-out testing is added.",
        ],
    }


# ==============================
# Backend Registry and Routing
# ==============================
# This block controls the active Pipeline V3 runtime artifacts used at inference time.
def load_model_registry(output_dir: Path) -> dict[str, Any] | None:
    return load_json_file(output_dir / "model_registry.json")


def _has_runtime_artifacts(folder: Path) -> bool:
    try:
        required_any = [
            get_pipeline_model_path("S1", folder),
            get_pipeline_model_path("S2", folder),
            (folder / RISK_NO_WEATHER_PIPELINE_NAME).resolve(),
            (folder / RISK_WITH_WEATHER_PIPELINE_NAME).resolve(),
            (folder / RISK_TEMPORAL_PIPELINE_NAME).resolve(),
        ]
        return any(path.exists() for path in required_any)
    except Exception:
        return False


def resolve_prediction_artifact_dir() -> tuple[Path, str]:
    configured = resolve_env_path(
        "FLOOD_OUTPUT_DIR",
        base_dir=PROJECT_BASE_DIR,
        default_relative=Path("outputs"),
    ).resolve()
    if _has_runtime_artifacts(configured):
        return configured, "configured"

    candidates = sorted(
        [
            path.resolve()
            for path in PROJECT_BASE_DIR.iterdir()
            if path.is_dir() and path.name.lower().startswith("outputs")
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if _has_runtime_artifacts(candidate):
            return candidate, "auto_latest_artifacts"
    return configured, "configured_missing_artifacts"


def get_pipeline_model_path(sensor: str, output_dir: Path) -> Path:
    if sensor == "S1":
        return (output_dir / PIPELINE_MODEL_S1_NAME).resolve()
    if sensor == "S2":
        return (output_dir / PIPELINE_MODEL_S2_NAME).resolve()
    raise ValueError(f"unknown sensor for pipeline path: {sensor}")


def get_pipeline_risk_model_paths(output_dir: Path) -> tuple[Path, Path]:
    return (
        (output_dir / RISK_WITH_WEATHER_PIPELINE_NAME).resolve(),
        (output_dir / RISK_NO_WEATHER_PIPELINE_NAME).resolve(),
    )


def load_active_backend_config(output_dir: Path) -> dict[str, Any]:
    p = output_dir / ACTIVE_BACKEND_NAME
    if not p.exists():
        return {
            "segmentation_backend": PIPELINE_V3_BACKEND_ID,
            "risk_backend": PIPELINE_V3_BACKEND_ID,
            "promotion_reason": "active_backend_missing",
            "metrics_snapshot": {},
        }
    try:
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("active backend payload is not a dict")
        payload["segmentation_backend"] = normalize_pipeline_backend_id(
            payload.get("segmentation_backend")
        )
        payload["risk_backend"] = normalize_pipeline_backend_id(payload.get("risk_backend"))
        return payload
    except Exception as ex:
        return {
            "segmentation_backend": PIPELINE_V3_BACKEND_ID,
            "risk_backend": PIPELINE_V3_BACKEND_ID,
            "promotion_reason": f"active_backend_invalid: {ex}",
            "metrics_snapshot": {},
        }


def available_backends(output_dir: Path, sensor: str | None = None) -> dict[str, bool]:
    if sensor in SENSOR_CHANNELS:
        pipeline_ready = get_pipeline_model_path(sensor, output_dir).exists()
    else:
        pipeline_ready = all(
            get_pipeline_model_path(s, output_dir).exists() for s in SENSOR_CHANNELS
        )
    return {
        PIPELINE_V3_BACKEND_ID: bool(pipeline_ready),
    }


def resolve_prediction_backend(
    *,
    requested_backend: str,
    output_dir: Path,
    sensor: str,
) -> dict[str, Any]:
    requested_backend = normalize_pipeline_backend_id(requested_backend)

    active = load_active_backend_config(output_dir)
    avail = available_backends(output_dir, sensor=sensor)
    fallback_reason = None
    if not bool(avail.get(PIPELINE_V3_BACKEND_ID)):
        fallback_reason = (
            "pipeline_v3_requested_but_not_available"
            if requested_backend == PIPELINE_V3_BACKEND_ID
            else "pipeline_models_missing"
        )

    return {
        "requested_backend": requested_backend,
        "segmentation_backend": PIPELINE_V3_BACKEND_ID,
        "risk_backend": PIPELINE_V3_BACKEND_ID,
        "fallback_reason": fallback_reason,
        "active_backend": active,
        "available_backends": avail,
    }


def load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


# Metrics used when deciding whether a freshly trained model version
# should replace the currently active production artifacts.
def _metric_from_summary(summary: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(summary, dict):
        return None
    value = summary.get(key)
    if value is None and isinstance(summary.get("cv_mean"), dict):
        value = summary["cv_mean"].get(key)
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def collect_promotion_metrics(output_dir: Path) -> dict[str, float | None]:
    seg_metrics = load_json_file(output_dir / "val_metrics_global.json")
    no_weather = load_json_file(output_dir / "risk_no_weather_cv_metrics.json")
    with_weather = load_json_file(output_dir / "risk_with_weather_cv_metrics.json")
    return {
        "seg_f1": _metric_from_summary(seg_metrics, "f1"),
        "risk_no_weather_auc": _metric_from_summary(no_weather, "roc_auc"),
        "risk_with_weather_auc": _metric_from_summary(with_weather, "roc_auc"),
    }


def backup_artifacts_for_promotion(
    output_dir: Path, artifact_paths: list[Path]
) -> dict[str, str]:
    backup_dir = output_dir / "_promotion_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    for src in artifact_paths:
        if not src.exists():
            continue
        rel = src.relative_to(output_dir)
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest[str(src.resolve())] = str(dst.resolve())
    save_json(backup_dir / "manifest.json", manifest)
    return manifest


def restore_artifacts_from_backup(manifest: dict[str, str]) -> None:
    for src_text, backup_text in manifest.items():
        src = Path(src_text)
        backup = Path(backup_text)
        if not backup.exists():
            continue
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, src)


def evaluate_promotion_decision(
    before: dict[str, float | None],
    after: dict[str, float | None],
    *,
    tolerance: float = 1e-6,
) -> tuple[bool, dict[str, Any]]:
    compared: list[dict[str, Any]] = []
    improved = False
    degraded = False
    for key in ["seg_f1", "risk_no_weather_auc", "risk_with_weather_auc"]:
        prev = before.get(key)
        curr = after.get(key)
        row: dict[str, Any] = {"metric": key, "before": prev, "after": curr}
        if prev is None or curr is None:
            row["status"] = "not_compared"
        elif curr + tolerance < prev:
            row["status"] = "degraded"
            degraded = True
        elif curr > prev + tolerance:
            row["status"] = "improved"
            improved = True
        else:
            row["status"] = "unchanged"
        compared.append(row)

    had_baseline = any(before.get(k) is not None for k in before.keys())
    if not had_baseline:
        return True, {"reason": "no_previous_baseline", "comparisons": compared}
    if degraded:
        return False, {"reason": "degraded_metrics_detected", "comparisons": compared}
    if improved:
        return True, {"reason": "improved_without_degradation", "comparisons": compared}
    return False, {"reason": "no_metric_improvement", "comparisons": compared}


# ==============================
# Prediction Audit and Feedback Intake
# ==============================
# Audit: append lightweight JSONL records per prediction.
# Feedback intake: keep unseen samples for future labeling/retraining.
def rotate_jsonl_if_needed(
    line_path: Path, *, max_mb: int = DEFAULT_AUDIT_ROTATE_MB
) -> None:
    if not line_path.exists():
        return
    limit_bytes = int(max_mb) * 1024 * 1024
    try:
        size = int(line_path.stat().st_size)
    except Exception:
        return
    if size <= limit_bytes:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rotated = line_path.with_name(f"{line_path.stem}_{stamp}{line_path.suffix}")
    try:
        line_path.rename(rotated)
    except Exception:
        pass


def append_prediction_audit(
    output_dir: Path, payload: dict[str, Any], *, max_mb: int = DEFAULT_AUDIT_ROTATE_MB
) -> None:
    line_path = output_dir / "prediction_audit_log.jsonl"
    rotate_jsonl_if_needed(line_path, max_mb=max_mb)
    record = {
        "prediction_id": payload.get("prediction_id"),
        "timestamp_utc": payload.get("timestamp_utc"),
        "image_path": payload.get("image_path"),
        "sensor": payload.get("sensor"),
        "risk_score": payload.get("risk_score"),
        "risk_score_percent": payload.get("risk_score_percent"),
        "risk_label": payload.get("risk_label"),
        "risk_threshold": payload.get("risk_threshold"),
        "risk_threshold_profile": payload.get("risk_threshold_profile"),
        "risk_model_used": payload.get("risk_model_used"),
        "inference_mode": payload.get("inference_mode"),
        "model_registry_path": payload.get("model_registry_path"),
    }
    with line_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_stem(name: str, fallback: str = "image") -> str:
    base = Path(name).stem if name else fallback
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in base)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def build_known_dataset_signatures(
    data_roots: list[Path],
) -> tuple[set[Path], set[str]]:
    key = tuple(sorted(str(Path(p).resolve()) for p in data_roots))
    cached_paths, cached_filenames = _build_known_dataset_signatures_cached(key)
    return {Path(x) for x in cached_paths}, set(cached_filenames)


@lru_cache(maxsize=8)
def _build_known_dataset_signatures_cached(
    data_roots_key: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    discovered = discover_dataset([Path(p).resolve() for p in data_roots_key])
    known_paths = tuple(sorted(str(p.resolve()) for p in discovered.pair_by_image.keys()))
    known_filenames = tuple(sorted(str(x) for x in discovered.image_index.keys()))
    return known_paths, known_filenames


def register_feedback_candidate(
    *,
    output_dir: Path,
    image_name: str,
    sensor: str,
    x_img: np.ndarray,
    weather_values: dict[str, Any] | None = None,
    source_image_path: Path | None = None,
    source_image_bytes: bytes | None = None,
    source_mode: str = "predict",
    known_dataset_paths: set[Path] | None = None,
    known_dataset_filenames: set[str] | None = None,
) -> dict[str, Any]:
    # Save only truly new samples (by path/name/hash) so we can
    # build a clean active-learning pool without duplicates.
    output_dir = Path(output_dir).resolve()
    feedback_root = output_dir / "feedback_pool"
    images_dir = feedback_root / "images"
    metadata_dir = feedback_root / "metadata"
    index_path = feedback_root / "feedback_index.json"
    queue_path = feedback_root / "labeling_queue.csv"

    for p in [feedback_root, images_dir, metadata_dir]:
        p.mkdir(parents=True, exist_ok=True)

    resolved_source = (
        source_image_path.resolve() if source_image_path is not None else None
    )
    if (
        known_dataset_paths is not None
        and resolved_source is not None
        and resolved_source in known_dataset_paths
    ):
        return {
            "status": "skipped",
            "reason": "already_in_base_dataset_path",
            "source_image_path": str(resolved_source),
        }
    if (
        known_dataset_filenames is not None
        and image_name
        and image_name in known_dataset_filenames
    ):
        return {
            "status": "skipped",
            "reason": "already_in_base_dataset_filename",
            "filename": image_name,
        }

    raw_bytes: bytes | None = source_image_bytes
    if raw_bytes is None and resolved_source is not None and resolved_source.exists():
        try:
            raw_bytes = resolved_source.read_bytes()
        except Exception:
            raw_bytes = None

    if raw_bytes is not None:
        sample_hash = hashlib.sha256(raw_bytes).hexdigest()
    else:
        arr_bytes = np.ascontiguousarray(x_img.astype(np.float32)).tobytes()
        sample_hash = hashlib.sha256(arr_bytes).hexdigest()

    index: dict[str, Any] = {}
    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as f:
                parsed = json.load(f)
            if isinstance(parsed, dict):
                index = parsed
        except Exception:
            index = {}

    if sample_hash in index:
        existing = (
            index.get(sample_hash, {})
            if isinstance(index.get(sample_hash), dict)
            else {}
        )
        return {
            "status": "skipped",
            "reason": "duplicate_in_feedback_pool",
            "sample_id": sample_hash,
            "stored_image_path": existing.get("stored_image_path"),
        }

    safe_stem = _safe_stem(image_name or "uploaded_image")
    stored_image_path = images_dir / f"{sample_hash[:16]}_{safe_stem}.tif"

    if raw_bytes is not None:
        stored_image_path.write_bytes(raw_bytes)
    else:
        tifffile.imwrite(stored_image_path, x_img.astype(np.float32))

    weather_norm, missing_weather = normalize_weather_features(weather_values)
    metadata = {
        "sample_id": sample_hash,
        "added_at_utc": utc_now_iso(),
        "status": "pending_label",
        "source_mode": source_mode,
        "sensor": sensor,
        "image_name": image_name,
        "source_image_path": (
            str(resolved_source) if resolved_source is not None else None
        ),
        "stored_image_path": str(stored_image_path),
        "weather_features": weather_norm if weather_norm is not None else {},
        "missing_weather_features": missing_weather,
    }
    save_json(metadata_dir / f"{sample_hash}.json", metadata)

    index[sample_hash] = {
        "added_at_utc": metadata["added_at_utc"],
        "status": metadata["status"],
        "source_mode": source_mode,
        "sensor": sensor,
        "image_name": image_name,
        "source_image_path": metadata["source_image_path"],
        "stored_image_path": str(stored_image_path),
    }
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    queue_row = {
        "sample_id": sample_hash,
        "status": "pending_label",
        "added_at_utc": metadata["added_at_utc"],
        "source_mode": source_mode,
        "sensor": sensor,
        "image_name": image_name,
        "stored_image_path": str(stored_image_path),
        "source_image_path": metadata["source_image_path"],
    }
    queue_exists = queue_path.exists()
    with queue_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(queue_row.keys()))
        if not queue_exists:
            writer.writeheader()
        writer.writerow(queue_row)

    return {
        "status": "collected",
        "reason": "new_sample_saved_for_feedback",
        "sample_id": sample_hash,
        "stored_image_path": str(stored_image_path),
        "label_status": "pending_label",
    }


# ==============================
# Feedback Label Import & Active Learning
# ==============================
def import_feedback_labels(
    *,
    labels_csv: Path,
    output_dir: Path,
    strict: bool = False,
) -> dict[str, Any]:
    labels_csv = Path(labels_csv).resolve()
    output_dir = Path(output_dir).resolve()
    feedback_root = output_dir / "feedback_pool"
    metadata_dir = feedback_root / "metadata"
    index_path = feedback_root / "feedback_index.json"
    manifest_path = feedback_root / "labeled_manifest.csv"
    labeled_root = feedback_root / "labeled"

    if not labels_csv.exists():
        raise FileNotFoundError(f"labels csv not found: {labels_csv}")
    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"feedback metadata directory not found: {metadata_dir}"
        )

    rows = pd.read_csv(labels_csv)
    required = {"sample_id", "label_mask_path"}
    missing_cols = sorted(required - set(rows.columns))
    if missing_cols:
        raise ValueError(f"labels csv missing columns: {', '.join(missing_cols)}")

    index_data: dict[str, Any] = {}
    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as f:
                parsed = json.load(f)
            if isinstance(parsed, dict):
                index_data = parsed
        except Exception:
            index_data = {}

    existing_rows: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            existing_rows = pd.read_csv(manifest_path).to_dict(orient="records")
        except Exception:
            existing_rows = []
    existing_ids = {str(r.get("sample_id", "")) for r in existing_rows}

    imported = 0
    skipped = 0
    issues: list[dict[str, Any]] = []
    out_rows: list[dict[str, Any]] = list(existing_rows)

    for row in rows.to_dict(orient="records"):
        sample_id = str(row.get("sample_id", "")).strip()
        label_mask_path = Path(str(row.get("label_mask_path", "")).strip()).resolve()
        if not sample_id:
            skipped += 1
            issues.append(make_issue("feedback_import", "missing_sample_id"))
            continue
        if not label_mask_path.exists():
            skipped += 1
            issues.append(
                make_issue(
                    "feedback_import",
                    "label_mask_not_found",
                    filename=sample_id,
                    mask_path=label_mask_path,
                )
            )
            continue

        meta_path = metadata_dir / f"{sample_id}.json"
        if not meta_path.exists():
            skipped += 1
            issues.append(
                make_issue(
                    "feedback_import",
                    "metadata_missing",
                    filename=sample_id,
                    image_path=meta_path,
                )
            )
            continue

        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as ex:
            skipped += 1
            issues.append(
                make_issue(
                    "feedback_import",
                    "metadata_parse_failed",
                    filename=sample_id,
                    details=str(ex),
                )
            )
            continue

        sensor = (
            str(row.get("sensor", "")).strip() or str(meta.get("sensor", "")).strip()
        )
        if sensor not in SENSOR_CHANNELS:
            skipped += 1
            issues.append(
                make_issue(
                    "feedback_import",
                    "invalid_sensor",
                    filename=sample_id,
                    details=str(sensor),
                )
            )
            continue

        image_path = Path(str(meta.get("stored_image_path", "")).strip())
        if not image_path.exists():
            skipped += 1
            issues.append(
                make_issue(
                    "feedback_import",
                    "stored_image_missing",
                    filename=sample_id,
                    image_path=image_path,
                )
            )
            continue

        safe_stem = _safe_stem(
            str(meta.get("image_name", sample_id)) or sample_id, fallback=sample_id
        )
        sensor_root = labeled_root / sensor
        images_dir = sensor_root / "images"
        masks_dir = sensor_root / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        target_image = images_dir / f"{sample_id}_{safe_stem}.tif"
        target_mask = masks_dir / f"{sample_id}_{safe_stem}.tif"

        shutil.copy2(image_path, target_image)
        shutil.copy2(label_mask_path, target_mask)

        now_iso = utc_now_iso()
        meta["status"] = "labeled"
        meta["labeled_at_utc"] = now_iso
        meta["label_mask_source_path"] = str(label_mask_path)
        meta["eligible_for_retrain"] = True
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        index_entry = index_data.get(sample_id)
        if isinstance(index_entry, dict):
            index_entry["status"] = "labeled"
            index_entry["labeled_at_utc"] = now_iso
            index_entry["eligible_for_retrain"] = True
            index_entry["labeled_mask_path"] = str(target_mask)
            index_entry["stored_image_path"] = str(target_image)
            index_data[sample_id] = index_entry

        if sample_id not in existing_ids:
            out_rows.append(
                {
                    "sample_id": sample_id,
                    "status": "eligible_for_retrain",
                    "sensor": sensor,
                    "image_name": str(meta.get("image_name", target_image.name)),
                    "image_path": str(target_image),
                    "mask_path": str(target_mask),
                    "weather_features_json": json.dumps(
                        meta.get("weather_features", {}), ensure_ascii=False
                    ),
                    "imported_at_utc": now_iso,
                }
            )
            existing_ids.add(sample_id)

        imported += 1

    if index_data:
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
    write_csv(
        manifest_path,
        out_rows,
        fieldnames=[
            "sample_id",
            "status",
            "sensor",
            "image_name",
            "image_path",
            "mask_path",
            "weather_features_json",
            "imported_at_utc",
        ],
    )
    write_csv(
        output_dir / "feedback_import_issues.csv",
        issues,
        fieldnames=[
            "stage",
            "issue_type",
            "sensor",
            "root",
            "filename",
            "image_path",
            "mask_path",
            "details",
            "candidate_count",
            "candidates",
        ],
    )

    report = {
        "status": "ok",
        "labels_csv": str(labels_csv),
        "output_dir": str(output_dir),
        "imported_count": int(imported),
        "skipped_count": int(skipped),
        "issues_count": int(len(issues)),
        "manifest_path": str(manifest_path),
    }
    save_json(output_dir / "feedback_import_report.json", report)

    if strict and issues:
        raise RuntimeError(
            f"feedback import completed with {len(issues)} issues in strict mode"
        )
    return report


# ==============================
# Runtime Risk Routing
# ==============================
# Chooses best available risk path:
# with-weather model if full weather is present, otherwise fallback.
def normalize_weather_features(
    raw: dict[str, Any] | None,
) -> tuple[dict[str, float] | None, list[str]]:
    if raw is None:
        return None, []
    # If user didn't provide any weather values at all, treat it as intentional no-weather mode.
    if all(raw.get(name) in (None, "") for name in WEATHER_FEATURE_NAMES):
        return None, []
    out: dict[str, float] = {}
    missing: list[str] = []
    for name in WEATHER_FEATURE_NAMES:
        value = raw.get(name)
        if value is None or value == "":
            missing.append(name)
            continue
        try:
            out[name] = float(value)
        except Exception:
            missing.append(name)
    if missing:
        return None, missing
    return out, []


def parse_weather_inputs(
    weather_json: str | None, weather_kv: list[str] | None
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    if weather_json:
        p = Path(weather_json)
        if not p.exists():
            raise FileNotFoundError(f"weather json not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("weather json must be an object")
        payload.update(data)
    if weather_kv:
        for item in weather_kv:
            if "=" not in item:
                raise ValueError(f"invalid --weather-kv item: {item}")
            key, value = item.split("=", 1)
            payload[key.strip()] = value.strip()
    return payload or None


def apply_sensor_risk_policy(
    *,
    sensor: str | None,
    pred_feats: dict[str, float],
    risk_score: float | None,
    risk_threshold: float,
    base_label: int | None,
) -> dict[str, Any]:
    sensor_key = str(sensor or "").upper()
    cfg = SENSOR_POLICY_CONFIG.get(sensor_key)
    flood_ratio = float(pred_feats.get("pred_flood_ratio", 0.0) or 0.0)
    if cfg is None:
        return {
            "risk_label": base_label,
            "flood_presence_label": int(flood_ratio > 0.0),
            "flood_presence_threshold": 0.0,
            "sensor_policy_applied": False,
            "sensor_policy_rule": None,
            "sensor_risk_threshold": float(risk_threshold),
            "risk_label_raw_threshold": base_label,
        }

    presence_threshold = float(cfg["presence_ratio_threshold"])
    flood_presence_label = int(flood_ratio >= presence_threshold)
    sensor_risk_threshold = float(
        np.clip(float(risk_threshold) + float(cfg["risk_threshold_offset"]), 0.0, 1.0)
    )
    if risk_score is None:
        final_label = int(flood_presence_label)
    else:
        score_label = int(float(risk_score) >= sensor_risk_threshold)
        if str(cfg["combine_rule"]).lower() == "and":
            final_label = int(score_label == 1 and flood_presence_label == 1)
        else:
            final_label = int(score_label == 1 or flood_presence_label == 1)

    return {
        "risk_label": int(final_label),
        "flood_presence_label": int(flood_presence_label),
        "flood_presence_threshold": float(presence_threshold),
        "sensor_policy_applied": True,
        "sensor_policy_rule": f"{sensor_key}_{str(cfg['combine_rule']).lower()}",
        "sensor_risk_threshold": float(sensor_risk_threshold),
        "risk_label_raw_threshold": base_label,
    }


def attach_detection_prediction_labels(out: dict[str, Any]) -> dict[str, Any]:
    det = out.get("flood_presence_label")
    det_i: int | None
    if det is None:
        det_i = None
        out["detection_label"] = None
        out["detection_text"] = "unknown"
    else:
        det_i = int(det)
        out["detection_label"] = det_i
        out["detection_text"] = "flood_detected" if det_i == 1 else "no_flood_detected"

    # Business rule:
    # if flood is already detected now, predictive label is not applicable.
    if det_i == 1:
        out["prediction_label"] = None
        out["prediction_text"] = "prediction_not_applicable_flood_already_detected"
        out["prediction_applicable"] = False
        out["prediction_status"] = "suppressed_due_to_detection"
        return out

    pred = out.get("risk_label")
    if pred is None:
        out["prediction_label"] = None
        out["prediction_text"] = "unknown"
        out["prediction_applicable"] = False
        out["prediction_status"] = "unavailable"
    else:
        pred_i = int(pred)
        out["prediction_label"] = pred_i
        out["prediction_text"] = (
            "flood_risk_predicted" if pred_i == 1 else "no_flood_risk_predicted"
        )
        out["prediction_applicable"] = True
        out["prediction_status"] = "active"
    return out


def route_risk_prediction(
    pred_feats: dict[str, float],
    weather_values: dict[str, Any] | None,
    risk_with_weather_model: Any | None,
    risk_no_weather_model: Any | None,
    *,
    risk_threshold: float = 0.5,
    risk_threshold_profile: str = DEFAULT_RISK_THRESHOLD_PROFILE,
    sensor: str | None = None,
) -> dict[str, Any]:
    weather_norm, missing = normalize_weather_features(weather_values)
    if risk_with_weather_model is not None and weather_norm is not None:
        row = np.array(
            [weather_norm[k] for k in WEATHER_FEATURE_NAMES]
            + [pred_feats[k] for k in IMAGE_FEATURE_NAMES],
            dtype=np.float32,
        ).reshape(1, -1)
        prob = float(risk_with_weather_model.predict_proba(row)[0, 1])
        base_label = int(prob >= risk_threshold)
        out = {
            "risk_score": prob,
            "risk_score_percent": to_percent(prob),
            "risk_label": base_label,
            "risk_threshold": float(risk_threshold),
            "risk_threshold_profile": risk_threshold_profile,
            "risk_model_used": "with_weather",
            "weather_features_used": True,
            "missing_weather_features": [],
        }
        out.update(
            apply_sensor_risk_policy(
                sensor=sensor,
                pred_feats=pred_feats,
                risk_score=prob,
                risk_threshold=float(risk_threshold),
                base_label=base_label,
            )
        )
        return attach_detection_prediction_labels(out)
    if risk_no_weather_model is not None:
        row = np.array(
            [pred_feats[k] for k in IMAGE_FEATURE_NAMES], dtype=np.float32
        ).reshape(1, -1)
        prob_raw = float(risk_no_weather_model.predict_proba(row)[0, 1])
        prob = prob_raw
        adjustment_note: str | None = None
        flood_ratio = float(pred_feats.get("pred_flood_ratio", 0.0) or 0.0)

        # Consistency guard:
        # if segmentation says almost no flooded pixels, cap risk score so summary
        # remains aligned with the observed flood footprint.
        if flood_ratio <= 0.002 and prob > 0.35:
            prob = 0.35
            adjustment_note = "capped_for_very_low_flood_ratio"
        elif flood_ratio <= 0.005 and prob > 0.45:
            prob = 0.45
            adjustment_note = "capped_for_low_flood_ratio"

        base_label = int(prob >= risk_threshold)
        out = {
            "risk_score": prob,
            "risk_score_percent": to_percent(prob),
            "risk_label": base_label,
            "risk_threshold": float(risk_threshold),
            "risk_threshold_profile": risk_threshold_profile,
            "risk_model_used": "no_weather_fallback",
            "weather_features_used": False,
            "missing_weather_features": missing,
            "risk_score_raw_model": prob_raw,
        }
        out.update(
            apply_sensor_risk_policy(
                sensor=sensor,
                pred_feats=pred_feats,
                risk_score=prob,
                risk_threshold=float(risk_threshold),
                base_label=base_label,
            )
        )
        if adjustment_note:
            out["risk_adjustment"] = adjustment_note
        # Warn only when weather input was attempted but incomplete.
        if risk_with_weather_model is not None and weather_norm is None and missing:
            out["risk_warning"] = "incomplete_weather_features_fallback_used"
        return attach_detection_prediction_labels(out)
    out = {
        "risk_score": None,
        "risk_score_percent": None,
        "risk_label": None,
        "risk_threshold": float(risk_threshold),
        "risk_threshold_profile": risk_threshold_profile,
        "risk_model_used": None,
        "weather_features_used": False,
        "missing_weather_features": missing,
        "risk_warning": "no_risk_model_available",
    }
    out.update(
        apply_sensor_risk_policy(
            sensor=sensor,
            pred_feats=pred_feats,
            risk_score=None,
            risk_threshold=float(risk_threshold),
            base_label=None,
        )
    )
    return attach_detection_prediction_labels(out)


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


# Decision-support helper that turns prediction outputs into practical actions.
def provide_solutions(
    prediction_row: dict[str, Any],
    *,
    flood_ratio_medium_threshold: float = 0.08,
    flood_ratio_high_threshold: float = 0.20,
    risk_score_medium_threshold: float = 0.45,
    risk_score_high_threshold: float = 0.75,
) -> dict[str, Any]:
    flood_ratio = _to_float_or_none(prediction_row.get("pred_flood_ratio"))
    if flood_ratio is None:
        flood_ratio = _to_float_or_none(prediction_row.get("img_flood_ratio"))
    flood_ratio = float(flood_ratio or 0.0)

    risk_score = _to_float_or_none(prediction_row.get("risk_score"))
    tp_sum = _to_float_or_none(prediction_row.get("tp_sum"))
    runoff_mean = _to_float_or_none(prediction_row.get("runoff_mean"))
    runoff_max = _to_float_or_none(prediction_row.get("runoff_max"))
    runoff_sum = _to_float_or_none(prediction_row.get("runoff_sum"))

    if risk_score is not None:
        if (
            risk_score >= risk_score_high_threshold
            or flood_ratio >= flood_ratio_high_threshold
        ):
            risk_level = "high"
        elif (
            risk_score >= risk_score_medium_threshold
            or flood_ratio >= flood_ratio_medium_threshold
        ):
            risk_level = "medium"
        else:
            risk_level = "low"
        status = "ok"
    else:
        if flood_ratio >= flood_ratio_high_threshold:
            risk_level = "high"
        elif flood_ratio >= flood_ratio_medium_threshold:
            risk_level = "medium"
        else:
            risk_level = "low"
        status = "partial_no_risk_score"

    drivers: list[str] = []
    warnings: list[str] = []
    actions: list[str] = []

    if flood_ratio >= flood_ratio_high_threshold:
        drivers.append("Large detected flooded footprint.")
    elif flood_ratio >= flood_ratio_medium_threshold:
        drivers.append("Moderate detected flooded footprint.")
    else:
        drivers.append("Limited detected flooded footprint.")

    if tp_sum is not None and tp_sum > 0:
        drivers.append("Precipitation accumulation is above zero.")

    if runoff_mean is not None:
        drivers.append("Runoff data is available for risk context.")
    if runoff_mean is not None and runoff_max is not None and runoff_mean > 0:
        if runoff_max / runoff_mean >= 1.75:
            warnings.append(
                "Runoff spike detected (runoff_max significantly above runoff_mean)."
            )
            actions.append(
                "Inspect culverts/drains near low-lying roads for sudden flow surges."
            )
    if runoff_sum is not None and runoff_sum > 0:
        drivers.append("Cumulative runoff is above zero.")

    if (
        prediction_row.get("risk_warning")
        == "incomplete_weather_features_fallback_used"
    ):
        warnings.append("Weather inputs were incomplete; fallback risk model was used.")
    if prediction_row.get("risk_model_used") is None:
        warnings.append(
            "No risk model was available; decisions are based on segmentation indicators only."
        )

    if risk_level == "high":
        actions.extend(
            [
                "Issue immediate flood watch/alert for exposed zones.",
                "Activate incident response and evacuation readiness plan.",
                "Deploy temporary flood barriers and emergency pump units.",
            ]
        )
    elif risk_level == "medium":
        actions.extend(
            [
                "Prepare field inspection teams for vulnerable locations.",
                "Pre-position pumps and verify stormwater drainage clearance.",
                "Increase monitoring frequency for next satellite overpass.",
            ]
        )
    else:
        actions.extend(
            [
                "Continue routine monitoring and archive this prediction.",
                "Schedule next assessment with updated weather inputs if available.",
            ]
        )

    dedup_actions: list[str] = []
    for action in actions:
        if action not in dedup_actions:
            dedup_actions.append(action)

    return {
        "status": status,
        "risk_level": risk_level,
        "risk_score_percent": to_percent(risk_score),
        "flood_ratio_percent": to_percent(flood_ratio),
        "primary_drivers": drivers[:4],
        "warnings": warnings,
        "recommended_actions": dedup_actions[:6],
    }


# ==============================
# Top-Level Commands
# ==============================
# run_predict_command: runs one prediction + QA checks + audit logging.
def _extract_loss_curve(history: Any) -> tuple[list[int], list[float]]:
    if not isinstance(history, list):
        return [], []
    epochs: list[int] = []
    losses: list[float] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        try:
            ep = int(row.get("epoch"))
            loss = float(row.get("train_loss"))
        except Exception:
            continue
        epochs.append(ep)
        losses.append(loss)
    return epochs, losses


def _extract_history_metric_curve(
    history: Any,
    *,
    metric_name: str,
) -> tuple[list[int], list[float], str]:
    if not isinstance(history, list):
        return [], [], "unavailable"
    epochs: list[int] = []
    values: list[float] = []
    sources: list[str] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        try:
            ep = int(row.get("epoch"))
        except Exception:
            continue
        metric_value = None
        metric_source = "unavailable"
        image_metrics = row.get("val_metrics_image_level")
        if isinstance(image_metrics, dict):
            metric_value = _to_float_or_none(image_metrics.get(metric_name))
            if metric_value is not None:
                metric_source = "image_level"
        if metric_value is None:
            patch_metrics = row.get("val_metrics_patch")
            if isinstance(patch_metrics, dict):
                metric_value = _to_float_or_none(patch_metrics.get(metric_name))
                if metric_value is not None:
                    metric_source = "patch_level"
        if metric_value is None:
            continue
        epochs.append(ep)
        values.append(float(metric_value))
        sources.append(metric_source)
    unique_sources = sorted(set(sources))
    if not unique_sources:
        source_label = "unavailable"
    elif len(unique_sources) == 1:
        source_label = unique_sources[0]
    else:
        source_label = "mixed"
    return epochs, values, source_label


def _merge_epoch_series(
    *series: tuple[list[int], list[float]],
) -> tuple[list[int], list[float]]:
    merged: dict[int, list[float]] = defaultdict(list)
    for epochs, values in series:
        for ep, val in zip(epochs, values):
            merged[int(ep)].append(float(val))
    if not merged:
        return [], []
    out_epochs = sorted(merged)
    out_values = [float(np.mean(merged[ep])) for ep in out_epochs]
    return out_epochs, out_values


def _build_monotone_smooth_curve(
    epochs: list[int],
    values: list[float],
    *,
    points_per_segment: int = 24,
) -> tuple[list[float], list[float]]:
    x = np.asarray(epochs, dtype=np.float32)
    y = np.asarray(values, dtype=np.float32)
    if x.size <= 2 or y.size <= 2:
        return [float(v) for v in x], [float(v) for v in y]

    # Monotone cubic Hermite interpolation keeps the smooth curve passing
    # through each epoch point instead of drifting away like a rolling average.
    h = np.diff(x)
    delta = np.diff(y) / np.maximum(h, 1e-6)
    m = np.zeros_like(y)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, len(y) - 1):
        if delta[i - 1] == 0.0 or delta[i] == 0.0 or np.sign(delta[i - 1]) != np.sign(delta[i]):
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / ((w1 / delta[i - 1]) + (w2 / delta[i]))

    x_dense: list[float] = []
    y_dense: list[float] = []
    seg_points = max(8, int(points_per_segment))
    for i in range(len(x) - 1):
        x0 = float(x[i])
        x1 = float(x[i + 1])
        y0 = float(y[i])
        y1 = float(y[i + 1])
        hi = float(max(x1 - x0, 1e-6))
        t_values = np.linspace(0.0, 1.0, seg_points, endpoint=False, dtype=np.float32)
        for t in t_values:
            tt = float(t)
            h00 = 2.0 * tt**3 - 3.0 * tt**2 + 1.0
            h10 = tt**3 - 2.0 * tt**2 + tt
            h01 = -2.0 * tt**3 + 3.0 * tt**2
            h11 = tt**3 - tt**2
            y_t = h00 * y0 + h10 * hi * float(m[i]) + h01 * y1 + h11 * hi * float(m[i + 1])
            x_dense.append(float(x0 + tt * hi))
            y_dense.append(float(y_t))
    x_dense.append(float(x[-1]))
    y_dense.append(float(y[-1]))
    return x_dense, y_dense


def _style_training_axis(
    ax: Any,
    *,
    ylabel: str,
    percent_axis: bool = False,
) -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.22)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    if percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    for spine in ax.spines.values():
        spine.set_color("#D9DEE7")
        spine.set_linewidth(1.0)


def _plot_sensor_series(
    ax: Any,
    *,
    epochs: list[int],
    values: list[float],
    color: str,
    label: str,
    linewidth: float = 2.0,
    alpha: float = 0.95,
    show_raw_points: bool = True,
) -> None:
    if not epochs or not values:
        return
    if show_raw_points:
        ax.plot(
            epochs,
            values,
            marker="o",
            markersize=4.8,
            linewidth=1.25,
            color=color,
            alpha=min(0.28, float(alpha)),
            label="_nolegend_",
        )
    dense_x, dense_y = _build_monotone_smooth_curve(epochs, values)
    ax.plot(
        dense_x,
        dense_y,
        marker=None,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        label=label,
    )


def _mark_best_epoch(
    ax: Any,
    *,
    best_epoch: int | None,
    epochs: list[int],
    values: list[float],
    color: str,
    label: str,
) -> None:
    if not best_epoch or not epochs or not values:
        return
    value_map = {int(ep): float(val) for ep, val in zip(epochs, values)}
    if int(best_epoch) not in value_map:
        return
    y_val = float(value_map[int(best_epoch)])
    ax.scatter(
        [int(best_epoch)],
        [y_val],
        marker="*",
        s=170,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        zorder=8,
    )


def _format_score_text(label: str, value: float | None) -> str:
    if value is None:
        return f"{label}: n/a"
    return f"{label}: {float(value) * 100.0:.1f}%"


def save_pipeline_training_progress_artifacts(
    report: dict[str, Any],
    *,
    output_dir: Path,
) -> tuple[Path | None, Path | None]:
    sensors_payload = report.get("sensors", {}) if isinstance(report, dict) else {}
    s1_hist = (
        (sensors_payload.get("S1") or {}).get("history")
        or (sensors_payload.get("S1") or {}).get("history_tail")
        or []
    )
    s2_hist = (
        (sensors_payload.get("S2") or {}).get("history")
        or (sensors_payload.get("S2") or {}).get("history_tail")
        or []
    )
    e1, l1 = _extract_loss_curve(s1_hist)
    e2, l2 = _extract_loss_curve(s2_hist)
    if not e1 and not e2:
        return None, None
    s1_f1_epochs, s1_f1_values, s1_f1_source = _extract_history_metric_curve(
        s1_hist,
        metric_name="f1",
    )
    s2_f1_epochs, s2_f1_values, s2_f1_source = _extract_history_metric_curve(
        s2_hist,
        metric_name="f1",
    )
    s1_iou_epochs, s1_iou_values, s1_iou_source = _extract_history_metric_curve(
        s1_hist,
        metric_name="iou",
    )
    s2_iou_epochs, s2_iou_values, s2_iou_source = _extract_history_metric_curve(
        s2_hist,
        metric_name="iou",
    )

    s1_map = {int(ep): float(loss) for ep, loss in zip(e1, l1)}
    s2_map = {int(ep): float(loss) for ep, loss in zip(e2, l2)}
    epochs = sorted(set(s1_map.keys()) | set(s2_map.keys()))
    curve_rows: list[dict[str, Any]] = []
    mean_epochs: list[int] = []
    mean_losses: list[float] = []
    mean_f1_epochs, mean_f1_values = _merge_epoch_series(
        (s1_f1_epochs, s1_f1_values),
        (s2_f1_epochs, s2_f1_values),
    )
    mean_iou_epochs, mean_iou_values = _merge_epoch_series(
        (s1_iou_epochs, s1_iou_values),
        (s2_iou_epochs, s2_iou_values),
    )
    s1_f1_map = {int(ep): float(val) for ep, val in zip(s1_f1_epochs, s1_f1_values)}
    s2_f1_map = {int(ep): float(val) for ep, val in zip(s2_f1_epochs, s2_f1_values)}
    mean_f1_map = {
        int(ep): float(val) for ep, val in zip(mean_f1_epochs, mean_f1_values)
    }
    s1_iou_map = {int(ep): float(val) for ep, val in zip(s1_iou_epochs, s1_iou_values)}
    s2_iou_map = {int(ep): float(val) for ep, val in zip(s2_iou_epochs, s2_iou_values)}
    mean_iou_map = {
        int(ep): float(val) for ep, val in zip(mean_iou_epochs, mean_iou_values)
    }
    for ep in epochs:
        vals: list[float] = []
        s1_loss = s1_map.get(ep)
        s2_loss = s2_map.get(ep)
        if s1_loss is not None:
            vals.append(float(s1_loss))
        if s2_loss is not None:
            vals.append(float(s2_loss))
        if not vals:
            continue
        mean_loss = float(np.mean(vals))
        mean_epochs.append(int(ep))
        mean_losses.append(mean_loss)
        curve_rows.append(
            {
                "epoch": int(ep),
                "s1_train_loss": s1_loss,
                "s2_train_loss": s2_loss,
                "mean_train_loss": mean_loss,
                "s1_val_f1": s1_f1_map.get(ep),
                "s2_val_f1": s2_f1_map.get(ep),
                "mean_val_f1": mean_f1_map.get(ep),
                "s1_val_iou": s1_iou_map.get(ep),
                "s2_val_iou": s2_iou_map.get(ep),
                "mean_val_iou": mean_iou_map.get(ep),
            }
        )

    curve_csv_path = output_dir / "unet_training_loss_curve.csv"
    write_csv(
        curve_csv_path,
        curve_rows,
        fieldnames=[
            "epoch",
            "s1_train_loss",
            "s2_train_loss",
            "mean_train_loss",
            "s1_val_f1",
            "s2_val_f1",
            "mean_val_f1",
            "s1_val_iou",
            "s2_val_iou",
            "mean_val_iou",
        ],
    )

    colors = {
        "S1": "#4E79A7",
        "S2": "#F28E2B",
        "mean": "#0F766E",
    }
    config = report.get("config", {}) if isinstance(report, dict) else {}
    scores = _extract_pipeline_report_scores(report)
    sensors_payload = sensors_payload if isinstance(sensors_payload, dict) else {}
    s1_payload = sensors_payload.get("S1") or {}
    s2_payload = sensors_payload.get("S2") or {}
    model_kind = str(config.get("model_kind", "unknown"))
    temporal_model_type = str(config.get("temporal_model_type", "unknown"))
    display_model_kind = model_kind.replace("_", " ")
    display_temporal_model_type = temporal_model_type.replace("_", " ")
    total_epochs = int(max(mean_epochs)) if mean_epochs else 0

    fig = plt.figure(figsize=(14.2, 8.8), facecolor="#F5F7FB")
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.25, 1.0],
        left=0.06,
        right=0.98,
        top=0.84,
        bottom=0.13,
        hspace=0.31,
        wspace=0.18,
    )
    ax_loss = fig.add_subplot(grid[0, :])
    ax_f1 = fig.add_subplot(grid[1, 0])
    ax_iou = fig.add_subplot(grid[1, 1])

    fig.suptitle(
        "Flood Model Training Overview",
        fontsize=18,
        fontweight="bold",
        x=0.06,
        y=0.962,
        ha="left",
    )
    fig.text(
        0.06,
        0.922,
        (
            f"Segmentation: {display_model_kind}   |   Temporal: {display_temporal_model_type}"
            f"   |   Epochs trained: {total_epochs}"
        ),
        fontsize=11.0,
        color="#475467",
    )

    summary_lines = [
        _format_score_text("Global F1", scores.get("seg_f1")),
        _format_score_text("Global IoU", scores.get("seg_iou")),
        _format_score_text("Temporal AUC", scores.get("temporal_auc")),
        (
            "S1 best epoch/thr: "
            f"{int(s1_payload.get('best_epoch', 0) or 0)} / "
            f"{float(s1_payload.get('decision_threshold', 0.5) or 0.5):.3f}"
        ),
        (
            "S2 best epoch/thr: "
            f"{int(s2_payload.get('best_epoch', 0) or 0)} / "
            f"{float(s2_payload.get('decision_threshold', 0.5) or 0.5):.3f}"
        ),
    ]
    fig.text(
        0.98,
        0.922,
        "\n".join(summary_lines),
        ha="right",
        va="top",
        fontsize=10.2,
        color="#344054",
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": "#F8FAFC",
            "edgecolor": "#D9DEE7",
            "alpha": 0.96,
        },
    )

    _style_training_axis(ax_loss, ylabel="Train Loss")
    _plot_sensor_series(
        ax_loss,
        epochs=e1,
        values=l1,
        color=colors["S1"],
        label=(
            f"S1 loss ({float(l1[-1]):.3f})" if l1 else "S1 loss"
        ),
        linewidth=2.1,
        alpha=0.78,
    )
    _plot_sensor_series(
        ax_loss,
        epochs=e2,
        values=l2,
        color=colors["S2"],
        label=(
            f"S2 loss ({float(l2[-1]):.3f})" if l2 else "S2 loss"
        ),
        linewidth=2.1,
        alpha=0.78,
    )
    _plot_sensor_series(
        ax_loss,
        epochs=mean_epochs,
        values=mean_losses,
        color=colors["mean"],
        label=(
            f"Mean loss ({float(mean_losses[-1]):.3f})"
            if mean_losses
            else "Mean loss"
        ),
        linewidth=2.8,
        alpha=1.0,
    )
    ax_loss.set_title("Training Loss Convergence", fontsize=12.8, fontweight="bold")
    if mean_epochs:
        x_max = float(max(mean_epochs))
        ax_loss.set_xlim(left=1.0, right=float(max(2.0, x_max + 0.4)))

    _style_training_axis(ax_f1, ylabel="Validation F1", percent_axis=True)
    _plot_sensor_series(
        ax_f1,
        epochs=s1_f1_epochs,
        values=s1_f1_values,
        color=colors["S1"],
        label="S1 F1",
    )
    _plot_sensor_series(
        ax_f1,
        epochs=s2_f1_epochs,
        values=s2_f1_values,
        color=colors["S2"],
        label="S2 F1",
    )
    _plot_sensor_series(
        ax_f1,
        epochs=mean_f1_epochs,
        values=mean_f1_values,
        color=colors["mean"],
        label="Mean F1",
        linewidth=2.6,
    )
    ax_f1.set_title("Validation F1", fontsize=12.5, fontweight="bold")

    _style_training_axis(ax_iou, ylabel="Validation IoU", percent_axis=True)
    _plot_sensor_series(
        ax_iou,
        epochs=s1_iou_epochs,
        values=s1_iou_values,
        color=colors["S1"],
        label="S1 IoU",
    )
    _plot_sensor_series(
        ax_iou,
        epochs=s2_iou_epochs,
        values=s2_iou_values,
        color=colors["S2"],
        label="S2 IoU",
    )
    _plot_sensor_series(
        ax_iou,
        epochs=mean_iou_epochs,
        values=mean_iou_values,
        color=colors["mean"],
        label="Mean IoU",
        linewidth=2.6,
    )
    ax_iou.set_title("Validation IoU", fontsize=12.5, fontweight="bold")

    selection_metric = str(config.get("selection_metric", "f1")).strip().lower()
    target_ax = ax_f1 if selection_metric == "f1" else ax_iou
    target_series = {
        "S1": (
            s1_f1_epochs if selection_metric == "f1" else s1_iou_epochs,
            s1_f1_values if selection_metric == "f1" else s1_iou_values,
        ),
        "S2": (
            s2_f1_epochs if selection_metric == "f1" else s2_iou_epochs,
            s2_f1_values if selection_metric == "f1" else s2_iou_values,
        ),
    }
    _mark_best_epoch(
        target_ax,
        best_epoch=(
            int(s1_payload.get("best_epoch"))
            if isinstance(s1_payload.get("best_epoch"), (int, float))
            else None
        ),
        epochs=target_series["S1"][0],
        values=target_series["S1"][1],
        color=colors["S1"],
        label="S1",
    )
    _mark_best_epoch(
        target_ax,
        best_epoch=(
            int(s2_payload.get("best_epoch"))
            if isinstance(s2_payload.get("best_epoch"), (int, float))
            else None
        ),
        epochs=target_series["S2"][0],
        values=target_series["S2"][1],
        color=colors["S2"],
        label="S2",
    )

    ax_loss.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        fontsize=9.6,
    )
    ax_f1.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9.8)
    ax_iou.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9.8)
    fig.text(
        0.06,
        0.045,
        (
            "Display uses monotone smoothing that passes through the epoch points; "
            "faint markers show raw epochs. "
            f"Validation sources: F1 [{s1_f1_source}/{s2_f1_source}], "
            f"IoU [{s1_iou_source}/{s2_iou_source}]."
        ),
        fontsize=9.2,
        color="#667085",
    )

    chart_path = output_dir / "unet_training_progress.png"
    fig.savefig(chart_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return chart_path, curve_csv_path


def _extract_pipeline_report_scores(report: dict[str, Any] | None) -> dict[str, float | None]:
    if not isinstance(report, dict):
        return {
            "seg_iou": None,
            "seg_f1": None,
            "seg_accuracy": None,
            "temporal_auc": None,
            "risk_weather_auc": None,
            "risk_fallback_auc": None,
        }
    global_metrics = report.get("global_val_metrics", {})
    risk_models = report.get("risk_models", {})
    temporal_cv = ((risk_models.get("temporal_metrics") or {}).get("cv_mean") or {})
    with_weather_cv = (
        ((risk_models.get("with_weather_metrics") or {}).get("cv_mean") or {})
    )
    fallback_cv = ((risk_models.get("no_weather_metrics") or {}).get("cv_mean") or {})
    return {
        "seg_iou": _to_float_or_none(global_metrics.get("iou")),
        "seg_f1": _to_float_or_none(global_metrics.get("f1")),
        "seg_accuracy": _to_float_or_none(global_metrics.get("accuracy")),
        "temporal_auc": _to_float_or_none(temporal_cv.get("roc_auc")),
        "risk_weather_auc": _to_float_or_none(with_weather_cv.get("roc_auc")),
        "risk_fallback_auc": _to_float_or_none(fallback_cv.get("roc_auc")),
    }


def _score_pipeline_candidate(scores: dict[str, float | None]) -> float:
    seg_iou = scores.get("seg_iou") if isinstance(scores.get("seg_iou"), float) else 0.0
    seg_f1 = scores.get("seg_f1") if isinstance(scores.get("seg_f1"), float) else 0.0
    temporal_auc = (
        scores.get("temporal_auc") if isinstance(scores.get("temporal_auc"), float) else 0.0
    )
    # Weighted objective: prioritize segmentation quality, but keep temporal forecasting meaningful.
    return float(seg_iou) + 0.5 * float(seg_f1) + 0.7 * float(temporal_auc)


def _should_keep_new_pipeline_report(
    *,
    previous_report: dict[str, Any] | None,
    candidate_report: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    previous_scores = _extract_pipeline_report_scores(previous_report)
    candidate_scores = _extract_pipeline_report_scores(candidate_report)
    if previous_scores.get("seg_iou") is None:
        return True, {
            "reason": "no_previous_baseline",
            "previous_scores": previous_scores,
            "candidate_scores": candidate_scores,
            "previous_composite": None,
            "candidate_composite": _score_pipeline_candidate(candidate_scores),
        }

    previous_seg_iou = float(previous_scores.get("seg_iou") or 0.0)
    candidate_seg_iou = float(candidate_scores.get("seg_iou") or 0.0)
    previous_temporal = previous_scores.get("temporal_auc")
    candidate_temporal = candidate_scores.get("temporal_auc")

    if candidate_seg_iou > previous_seg_iou + 1e-6:
        return True, {
            "reason": "seg_iou_improved",
            "previous_scores": previous_scores,
            "candidate_scores": candidate_scores,
            "previous_composite": _score_pipeline_candidate(previous_scores),
            "candidate_composite": _score_pipeline_candidate(candidate_scores),
        }

    # Allow a tiny segmentation drop only when temporal gain is substantial.
    if (
        previous_temporal is not None
        and candidate_temporal is not None
        and (previous_seg_iou - candidate_seg_iou) <= 0.003
        and (float(candidate_temporal) - float(previous_temporal)) >= 0.03
    ):
        return True, {
            "reason": "temporal_gain_with_small_seg_drop",
            "previous_scores": previous_scores,
            "candidate_scores": candidate_scores,
            "previous_composite": _score_pipeline_candidate(previous_scores),
            "candidate_composite": _score_pipeline_candidate(candidate_scores),
        }

    previous_composite = _score_pipeline_candidate(previous_scores)
    candidate_composite = _score_pipeline_candidate(candidate_scores)
    if candidate_composite > previous_composite + 1e-6:
        return True, {
            "reason": "composite_score_improved",
            "previous_scores": previous_scores,
            "candidate_scores": candidate_scores,
            "previous_composite": previous_composite,
            "candidate_composite": candidate_composite,
        }

    return False, {
        "reason": "candidate_not_better_than_baseline",
        "previous_scores": previous_scores,
        "candidate_scores": candidate_scores,
        "previous_composite": previous_composite,
        "candidate_composite": candidate_composite,
    }


def load_pipeline_best_profile(output_dir: Path) -> dict[str, Any] | None:
    return load_json_file(Path(output_dir).resolve() / PIPELINE_BEST_PROFILE_NAME)


def _extract_pipeline_config_from_report(report: dict[str, Any]) -> dict[str, Any]:
    cfg = report.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    epochs = _to_float_or_none(cfg.get("epochs"))
    val_ratio = _to_float_or_none(cfg.get("val_ratio"))
    model_kind = str(cfg.get("model_kind", "") or "").strip().lower()
    temporal_model_type = str(cfg.get("temporal_model_type", "") or "").strip().lower()
    out: dict[str, Any] = {}
    if epochs is not None and epochs > 0:
        out["epochs"] = int(round(epochs))
    if val_ratio is not None and 0.0 < val_ratio < 0.95:
        out["val_ratio"] = float(val_ratio)
    if model_kind:
        out["model_kind"] = model_kind
    if temporal_model_type in TEMPORAL_MODEL_TYPE_CHOICES:
        out["temporal_model_type"] = temporal_model_type
    return out


def _build_pipeline_profile_candidate(
    *,
    report: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any] | None:
    config = _extract_pipeline_config_from_report(report)
    if "epochs" not in config or "val_ratio" not in config:
        return None

    scores = _extract_pipeline_report_scores(report)
    candidate = {
        "schema_version": 1,
        "updated_at_utc": utc_now_iso(),
        "source": source,
        "config": {
            "epochs": int(config["epochs"]),
            "val_ratio": float(config["val_ratio"]),
            "model_kind": str(config.get("model_kind", "small_unet")),
            "temporal_model_type": str(config.get("temporal_model_type", "adaboost")),
        },
        "scores": {
            **scores,
            "composite": float(_score_pipeline_candidate(scores)),
        },
    }
    return candidate


def _profile_composite_value(profile: dict[str, Any] | None) -> float | None:
    if not isinstance(profile, dict):
        return None
    scores = profile.get("scores")
    if not isinstance(scores, dict):
        return None
    return _to_float_or_none(scores.get("composite"))


def _should_update_pipeline_best_profile(
    *,
    existing_profile: dict[str, Any] | None,
    candidate_profile: dict[str, Any],
) -> tuple[bool, str]:
    if not isinstance(existing_profile, dict):
        return True, "no_previous_profile"
    current = _profile_composite_value(existing_profile)
    candidate = _profile_composite_value(candidate_profile)
    if candidate is None:
        return False, "candidate_missing_composite"
    if current is None:
        return True, "previous_profile_missing_composite"
    if candidate > current + 1e-6:
        return True, "candidate_composite_improved"
    return False, "candidate_not_better"


def update_pipeline_best_profile_from_report(
    *,
    output_dir: Path,
    report: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    profile_path = output_dir / PIPELINE_BEST_PROFILE_NAME
    existing_profile = load_pipeline_best_profile(output_dir)
    candidate = _build_pipeline_profile_candidate(report=report, source=source)
    if candidate is None:
        decision = {
            "status": "skipped",
            "reason": "report_missing_required_config",
            "profile_path": str(profile_path.resolve()),
            "updated_at_utc": utc_now_iso(),
        }
        return decision

    should_update, reason = _should_update_pipeline_best_profile(
        existing_profile=existing_profile,
        candidate_profile=candidate,
    )
    if should_update:
        save_json(profile_path, candidate)
    decision = {
        "status": "updated" if should_update else "kept_previous",
        "reason": reason,
        "profile_path": str(profile_path.resolve()),
        "previous_composite": _profile_composite_value(existing_profile),
        "candidate_composite": _profile_composite_value(candidate),
        "active_config": (
            candidate.get("config")
            if should_update
            else (
                (existing_profile or {}).get("config")
                if isinstance(existing_profile, dict)
                else None
            )
        ),
        "updated_at_utc": utc_now_iso(),
    }
    return decision


def _is_truthy_env(name: str) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def maybe_launch_train_live_monitor(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    command_name: str,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if os.name != "nt":
        return {"status": "skipped", "reason": "non_windows"}
    if bool(getattr(args, "no_live_monitor", False)):
        return {"status": "skipped", "reason": "cli_disabled"}
    if _is_truthy_env(TRAIN_LIVE_MONITOR_DISABLE_ENV):
        return {"status": "skipped", "reason": "env_disabled"}
    if _is_truthy_env(TRAIN_LIVE_MONITOR_ACTIVE_ENV):
        return {"status": "skipped", "reason": "already_started_by_monitor"}

    monitor_script = (PROJECT_BASE_DIR / "scripts" / "run_train_pipeline_full_live.ps1").resolve()
    if not monitor_script.exists():
        return {
            "status": "skipped",
            "reason": "monitor_script_missing",
            "monitor_script": str(monitor_script),
        }

    powershell_exe = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell_exe:
        return {"status": "skipped", "reason": "powershell_missing"}

    try:
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        proc = subprocess.Popen(
            [
                powershell_exe,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(monitor_script),
                "-Mode",
                "attach",
                "-OutputDir",
                str(output_dir),
                "-PollSeconds",
                "5",
            ],
            cwd=str(PROJECT_BASE_DIR),
            creationflags=int(creationflags),
        )
        return {
            "status": "launched",
            "reason": "auto_enabled",
            "command": str(command_name),
            "pid": int(proc.pid),
            "monitor_script": str(monitor_script),
            "output_dir": str(output_dir),
        }
    except Exception as ex:
        return {
            "status": "failed",
            "reason": "launch_error",
            "error": str(ex),
            "command": str(command_name),
            "monitor_script": str(monitor_script),
            "output_dir": str(output_dir),
        }


def resolve_train_pipeline_auto_profile(
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    effective = {
        "epochs": int(args.epochs),
        "val_ratio": float(args.val_ratio),
        "model_kind": str(getattr(args, "model_kind", "small_unet")),
        "temporal_model_type": str(getattr(args, "temporal_model_type", "adaboost")),
    }
    if bool(getattr(args, "disable_auto_best_profile", False)):
        return effective, None

    profile = load_pipeline_best_profile(output_dir)
    if not isinstance(profile, dict):
        return effective, None
    cfg = profile.get("config")
    if not isinstance(cfg, dict):
        return effective, None

    epochs = _to_float_or_none(cfg.get("epochs"))
    if epochs is not None and epochs > 0:
        effective["epochs"] = int(round(epochs))
    val_ratio = _to_float_or_none(cfg.get("val_ratio"))
    if val_ratio is not None and 0.0 < val_ratio < 0.95:
        effective["val_ratio"] = float(val_ratio)
    model_kind = str(cfg.get("model_kind", "") or "").strip().lower()
    if model_kind:
        effective["model_kind"] = model_kind
    temporal_model_type = str(cfg.get("temporal_model_type", "") or "").strip().lower()
    if temporal_model_type in TEMPORAL_MODEL_TYPE_CHOICES:
        effective["temporal_model_type"] = temporal_model_type

    return effective, profile


def run_train_pipeline_command(args: argparse.Namespace) -> None:
    # CLI orchestration for end-to-end Pipeline V3 training:
    # prepare runtime state -> run training -> decide whether to keep the candidate
    # -> regenerate charts/previews from the selected report.
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_state = maybe_launch_train_live_monitor(
        output_dir=output_dir,
        args=args,
        command_name="train-pipeline",
    )
    if monitor_state.get("status") == "launched":
        print(
            f"[live] auto monitor launched pid={monitor_state.get('pid')} "
            f"output_dir={output_dir}"
        )
    elif monitor_state.get("status") == "failed":
        print(f"[live] auto monitor launch failed: {monitor_state.get('error')}")
    data_roots = [Path(p).resolve() for p in args.data_roots]
    no_flood_roots = [Path(p).resolve() for p in getattr(args, "no_flood_roots", [])]
    csv_path = Path(args.csv_path).resolve()
    temporal_csv_path, temporal_bridge_csv_path = resolve_temporal_paths(args, csv_path)
    effective_cfg, auto_profile = resolve_train_pipeline_auto_profile(
        output_dir=output_dir,
        args=args,
    )

    segmentation_pipeline = import_segmentation_pipeline()
    allowed_model_kinds = [
        str(x).strip().lower()
        for x in getattr(segmentation_pipeline, "MODEL_KIND_CHOICES", [effective_cfg["model_kind"]])
        if str(x).strip()
    ]
    if (
        allowed_model_kinds
        and str(effective_cfg["model_kind"]).strip().lower() not in allowed_model_kinds
    ):
        fallback_model = str(getattr(args, "model_kind", "small_unet")).strip().lower()
        if fallback_model not in allowed_model_kinds:
            fallback_model = allowed_model_kinds[0]
        effective_cfg["model_kind"] = fallback_model

    production_artifact_names = [
        PIPELINE_MODEL_S1_NAME,
        PIPELINE_MODEL_S2_NAME,
        RISK_WITH_WEATHER_PIPELINE_NAME,
        RISK_NO_WEATHER_PIPELINE_NAME,
        RISK_TEMPORAL_PIPELINE_NAME,
        "unet_train_report.json",
        "unet_val_metrics_global.json",
        "unet_val_metrics_s1.json",
        "unet_val_metrics_s2.json",
        "unet_val_report_s1.csv",
        "unet_val_report_s2.csv",
        "risk_with_weather_cv_metrics_unet.json",
        "risk_no_weather_cv_metrics_unet.json",
        RISK_TEMPORAL_METRICS_PIPELINE_NAME,
        "risk_with_weather_training_table_unet.csv",
        "risk_no_weather_training_table_unet.csv",
        RISK_TEMPORAL_TABLE_PIPELINE_NAME,
        "dataset_issues_unet.csv",
        "input_profile.json",
        "model_registry.json",
        DATASET_METADATA_CSV_NAME,
        DATASET_METADATA_SUMMARY_NAME,
        ACTIVE_BACKEND_NAME,
    ]
    production_artifacts = [(output_dir / x).resolve() for x in production_artifact_names]
    previous_report = load_json_file(output_dir / "unet_train_report.json")
    backup_manifest: dict[str, str] = {}
    if previous_report is not None:
        backup_manifest = backup_artifacts_for_promotion(output_dir, production_artifacts)

    try:
        # segmentation_pipeline owns the actual deep-learning work. project.py stays as the
        # shell that wires CLI, monitoring, artifact safety, and reporting together.
        candidate_report = segmentation_pipeline.train_pipeline_models(
            data_roots=data_roots,
            no_flood_roots=no_flood_roots,
            csv_path=csv_path,
            temporal_csv_path=temporal_csv_path,
            temporal_bridge_csv_path=temporal_bridge_csv_path,
            output_dir=output_dir,
            test_images=list(args.test_images),
            val_ratio=float(effective_cfg["val_ratio"]),
            seed=int(args.seed),
            threshold=float(args.threshold),
            patch_size=int(args.patch_size),
            stride=int(args.stride),
            epochs=int(effective_cfg["epochs"]),
            early_stopping_patience=int(args.early_stopping_patience),
            batch_size_s1=int(args.batch_size_s1),
            batch_size_s2=int(args.batch_size_s2),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            max_patches_per_image=int(args.max_patches_per_image),
            infer_batch_size=int(args.infer_batch_size),
            model_kind=str(effective_cfg["model_kind"]),
            temporal_model_type=str(effective_cfg["temporal_model_type"]),
            segmentation_mask_sync_policy=str(
                getattr(
                    args,
                    "segmentation_mask_sync_policy",
                    DEFAULT_SEGMENTATION_MASK_SYNC_POLICY,
                )
            ),
            segmentation_source_groups=getattr(
                args, "segmentation_source_groups", None
            ),
            segmentation_balance_policy=str(
                getattr(
                    args,
                    "segmentation_balance_policy",
                    DEFAULT_SEGMENTATION_IMAGE_BALANCE_POLICY,
                )
            ),
            segmentation_balance_min_flood_ratio=float(
                getattr(
                    args,
                    "segmentation_balance_min_flood_ratio",
                    DEFAULT_SEGMENTATION_BALANCE_MIN_FLOOD_RATIO,
                )
            ),
        )
    except Exception:
        if backup_manifest:
            restore_artifacts_from_backup(backup_manifest)
        raise

    # Keep the new candidate only if it beats or matches the promoted baseline
    # according to the project selection policy. Otherwise restore the previous run.
    accepted, decision = _should_keep_new_pipeline_report(
        previous_report=previous_report,
        candidate_report=candidate_report,
    )
    active_report = candidate_report
    decision_path = output_dir / "train_unet_selection_decision.json"
    if not accepted and backup_manifest:
        restore_artifacts_from_backup(backup_manifest)
        restored_report = load_json_file(output_dir / "unet_train_report.json")
        if isinstance(restored_report, dict):
            active_report = restored_report

    best_profile_update = update_pipeline_best_profile_from_report(
        output_dir=output_dir,
        report=active_report,
        source={
            "command": "train-pipeline",
            "selected_candidate": bool(accepted),
            "selection_reason": str(decision.get("reason")),
            "auto_profile_used": bool(auto_profile is not None),
            "requested_config": {
                "epochs": int(args.epochs),
                "val_ratio": float(args.val_ratio),
                "model_kind": str(getattr(args, "model_kind", "small_unet")),
                "temporal_model_type": str(
                    getattr(args, "temporal_model_type", "adaboost")
                ),
            },
            "effective_config": {
                "epochs": int(effective_cfg["epochs"]),
                "val_ratio": float(effective_cfg["val_ratio"]),
                "model_kind": str(effective_cfg["model_kind"]),
                "temporal_model_type": str(effective_cfg["temporal_model_type"]),
            },
            "report_path": str((output_dir / "unet_train_report.json").resolve()),
        },
    )
    decision_payload = {
        "status": "accepted_new_candidate" if accepted else "rejected_candidate_restored_previous",
        "decision": decision,
        "backup_manifest_count": int(len(backup_manifest)),
        "auto_profile_used": bool(auto_profile is not None),
        "auto_profile_config": (
            auto_profile.get("config")
            if isinstance(auto_profile, dict) and isinstance(auto_profile.get("config"), dict)
            else None
        ),
        "effective_training_config": {
            "epochs": int(effective_cfg["epochs"]),
            "val_ratio": float(effective_cfg["val_ratio"]),
            "model_kind": str(effective_cfg["model_kind"]),
            "temporal_model_type": str(effective_cfg["temporal_model_type"]),
        },
        "best_profile_update": best_profile_update,
        "timestamp_utc": utc_now_iso(),
    }
    save_json(decision_path, decision_payload)

    # Presentation artifacts must always reflect the selected runtime, not merely
    # the last candidate that happened to finish training.
    chart_path, curve_csv_path = save_pipeline_training_progress_artifacts(
        active_report, output_dir=output_dir
    )
    if chart_path is not None:
        print(f"[chart] training progress image: {chart_path}")
    if curve_csv_path is not None:
        print(f"[chart] training curve csv: {curve_csv_path}")
    preview_paths = (
        active_report.get("train_preview_paths")
        if isinstance(active_report, dict)
        else None
    )
    if isinstance(preview_paths, dict) and preview_paths:
        for sensor, path in sorted(preview_paths.items()):
            print(f"[preview] train {sensor}: {path}")
    if auto_profile is not None:
        print(
            "[auto-best] applied profile "
            f"epochs={effective_cfg['epochs']} val={float(effective_cfg['val_ratio']):.2f} "
            f"model={effective_cfg['model_kind']} temporal={effective_cfg['temporal_model_type']}"
        )
    else:
        print("[auto-best] no saved best profile found, using CLI config")
    print(f"[selection] status={decision_payload['status']} reason={decision.get('reason')}")
    print(
        "[auto-best] profile_update="
        f"{best_profile_update.get('status')} reason={best_profile_update.get('reason')}"
    )
    print(f"[selection] decision_json={decision_path.resolve()}")
    print(json.dumps(active_report, indent=2, ensure_ascii=False))


def run_compare_algorithms_command(args: argparse.Namespace) -> None:
    from benchmark_models import run_compare_algorithms_command as _run_compare_algorithms

    _run_compare_algorithms(args)


def run_compare_temporal_models_command(args: argparse.Namespace) -> None:
    from benchmark_models import run_compare_temporal_models_command as _run_compare

    _run_compare(args)


def run_predict_command(args: argparse.Namespace) -> None:
    # Single-image prediction entrypoint used by CLI and reused by API/GUI logic.
    # Steps: resolve backend -> infer mask/prob -> risk routing -> save artifacts.
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_threshold = resolve_risk_threshold(
        args.risk_threshold_profile, args.risk_threshold
    )

    # Start from the currently promoted runtime bundle unless the caller overrides
    # specific model paths. This keeps CLI/API/GUI consistent by default.
    default_model_dir, artifact_dir_source = resolve_prediction_artifact_dir()
    model_registry = load_model_registry(default_model_dir)
    input_profile = load_training_input_profile(default_model_dir)

    # Sensor resolution happens before model loading because it determines channel
    # count, preprocessing stats, and artifact paths.
    data_roots = [Path(p).resolve() for p in args.data_roots]
    roots_with_sensor = []
    for root in data_roots:
        sensor = infer_sensor_from_root(root)
        if sensor:
            roots_with_sensor.append((root, sensor))
    sensor = args.sensor or detect_sensor_for_image(image_path, roots_with_sensor)
    if sensor not in SENSOR_CHANNELS:
        raise ValueError(f"Could not determine sensor for image: {image_path}")

    backend_state = resolve_prediction_backend(
        requested_backend=str(getattr(args, "backend", "auto")),
        output_dir=default_model_dir,
        sensor=sensor,
    )
    seg_backend = backend_state["segmentation_backend"]
    risk_backend = backend_state["risk_backend"]

    pipeline_bundle: dict[str, Any] | None = None
    try:
        segmentation_pipeline = import_segmentation_pipeline()

        pipeline_bundle = segmentation_pipeline.load_pipeline_bundle(
            get_pipeline_model_path(sensor, default_model_dir), device="cpu"
        )
    except Exception as ex:
        raise RuntimeError(
            f"Pipeline V3 model could not be loaded for sensor {sensor}: {ex}"
        ) from ex

    # Validate geospatial metadata early so downstream weather/timeline export can
    # trust the coordinates and timestamps associated with this image.
    geo_meta = inspect_geospatial_metadata(image_path)
    enforce_geospatial_checks(geo_meta, strict=args.strict_geospatial_checks)

    try:
        x_img = load_image(image_path, SENSOR_CHANNELS[sensor])
    except ValueError:
        # If the requested sensor does not match the actual channel layout, try one
        # automatic correction pass and reload the matching runtime bundle.
        auto_sensor = detect_sensor_for_image(image_path, roots_with_sensor)
        if auto_sensor in SENSOR_CHANNELS and auto_sensor != sensor:
            print(
                f"[warn] sensor corrected from {sensor} to {auto_sensor} based on image channels."
            )
            sensor = auto_sensor
            x_img = load_image(image_path, SENSOR_CHANNELS[sensor])
            segmentation_pipeline = import_segmentation_pipeline()

            pipeline_bundle = segmentation_pipeline.load_pipeline_bundle(
                get_pipeline_model_path(sensor, default_model_dir), device="cpu"
            )
        else:
            raise
    drift_meta = detect_input_drift(
        x_img,
        sensor,
        input_profile,
        zscore_threshold=float(args.drift_zscore_threshold),
    )
    segmentation_pipeline = import_segmentation_pipeline()

    assert pipeline_bundle is not None
    # Inference defaults to the patch/stride/threshold stored in the trained bundle
    # so prediction uses the same geometry it was validated with.
    infer_patch_size = int(
        getattr(args, "pipeline_patch_size", 256)
        or pipeline_bundle.get("patch_size", 256)
    )
    infer_stride = int(
        getattr(args, "pipeline_stride", 192) or pipeline_bundle.get("stride", 192)
    )
    seg_threshold_used, seg_threshold_source = resolve_segmentation_threshold(
        getattr(args, "threshold", None),
        bundle_threshold=pipeline_bundle.get("decision_threshold"),
        default_threshold=0.5,
    )
    pred_mask, pred_prob, infer_meta = segmentation_pipeline.predict_pipeline_mask_auto(
        model=pipeline_bundle["model"],
        x_img=x_img,
        threshold=float(seg_threshold_used),
        patch_size=infer_patch_size,
        stride=infer_stride,
        batch_size=int(getattr(args, "pipeline_batch_size", 8)),
        device=pipeline_bundle["device"],
        model_kind=str(pipeline_bundle.get("model_kind", "small_unet")),
        normalization_stats=pipeline_bundle.get("normalization_stats"),
    )
    model_kind = str(pipeline_bundle.get("model_kind", "small_unet"))
    model_version = f"{model_kind}_epoch_{int(pipeline_bundle.get('epoch', 0))}"

    pred_feats = summarize_prediction_features(pred_mask, pred_prob)
    geo_summary = build_geo_summary(pred_mask, geo_meta)

    # Risk models sit on top of segmentation output and can be swapped independently
    # from the segmentation bundle, so they are resolved and loaded separately.
    risk_with_weather_path = (
        Path(args.risk_model_with_weather_path).resolve()
        if args.risk_model_with_weather_path
        else get_pipeline_risk_model_paths(default_model_dir)[0]
    )
    risk_no_weather_path = (
        Path(args.risk_model_no_weather_path).resolve()
        if args.risk_model_no_weather_path
        else get_pipeline_risk_model_paths(default_model_dir)[1]
    )
    risk_temporal_path = (
        Path(args.risk_model_temporal_path).resolve()
        if args.risk_model_temporal_path
        else get_temporal_model_path(output_dir=default_model_dir, backend=PIPELINE_V3_BACKEND_ID)
    )

    allow_untrusted_models = bool(getattr(args, "allow_untrusted_model_paths", False))
    trusted_model_roots = [PROJECT_BASE_DIR.resolve(), default_model_dir.resolve()]
    risk_with_weather_model = (
        safe_joblib_load(
            risk_with_weather_path,
            allowed_roots=trusted_model_roots,
            allow_untrusted=allow_untrusted_models,
        )
        if risk_with_weather_path.exists()
        else None
    )
    risk_no_weather_model = (
        safe_joblib_load(
            risk_no_weather_path,
            allowed_roots=trusted_model_roots,
            allow_untrusted=allow_untrusted_models,
        )
        if risk_no_weather_path.exists()
        else None
    )
    risk_temporal_model = (
        safe_joblib_load(
            risk_temporal_path,
            allowed_roots=trusted_model_roots,
            allow_untrusted=allow_untrusted_models,
        )
        if risk_temporal_path.exists()
        else None
    )

    weather_values_raw = parse_weather_inputs(args.weather_json, args.weather_kv)
    weather_values: dict[str, Any] = (
        dict(weather_values_raw) if isinstance(weather_values_raw, dict) else {}
    )
    weather_source = "manual_input" if weather_values else "none"
    weather_csv_status: str | None = None
    weather_anchor_status: str | None = None
    weather_anchor_meta: dict[str, Any] = {}

    weather_csv_path = resolve_env_path(
        "WEATHER_CSV_PATH",
        base_dir=PROJECT_BASE_DIR,
        default_relative=DEFAULT_CSV_PATH,
    )
    temporal_csv_path, temporal_bridge_csv_path = resolve_temporal_paths(
        args, weather_csv_path
    )

    # Weather fill priority:
    # 1) explicit user-provided values
    # 2) filename match from weather CSV
    # 3) ERA5 anchor inferred from image metadata
    if image_path.name and weather_csv_path.exists():
        auto_weather_values, auto_warn = lookup_weather_features_for_filename(
            weather_csv_path, image_path.name
        )
        if auto_warn is None and auto_weather_values:
            if weather_values:
                filled = 0
                for name in WEATHER_FEATURE_NAMES:
                    if weather_values.get(name) in (None, ""):
                        auto_value = auto_weather_values.get(name)
                        if auto_value not in (None, ""):
                            weather_values[name] = auto_value
                            filled += 1
                if filled > 0:
                    weather_source = "manual_plus_csv_fill"
            else:
                weather_values = dict(auto_weather_values)
                weather_source = "auto_csv"
        else:
            weather_csv_status = auto_warn
    elif image_path.name:
        weather_csv_status = "weather_csv_path_not_found"

    if image_path.name and temporal_csv_path.exists():
        missing_after_csv = [
            name for name in WEATHER_FEATURE_NAMES if weather_values.get(name) in (None, "")
        ]
        if not weather_values or missing_after_csv:
            anchor_weather, anchor_status, anchor_meta = (
                lookup_weather_features_for_image_from_temporal(
                    csv_path=temporal_csv_path,
                    image_filename=image_path.name,
                    image_path=image_path,
                    geo_meta=geo_meta,
                    bridge_csv_path=temporal_bridge_csv_path,
                )
            )
            weather_anchor_meta = anchor_meta if isinstance(anchor_meta, dict) else {}
            weather_anchor_status = anchor_status
            if anchor_status is None and anchor_weather:
                if weather_values:
                    filled_anchor = 0
                    for name in WEATHER_FEATURE_NAMES:
                        if weather_values.get(name) in (None, ""):
                            anchor_value = anchor_weather.get(name)
                            if anchor_value not in (None, ""):
                                weather_values[name] = anchor_value
                                filled_anchor += 1
                    if filled_anchor > 0:
                        if weather_source == "manual_plus_csv_fill":
                            weather_source = "manual_plus_csv_era5_fill"
                        elif weather_source == "manual_input":
                            weather_source = "manual_plus_era5_fill"
                        elif weather_source == "none":
                            weather_source = "manual_plus_era5_fill"
                else:
                    weather_values = dict(anchor_weather)
                    weather_source = "auto_era5_anchor"

    risk = route_risk_prediction(
        pred_feats,
        weather_values if weather_values else None,
        risk_with_weather_model,
        risk_no_weather_model,
        risk_threshold=risk_threshold,
        risk_threshold_profile=args.risk_threshold_profile,
        sensor=sensor,
    )
    if int(risk.get("detection_label", 0) or 0) == 1:
        temporal_risk = {
            "temporal_status": "skipped_due_to_detection",
            "temporal_risk_score": None,
            "temporal_risk_score_percent": None,
            "temporal_risk_label": None,
            "temporal_risk_text": "skipped_due_to_detection",
            "temporal_model_used": None,
            "temporal_weather_match_status": "skipped_due_to_detection",
            "temporal_horizon": "short_term_sequence_window",
        }
    else:
        temporal_risk = predict_temporal_risk(
            image_filename=image_path.name,
            sensor=sensor,
            pred_feats=pred_feats,
            csv_path=temporal_csv_path,
            temporal_model=risk_temporal_model,
            risk_threshold=risk_threshold,
            bridge_csv_path=temporal_bridge_csv_path,
            image_path=image_path,
            geo_meta=geo_meta,
        )
    prediction_eta = build_prediction_eta(
        detection_label=risk.get("detection_label"),
        prediction_label=risk.get("prediction_label"),
        temporal_payload=temporal_risk,
    )
    decision_support = provide_solutions(
        {
            **pred_feats,
            "risk_score": risk.get("risk_score"),
            "risk_model_used": risk.get("risk_model_used"),
            "risk_warning": risk.get("risk_warning"),
            **(weather_values or {}),
        }
    )

    feedback_dir = (
        Path(args.feedback_output_dir).resolve()
        if args.feedback_output_dir
        else output_dir
    )
    if args.disable_feedback_collection:
        feedback_info: dict[str, Any] = {
            "status": "disabled",
            "reason": "feedback_collection_disabled_by_flag",
        }
    else:
        known_dataset_paths, known_dataset_filenames = build_known_dataset_signatures(
            data_roots
        )
        feedback_info = register_feedback_candidate(
            output_dir=feedback_dir,
            image_name=image_path.name,
            sensor=sensor,
            x_img=x_img,
            weather_values=weather_values,
            source_image_path=image_path,
            source_mode="predict_cli",
            known_dataset_paths=known_dataset_paths,
            known_dataset_filenames=known_dataset_filenames,
        )

    stem = image_path.stem
    mask_path = output_dir / f"{stem}_pred_mask.tif"
    prob_path = output_dir / f"{stem}_pred_prob.npy"
    preview_path = output_dir / f"{stem}_preview.png"
    prediction_json_path = output_dir / args.prediction_json_name

    tifffile.imwrite(mask_path, pred_mask.astype(np.uint8))
    np.save(prob_path, pred_prob.astype(np.float32))
    save_preview(preview_path, x_img, pred_prob, pred_mask)
    weather_statuses = [
        str(x)
        for x in [weather_csv_status, weather_anchor_status]
        if x not in (None, "", "ok")
    ]
    prediction_analysis = build_prediction_analysis(
        pred_mask=pred_mask,
        pred_prob=pred_prob,
        pred_feats=pred_feats,
        risk_payload=risk,
        temporal_payload=temporal_risk,
        prediction_eta=prediction_eta,
        decision_support=decision_support,
        drift_meta=drift_meta,
        geo_meta=geo_meta,
        weather_statuses=weather_statuses,
        seg_threshold=float(seg_threshold_used),
    )
    geo_exports = export_prediction_geo_artifacts(
        output_dir=output_dir,
        stem=stem,
        zone_mask=prediction_analysis["zone_mask"],
        pred_prob=pred_prob,
        geo_meta=geo_meta,
        zone_meta=prediction_analysis["zone_meta"],
    )

    payload = {
        "prediction_id": str(uuid4()),
        "timestamp_utc": utc_now_iso(),
        "image_path": str(image_path),
        "sensor": sensor,
        "segmentation_backend_used": seg_backend,
        "segmentation_model_kind_used": model_kind,
        "segmentation_model_epoch_used": int(pipeline_bundle.get("epoch", 0)),
        "risk_backend_used": risk_backend,
        "segmentation_threshold_used": float(seg_threshold_used),
        "segmentation_threshold_source": str(seg_threshold_source),
        "requested_backend": backend_state.get("requested_backend"),
        "available_backends": backend_state.get("available_backends"),
        "artifacts_dir_used": str(default_model_dir.resolve()),
        "artifacts_dir_source": str(artifact_dir_source),
        "model_version": model_version,
        "promotion_state": backend_state.get("active_backend"),
        "pred_flood_ratio": pred_feats["pred_flood_ratio"],
        "pred_flood_ratio_percent": to_percent(pred_feats["pred_flood_ratio"]),
        "pred_prob_mean": pred_feats["pred_prob_mean"],
        "pred_prob_mean_percent": to_percent(pred_feats["pred_prob_mean"]),
        "pred_prob_p90": pred_feats["pred_prob_p90"],
        "pred_prob_p90_percent": to_percent(pred_feats["pred_prob_p90"]),
        "risk_score": risk["risk_score"],
        "risk_score_percent": risk["risk_score_percent"],
        "risk_label": risk["risk_label"],
        "risk_label_raw_threshold": risk.get("risk_label_raw_threshold"),
        "sensor_policy_applied": risk.get("sensor_policy_applied"),
        "sensor_policy_rule": risk.get("sensor_policy_rule"),
        "sensor_risk_threshold": risk.get("sensor_risk_threshold"),
        "flood_presence_label": risk.get("flood_presence_label"),
        "flood_presence_threshold": risk.get("flood_presence_threshold"),
        "detection_label": risk.get("detection_label"),
        "detection_text": risk.get("detection_text"),
        "prediction_label": risk.get("prediction_label"),
        "prediction_text": risk.get("prediction_text"),
        "prediction_applicable": risk.get("prediction_applicable"),
        "prediction_status": risk.get("prediction_status"),
        "risk_threshold_profile": risk["risk_threshold_profile"],
        "risk_threshold": risk["risk_threshold"],
        "risk_model_used": risk["risk_model_used"],
        "weather_features_used": risk["weather_features_used"],
        "missing_weather_features": risk.get("missing_weather_features", []),
        "temporal_status": temporal_risk.get("temporal_status"),
        "temporal_risk_score": temporal_risk.get("temporal_risk_score"),
        "temporal_risk_score_percent": temporal_risk.get("temporal_risk_score_percent"),
        "temporal_risk_label": temporal_risk.get("temporal_risk_label"),
        "temporal_risk_text": temporal_risk.get("temporal_risk_text"),
        "temporal_model_used": temporal_risk.get("temporal_model_used"),
        "temporal_weather_match_status": temporal_risk.get(
            "temporal_weather_match_status"
        ),
        "temporal_lookup_mode": temporal_risk.get("temporal_lookup_mode"),
        "temporal_feature_snapshot": temporal_risk.get("temporal_feature_snapshot"),
        "temporal_anchor_source": temporal_risk.get("temporal_anchor_source"),
        "temporal_anchor_time_utc": temporal_risk.get("temporal_anchor_time_utc"),
        "temporal_anchor_lat": temporal_risk.get("temporal_anchor_lat"),
        "temporal_anchor_lon": temporal_risk.get("temporal_anchor_lon"),
        "prediction_eta_text": prediction_eta.get("prediction_eta_text"),
        "prediction_eta_start_utc": prediction_eta.get("prediction_eta_start_utc"),
        "prediction_eta_end_utc": prediction_eta.get("prediction_eta_end_utc"),
        "prediction_eta_source": prediction_eta.get("prediction_eta_source"),
        "prediction_eta_note": prediction_eta.get("prediction_eta_note"),
        "prediction_eta_days_min": prediction_eta.get("prediction_eta_days_min"),
        "prediction_eta_days_max": prediction_eta.get("prediction_eta_days_max"),
        "prediction_eta_hours_min": prediction_eta.get("prediction_eta_hours_min"),
        "prediction_eta_hours_max": prediction_eta.get("prediction_eta_hours_max"),
        "prediction_eta_horizon": prediction_eta.get("prediction_eta_horizon"),
        "prediction_eta_confidence_percent": prediction_eta.get(
            "prediction_eta_confidence_percent"
        ),
        "prediction_eta_confidence_level": prediction_eta.get(
            "prediction_eta_confidence_level"
        ),
        "mask_flood_policy": str(ACTIVE_MASK_FLOOD_POLICY),
        "weather_source": weather_source,
        "weather_csv_path_used": str(weather_csv_path),
        "temporal_csv_path_used": str(temporal_csv_path),
        "temporal_bridge_csv_path_used": str(temporal_bridge_csv_path),
        "decision_support": decision_support,
        "prediction_confidence": prediction_analysis["confidence"],
        "prediction_explanation": prediction_analysis["explanation"],
        "forecast_timeline": prediction_analysis["timeline"],
        "prediction_zone": prediction_analysis["zone_meta"],
        "geo_export": geo_exports,
        "inference_mode": infer_meta["mode"],
        "inference_config": infer_meta,
        "geospatial_checks": geo_meta,
        "geospatial_summary": geo_summary,
        "flood_area_km2": geo_summary.get("flood_area_km2"),
        "drift_check": drift_meta,
        "model_registry_path": str(
            (default_model_dir / "model_registry.json").resolve()
        ),
        "model_run_id": (
            model_registry.get("run_id") if isinstance(model_registry, dict) else None
        ),
        "feedback_collection": feedback_info,
        "artifacts": {
            "pred_mask_tif": str(mask_path),
            "pred_prob_npy": str(prob_path),
            "preview_png": str(preview_path),
            "prediction_zone_geojson": geo_exports.get("prediction_zone_geojson"),
            "prediction_zone_mask_geotiff": geo_exports.get(
                "prediction_zone_mask_geotiff"
            ),
            "prediction_probability_geotiff": geo_exports.get(
                "prediction_probability_geotiff"
            ),
        },
    }
    if backend_state.get("fallback_reason"):
        payload["backend_fallback_reason"] = backend_state["fallback_reason"]
    if weather_csv_status:
        payload["weather_csv_status"] = weather_csv_status
    if weather_anchor_status:
        payload["weather_anchor_status"] = weather_anchor_status
    if weather_anchor_meta:
        payload["weather_anchor_meta"] = weather_anchor_meta
    if "risk_warning" in risk:
        payload["risk_warning"] = risk["risk_warning"]
    save_json(prediction_json_path, payload)
    append_prediction_audit(output_dir, payload)
    print(json.dumps(payload, indent=2))


def main() -> None:
    # Entry-point dispatcher for CLI subcommands.
    args = parse_args()
    set_active_mask_flood_policy(getattr(args, "mask_flood_policy", None))
    if args.command == "import-labels":
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        report = import_feedback_labels(
            labels_csv=Path(args.csv),
            output_dir=output_dir,
            strict=bool(args.strict),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if args.command in {"train-pipeline", "train-pipeline"}:
        run_train_pipeline_command(args)
        return
    if args.command == "compare-algorithms":
        run_compare_algorithms_command(args)
        return
    if args.command == "train-benchmark":
        run_compare_algorithms_command(args)
        return
    if args.command == "compare-temporal-models":
        run_compare_temporal_models_command(args)
        return
    if args.command == "predict":
        run_predict_command(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

