from __future__ import annotations

import json
import os
import re
from io import BytesIO
import io
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from time import strftime
from typing import Any
from uuid import uuid4
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import tifffile

import project
from model_security import safe_joblib_load
try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

try:
    from env_utils import resolve_env_path, load_dotenv
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


st.set_page_config(
    page_title="Flood Intelligence Platform",
    page_icon="~",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)
OUTPUT_DIR_ENV_VAR = "FLOOD_OUTPUT_DIR"
DEFAULT_OUTPUT_DIR_FALLBACK = Path("outputs")
OUTPUT_DIR = resolve_env_path(
    OUTPUT_DIR_ENV_VAR,
    base_dir=BASE_DIR,
    default_relative=DEFAULT_OUTPUT_DIR_FALLBACK,
)

DEFAULT_DATA_ROOTS = "\n".join(project.DEFAULT_DATA_ROOTS)
DEFAULT_MODEL_S1 = str(project.get_pipeline_model_path("S1", OUTPUT_DIR))
DEFAULT_MODEL_S2 = str(project.get_pipeline_model_path("S2", OUTPUT_DIR))
DEFAULT_RISK_WITH_WEATHER = str(OUTPUT_DIR / project.RISK_WITH_WEATHER_PIPELINE_NAME)
DEFAULT_RISK_NO_WEATHER = str(OUTPUT_DIR / project.RISK_NO_WEATHER_PIPELINE_NAME)
DEFAULT_EXPORT_DIR = str(OUTPUT_DIR / "gui")
DEFAULT_THRESHOLD = 0.50
WEATHER_CSV_ENV_VAR = "WEATHER_CSV_PATH"
DEFAULT_WEATHER_CSV_FALLBACK = Path("dataset") / "Final_Full_Data_Matched.csv"
TEMPORAL_CSV_ENV_VAR = "TEMPORAL_CSV_PATH"
TEMPORAL_BRIDGE_CSV_ENV_VAR = "TEMPORAL_BRIDGE_CSV_PATH"
DEFAULT_TEMPORAL_CSV_FALLBACK = Path("dataset") / "ERA5_Final1_Combined.csv"
DEFAULT_TEMPORAL_BRIDGE_FALLBACK = Path("dataset") / "Final_Full_Data_Matched.csv"
DEFAULT_RISK_THRESHOLD_PROFILE = project.DEFAULT_RISK_THRESHOLD_PROFILE
DEFAULT_DRIFT_ZSCORE_THRESHOLD = project.DEFAULT_DRIFT_ZSCORE_THRESHOLD
DEFAULT_AUTO_TILING_PIXELS = project.DEFAULT_AUTO_TILING_PIXELS
DEFAULT_TILE_SIZE = project.DEFAULT_TILE_SIZE
DEFAULT_TILE_OVERLAP = project.DEFAULT_TILE_OVERLAP
DEFAULT_PREDICT_BATCH_ROWS = project.DEFAULT_PREDICT_BATCH_ROWS
DEFAULT_BACKEND = "auto"


def import_segmentation_pipeline_safe() -> Any:
    try:
        import segmentation_pipeline
    except Exception as ex:
        raise RuntimeError(
            "segmentation_pipeline.py is required for Pipeline V3 in this dashboard. "
            "Keep the file beside app.py and install dependencies via "
            "`pip install -r requirements-pipeline.txt`."
        ) from ex
    return segmentation_pipeline


def resolve_weather_csv_path() -> Path:
    return resolve_env_path(
        WEATHER_CSV_ENV_VAR,
        base_dir=BASE_DIR,
        default_relative=DEFAULT_WEATHER_CSV_FALLBACK,
    )


def resolve_temporal_csv_paths() -> tuple[Path, Path]:
    temporal_csv = resolve_env_path(
        TEMPORAL_CSV_ENV_VAR,
        base_dir=BASE_DIR,
        default_relative=DEFAULT_TEMPORAL_CSV_FALLBACK,
    )
    temporal_bridge_csv = resolve_env_path(
        TEMPORAL_BRIDGE_CSV_ENV_VAR,
        base_dir=BASE_DIR,
        default_relative=DEFAULT_TEMPORAL_BRIDGE_FALLBACK,
    )
    return temporal_csv, temporal_bridge_csv


# ==============================
# Runtime Artifact Loading
# ==============================
# Streamlit reruns frequently, so model/bundle loading is cached and keyed by file
# mtime. This keeps the app responsive while still reloading when artifacts change.
@st.cache_resource
def load_model(path: str, mtime_token: int = -1) -> Any | None:
    p = Path(path)
    if not p.exists():
        return None
    trusted_roots = [BASE_DIR.resolve(), OUTPUT_DIR.resolve()]
    try:
        return safe_joblib_load(p, allowed_roots=trusted_roots, allow_untrusted=False)
    except Exception:
        return None


def _load_pipeline_bundle_impl(path: str) -> tuple[Any | None, str | None]:
    p = Path(path)
    if not p.exists():
        return None, f"file_not_found: {p}"
    try:
        segmentation_pipeline = import_segmentation_pipeline_safe()
        bundle = segmentation_pipeline.load_pipeline_bundle(p, device="cpu")
        return bundle, None
    except Exception as ex:
        return None, f"{type(ex).__name__}: {ex}"


@st.cache_resource
def load_pipeline_bundle(path: str, mtime_token: int = -1) -> tuple[Any | None, str | None]:
    return _load_pipeline_bundle_impl(path)


def load_pipeline_bundle_uncached(path: str) -> tuple[Any | None, str | None]:
    return _load_pipeline_bundle_impl(path)


def file_mtime_token(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return -1
    try:
        return int(p.stat().st_mtime_ns)
    except Exception:
        return -1


def preload_pipeline_bundles(
    pipeline_model_paths: dict[str, Path],
) -> dict[str, str]:
    # Warm the cache up front so the first actual prediction click does not pay the
    # full bundle-load cost for both sensors.
    status: dict[str, str] = {}
    for sensor, path in pipeline_model_paths.items():
        if not path.exists():
            status[sensor] = "missing"
            continue
        bundle, err = load_pipeline_bundle(str(path), file_mtime_token(path))
        if bundle is not None:
            status[sensor] = "loaded"
        else:
            status[sensor] = f"error: {err or 'unknown'}"
    return status


@st.cache_data
def discover_all_pairs(data_roots: tuple[str, ...]) -> project.DiscoveryResult:
    roots = [Path(p) for p in data_roots]
    return project.discover_dataset(roots)


@st.cache_data
def load_weather_aggregate(csv_path: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    return project.aggregate_csv_features(Path(csv_path))


@st.cache_data(show_spinner=False, ttl=20)
def load_experiment_rows_cached(
    project_dir: str, limit: int
) -> list[dict[str, Any]]:
    return project.collect_experiment_run_summaries(Path(project_dir), limit=limit)


@st.cache_data(show_spinner=False, ttl=20)
def load_failure_report_cached(artifact_dir: str) -> dict[str, Any]:
    return project.build_failure_analysis_report(Path(artifact_dir))


def inject_ui_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

        :root {
          --bg-soft: linear-gradient(135deg, #0b1220 0%, #101a2e 55%, #16263a 100%);
          --card-bg: rgba(14, 24, 40, 0.78);
          --card-border: rgba(149, 176, 211, 0.20);
          --text-main: #ecf3ff;
          --text-muted: #9fb1c9;
          --ok: #17c964;
          --warn: #f5a524;
          --danger: #f31260;
        }

        html, body, [class*="css"]  {
          font-family: "Manrope", "Segoe UI", Tahoma, sans-serif;
        }

        .stApp {
          background: var(--bg-soft);
          color: var(--text-main);
        }

        [data-testid="stSidebar"] {
          background: rgba(8, 14, 26, 0.92);
          border-right: 1px solid rgba(149, 176, 211, 0.20);
        }

        [data-testid="stMetric"] {
          background: var(--card-bg);
          border: 1px solid var(--card-border);
          border-radius: 14px;
          padding: 10px 14px;
        }

        [data-testid="stMetricLabel"] {
          white-space: normal !important;
          overflow-wrap: anywhere;
          line-height: 1.2;
        }

        [data-testid="stMetricValue"] {
          line-height: 1.05;
          font-size: 2rem;
        }

        @media (max-width: 1400px) {
          [data-testid="stMetricValue"] {
            font-size: 1.7rem;
          }
        }

        .block-container {
          padding-top: 1.6rem;
        }

        h1, h2, h3 {
          letter-spacing: 0.2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def to_channels_last(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image dimensions: {arr.shape}")
    if arr.shape[0] <= 16 and arr.shape[1] == arr.shape[2]:
        return np.moveaxis(arr, 0, -1)
    return arr


def detect_sensor_from_raw(raw: bytes) -> str | None:
    arr = np.asarray(tifffile.imread(BytesIO(raw)))
    arr = to_channels_last(arr)
    ch = arr.shape[-1]
    if ch == project.SENSOR_CHANNELS["S1"] or ch == 1:
        return "S1"
    if ch == project.SENSOR_CHANNELS["S2"]:
        return "S2"
    return None


def load_uploaded_image(raw: bytes, required_channels: int) -> np.ndarray:
    arr = np.asarray(tifffile.imread(BytesIO(raw)))
    arr = to_channels_last(arr)
    if arr.shape[-1] < required_channels:
        if required_channels == 2 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 2, axis=-1)
        else:
            raise ValueError(
                f"Uploaded image has {arr.shape[-1]} channels but needs {required_channels}"
            )
    arr = arr[..., :required_channels].astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_channel(channel: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(channel.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.quantile(x, 0.02))
    hi = float(np.quantile(x, 0.98))
    if hi - lo < 1e-6:
        lo = float(np.min(x))
        hi = float(np.max(x))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def normalize_prob_for_display(prob: np.ndarray) -> np.ndarray:
    # Improve visual contrast when probabilities occupy a narrow range.
    p = np.nan_to_num(prob.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.quantile(p, 0.02))
    hi = float(np.quantile(p, 0.98))
    if hi - lo < 1e-6:
        lo = float(np.min(p))
        hi = float(np.max(p))
    if hi - lo < 1e-6:
        return np.zeros_like(p, dtype=np.float32)
    p = np.clip((p - lo) / (hi - lo), 0.0, 1.0)
    return p


def build_input_overview(
    x_img: np.ndarray, sensor: str | None = None
) -> tuple[np.ndarray, str]:
    x = np.nan_to_num(x_img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    channels = int(x.shape[-1]) if x.ndim == 3 else 1
    sensor_tag = (sensor or "").strip().upper()
    # Explicit S1 false-color RGB preview.
    if x.ndim == 3 and sensor_tag == "S1" and channels >= 2:
        c0 = normalize_channel(x[..., 0])
        c1 = normalize_channel(x[..., 1])
        rgb = np.stack([c1, c0, np.zeros_like(c0)], axis=-1)
        return rgb, "Input Overview (RGB)"
    # Explicit S2 RGB composite preview.
    if x.ndim == 3 and sensor_tag == "S2" and channels >= 3:
        if channels >= 4:
            # Common ordering fallback: [.., B2, B3, B4, ...] => RGB from (B4,B3,B2).
            idx_r, idx_g, idx_b = 3, 2, 1
        else:
            idx_r, idx_g, idx_b = 2, 1, 0
        rgb = np.stack(
            [
                normalize_channel(x[..., idx_r]),
                normalize_channel(x[..., idx_g]),
                normalize_channel(x[..., idx_b]),
            ],
            axis=-1,
        )
        return rgb, "Input Overview (RGB)"
    # Fallback by channel count when sensor tag is unavailable.
    if x.ndim == 3 and channels >= 3:
        if channels >= 4:
            idx_r, idx_g, idx_b = 3, 2, 1
        else:
            idx_r, idx_g, idx_b = 2, 1, 0
        rgb = np.stack(
            [
                normalize_channel(x[..., idx_r]),
                normalize_channel(x[..., idx_g]),
                normalize_channel(x[..., idx_b]),
            ],
            axis=-1,
        )
        return rgb, "Input Overview (RGB)"
    if x.ndim == 3 and channels >= 2:
        c0 = normalize_channel(x[..., 0])
        c1 = normalize_channel(x[..., 1])
        rgb = np.stack([c1, c0, np.zeros_like(c0)], axis=-1)
        return rgb, "Input Overview (RGB)"
    # Single-channel fallback: replicate channel to RGB.
    gray = normalize_channel(x[..., 0] if x.ndim == 3 else x)
    rgb = np.stack([gray, gray, gray], axis=-1)
    return rgb, "Input Overview (RGB)"


def downsample_for_display(arr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    if arr.ndim < 2:
        return arr
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h <= max_side and w <= max_side:
        return arr
    step_h = max(1, int(np.ceil(h / max_side)))
    step_w = max(1, int(np.ceil(w / max_side)))
    return arr[::step_h, ::step_w, ...]


def dilate_binary_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2:
        return m
    r = max(0, int(radius))
    if r == 0:
        return m
    if ndi is not None:
        kernel = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
        return ndi.binary_dilation(m, structure=kernel)
    h, w = m.shape
    out = m.copy()
    for dy in range(-r, r + 1):
        y_src0 = max(0, -dy)
        y_src1 = min(h, h - dy)
        y_dst0 = max(0, dy)
        y_dst1 = min(h, h + dy)
        if y_src0 >= y_src1 or y_dst0 >= y_dst1:
            continue
        for dx in range(-r, r + 1):
            x_src0 = max(0, -dx)
            x_src1 = min(w, w - dx)
            x_dst0 = max(0, dx)
            x_dst1 = min(w, w + dx)
            if x_src0 >= x_src1 or x_dst0 >= x_dst1:
                continue
            out[y_dst0:y_dst1, x_dst0:x_dst1] |= m[y_src0:y_src1, x_src0:x_src1]
    return out


def keep_largest_components(
    mask: np.ndarray, *, min_pixels: int = 16, max_regions: int = 3
) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2 or not bool(np.any(m)):
        return m
    min_pixels = max(1, int(min_pixels))
    max_regions = max(1, int(max_regions))
    if ndi is not None:
        labeled, n_comp = ndi.label(m)
        if int(n_comp) <= 0:
            return np.zeros_like(m, dtype=bool)
        counts = np.bincount(labeled.ravel())
        if counts.size <= 1:
            return np.zeros_like(m, dtype=bool)
        counts[0] = 0
        valid = np.where(counts >= min_pixels)[0]
        if valid.size == 0:
            best = int(np.argmax(counts))
            if best <= 0:
                return np.zeros_like(m, dtype=bool)
            return labeled == best
        top = valid[np.argsort(counts[valid])[::-1][:max_regions]]
        return np.isin(labeled, top)
    h, w = m.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    components: list[list[tuple[int, int]]] = []
    for y0 in range(h):
        for x0 in range(w):
            if (not m[y0, x0]) or visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = 1
            comp: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                y1 = max(0, y - 1)
                y2 = min(h - 1, y + 1)
                x1 = max(0, x - 1)
                x2 = min(w - 1, x + 1)
                for ny in range(y1, y2 + 1):
                    for nx in range(x1, x2 + 1):
                        if not m[ny, nx] or visited[ny, nx]:
                            continue
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
            components.append(comp)

    if not components:
        return np.zeros_like(m, dtype=bool)

    components.sort(key=len, reverse=True)
    keep: list[list[tuple[int, int]]] = [
        c for c in components if len(c) >= max(1, int(min_pixels))
    ][: max(1, int(max_regions))]
    if not keep:
        keep = [components[0]]

    out = np.zeros_like(m, dtype=bool)
    for comp in keep:
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        out[np.array(ys, dtype=np.int32), np.array(xs, dtype=np.int32)] = True
    return out


def summarize_zone_components(
    mask: np.ndarray, *, min_pixels: int = 8, max_regions: int = 3
) -> list[dict[str, float | int]]:
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2 or not bool(np.any(m)):
        return []
    min_pixels = max(1, int(min_pixels))
    max_regions = max(1, int(max_regions))
    if ndi is not None:
        labeled, n_comp = ndi.label(m)
        if int(n_comp) <= 0:
            return []
        counts = np.bincount(labeled.ravel())
        if counts.size <= 1:
            return []
        counts[0] = 0
        valid = np.where(counts >= min_pixels)[0]
        if valid.size == 0:
            return []
        top = valid[np.argsort(counts[valid])[::-1][:max_regions]]
        h, w = m.shape
        components_fast: list[dict[str, float | int]] = []
        for lab in top:
            ys, xs = np.where(labeled == int(lab))
            if ys.size == 0:
                continue
            area = int(ys.size)
            components_fast.append(
                {
                    "area": area,
                    "scene_pct": float((area / float(h * w)) * 100.0),
                    "x_min": int(xs.min()),
                    "x_max": int(xs.max()),
                    "y_min": int(ys.min()),
                    "y_max": int(ys.max()),
                    "x_center": int(round(float(xs.mean()))),
                    "y_center": int(round(float(ys.mean()))),
                }
            )
        components_fast.sort(key=lambda c: int(c["area"]), reverse=True)
        return components_fast
    h, w = m.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    components: list[dict[str, float | int]] = []
    for y0 in range(h):
        for x0 in range(w):
            if (not m[y0, x0]) or visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = 1
            ys: list[int] = []
            xs: list[int] = []
            while stack:
                y, x = stack.pop()
                ys.append(int(y))
                xs.append(int(x))
                y1 = max(0, y - 1)
                y2 = min(h - 1, y + 1)
                x1 = max(0, x - 1)
                x2 = min(w - 1, x + 1)
                for ny in range(y1, y2 + 1):
                    for nx in range(x1, x2 + 1):
                        if not m[ny, nx] or visited[ny, nx]:
                            continue
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
            area = len(ys)
            if area < max(1, int(min_pixels)):
                continue
            y_min, y_max = min(ys), max(ys)
            x_min, x_max = min(xs), max(xs)
            y_center = int(round(float(np.mean(ys))))
            x_center = int(round(float(np.mean(xs))))
            components.append(
                {
                    "area": int(area),
                    "scene_pct": float((area / float(h * w)) * 100.0),
                    "x_min": int(x_min),
                    "x_max": int(x_max),
                    "y_min": int(y_min),
                    "y_max": int(y_max),
                    "x_center": int(x_center),
                    "y_center": int(y_center),
                }
            )
    components.sort(key=lambda c: int(c["area"]), reverse=True)
    return components[: max(1, int(max_regions))]


def build_mask_canvas(
    zone_mask: np.ndarray,
    *,
    flood_color: tuple[float, float, float] = (1.0, 0.20, 0.05),
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    draw_border: bool = True,
) -> np.ndarray:
    m = np.asarray(zone_mask).astype(bool)
    if m.ndim == 3:
        m = m[..., 0]
    if m.ndim != 2:
        return np.zeros((1, 1, 3), dtype=np.float32)
    h, w = int(m.shape[0]), int(m.shape[1])
    out = np.zeros((h, w, 3), dtype=np.float32)
    out[:, :] = np.array(background, dtype=np.float32)
    if bool(np.any(m)):
        out[m] = np.array(flood_color, dtype=np.float32)
        if draw_border:
            border = dilate_binary_mask(m, radius=1) & (~m)
            out[border] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def show_result_images(
    x_img: np.ndarray,
    pred_prob: np.ndarray,
    pred_mask: np.ndarray,
    zone_mask: np.ndarray | None = None,
    sensor: str | None = None,
    detection_label: int | None = None,
    prediction_label: int | None = None,
    prediction_status: str | None = None,
    seg_threshold: float = DEFAULT_THRESHOLD,
) -> None:
    overview, overview_caption = build_input_overview(x_img, sensor=sensor)
    prob_view = normalize_prob_for_display(pred_prob)
    current_mask = (pred_mask > 0).astype(np.uint8)
    zone_mask_arr = (
        np.asarray(zone_mask).astype(np.uint8) if zone_mask is not None else current_mask.copy()
    )
    # Detect now: white mask. Prediction zones: red mask.
    mask_color = (1.0, 1.0, 1.0)
    mask_draw_border = False
    mask_caption = "Detected Flood Mask (Current Scene)"
    zone_title = "Detected zones"
    mask_note: str | None = None
    if (
        zone_mask is None
        and detection_label == 0
        and str(prediction_status or "active") == "active"
    ):
        zone_thr = float(np.clip(float(seg_threshold) - 0.15, 0.20, 0.45))
        zone_mask_arr = (pred_prob >= zone_thr).astype(np.uint8)
        mask_color = (1.0, 0.30, 0.05)
        mask_draw_border = True
        zone_title = "Predicted zones"
        if int(np.sum(zone_mask_arr)) == 0:
            q90 = float(np.quantile(pred_prob, 0.90))
            zone_mask_arr = (pred_prob >= q90).astype(np.uint8)
            mask_note = (
                f"No pixels above zone threshold {zone_thr:.2f}; "
                "showing top 10% probability areas as weak prediction hints."
            )
        else:
            mask_note = f"Prediction zones extracted from probability map (threshold={zone_thr:.2f})."
        if prediction_label == 1:
            mask_caption = "Predicted Flood Mask (Expected Zones)"
        else:
            mask_caption = "Prediction Mask (Low Confidence Zones)"
            # Remove tiny noisy blobs for a cleaner low-confidence preview.
            zone_bool = zone_mask_arr > 0
            min_blob = max(12, int(zone_bool.size * 0.0015))
            zone_bool = keep_largest_components(
                zone_bool, min_pixels=min_blob, max_regions=2
            )
            zone_mask_arr = zone_bool.astype(np.uint8)
            mask_color = (1.0, 0.18, 0.05)
            mask_draw_border = True
            if mask_note is None:
                mask_note = (
                    "Display cleaned: tiny scattered points are removed to show "
                    "main predicted regions only."
                )
    elif zone_mask is not None and detection_label == 0:
        mask_color = (1.0, 0.18, 0.05)
        mask_draw_border = True
        zone_title = "Predicted zones"
        mask_caption = (
            "Predicted Flood Mask (Expected Zones)"
            if int(prediction_label or 0) == 1
            else "Prediction Mask (Low Confidence Zones)"
        )
    zone_pct = float(np.mean(zone_mask_arr > 0) * 100.0)
    zone_mask_display = zone_mask_arr > 0
    tiny_zone_visualized = False
    if 0.0 < zone_pct < 1.0:
        # Display-only expansion for tiny zones; does not change model output.
        zone_mask_display = dilate_binary_mask(zone_mask_display, radius=2)
        tiny_zone_visualized = True
    min_zone_pixels = max(6, int(zone_mask_display.size * 0.0008))
    zone_components = summarize_zone_components(
        zone_mask_arr > 0, min_pixels=min_zone_pixels, max_regions=3
    )
    overview_view = downsample_for_display(overview)
    prob_view_ds = downsample_for_display(prob_view)
    mask_view = build_mask_canvas(
        zone_mask_display,
        flood_color=mask_color,
        draw_border=mask_draw_border,
    )
    mask_view_ds = downsample_for_display(mask_view)
    col1, col2, col3 = st.columns(3)
    col1.image(overview_view, caption=overview_caption, width="stretch", clamp=True)
    col2.image(
        prob_view_ds,
        caption="Flood Probability Map (contrast-enhanced)",
        width="stretch",
        clamp=True,
    )
    col2.caption(
        f"Raw prob stats: min={float(np.min(pred_prob)):.4f}, "
        f"mean={float(np.mean(pred_prob)):.4f}, max={float(np.max(pred_prob)):.4f}"
    )
    col3.image(mask_view_ds, caption=mask_caption, width="stretch", clamp=True)
    col3.caption(f"Highlighted zone coverage: {zone_pct:.2f}% of scene pixels.")
    if zone_components:
        comp_lines = []
        for idx, comp in enumerate(zone_components, start=1):
            comp_lines.append(
                f"{zone_title} {idx}: center=({int(comp['x_center'])}, {int(comp['y_center'])}), "
                f"box x[{int(comp['x_min'])}:{int(comp['x_max'])}], "
                f"y[{int(comp['y_min'])}:{int(comp['y_max'])}], area={float(comp['scene_pct']):.2f}%"
            )
        col3.caption(" | ".join(comp_lines))
    if tiny_zone_visualized:
        col3.caption(
            "Tiny predicted area: visual highlight was thickened for clarity only."
        )
    if mask_note:
        col3.caption(mask_note)
    elif int(np.sum(zone_mask_arr > 0)) == 0:
        col3.caption("No flood pixels at current threshold.")


def parse_manual_weather_values() -> dict[str, float | str]:
    values: dict[str, float | str] = {}
    cols = st.columns(2)
    for idx, feat in enumerate(project.WEATHER_FEATURE_NAMES):
        values[feat] = cols[idx % 2].text_input(feat, value="", key=f"manual_{feat}")
    return values


def parse_weather_from_csv(upload) -> dict[str, float | str]:
    if upload is None:
        return {}
    df = pd.read_csv(upload)
    if df.empty:
        raise ValueError("Uploaded weather CSV is empty.")
    row = df.iloc[0].to_dict()
    return {k: row.get(k, "") for k in project.WEATHER_FEATURE_NAMES}


def filename_match_keys(raw_name: str | None) -> tuple[str, ...]:
    return project.filename_match_keys(raw_name)


@lru_cache(maxsize=8)
def _cached_weather_lookup_from_csv(
    csv_path_text: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    df, issues = load_weather_aggregate(csv_path_text)
    lookup: dict[str, list[dict[str, Any]]] = {}
    if not df.empty:
        for row in df.itertuples(index=False):
            row_name = str(getattr(row, "filename", "")).strip()
            if not row_name:
                continue
            record = {name: getattr(row, name, "") for name in project.WEATHER_FEATURE_NAMES}
            record["filename"] = row_name
            for key in filename_match_keys(row_name):
                lookup.setdefault(key, []).append(record)
    return lookup, issues


def resolve_weather_record_from_lookup(
    lookup: dict[str, list[dict[str, Any]]], filename: str
) -> tuple[dict[str, Any] | None, str | None]:
    query_keys = filename_match_keys(filename)
    if not query_keys:
        return None, "no_filename_selected"
    matched: dict[str, dict[str, Any]] = {}
    for key in query_keys:
        for record in lookup.get(key, []):
            row_name = str(record.get("filename", "")).strip()
            if row_name:
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


def lookup_weather_for_filename(
    csv_path: str, filename: str
) -> tuple[dict[str, float | str], str | None]:
    if not filename:
        return {}, "no_filename_selected"
    lookup, issues = _cached_weather_lookup_from_csv(csv_path)
    if issues:
        return {}, "weather_csv_invalid"
    if not lookup:
        return {}, "weather_csv_empty"
    record, match_status = resolve_weather_record_from_lookup(lookup, filename)
    if record is None:
        return {}, match_status or "weather_csv_missing_filename"
    return {k: record.get(k, "") for k in project.WEATHER_FEATURE_NAMES}, None


def fill_missing_weather_values(target: dict[str, Any], source: dict[str, Any]) -> int:
    filled = 0
    for name in project.WEATHER_FEATURE_NAMES:
        if target.get(name) in (None, ""):
            val = source.get(name)
            if val not in (None, ""):
                target[name] = val
                filled += 1
    return filled


def inspect_uploaded_geospatial_metadata(raw: bytes) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ok",
        "path": None,
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
        with tifffile.TiffFile(BytesIO(raw)) as tf:
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
            epsg = geo.get("ProjectedCSTypeGeoKey") or geo.get("GeographicTypeGeoKey")
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
            parsed_dt = project._parse_utc_timestamp(raw_dt)
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
    if not payload["has_model_tiepoint"]:
        payload["warnings"].append("missing_model_tiepoint")
    center_lat, center_lon, _ = project._extract_lat_lon_from_geo_meta(payload)
    if center_lat is not None and center_lon is not None:
        payload["center_lat"] = float(center_lat)
        payload["center_lon"] = float(center_lon)
    return payload


def resolve_artifact_dir(
    model_s1_path: str,
    model_s2_path: str,
    risk_with_weather_path: str,
    risk_no_weather_path: str,
) -> Path:
    # Reuse the same runtime artifact discovery used by the CLI/web paths first, then
    # fall back to any explicit user-supplied model folder that still looks coherent.
    def has_pipeline_artifacts(folder: Path) -> bool:
        try:
            s1 = project.get_pipeline_model_path("S1", folder)
            s2 = project.get_pipeline_model_path("S2", folder)
            risk_no_weather = (folder / project.RISK_NO_WEATHER_PIPELINE_NAME).resolve()
            return bool(s1.exists() or s2.exists() or risk_no_weather.exists())
        except Exception:
            return False

    preferred, _ = project.resolve_prediction_artifact_dir()
    if has_pipeline_artifacts(preferred):
        return preferred

    candidate_paths = list(
        dict.fromkeys(
            [
                risk_with_weather_path,
                risk_no_weather_path,
                model_s1_path,
                model_s2_path,
            ]
        )
    )

    for candidate in candidate_paths:
        p = Path(candidate)
        if p.exists():
            parent = p.resolve().parent
            if has_pipeline_artifacts(parent):
                return parent

    for candidate in candidate_paths:
        p = Path(candidate)
        if p.exists():
            return p.resolve().parent
    return preferred


def export_prediction_package(
    save_dir: Path,
    stem: str,
    payload: dict[str, Any],
    pred_mask: np.ndarray,
    pred_prob: np.ndarray,
    zone_mask: np.ndarray | None = None,
) -> dict[str, str]:
    # Persist the core outputs of one prediction in a timestamped folder so the UI
    # can offer download links and the user can inspect artifacts later.
    save_dir.mkdir(parents=True, exist_ok=True)
    folder = save_dir / f"{stem}_{strftime('%Y%m%d_%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=True)
    mask_path = folder / f"{stem}_pred_mask.tif"
    prob_path = folder / f"{stem}_pred_prob.npy"
    json_path = folder / "prediction.json"
    tifffile.imwrite(mask_path, pred_mask.astype(np.uint8))
    np.save(prob_path, pred_prob.astype(np.float32))
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    export_paths = {"mask": str(mask_path), "prob": str(prob_path), "json": str(json_path)}
    if zone_mask is not None:
        try:
            geo_exports = project.export_prediction_geo_artifacts(
                output_dir=folder,
                stem=stem,
                zone_mask=np.asarray(zone_mask).astype(np.uint8),
                pred_prob=pred_prob,
                geo_meta=payload.get("geospatial_checks"),
                zone_meta=payload.get("prediction_zone"),
            )
            export_paths.update(
                {
                    "geojson": str(geo_exports.get("prediction_zone_geojson", "")),
                    "zone_geotiff": str(
                        geo_exports.get("prediction_zone_mask_geotiff", "")
                    ),
                    "prob_geotiff": str(
                        geo_exports.get("prediction_probability_geotiff", "")
                    ),
                }
            )
        except Exception:
            pass
    return export_paths


def build_geotiff_bytes(array: np.ndarray, geo_meta: dict[str, Any] | None) -> bytes:
    geo = geo_meta if isinstance(geo_meta, dict) else {}
    buf = io.BytesIO()
    tifffile.imwrite(
        buf,
        np.asarray(array),
        extratags=project._build_geotiff_extratags(geo),
    )
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def build_download_package(
    stem: str,
    payload: dict[str, Any],
    pred_mask: np.ndarray,
    pred_prob: np.ndarray,
    zone_mask: np.ndarray | None = None,
) -> dict[str, tuple[str, bytes, str]]:
    # Download artifacts are deterministic for one prediction, so caching avoids
    # rebuilding TIFF/NumPy/ZIP bytes on every Streamlit rerun.
    pred_json = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    mask_buf = io.BytesIO()
    tifffile.imwrite(mask_buf, pred_mask.astype(np.uint8))
    mask_bytes = mask_buf.getvalue()
    prob_buf = io.BytesIO()
    np.save(prob_buf, pred_prob.astype(np.float32))
    prob_bytes = prob_buf.getvalue()
    geo_meta = payload.get("geospatial_checks") if isinstance(payload, dict) else {}
    zone_geojson_bytes: bytes | None = None
    zone_geotiff_bytes: bytes | None = None
    prob_geotiff_bytes: bytes | None = None
    if zone_mask is not None:
        try:
            geo_export = payload.get("geo_export") if isinstance(payload, dict) else {}
            zone_geojson_data = None
            if isinstance(geo_export, dict):
                zone_geojson_data = geo_export.get("prediction_zone_geojson_data")
            if not isinstance(zone_geojson_data, dict):
                zone_geojson_data = project.build_prediction_zone_geojson(
                    zone_mask=np.asarray(zone_mask).astype(np.uint8),
                    zone_meta=(
                        payload.get("prediction_zone")
                        if isinstance(payload.get("prediction_zone"), dict)
                        else {}
                    ),
                    geo_meta=(geo_meta if isinstance(geo_meta, dict) else {}),
                )
            zone_geojson_bytes = json.dumps(
                zone_geojson_data, indent=2, ensure_ascii=False
            ).encode("utf-8")
            zone_geotiff_bytes = build_geotiff_bytes(
                np.asarray(zone_mask).astype(np.uint8),
                geo_meta if isinstance(geo_meta, dict) else {},
            )
            prob_geotiff_bytes = build_geotiff_bytes(
                np.asarray(pred_prob).astype(np.float32),
                geo_meta if isinstance(geo_meta, dict) else {},
            )
        except Exception:
            zone_geojson_bytes = None
            zone_geotiff_bytes = None
            prob_geotiff_bytes = None
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", pred_json)
        zf.writestr(f"{stem}_pred_mask.tif", mask_bytes)
        zf.writestr(f"{stem}_pred_prob.npy", prob_bytes)
        if zone_geojson_bytes is not None:
            zf.writestr(f"{stem}_prediction_zone.geojson", zone_geojson_bytes)
        if zone_geotiff_bytes is not None:
            zf.writestr(f"{stem}_prediction_zone_mask_georef.tif", zone_geotiff_bytes)
        if prob_geotiff_bytes is not None:
            zf.writestr(f"{stem}_prediction_probability_georef.tif", prob_geotiff_bytes)
    zip_bytes = zip_buf.getvalue()
    out = {
        "json": (f"{stem}_prediction.json", pred_json, "application/json"),
        "mask": (f"{stem}_pred_mask.tif", mask_bytes, "image/tiff"),
        "prob": (f"{stem}_pred_prob.npy", prob_bytes, "application/octet-stream"),
        "zip": (f"{stem}_prediction_package.zip", zip_bytes, "application/zip"),
    }
    if zone_geojson_bytes is not None:
        out["geojson"] = (
            f"{stem}_prediction_zone.geojson",
            zone_geojson_bytes,
            "application/geo+json",
        )
    if zone_geotiff_bytes is not None:
        out["zone_geotiff"] = (
            f"{stem}_prediction_zone_mask_georef.tif",
            zone_geotiff_bytes,
            "image/tiff",
        )
    return out


def risk_level_from_percent(
    risk_pct: float | None,
    *,
    risk_label: int | None = None,
    risk_threshold: float | None = None,
) -> tuple[str, str]:
    if risk_pct is None:
        return "Unavailable", "gray"
    try:
        threshold_pct = (
            float(risk_threshold if risk_threshold is not None else 0.5) * 100.0
        )
    except Exception:
        threshold_pct = 50.0
    high_cut = max(70.0, threshold_pct + 25.0)
    # If model already produced binary decision, keep UI consistent with it.
    if risk_label in (0, 1):
        if int(risk_label) == 0:
            return "Low", "green"
        if risk_pct >= high_cut:
            return "High", "red"
        return "Medium", "orange"
    # Fallback if risk_label is unavailable.
    if risk_pct < threshold_pct:
        return "Low", "green"
    if risk_pct >= high_cut:
        return "High", "red"
    return "Medium", "orange"


def risk_path_label(risk_model_used: str | None) -> str:
    mapping = {
        "with_weather": "With Weather",
        "no_weather_fallback": "No Weather (Fallback)",
        None: "Unavailable",
    }
    return mapping.get(risk_model_used, str(risk_model_used))


def pipeline_label(value: str | None, *, default: str = "Pipeline") -> str:
    return project.pipeline_display_name(value, default=default)


def temporal_status_label(status: str | None) -> str:
    mapping = {
        "ok": "Temporal forecast available.",
        "temporal_csv_missing_filename": "No matching weather time-series row for this image filename.",
        "temporal_csv_ambiguous_filename": "More than one weather row matched this filename.",
        "temporal_csv_invalid": "Weather CSV format is invalid.",
        "temporal_csv_empty": "Weather CSV has no usable rows.",
        "temporal_anchor_missing_coordinates": "Temporal forecast needs geospatial coordinates (lat/lon) for this image.",
        "temporal_era5_grid_not_available": "No nearby ERA5 weather grid was found for this image location.",
        "temporal_era5_grid_slice_empty": "ERA5 grid slice is empty for the selected location.",
        "temporal_era5_sequence_empty": "Temporal sequence is empty after location/time filtering.",
        "temporal_model_missing": "Temporal model is not loaded.",
        "missing_filename": "Image filename is missing, temporal lookup cannot run.",
        "feature_schema_mismatch": "Temporal model/features mismatch.",
        "prediction_failed": "Temporal model inference failed.",
        "not_supported_for_sensor": "Temporal forecast is not supported for this sensor.",
        "skipped_due_to_detection": "Temporal forecast skipped because flood is already detected in current scene.",
        "unavailable": "Temporal forecast is unavailable.",
    }
    if status is None:
        return "Temporal forecast is unavailable."
    return mapping.get(str(status), f"Temporal forecast unavailable ({status}).")


def eta_text_label(code: str | None) -> str:
    mapping = {
        "not_applicable_flood_already_detected": "ETA is not applicable because flood is already detected.",
        "prediction_unavailable": "ETA is unavailable because forecast is unavailable.",
        "time_window_unavailable": "ETA window is unavailable.",
        "time_window_unavailable_temporal_data_required": "ETA requires matching temporal weather data.",
        "no_elevated_short_term_signal_next_30_days": "No elevated short-term flood signal in the next 30 days.",
        "no_elevated_signal_next_5_years": "No elevated flood signal in the next 5 years.",
    }
    if code is None:
        return "ETA unavailable."
    text = str(code)
    m_hours = re.match(r"^possible_flood_window_(\d+)_to_(\d+)_hours$", text)
    if m_hours:
        h_start = int(m_hours.group(1))
        h_end = int(m_hours.group(2))
        if h_end >= 24:
            d1 = h_start / 24.0
            d2 = h_end / 24.0
            return f"Possible flood window in about {d1:.1f} to {d2:.1f} days ({h_start}-{h_end} hours)."
        return f"Possible flood window in {h_start}-{h_end} hours."
    m = re.match(r"^possible_flood_window_(\d+)_to_(\d+)_days$", text)
    if m:
        d_start = int(m.group(1))
        d_end = int(m.group(2))
        if d_end >= 365:
            y1 = d_start / 365.0
            y2 = d_end / 365.0
            return f"Possible flood window in about {y1:.1f} to {y2:.1f} years ({d_start}-{d_end} days)."
        if d_end >= 30:
            m1 = d_start / 30.0
            m2 = d_end / 30.0
            return f"Possible flood window in about {m1:.1f} to {m2:.1f} months ({d_start}-{d_end} days)."
        if d_end >= 14:
            w1 = d_start / 7.0
            w2 = d_end / 7.0
            return f"Possible flood window in about {w1:.1f} to {w2:.1f} weeks ({d_start}-{d_end} days)."
        return f"Possible flood window in {d_start}-{d_end} days."
    return mapping.get(text, text.replace("_", " "))


def eta_window_relative_label(
    *,
    days_min: Any,
    days_max: Any,
    hours_min: Any | None = None,
    hours_max: Any | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
) -> str | None:
    h_min: float | None = None
    h_max: float | None = None
    try:
        if hours_min is not None:
            h_min = float(hours_min)
        if hours_max is not None:
            h_max = float(hours_max)
    except Exception:
        h_min = None
        h_max = None
    if h_min is not None and h_max is not None:
        if h_max < 24.0:
            return f"{h_min:.0f}-{h_max:.0f} hours"
        return f"about {h_min / 24.0:.1f}-{h_max / 24.0:.1f} days ({h_min:.0f}-{h_max:.0f} hours)"

    d_min: float | None = None
    d_max: float | None = None
    try:
        if days_min is not None:
            d_min = float(days_min)
        if days_max is not None:
            d_max = float(days_max)
    except Exception:
        d_min = None
        d_max = None

    if d_min is None or d_max is None:
        if start_utc and end_utc:
            try:
                start_dt = datetime.fromisoformat(str(start_utc).replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(str(end_utc).replace("Z", "+00:00"))
                delta_hours = max(
                    0.0, float((end_dt - start_dt).total_seconds()) / 3600.0
                )
                if delta_hours < 48.0:
                    return f"0-{delta_hours:.0f} hours"
                delta_days = max(0.0, delta_hours / 24.0)
                d_min = 0.0
                d_max = delta_days
            except Exception:
                return None
        else:
            return None

    if d_max < 2.0:
        h1 = d_min * 24.0
        h2 = d_max * 24.0
        return f"{h1:.0f}-{h2:.0f} hours"
    if d_max >= 365.0:
        return f"about {d_min / 365.0:.1f}-{d_max / 365.0:.1f} years"
    if d_max >= 30.0:
        return f"about {d_min / 30.0:.1f}-{d_max / 30.0:.1f} months"
    if d_max >= 14.0:
        return f"about {d_min / 7.0:.1f}-{d_max / 7.0:.1f} weeks"
    return f"{d_min:.0f}-{d_max:.0f} days"


def confidence_from_score(
    *,
    score: Any,
    threshold: Any = 0.5,
    predicted_label: Any = None,
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
    if pct >= 85.0:
        level = "High"
    elif pct >= 70.0:
        level = "Medium"
    else:
        level = "Low"
    return pct, level


def format_utc_short(value: Any) -> str:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(value)


def feedback_reason_label(reason: str | None) -> str:
    mapping = {
        "already_in_base_dataset_path": "Image already exists in the base training dataset (same path).",
        "already_in_base_dataset_filename": "Image filename already exists in the base training dataset.",
        "duplicate_in_feedback_pool": "Image is already queued in feedback pool.",
        "new_sample_saved_for_feedback": "New image saved for feedback labeling.",
        "feedback_collection_disabled_by_flag": "Feedback collection is disabled by current run setting.",
    }
    if reason is None:
        return "No feedback status."
    return mapping.get(str(reason), str(reason))


def is_missing_weather_fallback_warning(payload: dict[str, Any]) -> bool:
    return (
        payload.get("risk_model_used") == "no_weather_fallback"
        and payload.get("risk_warning") == "incomplete_weather_features_fallback_used"
    )


def parse_data_roots(text: str) -> tuple[str, ...]:
    return tuple([x.strip() for x in text.splitlines() if x.strip()])


def collect_roots_with_sensor(data_roots: tuple[str, ...]) -> list[tuple[Path, str]]:
    roots_with_sensor: list[tuple[Path, str]] = []
    for root in data_roots:
        sensor = project.infer_sensor_from_root(Path(root))
        if sensor:
            roots_with_sensor.append((Path(root).resolve(), sensor))
    return roots_with_sensor


def sensor_with_auto_correction_for_dataset(
    selected_path: Path,
    sensor_override: str,
    roots_with_sensor: list[tuple[Path, str]],
) -> str | None:
    if sensor_override == "Auto":
        return project.detect_sensor_for_image(selected_path, roots_with_sensor)
    sensor = sensor_override
    auto_sensor = project.detect_sensor_for_image(selected_path, roots_with_sensor)
    if auto_sensor in project.SENSOR_CHANNELS and auto_sensor != sensor:
        st.warning(
            f"Sensor corrected from {sensor} to {auto_sensor} based on image channels."
        )
        return auto_sensor
    return sensor


def sensor_with_auto_correction_for_upload(
    raw: bytes, sensor_override: str
) -> str | None:
    if sensor_override == "Auto":
        return detect_sensor_from_raw(raw)
    sensor = sensor_override
    auto_sensor = detect_sensor_from_raw(raw)
    if auto_sensor in project.SENSOR_CHANNELS and auto_sensor != sensor:
        st.warning(
            f"Sensor corrected from {sensor} to {auto_sensor} based on image channels."
        )
        return auto_sensor
    return sensor


def build_payload(
    sensor: str,
    image_name: str,
    pred_feats: dict[str, float],
    risk: dict[str, Any],
    weather_values: dict[str, Any] | None = None,
    temporal_risk: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eta_info = project.build_prediction_eta(
        detection_label=risk.get("detection_label"),
        prediction_label=risk.get("prediction_label"),
        temporal_payload=temporal_risk if isinstance(temporal_risk, dict) else None,
    )
    decision_support = project.provide_solutions(
        {
            **pred_feats,
            "risk_score": risk.get("risk_score"),
            "risk_model_used": risk.get("risk_model_used"),
            "risk_warning": risk.get("risk_warning"),
            **(weather_values or {}),
        }
    )
    payload = {
        "sensor": sensor,
        "filename": image_name,
        "pred_flood_ratio": pred_feats["pred_flood_ratio"],
        "pred_flood_ratio_percent": project.to_percent(pred_feats["pred_flood_ratio"]),
        "pred_prob_mean": pred_feats["pred_prob_mean"],
        "pred_prob_mean_percent": project.to_percent(pred_feats["pred_prob_mean"]),
        "pred_prob_p90": pred_feats["pred_prob_p90"],
        "pred_prob_p90_percent": project.to_percent(pred_feats["pred_prob_p90"]),
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
        "risk_model_used": risk["risk_model_used"],
        "weather_features_used": risk["weather_features_used"],
        "missing_weather_features": risk.get("missing_weather_features", []),
        "decision_support": decision_support,
        "prediction_eta_text": eta_info.get("prediction_eta_text"),
        "prediction_eta_start_utc": eta_info.get("prediction_eta_start_utc"),
        "prediction_eta_end_utc": eta_info.get("prediction_eta_end_utc"),
        "prediction_eta_source": eta_info.get("prediction_eta_source"),
        "prediction_eta_note": eta_info.get("prediction_eta_note"),
        "prediction_eta_days_min": eta_info.get("prediction_eta_days_min"),
        "prediction_eta_days_max": eta_info.get("prediction_eta_days_max"),
        "prediction_eta_hours_min": eta_info.get("prediction_eta_hours_min"),
        "prediction_eta_hours_max": eta_info.get("prediction_eta_hours_max"),
        "prediction_eta_horizon": eta_info.get("prediction_eta_horizon"),
        "prediction_eta_confidence_percent": eta_info.get(
            "prediction_eta_confidence_percent"
        ),
        "prediction_eta_confidence_level": eta_info.get(
            "prediction_eta_confidence_level"
        ),
    }
    if "risk_warning" in risk:
        payload["risk_warning"] = risk["risk_warning"]
    if isinstance(temporal_risk, dict):
        payload.update(
            {
                "temporal_status": temporal_risk.get("temporal_status"),
                "temporal_risk_score": temporal_risk.get("temporal_risk_score"),
                "temporal_risk_score_percent": temporal_risk.get(
                    "temporal_risk_score_percent"
                ),
                "temporal_risk_label": temporal_risk.get("temporal_risk_label"),
                "temporal_risk_text": temporal_risk.get("temporal_risk_text"),
                "temporal_model_used": temporal_risk.get("temporal_model_used"),
                "temporal_weather_match_status": temporal_risk.get(
                    "temporal_weather_match_status"
                ),
                "temporal_horizon": temporal_risk.get("temporal_horizon"),
                "temporal_lookup_mode": temporal_risk.get("temporal_lookup_mode"),
                "temporal_risk_threshold": temporal_risk.get("temporal_risk_threshold"),
                "temporal_feature_snapshot": temporal_risk.get(
                    "temporal_feature_snapshot"
                ),
                "temporal_anchor_source": temporal_risk.get("temporal_anchor_source"),
                "temporal_anchor_time_utc": temporal_risk.get(
                    "temporal_anchor_time_utc"
                ),
                "temporal_anchor_lat": temporal_risk.get("temporal_anchor_lat"),
                "temporal_anchor_lon": temporal_risk.get("temporal_anchor_lon"),
            }
        )
    return payload


def render_system_status(
    seg_ready: dict[str, bool],
    risk_with_weather_ready: bool,
    risk_no_weather_ready: bool,
    risk_temporal_ready: bool,
) -> None:
    st.subheader("Runtime Status")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("S1 Segmentation", "Ready" if bool(seg_ready.get("S1")) else "Missing")
    c2.metric("S2 Segmentation", "Ready" if bool(seg_ready.get("S2")) else "Missing")
    c3.metric("Risk (Weather)", "Ready" if risk_with_weather_ready else "Missing")
    c4.metric("Risk (Fallback)", "Ready" if risk_no_weather_ready else "Missing")
    c5.metric("Temporal", "Ready" if risk_temporal_ready else "Missing")


def render_dashboard_submission_brief() -> None:
    st.caption(
        "Primary technical dashboard for Flood Intelligence Platform: run scene-level "
        "flood detection, risk scoring, temporal forecasting, and geospatial export "
        "from one place."
    )
    st.caption(
        "Recommended submission demo path: choose or upload a TIFF scene, run the "
        "pipeline, then walk through Decision Summary, preview panels, and Download "
        "Artifacts."
    )
    with st.expander("Product Story and Demo Path", expanded=False):
        st.markdown("**Official product name:** Flood Intelligence Platform")
        st.markdown(
            "**Primary interface:** This dashboard is the fastest technical demo for "
            "the full pipeline, including prediction, interpretation, and export."
        )
        st.markdown(
            "**Other interfaces:** HYDROVISION is an alternate exploratory UI built on "
            "the same core pipeline. They are optional surfaces, not the primary demo path."
        )
        st.markdown("**How to demo this dashboard:**")
        st.markdown("1. Choose an image source: upload a TIFF or select an internal dataset scene.")
        st.markdown("2. Keep `Sensor = Auto` unless you need to force S1 or S2 for testing.")
        st.markdown(
            "3. Optionally enable weather inputs if you want to show the with-weather risk path."
        )
        st.markdown("4. Click `Run Detection & Forecast`.")
        st.markdown(
            "5. Explain the result in this order: Decision Summary -> Operational Summary "
            "-> preview panels -> Download Artifacts / Export Analysis Package."
        )
        st.markdown(
            "6. If flood is already detected, explain that temporal prediction is "
            "intentionally suppressed because the scene already contains a current event."
        )


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def to_percent_text(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except Exception:
        return "N/A"


def pick_metrics_block(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    block = payload.get("cv_mean")
    if isinstance(block, dict):
        return block
    return payload


def load_quality_metrics_for_backend(artifact_dir: Path) -> tuple[
    str,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    label = pipeline_label(project.PIPELINE_V3_BACKEND_ID, default="Pipeline V3")
    global_metrics = read_json_if_exists(artifact_dir / "unet_val_metrics_global.json")
    s1_metrics = read_json_if_exists(artifact_dir / "unet_val_metrics_s1.json")
    s2_metrics = read_json_if_exists(artifact_dir / "unet_val_metrics_s2.json")
    risk_fallback = read_json_if_exists(
        artifact_dir / "risk_no_weather_cv_metrics_unet.json"
    )
    risk_weather = read_json_if_exists(
        artifact_dir / "risk_with_weather_cv_metrics_unet.json"
    )
    return label, global_metrics, s1_metrics, s2_metrics, risk_fallback, risk_weather


def render_model_quality(artifact_dir: Path) -> None:
    (
        metrics_label,
        global_metrics,
        s1_metrics,
        s2_metrics,
        risk_fallback,
        risk_weather,
    ) = load_quality_metrics_for_backend(artifact_dir)
    temporal_metrics = read_json_if_exists(
        artifact_dir / project.RISK_TEMPORAL_METRICS_PIPELINE_NAME
    )
    fallback_block = pick_metrics_block(risk_fallback)
    weather_block = pick_metrics_block(risk_weather)
    temporal_block = pick_metrics_block(temporal_metrics)
    has_any = any(
        x is not None
        for x in [
            global_metrics,
            s1_metrics,
            s2_metrics,
            risk_fallback,
            risk_weather,
            temporal_metrics,
        ]
    )
    if not has_any:
        st.info("Model quality metrics are not available yet. Run training first.")
        return
    global_block = global_metrics or {}
    st.subheader("Model Quality Snapshot")
    st.caption(f"Metrics pipeline in view: {metrics_label}")
    row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
    row1_c1.metric("Seg F1 (Global)", to_percent_text(global_block.get("f1")))
    row1_c2.metric("Seg IoU (Global)", to_percent_text(global_block.get("iou")))
    row1_c3.metric("Seg Recall (Global)", to_percent_text(global_block.get("recall")))
    row1_c4.metric(
        "Seg Accuracy (Secondary)", to_percent_text(global_block.get("accuracy"))
    )
    row2_c1, row2_c2, row2_c3 = st.columns(3)
    row2_c1.metric("Risk ROC AUC (Weather)", to_percent_text(weather_block.get("roc_auc")))
    row2_c2.metric(
        "Risk ROC AUC (Fallback)", to_percent_text(fallback_block.get("roc_auc"))
    )
    row2_c3.metric("Temporal ROC AUC", to_percent_text(temporal_block.get("roc_auc")))
    st.caption(
        "For flood segmentation, F1 / IoU / Recall are the headline metrics because "
        "flood pixels occupy much less area than background. Accuracy is still shown, "
        "but it should be read as a secondary metric."
    )
    with st.expander("How to Read These Metrics", expanded=False):
        st.markdown(
            "**F1:** best single summary when you want balance between false positives "
            "and false negatives on flood pixels."
        )
        st.markdown(
            "**IoU:** overlap quality between predicted flooded area and ground truth flooded area."
        )
        st.markdown(
            "**Recall:** how much real flooded area the pipeline successfully captures."
        )
        st.markdown(
            "**ROC AUC:** ranking quality for the risk/forecast models across decision thresholds."
        )
        st.markdown(
            "**Accuracy:** still useful, but easier to inflate in flood segmentation because "
            "background pixels dominate many scenes."
        )
    temporal_coverage_ratio = (
        float(temporal_metrics.get("temporal_coverage_ratio"))
        if isinstance(temporal_metrics, dict)
        and temporal_metrics.get("temporal_coverage_ratio") is not None
        else None
    )
    temporal_rows_after_merge = (
        int(temporal_metrics.get("temporal_rows_after_merge"))
        if isinstance(temporal_metrics, dict)
        and temporal_metrics.get("temporal_rows_after_merge") is not None
        else None
    )
    temporal_base_rows = (
        int(temporal_metrics.get("base_rows_count"))
        if isinstance(temporal_metrics, dict)
        and temporal_metrics.get("base_rows_count") is not None
        else None
    )
    if (
        temporal_coverage_ratio is not None
        and temporal_rows_after_merge is not None
        and temporal_base_rows is not None
    ):
        st.caption(
            "Temporal training coverage: "
            f"{temporal_rows_after_merge}/{temporal_base_rows} rows "
            f"({temporal_coverage_ratio * 100.0:.2f}%)."
        )
        if temporal_coverage_ratio < 0.05:
            st.warning(
                "Temporal model coverage is very low. Temporal scores may be weaker than segmentation/risk metrics."
            )
    details_rows: list[dict[str, str]] = []
    if isinstance(s1_metrics, dict):
        details_rows.append(
            {
                "Model": "Segmentation S1",
                "Accuracy": to_percent_text(s1_metrics.get("accuracy")),
                "Recall": to_percent_text(s1_metrics.get("recall")),
                "F1": to_percent_text(s1_metrics.get("f1")),
                "IoU": to_percent_text(s1_metrics.get("iou")),
                "ROC AUC": "N/A",
            }
        )
    if isinstance(s2_metrics, dict):
        details_rows.append(
            {
                "Model": "Segmentation S2",
                "Accuracy": to_percent_text(s2_metrics.get("accuracy")),
                "Recall": to_percent_text(s2_metrics.get("recall")),
                "F1": to_percent_text(s2_metrics.get("f1")),
                "IoU": to_percent_text(s2_metrics.get("iou")),
                "ROC AUC": "N/A",
            }
        )
    if fallback_block:
        details_rows.append(
            {
                "Model": "Risk Fallback Model",
                "Accuracy": to_percent_text(fallback_block.get("accuracy")),
                "Recall": to_percent_text(fallback_block.get("recall")),
                "F1": to_percent_text(fallback_block.get("f1")),
                "IoU": "N/A",
                "ROC AUC": to_percent_text(fallback_block.get("roc_auc")),
            }
        )
    if weather_block:
        details_rows.append(
            {
                "Model": "Risk With Weather",
                "Accuracy": to_percent_text(weather_block.get("accuracy")),
                "Recall": to_percent_text(weather_block.get("recall")),
                "F1": to_percent_text(weather_block.get("f1")),
                "IoU": "N/A",
                "ROC AUC": to_percent_text(weather_block.get("roc_auc")),
            }
        )
    if temporal_block:
        details_rows.append(
            {
                "Model": "Temporal Risk",
                "Accuracy": to_percent_text(temporal_block.get("accuracy")),
                "Recall": to_percent_text(temporal_block.get("recall")),
                "F1": to_percent_text(temporal_block.get("f1")),
                "IoU": "N/A",
                "ROC AUC": to_percent_text(temporal_block.get("roc_auc")),
            }
        )
    with st.expander("Detailed Validation Metrics", expanded=False):
        if details_rows:
            st.dataframe(
                pd.DataFrame(details_rows)[
                    ["Model", "Accuracy", "Recall", "F1", "IoU", "ROC AUC"]
                ],
                width="stretch",
                hide_index=True,
            )
        st.caption(f"Metrics source: `{artifact_dir}`")
        st.caption(f"Metrics pipeline: {metrics_label}")
        st.caption(
            "These are validation/training metrics for the promoted runtime artifacts, "
            "not the current single-scene prediction."
        )
    experiment_rows = load_experiment_rows_cached(
        str(project.PROJECT_BASE_DIR), limit=10
    )
    with st.expander("Experiment Tracking", expanded=False):
        if experiment_rows:
            exp_df = pd.DataFrame(experiment_rows)
            rename_map = {
                "run_name": "Run",
                "updated_at_utc": "Updated UTC",
                "model_kind": "Model",
                "patch_size": "Patch",
                "stride": "Stride",
                "epochs": "Epochs",
                "threshold": "Threshold",
                "seg_f1": "Seg F1",
                "seg_iou": "Seg IoU",
                "risk_auc_fallback": "Risk AUC",
                "temporal_auc": "Temporal AUC",
            }
            display_cols = [x for x in rename_map.keys() if x in exp_df.columns]
            exp_df = exp_df[display_cols].rename(columns=rename_map).copy()
            for col in ["Seg F1", "Seg IoU", "Risk AUC", "Temporal AUC"]:
                if col in exp_df.columns:
                    exp_df[col] = exp_df[col].apply(to_percent_text)
            st.dataframe(exp_df, width="stretch", hide_index=True)
        else:
            st.caption("No run reports were discovered.")
    failure_report = load_failure_report_cached(str(artifact_dir))
    with st.expander("Failure Analysis", expanded=False):
        if failure_report.get("status") != "ok":
            st.caption("Failure analysis is not available yet.")
        else:
            st.caption(str(failure_report.get("comparison_note", "")))
            st.caption(
                f"Best sensor by mean F1: {failure_report.get('best_sensor_by_f1') or 'N/A'}"
            )
            for sensor in ("S1", "S2"):
                block = failure_report.get("sensors", {}).get(sensor, {})
                if not isinstance(block, dict) or block.get("status") != "ok":
                    st.caption(f"{sensor}: report unavailable.")
                    continue
                st.markdown(f"**{sensor}**")
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("Mean F1", to_percent_text(block.get("mean_f1")))
                fc2.metric("Mean IoU", to_percent_text(block.get("mean_iou")))
                fc3.metric(
                    "Pred/True Flood Ratio",
                    (
                        f"{block.get('mean_pred_ratio_percent', 'N/A')} / "
                        f"{block.get('mean_true_ratio_percent', 'N/A')}"
                    ),
                )
                st.caption(f"Issue pattern: {block.get('issue_summary', 'unknown')}")
                fp_examples = block.get("false_positive_examples") or []
                fn_examples = block.get("false_negative_examples") or []
                if fp_examples:
                    st.caption("False positives")
                    st.dataframe(
                        pd.DataFrame(fp_examples),
                        width="stretch",
                        hide_index=True,
                    )
                if fn_examples:
                    st.caption("False negatives")
                    st.dataframe(
                        pd.DataFrame(fn_examples),
                        width="stretch",
                        hide_index=True,
                    )


def app() -> None:
    inject_ui_styles()
    st.title("Flood Intelligence Platform")
    render_dashboard_submission_brief()
    data_roots = parse_data_roots(DEFAULT_DATA_ROOTS)
    model_s1_path = DEFAULT_MODEL_S1
    model_s2_path = DEFAULT_MODEL_S2
    risk_with_weather_path = DEFAULT_RISK_WITH_WEATHER
    risk_no_weather_path = DEFAULT_RISK_NO_WEATHER
    save_dir = Path(DEFAULT_EXPORT_DIR)
    threshold = DEFAULT_THRESHOLD
    risk_threshold_profile = DEFAULT_RISK_THRESHOLD_PROFILE
    drift_zscore_threshold = DEFAULT_DRIFT_ZSCORE_THRESHOLD
    backend_choice = DEFAULT_BACKEND
    with st.sidebar:
        st.header("Run Controls")
        with st.expander("Settings", expanded=False):
            model_s1_path = st.text_input(
                "S1 pipeline bundle", DEFAULT_MODEL_S1
            )
            model_s2_path = st.text_input(
                "S2 pipeline bundle", DEFAULT_MODEL_S2
            )
            risk_with_weather_path = st.text_input(
                "Risk model (with weather)", DEFAULT_RISK_WITH_WEATHER
            )
            risk_no_weather_path = st.text_input(
                "Risk model (fallback no-weather)", DEFAULT_RISK_NO_WEATHER
            )
            threshold = st.slider(
                "Segmentation threshold",
                min_value=0.05,
                max_value=0.95,
                value=DEFAULT_THRESHOLD,
                step=0.01,
            )
            risk_threshold_profile = st.selectbox(
                "Risk threshold profile",
                options=sorted(project.RISK_THRESHOLD_PROFILES.keys()),
                index=sorted(project.RISK_THRESHOLD_PROFILES.keys()).index(
                    DEFAULT_RISK_THRESHOLD_PROFILE
                ),
            )
            backend_options = list(project.BACKEND_CHOICES)
            backend_choice = st.selectbox(
                "Prediction pipeline",
                options=backend_options,
                index=backend_options.index(DEFAULT_BACKEND),
                format_func=lambda value: pipeline_label(value, default="Pipeline"),
            )
            save_dir = Path(st.text_input("Export directory", DEFAULT_EXPORT_DIR))
            st.caption(
                "With-weather runs when weather features are available (CSV/ERA5 anchor). "
                "No-weather fallback is global for S1+S2."
            )
        st.caption(f"Export folder: `{save_dir}`")
    artifact_dir = resolve_artifact_dir(
        model_s1_path, model_s2_path, risk_with_weather_path, risk_no_weather_path
    )
    pipeline_model_paths = {
        "S1": project.get_pipeline_model_path("S1", artifact_dir),
        "S2": project.get_pipeline_model_path("S2", artifact_dir),
    }
    risk_with_weather_pipeline_path = (
        artifact_dir / project.RISK_WITH_WEATHER_PIPELINE_NAME
    ).resolve()
    risk_no_weather_pipeline_path = (
        artifact_dir / project.RISK_NO_WEATHER_PIPELINE_NAME
    ).resolve()
    risk_temporal_pipeline_path = (artifact_dir / project.RISK_TEMPORAL_PIPELINE_NAME).resolve()
    pipeline_ready = {sensor: path.exists() for sensor, path in pipeline_model_paths.items()}
    risk_with_weather_ready = risk_with_weather_pipeline_path.exists()
    risk_no_weather_ready = risk_no_weather_pipeline_path.exists()
    risk_temporal_ready = risk_temporal_pipeline_path.exists()

    # Preload bundles right after page refresh so first prediction is instant.
    preload_key = "|".join(
        [
            str(artifact_dir.resolve()),
            str(file_mtime_token(pipeline_model_paths["S1"])),
            str(file_mtime_token(pipeline_model_paths["S2"])),
        ]
    )
    if st.session_state.get("pipeline_preload_key") != preload_key:
        with st.spinner("Loading pipeline bundles..."):
            st.session_state["pipeline_preload_status"] = preload_pipeline_bundles(
                pipeline_model_paths
            )
            st.session_state["pipeline_preload_key"] = preload_key

    active_backend_state = project.load_active_backend_config(artifact_dir)
    model_registry = project.load_model_registry(artifact_dir)
    input_profile = project.load_training_input_profile(artifact_dir)
    risk_threshold = project.resolve_risk_threshold(risk_threshold_profile, None)
    weather_csv_path = resolve_weather_csv_path()
    weather_csv_exists = weather_csv_path.exists()
    render_system_status(
        seg_ready=pipeline_ready,
        risk_with_weather_ready=risk_with_weather_ready,
        risk_no_weather_ready=risk_no_weather_ready,
        risk_temporal_ready=risk_temporal_ready,
    )
    st.caption(
        "Active pipeline: "
        f"{pipeline_label(active_backend_state.get('segmentation_backend'), default='Pipeline')} "
        f"(risk: {pipeline_label(active_backend_state.get('risk_backend'), default='Pipeline')})"
    )
    st.caption(
        "Production pipeline availability: "
        f"{pipeline_label(project.PIPELINE_V3_BACKEND_ID, default='Pipeline')}="
        f"{'ready' if any(pipeline_ready.values()) else 'missing'}"
    )
    render_model_quality(artifact_dir)
    if not any(pipeline_ready.values()):
        st.error(
            "No production segmentation pipeline bundle is loaded. Run: python project.py train-pipeline"
        )
        return
    st.subheader("Run a Scene")
    mode = st.radio(
        "Image source",
        ["Upload TIFF", "Choose from internal dataset"],
        index=0,
        horizontal=True,
    )
    sensor_override = st.radio("Sensor", ["Auto", "S1", "S2"], index=0, horizontal=True)
    selected_path: Path | None = None
    uploaded_file = None
    image_name = ""
    discovered: project.DiscoveryResult | None = None
    if mode == "Choose from internal dataset":
        discovered = discover_all_pairs(data_roots)
        options = sorted([str(p.image_path) for p in discovered.pairs])
        if not options:
            st.error("No discovered image pairs found. Check data roots.")
            return
        pick = st.selectbox("Select image path", options, index=0)
        selected_path = Path(pick)
        image_name = selected_path.name
    else:
        uploaded_file = st.file_uploader("Upload .tif image", type=["tif", "tiff"])
        if uploaded_file is not None:
            image_name = uploaded_file.name
    st.caption(
        "After a run, this page shows Decision Summary, preview panels, location/export "
        "artifacts, and the final download buttons in the lower sections."
    )
    st.subheader("Optional Weather Inputs")
    st.caption(
        f"Weather CSV in use: `{weather_csv_path}` ({'exists' if weather_csv_exists else 'missing'})"
    )
    use_extra_values = st.checkbox(
        "Include weather/extra values for improved risk scoring", value=False
    )
    weather_values_raw: dict[str, float | str] = {}
    if use_extra_values:
        weather_sources = ["Manual form", "Upload 1-row CSV"]
        if mode == "Choose from internal dataset":
            weather_sources.insert(0, "Auto from training CSV")
        weather_mode = st.radio("Extra values source", weather_sources, horizontal=True)
        if weather_mode == "Auto from training CSV":
            if weather_csv_exists:
                weather_values_raw, warn = lookup_weather_for_filename(
                    str(weather_csv_path), image_name
                )
                if warn is None:
                    st.success("Weather values loaded from CSV.")
                elif warn == "weather_csv_missing_filename":
                    st.warning(
                        f"No weather row found for selected filename in CSV: {weather_csv_path.name}"
                    )
                elif warn == "weather_csv_ambiguous_filename":
                    st.warning(
                        "More than one weather row matched this filename. Enter values manually."
                    )
                else:
                    st.warning("Weather CSV not ready or invalid.")
            else:
                st.error(f"Weather CSV path not found: {weather_csv_path}")
        elif weather_mode == "Manual form":
            weather_values_raw = parse_manual_weather_values()
        else:
            csv_weather_upload = st.file_uploader(
                "Upload one-row CSV with weather features", type=["csv"]
            )
            if csv_weather_upload is not None:
                try:
                    weather_values_raw = parse_weather_from_csv(csv_weather_upload)
                    st.success("Weather CSV parsed successfully.")
                except Exception as ex:
                    st.error(str(ex))
    run = st.button("Run Detection & Forecast", type="primary")
    if run:
        if mode == "Upload TIFF" and uploaded_file is None:
            st.warning("Please upload a TIFF image first.")
            return
        try:
            roots_with_sensor = collect_roots_with_sensor(data_roots)
            known_dataset_paths: set[Path] | None = None
            known_dataset_filenames: set[str] | None = None
            if mode == "Choose from internal dataset" and discovered is not None:
                known_dataset_paths = set(discovered.pair_by_image.keys())
                known_dataset_filenames = set(discovered.image_index.keys())
            geo_meta: dict[str, Any]
            source_image_path: Path | None = None
            source_image_bytes: bytes | None = None
            source_mode = "upload"
            if mode == "Choose from internal dataset":
                assert selected_path is not None
                sensor = sensor_with_auto_correction_for_dataset(
                    selected_path, sensor_override, roots_with_sensor
                )
                if sensor not in project.SENSOR_CHANNELS:
                    st.error("Could not detect sensor for selected image.")
                    return
                x_img = project.load_image(
                    selected_path, project.SENSOR_CHANNELS[sensor]
                )
                geo_meta = project.inspect_geospatial_metadata(selected_path)
                source_image_path = selected_path
                source_mode = "internal_dataset"
            else:
                source_image_bytes = uploaded_file.getvalue()
                sensor = sensor_with_auto_correction_for_upload(
                    source_image_bytes, sensor_override
                )
                if sensor not in project.SENSOR_CHANNELS:
                    st.error("Could not detect sensor from uploaded image.")
                    return
                x_img = load_uploaded_image(
                    source_image_bytes, project.SENSOR_CHANNELS[sensor]
                )
                geo_meta = inspect_uploaded_geospatial_metadata(source_image_bytes)
            backend_state = project.resolve_prediction_backend(
                requested_backend=backend_choice,
                output_dir=artifact_dir,
                sensor=sensor,
            )
            seg_backend = backend_state.get("segmentation_backend", project.PIPELINE_V3_BACKEND_ID)
            risk_backend = backend_state.get("risk_backend", project.PIPELINE_V3_BACKEND_ID)
            backend_fallback_reason = backend_state.get("fallback_reason")
            pipeline_bundle, pipeline_load_error = load_pipeline_bundle(
                str(pipeline_model_paths[sensor]),
                file_mtime_token(pipeline_model_paths[sensor]),
            )
            if pipeline_bundle is None:
                # Retry once without Streamlit cache in case dependencies were just installed.
                uncached_bundle, uncached_error = load_pipeline_bundle_uncached(
                    str(pipeline_model_paths[sensor])
                )
                if uncached_bundle is not None:
                    load_pipeline_bundle.clear()
                    pipeline_bundle = uncached_bundle
                    pipeline_load_error = None
                elif uncached_error:
                    pipeline_load_error = uncached_error
            if pipeline_bundle is None:
                st.error(
                    f"Pipeline V3 segmentation model for {sensor} is missing: {pipeline_model_paths[sensor]}"
                )
                st.caption(f"Artifacts directory in use: `{artifact_dir}`")
                if pipeline_load_error:
                    st.caption(f"Load failure details: `{pipeline_load_error}`")
                    if any(
                        key in pipeline_load_error.lower()
                        for key in [
                            "no module named",
                            "segmentation-models-pytorch",
                            "timm",
                            "torch",
                            "albumentations",
                        ]
                    ):
                        st.warning(
                            "Pipeline V3 dependencies are likely missing. Run: `pip install -r requirements-pipeline.txt` "
                            "then restart the GUI."
                        )
                return

            temporal_csv_path, temporal_bridge_csv_path = resolve_temporal_csv_paths()

            # Auto-check CSV by filename, then fallback to ERA5 anchor (coords/time).
            weather_values_for_predict = (
                dict(weather_values_raw) if weather_values_raw else {}
            )
            weather_source = "manual_input" if weather_values_for_predict else "none"
            weather_csv_status: str | None = None
            weather_anchor_status: str | None = None
            weather_anchor_meta: dict[str, Any] = {}
            if image_name and weather_csv_exists:
                auto_weather_values, auto_warn = lookup_weather_for_filename(
                    str(weather_csv_path), image_name
                )
                if auto_warn is None and auto_weather_values:
                    if weather_values_for_predict:
                        filled = fill_missing_weather_values(
                            weather_values_for_predict, auto_weather_values
                        )
                        if filled > 0:
                            weather_source = "manual_plus_csv_fill"
                    else:
                        weather_values_for_predict = auto_weather_values
                        weather_source = "auto_csv"
                else:
                    weather_csv_status = auto_warn
            elif image_name:
                weather_csv_status = "weather_csv_path_not_found"

            if image_name and temporal_csv_path.exists():
                missing_after_csv = [
                    name
                    for name in project.WEATHER_FEATURE_NAMES
                    if weather_values_for_predict.get(name) in (None, "")
                ]
                if not weather_values_for_predict or missing_after_csv:
                    anchor_weather, anchor_status, anchor_meta = (
                        project.lookup_weather_features_for_image_from_temporal(
                            csv_path=temporal_csv_path,
                            image_filename=image_name,
                            image_path=(
                                selected_path if selected_path is not None else None
                            ),
                            geo_meta=geo_meta,
                            bridge_csv_path=temporal_bridge_csv_path,
                        )
                    )
                    weather_anchor_meta = (
                        anchor_meta if isinstance(anchor_meta, dict) else {}
                    )
                    weather_anchor_status = anchor_status
                    if anchor_status is None and anchor_weather:
                        if weather_values_for_predict:
                            filled_anchor = fill_missing_weather_values(
                                weather_values_for_predict, anchor_weather
                            )
                            if filled_anchor > 0:
                                if weather_source == "manual_plus_csv_fill":
                                    weather_source = "manual_plus_csv_era5_fill"
                                elif weather_source == "manual_input":
                                    weather_source = "manual_plus_era5_fill"
                                elif weather_source == "none":
                                    weather_source = "manual_plus_era5_fill"
                        else:
                            weather_values_for_predict = anchor_weather
                            weather_source = "auto_era5_anchor"
            with st.spinner(
                "Analyzing satellite imagery and weather data... Please wait."
            ):
                drift_meta = project.detect_input_drift(
                    x_img,
                    sensor,
                    input_profile,
                    zscore_threshold=float(drift_zscore_threshold),
                )
                segmentation_pipeline = import_segmentation_pipeline_safe()
                seg_threshold_used, seg_threshold_source = (
                    project.resolve_segmentation_threshold(
                        float(threshold),
                        bundle_threshold=pipeline_bundle.get("decision_threshold"),
                        default_threshold=DEFAULT_THRESHOLD,
                    )
                )
                pred_mask, pred_prob, infer_meta = segmentation_pipeline.predict_pipeline_mask_auto(
                    model=pipeline_bundle["model"],
                    x_img=x_img,
                    threshold=float(seg_threshold_used),
                    patch_size=int(pipeline_bundle.get("patch_size", 256)),
                    stride=int(pipeline_bundle.get("stride", 192)),
                    batch_size=8,
                    device=pipeline_bundle["device"],
                    model_kind=str(pipeline_bundle.get("model_kind", "small_unet")),
                    normalization_stats=pipeline_bundle.get("normalization_stats"),
                )
                model_kind = str(pipeline_bundle.get("model_kind", "small_unet"))
                model_version = f"{model_kind}_epoch_{int(pipeline_bundle.get('epoch', 0))}"
                pred_feats = project.summarize_prediction_features(pred_mask, pred_prob)
                risk_with_weather_model = load_model(
                    str(risk_with_weather_pipeline_path),
                    file_mtime_token(risk_with_weather_pipeline_path),
                )
                risk_no_weather_model = load_model(
                    str(risk_no_weather_pipeline_path),
                    file_mtime_token(risk_no_weather_pipeline_path),
                )
                risk = project.route_risk_prediction(
                    pred_feats=pred_feats,
                    weather_values=(
                        weather_values_for_predict
                        if weather_values_for_predict
                        else None
                    ),
                    risk_with_weather_model=risk_with_weather_model,
                    risk_no_weather_model=risk_no_weather_model,
                    risk_threshold=risk_threshold,
                    risk_threshold_profile=risk_threshold_profile,
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
                    temporal_risk = project.predict_temporal_risk(
                        image_filename=image_name,
                        sensor=sensor,
                        pred_feats=pred_feats,
                        csv_path=temporal_csv_path,
                        temporal_model=load_model(
                            str(risk_temporal_pipeline_path),
                            file_mtime_token(risk_temporal_pipeline_path),
                        ),
                        risk_threshold=risk_threshold,
                        bridge_csv_path=temporal_bridge_csv_path,
                        image_path=selected_path if selected_path is not None else None,
                        geo_meta=geo_meta,
                    )
            payload = build_payload(
                sensor=sensor,
                image_name=image_name,
                pred_feats=pred_feats,
                risk=risk,
                weather_values=(
                    weather_values_for_predict if weather_values_for_predict else None
                ),
                temporal_risk=temporal_risk,
            )
            payload["prediction_id"] = str(uuid4())
            payload["timestamp_utc"] = project.utc_now_iso()
            payload["inference_mode"] = infer_meta["mode"]
            payload["inference_config"] = infer_meta
            payload["risk_threshold_profile"] = risk["risk_threshold_profile"]
            payload["risk_threshold"] = risk["risk_threshold"]
            payload["segmentation_backend_used"] = seg_backend
            payload["segmentation_model_kind_used"] = model_kind
            payload["segmentation_model_epoch_used"] = int(
                pipeline_bundle.get("epoch", 0)
            )
            payload["risk_backend_used"] = risk_backend
            payload["segmentation_threshold_used"] = float(seg_threshold_used)
            payload["segmentation_threshold_source"] = str(seg_threshold_source)
            payload["requested_backend"] = backend_state.get("requested_backend")
            payload["available_backends"] = backend_state.get("available_backends")
            payload["promotion_state"] = backend_state.get("active_backend")
            payload["model_version"] = model_version
            payload["geospatial_checks"] = geo_meta
            payload["geospatial_summary"] = project.build_geo_summary(
                pred_mask, geo_meta
            )
            payload["flood_area_km2"] = payload["geospatial_summary"].get(
                "flood_area_km2"
            )
            payload["drift_check"] = drift_meta
            payload["weather_source"] = weather_source
            payload["weather_csv_path_used"] = str(weather_csv_path)
            payload["temporal_csv_path_used"] = str(temporal_csv_path)
            payload["temporal_bridge_csv_path_used"] = str(temporal_bridge_csv_path)
            if weather_csv_status:
                payload["weather_csv_status"] = weather_csv_status
            if weather_anchor_status:
                payload["weather_anchor_status"] = weather_anchor_status
            if weather_anchor_meta:
                payload["weather_anchor_meta"] = weather_anchor_meta
            if backend_fallback_reason:
                payload["backend_fallback_reason"] = backend_fallback_reason
            payload["model_registry_path"] = str(
                (artifact_dir / "model_registry.json").resolve()
            )
            payload["model_run_id"] = (
                model_registry.get("run_id")
                if isinstance(model_registry, dict)
                else None
            )
            weather_statuses = [
                str(x)
                for x in [weather_csv_status, weather_anchor_status]
                if x not in (None, "", "ok")
            ]
            prediction_analysis = project.build_prediction_analysis(
                pred_mask=pred_mask,
                pred_prob=pred_prob,
                pred_feats=pred_feats,
                risk_payload=risk,
                temporal_payload=temporal_risk,
                prediction_eta={
                    key: payload.get(key)
                    for key in [
                        "prediction_eta_text",
                        "prediction_eta_start_utc",
                        "prediction_eta_end_utc",
                        "prediction_eta_source",
                        "prediction_eta_note",
                        "prediction_eta_days_min",
                        "prediction_eta_days_max",
                        "prediction_eta_hours_min",
                        "prediction_eta_hours_max",
                        "prediction_eta_horizon",
                        "prediction_eta_confidence_percent",
                        "prediction_eta_confidence_level",
                    ]
                },
                decision_support=payload.get("decision_support"),
                drift_meta=drift_meta,
                geo_meta=geo_meta,
                weather_statuses=weather_statuses,
                seg_threshold=float(seg_threshold_used),
            )
            payload["prediction_confidence"] = prediction_analysis["confidence"]
            payload["prediction_explanation"] = prediction_analysis["explanation"]
            payload["forecast_timeline"] = prediction_analysis["timeline"]
            payload["prediction_zone"] = prediction_analysis["zone_meta"]
            payload["geo_export"] = {
                "prediction_zone_geojson_data": project.build_prediction_zone_geojson(
                    zone_mask=prediction_analysis["zone_mask"],
                    zone_meta=prediction_analysis["zone_meta"],
                    geo_meta=geo_meta,
                )
            }
            feedback_info = project.register_feedback_candidate(
                output_dir=artifact_dir,
                image_name=image_name or "uploaded_image.tif",
                sensor=sensor,
                x_img=x_img,
                weather_values=(
                    weather_values_for_predict if weather_values_for_predict else None
                ),
                source_image_path=source_image_path,
                source_image_bytes=source_image_bytes,
                source_mode=source_mode,
                known_dataset_paths=known_dataset_paths,
                known_dataset_filenames=known_dataset_filenames,
            )
            payload["feedback_collection"] = feedback_info
            project.append_prediction_audit(artifact_dir, payload)
            st.session_state["last_prediction"] = {
                "payload": payload,
                "pred_mask": pred_mask,
                "pred_prob": pred_prob,
                "zone_mask": prediction_analysis["zone_mask"],
                "x_img": x_img,
                "image_name": image_name or "uploaded_image.tif",
            }
            st.success(f"Analysis completed ({sensor}).")
            if weather_source == "auto_csv":
                st.info(
                    "Weather values were auto-loaded from CSV using image filename."
                )
            elif weather_source == "auto_era5_anchor":
                st.info(
                    "Weather values were auto-loaded from ERA5 using location/time anchor."
                )
            elif weather_source == "manual_plus_csv_fill":
                st.info(
                    "Missing weather values were auto-filled from CSV using image filename."
                )
            elif weather_source == "manual_plus_era5_fill":
                st.info(
                    "Missing weather values were auto-filled from ERA5 using location/time anchor."
                )
            elif weather_source == "manual_plus_csv_era5_fill":
                st.info(
                    "Missing weather values were auto-filled using CSV first, then ERA5 anchor."
                )
            elif (
                weather_source == "none"
                and weather_csv_status == "weather_csv_missing_filename"
            ):
                st.caption(
                    "No matching filename was found in weather CSV. Fallback path was used."
                )
            elif (
                weather_source == "none"
                and weather_csv_status == "weather_csv_ambiguous_filename"
            ):
                st.caption(
                    "More than one CSV filename matched this image. Fallback path was used."
                )
            elif (
                weather_source == "none"
                and weather_csv_status == "weather_csv_path_not_found"
            ):
                st.caption(
                    f"Weather CSV is missing at `{weather_csv_path}`. Fallback path was used."
                )
            if weather_source == "none" and weather_anchor_status is not None:
                st.caption(
                    f"ERA5 anchor weather lookup unavailable: {weather_anchor_status}."
                )
            if feedback_info.get("status") == "collected":
                st.info("New image saved for future model improvement (pending label).")
            elif feedback_info.get("status") == "skipped":
                st.caption(
                    f"Feedback queue: {feedback_reason_label(feedback_info.get('reason'))}"
                )
        except Exception as ex:
            st.exception(ex)
    if "last_prediction" in st.session_state:
        last = st.session_state["last_prediction"]
        payload = last["payload"]
        risk_pct = payload.get("risk_score_percent")
        flood_pct = payload.get("pred_flood_ratio_percent")
        risk_level, color = risk_level_from_percent(
            risk_pct,
            risk_label=payload.get("risk_label"),
            risk_threshold=payload.get("risk_threshold"),
        )
        try:
            base_thr = float(payload.get("risk_threshold", 0.5))
        except Exception:
            base_thr = 0.5
        try:
            effective_score_thr = float(payload.get("sensor_risk_threshold", base_thr))
        except Exception:
            effective_score_thr = float(base_thr)
        policy_applied = bool(payload.get("sensor_policy_applied"))
        policy_rule = str(payload.get("sensor_policy_rule", "") or "")
        st.subheader("Decision Summary")
        row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
        row1_c1.metric(
            "Flood Risk %", f"{risk_pct:.2f}%" if risk_pct is not None else "N/A"
        )
        row1_c2.metric(
            "Flooded Area %", f"{flood_pct:.2f}%" if flood_pct is not None else "N/A"
        )
        row1_c3.metric("Risk Level", risk_level)
        row1_c4.metric("Sensor", payload.get("sensor", "N/A"))
        st.markdown(f"**Current Risk Level:** :{color}[{risk_level}]")
        st.caption(
            f"Model decision: {'Alert' if payload.get('risk_label') == 1 else 'No Alert'} "
            f"(effective score threshold={effective_score_thr:.2f})"
        )
        if policy_applied and policy_rule:
            try:
                det_thr_pct = (
                    float(payload.get("flood_presence_threshold", 0.0)) * 100.0
                )
            except Exception:
                det_thr_pct = 0.0
            combine_txt = "AND" if policy_rule.lower().endswith("_and") else "OR"
            st.caption(
                f"Sensor policy: {policy_rule} (score >= {effective_score_thr*100.0:.2f}% {combine_txt} flooded area >= {det_thr_pct:.2f}%)."
            )
        risk_path = risk_path_label(payload.get("risk_model_used"))
        seg_model_kind = str(
            payload.get("segmentation_model_kind_used")
            or payload.get("model_version")
            or pipeline_label(payload.get("segmentation_backend_used"), default="Pipeline V3")
        )
        st.caption(
            f"Risk path: {risk_path} | Inference mode: {payload.get('inference_mode', 'unknown')} | "
            f"Risk pipeline: {pipeline_label(payload.get('risk_backend_used'), default='Pipeline')} | Model: {seg_model_kind}"
        )
        detection_label = payload.get("detection_label")
        if detection_label is None:
            flood_ratio_val = payload.get("pred_flood_ratio")
            detection_label = (
                int(float(flood_ratio_val) > 0.0)
                if flood_ratio_val is not None
                else None
            )
        detection_text = (
            "Flood Detected"
            if detection_label == 1
            else ("No Flood Detected" if detection_label == 0 else "Unknown")
        )
        prediction_label = payload.get("prediction_label")
        prediction_status = str(payload.get("prediction_status", "unavailable"))
        is_detected_now = detection_label == 1
        prediction_text = (
            "Flood Risk Predicted"
            if prediction_label == 1
            else ("No Flood Risk Predicted" if prediction_label == 0 else "Unknown")
        )
        if is_detected_now:
            row2_c1, row2_c2 = st.columns(2)
            row2_c1.metric(
                "Model", seg_model_kind
            )
            row2_c2.metric("Detection", detection_text)
        else:
            row2_c1, row2_c2, row2_c3 = st.columns(3)
            row2_c1.metric(
                "Model", seg_model_kind
            )
            row2_c2.metric("Detection", detection_text)
            row2_c3.metric("Prediction", prediction_text)

        # Build concise, user-facing summary lines.
        summary_lines: list[str] = []
        if is_detected_now:
            summary_lines.append(
                "Flood is currently detected, so prediction and ETA are hidden."
            )
        else:
            summary_lines.append(f"{detection_text} | {prediction_text}")

        flood_presence_thr = payload.get("flood_presence_threshold")
        if (
            detection_label == 0
            and flood_pct is not None
            and flood_presence_thr is not None
            and float(flood_pct) > 0.0
        ):
            try:
                det_thr_pct = float(flood_presence_thr) * 100.0
                summary_lines.append(
                    f"Detection logic: flooded area {float(flood_pct):.2f}% is below detection threshold {det_thr_pct:.2f}%."
                )
            except Exception:
                pass

        prediction_logic_text: str | None = None
        if (not is_detected_now) and prediction_status == "active" and risk_pct is not None:
            try:
                risk_thr_pct = float(effective_score_thr) * 100.0
                if prediction_label == 0:
                    prediction_logic_text = f"Risk score {float(risk_pct):.2f}% is below threshold {risk_thr_pct:.2f}%."
                elif prediction_label == 1:
                    prediction_logic_text = f"Risk score {float(risk_pct):.2f}% reached threshold {risk_thr_pct:.2f}%."
            except Exception:
                pass

        pred_conf_pct, pred_conf_level = confidence_from_score(
            score=payload.get("risk_score"),
            threshold=effective_score_thr,
            predicted_label=prediction_label,
        )
        temporal_pct = payload.get("temporal_risk_score_percent")
        temporal_status = str(payload.get("temporal_status", "unavailable"))
        temporal_label = payload.get("temporal_risk_label")
        temporal_lookup_mode = str(payload.get("temporal_lookup_mode", "") or "")
        temporal_anchor_source = str(payload.get("temporal_anchor_source", "") or "")
        temporal_summary_text: str | None = None
        if (not is_detected_now) and temporal_pct is not None:
            temporal_text = (
                "Flood Risk Predicted"
                if int(temporal_label or 0) == 1
                else "No Flood Risk Predicted"
            )
            temporal_summary_text = (
                f"Temporal forecast: {temporal_pct:.2f}% | {temporal_text}"
            )
        elif temporal_status not in {"ok", "not_supported_for_sensor", "unavailable"}:
            temporal_summary_text = temporal_status_label(temporal_status)
        elif temporal_status in {"not_supported_for_sensor", "unavailable"}:
            temporal_summary_text = temporal_status_label(temporal_status)

        eta_text = payload.get("prediction_eta_text")
        eta_start = payload.get("prediction_eta_start_utc")
        eta_end = payload.get("prediction_eta_end_utc")
        eta_note = payload.get("prediction_eta_note")
        eta_days_min = payload.get("prediction_eta_days_min")
        eta_days_max = payload.get("prediction_eta_days_max")
        eta_hours_min = payload.get("prediction_eta_hours_min")
        eta_hours_max = payload.get("prediction_eta_hours_max")
        eta_horizon = payload.get("prediction_eta_horizon")
        eta_brief = "Unavailable"
        eta_window_detail: str | None = None
        if (not is_detected_now) and eta_start and eta_end:
            rel = eta_window_relative_label(
                days_min=eta_days_min,
                days_max=eta_days_max,
                hours_min=eta_hours_min,
                hours_max=eta_hours_max,
                start_utc=str(eta_start),
                end_utc=str(eta_end),
            )
            if rel:
                eta_brief = rel
            else:
                eta_brief = "Time window available"
            eta_window_detail = (
                f"{format_utc_short(eta_start)} -> {format_utc_short(eta_end)}"
            )
        elif (not is_detected_now) and eta_text not in (None, "", "unavailable", "prediction_unavailable"):
            eta_brief = eta_text_label(str(eta_text))

        pred_conf_text = "N/A"
        if (not is_detected_now) and prediction_status == "active" and pred_conf_pct is not None:
            pred_conf_text = f"{pred_conf_pct:.2f}% ({pred_conf_level})"
        confidence_block = payload.get("prediction_confidence", {})
        if (
            (not is_detected_now)
            and isinstance(confidence_block, dict)
            and confidence_block.get("overall_confidence_percent") is not None
        ):
            try:
                pred_conf_text = (
                    f"{float(confidence_block['overall_confidence_percent']):.2f}% "
                    f"({confidence_block.get('overall_confidence_level', 'N/A')})"
                )
            except Exception:
                pass
        eta_conf_text = "N/A"
        eta_conf_pct = payload.get("prediction_eta_confidence_percent")
        eta_conf_level = payload.get("prediction_eta_confidence_level")
        if (not is_detected_now) and eta_conf_pct is not None:
            try:
                eta_conf_text = (
                    f"{float(eta_conf_pct):.2f}% ({eta_conf_level or 'N/A'})"
                )
            except Exception:
                eta_conf_text = "N/A"

        if is_detected_now:
            st.markdown("**Operational Summary**")
            st.caption("Current scene contains detected flood. Prediction and ETA are hidden.")
        else:
            st.markdown("**Operational Summary**")
            csum1, csum2, csum3 = st.columns(3)
            csum1.metric("Prediction Confidence", pred_conf_text)
            csum2.metric("Temporal Confidence", eta_conf_text)
            csum3.metric("ETA", eta_brief)
        if prediction_logic_text:
            summary_lines.append(prediction_logic_text)
        if temporal_summary_text:
            summary_lines.append(temporal_summary_text)
        if eta_window_detail:
            summary_lines.append(f"ETA window: {eta_window_detail}")
        for line in summary_lines:
            st.caption(line)

        with st.expander("Technical details", expanded=False):
            if policy_applied and policy_rule:
                st.caption(
                    f"Sensor policy: {policy_rule} | effective score threshold={effective_score_thr:.2f}"
                )
            st.caption(
                f"Risk path: {risk_path} | Inference mode: {payload.get('inference_mode', 'unknown')} | "
                f"Risk pipeline: {pipeline_label(payload.get('risk_backend_used'), default='Pipeline')} | Model: {seg_model_kind}"
            )
            st.caption(
                f"Segmentation threshold used: {float(payload.get('segmentation_threshold_used', threshold)):.4f} "
                f"({payload.get('segmentation_threshold_source', 'runtime_argument')})"
            )
            if (not is_detected_now) and temporal_lookup_mode:
                st.caption(f"Temporal lookup mode: {temporal_lookup_mode}")
            if (not is_detected_now) and temporal_anchor_source:
                st.caption(f"Temporal anchor source: {temporal_anchor_source}")
            if (not is_detected_now) and eta_horizon:
                st.caption(f"ETA horizon: {str(eta_horizon).replace('_', ' ')}")
            if (not is_detected_now) and eta_note:
                st.caption(f"ETA note: {eta_note}")
            st.caption(
                f"Risk threshold (base): {base_thr:.2f} ({payload.get('risk_threshold_profile', 'N/A')}) | "
                f"Effective score threshold: {effective_score_thr:.2f}"
            )
            if payload.get("risk_adjustment"):
                st.caption(
                    "Risk score consistency guard applied for very low flooded-area prediction."
                )
        if is_missing_weather_fallback_warning(payload):
            missing = payload.get("missing_weather_features") or []
            details = f" Missing: {', '.join(missing)}." if missing else ""
            st.warning(
                "No Weather (Fallback) was used because weather inputs are incomplete."
                + details
            )
        geo_warn = payload.get("geospatial_checks", {}).get("warnings", [])
        if geo_warn:
            st.warning("Geospatial checks: " + ", ".join(geo_warn))
        drift_warn = payload.get("drift_check", {}).get("warnings", [])
        if drift_warn:
            st.warning("Input drift checks: " + ", ".join(drift_warn))
        if isinstance(confidence_block, dict) and confidence_block:
            st.subheader("Prediction Confidence")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric(
                "Overall Confidence",
                (
                    f"{float(confidence_block.get('overall_confidence_percent')):.2f}%"
                    if confidence_block.get("overall_confidence_percent") is not None
                    else "N/A"
                ),
            )
            cc2.metric(
                "Confidence Level",
                str(confidence_block.get("overall_confidence_level", "N/A")).title(),
            )
            cc3.metric(
                "Data Quality",
                str(confidence_block.get("data_quality_level", "N/A")).title(),
            )
            if confidence_block.get("out_of_distribution_warning"):
                st.warning(
                    "Out-of-distribution warning: current image statistics differ from training profile."
                )
            for warn in confidence_block.get("data_quality_warnings") or []:
                st.caption(f"Data quality: {warn}")
        explanation_block = payload.get("prediction_explanation", {})
        if isinstance(explanation_block, dict) and explanation_block:
            st.subheader("Why This Prediction")
            summary_text = str(explanation_block.get("summary", "")).strip()
            if summary_text:
                st.write(summary_text)
            drivers = explanation_block.get("key_drivers") or []
            if drivers:
                st.caption("Key drivers")
                for item in drivers:
                    st.write(f"- {item}")
            caveats = explanation_block.get("caveats") or []
            if caveats:
                st.caption("Caveats")
                for item in caveats:
                    st.write(f"- {item}")
        timeline_block = payload.get("forecast_timeline", {})
        if (
            isinstance(timeline_block, dict)
            and isinstance(timeline_block.get("items"), list)
            and timeline_block.get("items")
        ):
            st.subheader("Timeline View")
            timeline_rows: list[dict[str, str]] = []
            for item in timeline_block.get("items") or []:
                label = str(item.get("label", "unavailable")).replace("_", " ").title()
                note = str(item.get("note", "") or "")
                conf_text = "N/A"
                try:
                    if item.get("confidence_percent") is not None:
                        conf_text = (
                            f"{float(item.get('confidence_percent')):.2f}% "
                            f"({str(item.get('confidence_level', '') or 'n/a')})"
                        )
                except Exception:
                    conf_text = "N/A"
                timeline_rows.append(
                    {
                        "Horizon": (
                            "Now"
                            if int(item.get("horizon_days", 0) or 0) == 0
                            else f"+{int(item.get('horizon_days', 0) or 0)}d"
                        ),
                        "Status": label,
                        "Confidence": conf_text,
                        "Note": note,
                    }
                )
            st.dataframe(pd.DataFrame(timeline_rows), width="stretch", hide_index=True)
        geo_checks = payload.get("geospatial_checks", {})
        try:
            center_lat = (
                float(geo_checks.get("center_lat"))
                if geo_checks.get("center_lat") is not None
                else None
            )
            center_lon = (
                float(geo_checks.get("center_lon"))
                if geo_checks.get("center_lon") is not None
                else None
            )
        except Exception:
            center_lat, center_lon = None, None
        if (
            center_lat is not None
            and center_lon is not None
            and -90.0 <= center_lat <= 90.0
            and -180.0 <= center_lon <= 180.0
        ):
            st.subheader("Scene Location")
            st.map(
                pd.DataFrame({"lat": [center_lat], "lon": [center_lon]}),
                width="stretch",
            )
            st.caption(
                f"Center coordinates: lat={center_lat:.6f}, lon={center_lon:.6f}"
            )
        decision_support = payload.get("decision_support", {})
        if isinstance(decision_support, dict):
            actions = decision_support.get("recommended_actions") or []
            drivers = decision_support.get("primary_drivers") or []
            support_warnings = decision_support.get("warnings") or []
            if actions:
                st.subheader("Recommended Actions")
                if drivers:
                    st.caption("Drivers: " + " | ".join(str(x) for x in drivers))
                for action in actions:
                    st.write(f"- {action}")
            for warn in support_warnings:
                st.warning(str(warn))
        show_result_images(
            last["x_img"],
            last["pred_prob"],
            last["pred_mask"],
            zone_mask=last.get("zone_mask"),
            sensor=payload.get("sensor"),
            detection_label=detection_label if detection_label in (0, 1) else None,
            prediction_label=prediction_label if prediction_label in (0, 1) else None,
            prediction_status=prediction_status,
            seg_threshold=float(payload.get("segmentation_threshold_used", threshold)),
        )
        stem = Path(last["image_name"]).stem
        download_pack = build_download_package(
            stem,
            payload,
            last["pred_mask"],
            last["pred_prob"],
            zone_mask=last.get("zone_mask"),
        )
        dl_json_name, dl_json_bytes, dl_json_mime = download_pack["json"]
        dl_mask_name, dl_mask_bytes, dl_mask_mime = download_pack["mask"]
        dl_prob_name, dl_prob_bytes, dl_prob_mime = download_pack["prob"]
        dl_zip_name, dl_zip_bytes, dl_zip_mime = download_pack["zip"]
        dl_geojson = download_pack.get("geojson")
        dl_zone_geotiff = download_pack.get("zone_geotiff")
        pred_id = str(payload.get("prediction_id", stem))
        st.subheader("Download Artifacts")
        col_count = 6 if (dl_geojson is not None or dl_zone_geotiff is not None) else 4
        cols = st.columns(col_count)
        d1, d2, d3, d4 = cols[:4]
        d1.download_button(
            "Download JSON",
            data=dl_json_bytes,
            file_name=dl_json_name,
            mime=dl_json_mime,
            key=f"dl_json_{pred_id}",
            width="stretch",
        )
        d2.download_button(
            "Download Mask",
            data=dl_mask_bytes,
            file_name=dl_mask_name,
            mime=dl_mask_mime,
            key=f"dl_mask_{pred_id}",
            width="stretch",
        )
        d3.download_button(
            "Download Prob",
            data=dl_prob_bytes,
            file_name=dl_prob_name,
            mime=dl_prob_mime,
            key=f"dl_prob_{pred_id}",
            width="stretch",
        )
        d4.download_button(
            "Download ZIP",
            data=dl_zip_bytes,
            file_name=dl_zip_name,
            mime=dl_zip_mime,
            key=f"dl_zip_{pred_id}",
            width="stretch",
        )
        if dl_geojson is not None:
            geo_name, geo_bytes, geo_mime = dl_geojson
            cols[4].download_button(
                "Download GeoJSON",
                data=geo_bytes,
                file_name=geo_name,
                mime=geo_mime,
                key=f"dl_geojson_{pred_id}",
                width="stretch",
            )
        if dl_zone_geotiff is not None:
            zone_name, zone_bytes, zone_mime = dl_zone_geotiff
            cols[5].download_button(
                "Download GeoTIFF",
                data=zone_bytes,
                file_name=zone_name,
                mime=zone_mime,
                key=f"dl_zone_geotiff_{pred_id}",
                width="stretch",
            )
        if st.button("Export Analysis Package"):
            export_paths = export_prediction_package(
                save_dir,
                stem,
                payload,
                last["pred_mask"],
                last["pred_prob"],
                zone_mask=last.get("zone_mask"),
            )
            st.success("Analysis package exported.")
            st.write(export_paths)


if __name__ == "__main__":
    app()
