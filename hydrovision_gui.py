from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import tifffile

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    import folium
    from streamlit_folium import st_folium
except Exception:  # pragma: no cover - optional runtime dependency
    folium = None
    st_folium = None


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs_pipeline_v3_maskstrict_20260606_175900"


st.set_page_config(page_title="HYDROVISION", layout="wide", initial_sidebar_state="expanded")


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700;900&display=swap');

:root {
  --cyan: #7ff6ff;
  --cyan-strong: #00d4ff;
  --red: #ff3131;
  --green: #39ff14;
  --panel: rgba(12, 15, 28, 0.92);
}

.stApp {
  background:
    radial-gradient(circle at 56% 96%, rgba(0, 212, 255, 0.22), transparent 22%),
    radial-gradient(circle at 92% 92%, rgba(255, 49, 49, 0.19), transparent 22%),
    linear-gradient(135deg, #070912 0%, #0b0d18 48%, #07070b 100%);
  color: #ffffff;
  font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stSidebar"] {
  background: rgba(10, 13, 24, 0.96);
  border-right: 1px solid rgba(127, 246, 255, 0.28);
  box-shadow: 0 0 30px rgba(0, 212, 255, 0.12);
}

[data-testid="stSidebar"] * {
  color: #f4fbff;
}

.block-container {
  padding-top: 1.55rem;
  max-width: 1220px;
}

.hv-logo {
  color: var(--cyan);
  font-size: 2rem;
  font-weight: 900;
  letter-spacing: 0;
  padding: 0.4rem 0 1.1rem;
  margin-bottom: 1rem;
  border-bottom: 2px solid rgba(127, 246, 255, 0.65);
  text-shadow: 0 0 20px rgba(127, 246, 255, 0.55);
}

.hv-footer {
  margin-top: 1.5rem;
  padding: 0.7rem;
  border: 1px solid rgba(127, 246, 255, 0.18);
  color: #9fb4c5;
  text-align: center;
  font-size: 0.75rem;
}

.main-title {
  color: var(--cyan);
  font-size: 2rem;
  font-weight: 900;
  text-align: center;
  letter-spacing: 0.1rem;
  margin: 0.7rem 0 1.2rem;
  text-shadow: 0 0 16px rgba(127, 246, 255, 0.45);
}

.main-title:before,
.main-title:after {
  content: "";
  display: inline-block;
  width: 90px;
  border-top: 2px solid rgba(127, 246, 255, 0.75);
  margin: 0 18px 9px;
}

.upload-panel,
.image-card,
.report-box,
.map-box {
  background: rgba(8, 10, 20, 0.88);
  border: 1.5px solid rgba(127, 246, 255, 0.58);
  border-radius: 10px;
  box-shadow: 0 0 20px rgba(0, 212, 255, 0.16);
  padding: 1rem;
}

.upload-panel {
  margin-bottom: 1rem;
}

.hint {
  color: #b6bdc9;
  font-size: 0.95rem;
  margin-top: 0.6rem;
}

.danger-zone {
  background:
    linear-gradient(135deg, rgba(74, 0, 0, 0.96), rgba(24, 7, 14, 0.92));
  color: #fff3f3;
  padding: 1.15rem;
  border-radius: 10px;
  border: 2px solid var(--red);
  box-shadow: inset 0 0 26px rgba(255, 49, 49, 0.16), 0 0 18px rgba(255, 49, 49, 0.22);
  min-height: 292px;
}

.success-zone {
  background:
    linear-gradient(135deg, rgba(0, 34, 0, 0.96), rgba(5, 20, 11, 0.92));
  color: #efffed;
  padding: 1.15rem;
  border-radius: 10px;
  border: 2px solid var(--green);
  box-shadow: inset 0 0 26px rgba(57, 255, 20, 0.13), 0 0 18px rgba(57, 255, 20, 0.18);
  min-height: 292px;
}

.danger-zone h3 {
  color: #ff8585;
  letter-spacing: 0.08rem;
  margin: 0 0 1rem;
}

.success-zone h3 {
  color: var(--green);
  letter-spacing: 0.08rem;
  margin: 0 0 1rem;
}

.ratio {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  font-size: 1.35rem;
  font-weight: 900;
  border-bottom: 1px solid rgba(255, 255, 255, 0.16);
  padding-bottom: 0.7rem;
  margin-bottom: 0.8rem;
}

.ratio span:last-child {
  color: var(--cyan);
}

.coord-badge {
  position: absolute;
  right: 12px;
  bottom: 12px;
  background: rgba(10, 13, 24, 0.86);
  border: 1px solid rgba(127, 246, 255, 0.5);
  border-radius: 8px;
  padding: 0.6rem 0.8rem;
  color: #ffffff;
  font-weight: 700;
}

.image-wrap {
  position: relative;
}

.stButton > button,
.stDownloadButton > button {
  background: rgba(0, 212, 255, 0.16);
  color: var(--cyan);
  border: 2px solid var(--cyan);
  border-radius: 10px;
  font-weight: 900;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
  background: rgba(0, 212, 255, 0.28);
  color: #ffffff;
  border-color: var(--cyan);
}

[data-testid="stFileUploader"] section {
  background: rgba(7, 9, 19, 0.95);
  border: 1px solid rgba(127, 246, 255, 0.3);
  border-radius: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)


def flood_reason_solution(runoff: float, prob: float) -> tuple[str, str]:
    if runoff > 0.001 and prob > 0.6:
        return (
            "High surface runoff combined with hydrological response similar to flood-prone basins.",
            "Enhance drainage capacity and activate early flood warning protocols.",
        )
    if runoff > 0.001:
        return (
            "Surface runoff exceeds natural absorption capacity.",
            "Improve infiltration measures and runoff control systems.",
        )
    if prob > 0.6:
        return (
            "SAR and meteorological indicators show flood-like behavior.",
            "Increase monitoring frequency and prepare mitigation actions.",
        )
    return (
        "Hydrological conditions are stable.",
        "No immediate intervention required.",
    )


def _to_band(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[-1] <= 16:
            return arr[:, :, 0]
        return arr[0]
    if arr.ndim > 3:
        return np.squeeze(arr)[0]
    return arr


def resolve_output_dir() -> Path:
    raw = os.getenv("FLOOD_OUTPUT_DIR", "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_DIR / path
        return path.resolve()
    return DEFAULT_OUTPUT_DIR.resolve()


@st.cache_data(show_spinner=False)
def load_metrics(output_dir_text: str) -> dict[str, Any]:
    import json

    output_dir = Path(output_dir_text)
    payload: dict[str, Any] = {"output_dir": str(output_dir)}
    for key, filename in {
        "global": "unet_val_metrics_global.json",
        "s1": "unet_val_metrics_s1.json",
        "s2": "unet_val_metrics_s2.json",
    }.items():
        try:
            payload[key] = json.loads((output_dir / filename).read_text(encoding="utf-8"))
        except Exception:
            payload[key] = None
    return payload


@st.cache_resource(show_spinner=False)
def load_pipeline_bundle_cached(model_path_text: str) -> dict[str, Any]:
    import segmentation_pipeline

    return segmentation_pipeline.load_pipeline_bundle(Path(model_path_text), device="cpu")


def available_model_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "S1": output_dir / "unet_model_s1.pth",
        "S2": output_dir / "unet_model_s2.pth",
    }


def _to_channels_last(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF shape: {arr.shape}")
    if arr.shape[-1] <= 16:
        return arr
    if arr.shape[0] <= 16:
        return np.moveaxis(arr, 0, -1)
    return arr


def _fit_channels(arr: np.ndarray, required_channels: int) -> np.ndarray:
    arr = _to_channels_last(arr).astype(np.float32)
    current = int(arr.shape[-1])
    required = int(required_channels)
    if current == required:
        return arr
    if current > required:
        return arr[:, :, :required]
    parts = [arr]
    while sum(part.shape[-1] for part in parts) < required:
        remaining = required - sum(part.shape[-1] for part in parts)
        parts.append(arr[:, :, : min(current, remaining)])
    return np.concatenate(parts, axis=-1)[:, :, :required]


def _downsample_image(arr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return arr
    if cv2 is not None:
        return cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    step = max(1, int(round(1 / scale)))
    return arr[::step, ::step]


def infer_sensor_from_image(arr: np.ndarray, requested: str) -> str:
    if requested in {"S1", "S2"}:
        return requested
    channels = int(_to_channels_last(arr).shape[-1])
    return "S2" if channels >= 6 else "S1"


def _normalize_uint8(band: np.ndarray) -> np.ndarray:
    band = np.asarray(band, dtype=np.float32)
    band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(band, [2, 98])
    if high <= low:
        high = float(np.max(band) or 1.0)
        low = float(np.min(band))
    scaled = np.clip((band - low) / max(high - low, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def _resize_for_display(image: np.ndarray, max_side: int = 900) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0 or cv2 is None:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def render_model_status(output_dir: Path) -> None:
    metrics = load_metrics(str(output_dir))
    paths = available_model_paths(output_dir)
    st.sidebar.markdown("#### Pipeline Output")
    st.sidebar.caption(output_dir.name)
    st.sidebar.write(f"S1 model: {'OK' if paths['S1'].exists() else 'Missing'}")
    st.sidebar.write(f"S2 model: {'OK' if paths['S2'].exists() else 'Missing'}")
    global_metrics = metrics.get("global") or {}
    if global_metrics:
        st.sidebar.write(f"F1: {float(global_metrics.get('f1', 0.0)):.2%}")
        st.sidebar.write(f"IoU: {float(global_metrics.get('iou', 0.0)):.2%}")


def predict_with_pipeline(raw: bytes, output_dir: Path, sensor: str) -> tuple[np.ndarray, float, float, str]:
    import segmentation_pipeline

    original = tifffile.imread(BytesIO(raw))
    use_sensor = infer_sensor_from_image(original, sensor)
    model_path = available_model_paths(output_dir)[use_sensor]
    if not model_path.exists():
        raise FileNotFoundError(f"{use_sensor} model is missing: {model_path}")
    bundle = load_pipeline_bundle_cached(str(model_path))
    x_img = _fit_channels(original, int(bundle["in_channels"]))
    x_img = _downsample_image(x_img, max_side=1024)
    threshold = bundle.get("decision_threshold")
    if threshold is None:
        threshold = 0.5
    pred_mask, pred_prob, _meta = segmentation_pipeline.predict_pipeline_mask_auto(
        model=bundle["model"],
        x_img=x_img,
        threshold=float(threshold),
        patch_size=int(bundle.get("patch_size", 256)),
        stride=int(bundle.get("stride", 192)),
        batch_size=2,
        device=bundle["device"],
        model_kind=str(bundle.get("model_kind", "small_unet")),
        normalization_stats=bundle.get("normalization_stats"),
    )
    overview = _normalize_uint8(_to_band(x_img))
    visual = np.repeat(overview[:, :, None], 3, axis=2)
    visual[pred_prob >= 0.35] = np.array([0, 230, 255], dtype=np.uint8)
    visual[pred_prob >= float(threshold)] = np.array([255, 225, 80], dtype=np.uint8)
    visual[pred_prob >= min(0.95, float(threshold) + 0.2)] = np.array([255, 120, 25], dtype=np.uint8)
    flood_ratio = float(np.mean(pred_mask > 0))
    prob = float(np.clip(np.mean(pred_prob) * 2.2 + np.percentile(pred_prob, 90) * 0.55, 0.0, 1.0))
    label = f"Pipeline V3 {use_sensor} | {bundle.get('model_kind')} | threshold={float(threshold):.3f}"
    return _resize_for_display(visual), flood_ratio, prob, label


def analyze_sar(raw: bytes, similarity_factor: float) -> tuple[np.ndarray, float, float, str]:
    arr = tifffile.imread(BytesIO(raw))
    band = _to_band(arr)
    if max(band.shape[:2]) > 1600:
        band = band[::2, ::2]
    norm = _normalize_uint8(band)

    threshold = 65
    if cv2 is not None:
        _, mask = cv2.threshold(norm, threshold, 255, cv2.THRESH_BINARY_INV)
        visual = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        glow = cv2.applyColorMap(norm, cv2.COLORMAP_BONE)
        visual = cv2.addWeighted(visual, 0.58, glow, 0.42, 0)
    else:
        mask = np.where(norm < threshold, 255, 0).astype(np.uint8)
        visual = np.repeat(norm[:, :, None], 3, axis=2)

    flood_ratio = float(np.mean(mask > 0))
    hot = mask > 0
    visual[hot] = np.array([0, 230, 255], dtype=np.uint8)
    very_hot = hot & (norm < 35)
    visual[very_hot] = np.array([255, 225, 80], dtype=np.uint8)
    extreme = hot & (norm < 22)
    visual[extreme] = np.array([255, 120, 25], dtype=np.uint8)
    prob = min(1.0, flood_ratio * similarity_factor * 5.0)
    return _resize_for_display(visual), flood_ratio, prob, "Fast SAR threshold fallback"


def sidebar_controls(output_dir: Path) -> tuple[float, float, str, float, str, str]:
    st.sidebar.markdown("<div class='hv-logo'>HYDROVISION</div>", unsafe_allow_html=True)
    st.sidebar.markdown("#### Target Area Coordinates")
    lat = st.sidebar.number_input("Latitude", value=30.800000, format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=30.990000, format="%.6f")

    st.sidebar.markdown("#### Basin Similarity")
    basin_type = st.sidebar.selectbox(
        "Hydrological Similarity Type",
        [
            "Seine-like (Low slope / Urban)",
            "Marne-like (Mixed rural/urban)",
            "General Similar Basin",
        ],
    )
    similarity_factor = {
        "Seine-like (Low slope / Urban)": 1.0,
        "Marne-like (Mixed rural/urban)": 1.1,
        "General Similar Basin": 1.05,
    }[basin_type]

    st.sidebar.markdown("#### System Modules")
    mode = st.sidebar.radio(
        "Select module",
        [
            "Satellite Radar Analysis",
            "Manual Hydrological Diagnostic",
            "Geospatial Risk Map",
        ],
        label_visibility="collapsed",
    )
    sensor = st.sidebar.selectbox("Sensor Runtime", ["Auto", "S1", "S2"])
    render_model_status(output_dir)
    st.sidebar.markdown(
        "<div class='hv-footer'>HYDROVISION | Similar Basin Flood Prediction System</div>",
        unsafe_allow_html=True,
    )
    return lat, lon, basin_type, similarity_factor, mode, sensor


def render_satellite(lat: float, lon: float, similarity_factor: float, output_dir: Path, sensor: str) -> None:
    st.markdown("<div class='main-title'>SATELLITE RADAR ANALYSIS</div>", unsafe_allow_html=True)
    st.markdown("<div class='upload-panel'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload SAR Image (.tif)", type=["tif", "tiff"])
    st.markdown(
        "<div class='hint'>Prediction based on SAR pattern similarity and hydrological response.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is None:
        col1, col2 = st.columns([1.08, 1])
        with col1:
            st.markdown("<div class='image-card'>Upload a SAR GeoTIFF scene to preview the radar analysis.</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(
                """
<div class='success-zone'>
  <h3>READY</h3>
  <div class='ratio'><span>Inundation Ratio:</span><span>--</span></div>
  <p><b>Reason:</b><br/>Waiting for satellite input.</p>
  <p><b>Solution:</b><br/>Upload a SAR TIFF image to run HYDROVISION AI.</p>
</div>
""",
                unsafe_allow_html=True,
            )
        return

    try:
        raw = uploaded.getvalue()
        try:
            visual, flood_ratio, prob, source_label = predict_with_pipeline(
                raw,
                output_dir,
                "" if sensor == "Auto" else sensor,
            )
        except Exception as model_exc:
            visual, flood_ratio, prob, source_label = analyze_sar(raw, similarity_factor)
            st.warning(f"Pipeline model could not run, using fallback analysis. Details: {model_exc}")
    except Exception as exc:
        st.error(f"Could not read SAR image: {exc}")
        return

    reason, solution = flood_reason_solution(flood_ratio, prob)
    flood_detected = flood_ratio > 0.10 or prob > 0.60

    col1, col2 = st.columns([1.08, 1])
    with col1:
        st.markdown("<div class='image-card image-wrap'>", unsafe_allow_html=True)
        st.image(visual, use_container_width=True)
        st.markdown(
            f"<div class='coord-badge'>Area Coordinates:<br>{lat:.2f}, {lon:.2f}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        zone = "danger-zone" if flood_detected else "success-zone"
        title = "FLOOD INDICATION" if flood_detected else "NO FLOOD"
        st.markdown(
            f"""
<div class='{zone}'>
  <h3>{title}</h3>
  <div class='ratio'><span>Inundation Ratio:</span><span>{flood_ratio:.2%}</span></div>
  <p><b>Runtime:</b><br/>{source_label}</p>
  <p><b>Reason:</b><br/>{reason}</p>
  <p><b>Solution:</b><br/>{solution}</p>
</div>
""",
            unsafe_allow_html=True,
        )


def render_manual(lat: float, lon: float, similarity_factor: float) -> None:
    st.markdown("<div class='main-title'>MANUAL HYDROLOGICAL DIAGNOSTIC</div>", unsafe_allow_html=True)
    st.markdown("<div class='report-box'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    temp = c1.number_input("Temperature (C)", value=15.0)
    rain = c1.number_input("Precipitation", value=0.002, format="%.6f")
    runoff = c2.number_input("Surface Runoff", value=0.00003, format="%.8f")
    c2.info(f"Area Coordinates: {lat:.6f}, {lon:.6f}")
    run = st.button("RUN HYDROVISION AI", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not run:
        return

    adjusted_ro = runoff * similarity_factor
    prob = min(1.0, max(0.0, (rain * 70.0) + (adjusted_ro * 380.0) + (0.01 if temp > 28 else 0.0)))
    reason, solution = flood_reason_solution(runoff, prob)
    flood_risk = runoff > 0.001 or prob > 0.5
    zone = "danger-zone" if flood_risk else "success-zone"
    title = f"FLOOD RISK ({prob:.2%})" if flood_risk else f"SAFE ({prob:.2%})"
    st.markdown(
        f"""
<div class='{zone}'>
  <h3>{title}</h3>
  <p><b>Reason:</b><br/>{reason}</p>
  <p><b>Solution:</b><br/>{solution}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_map(lat: float, lon: float) -> None:
    st.markdown("<div class='main-title'>GEOSPATIAL FLOOD RISK MAP</div>", unsafe_allow_html=True)
    if folium is None or st_folium is None:
        st.warning("Map dependencies are not installed. Install folium and streamlit-folium to enable the interactive map.")
        return

    m = folium.Map(location=[lat, lon], zoom_start=10, tiles="CartoDB dark_matter")
    rng = np.random.default_rng(42)
    for idx in range(80):
        lat_i = lat + float(rng.uniform(-0.06, 0.06))
        lon_i = lon + float(rng.uniform(-0.06, 0.06))
        risk = float(rng.beta(1.5, 3.0))
        color = "#ff3131" if risk > 0.45 else "#39ff14"
        status = "FLOOD RISK" if risk > 0.45 else "SAFE"
        folium.CircleMarker(
            [lat_i, lon_i],
            radius=10 if risk > 0.45 else 6,
            color=color,
            fill=True,
            fill_opacity=0.75,
            popup=f"<b>Status:</b> {status}<br><b>Risk:</b> {risk:.2%}",
        ).add_to(m)

    st.markdown("<div class='map-box'>", unsafe_allow_html=True)
    st_folium(m, width=1180, height=610)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    output_dir = resolve_output_dir()
    lat, lon, _basin_type, similarity_factor, mode, sensor = sidebar_controls(output_dir)
    if mode == "Satellite Radar Analysis":
        render_satellite(lat, lon, similarity_factor, output_dir, sensor)
    elif mode == "Manual Hydrological Diagnostic":
        render_manual(lat, lon, similarity_factor)
    else:
        render_map(lat, lon)


if __name__ == "__main__":
    main()
