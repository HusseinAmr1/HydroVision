from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tifffile

import project
from env_utils import load_dotenv, resolve_env_path

try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None


st.set_page_config(page_title="HYDROVISION", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)
OUTPUT_DIR = BASE_DIR / "outputs"
WEATHER_CSV_ENV_VAR = "WEATHER_CSV_PATH"
DEFAULT_WEATHER_CSV_FALLBACK = Path("dataset") / "Final_Full_Data_Matched.csv"
DEFAULT_RISK_PROFILE = project.DEFAULT_RISK_THRESHOLD_PROFILE


def inject_styles() -> None:
    st.markdown(
        """
<style>
.stApp { background-color: #000000; color: #ffffff; }
.main-title { color: #00d4ff; font-size: 40px; font-weight: 900; text-align: center;
border-bottom: 3px solid #00d4ff; padding-bottom: 10px; margin-bottom: 20px;}
.report-box { background-color: #111111; padding: 25px; border-radius: 15px;
border: 2px solid #00d4ff; }
.danger-zone { background-color: #4a0000; color: #ff3131; padding: 20px;
border-radius: 10px; border: 2px solid #ff3131; }
.success-zone { background-color: #002200; color: #39ff14; padding: 20px;
border-radius: 10px; border: 2px solid #39ff14; }
b { color: #00d4ff; }
</style>
""",
        unsafe_allow_html=True,
    )


def resolve_weather_csv_path() -> Path:
    return resolve_env_path(
        WEATHER_CSV_ENV_VAR,
        base_dir=BASE_DIR,
        default_relative=DEFAULT_WEATHER_CSV_FALLBACK,
    )


def load_pickle(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_pipeline_bundle(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        import segmentation_pipeline

        return segmentation_pipeline.load_pipeline_bundle(path, device="cpu")
    except Exception:
        return None


@st.cache_data
def load_weather_aggregate(csv_path: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    return project.aggregate_csv_features(Path(csv_path))


@st.cache_resource
def load_resources() -> dict[str, Any]:
    weather_csv_path = resolve_weather_csv_path()
    weather_df = pd.DataFrame()
    weather_issues: list[dict[str, Any]] = []
    if weather_csv_path.exists():
        weather_df, weather_issues = load_weather_aggregate(str(weather_csv_path))

    pipeline_seg = {
        "S1": load_pipeline_bundle(project.get_pipeline_model_path("S1", OUTPUT_DIR)),
        "S2": load_pipeline_bundle(project.get_pipeline_model_path("S2", OUTPUT_DIR)),
    }
    pipeline_w, pipeline_n = project.get_pipeline_risk_model_paths(OUTPUT_DIR)
    risk_pipeline = {
        "with_weather": load_pickle(pipeline_w),
        "no_weather": load_pickle(pipeline_n),
        "temporal": load_pickle(project.get_temporal_model_path(output_dir=OUTPUT_DIR, backend=project.PIPELINE_V3_BACKEND_ID)),
    }

    return {
        "output_dir": OUTPUT_DIR,
        "active_backend": project.load_active_backend_config(OUTPUT_DIR),
        "pipeline_seg": pipeline_seg,
        "risk_pipeline": risk_pipeline,
        "weather_csv_path": weather_csv_path,
        "weather_df": weather_df,
        "weather_issues": weather_issues,
    }


def flood_reason_solution(runoff: float, prob: float) -> tuple[str, str]:
    if runoff > 0.001 and prob > 0.6:
        return (
            "High surface runoff and high flood probability were detected together.",
            "Increase drainage capacity and activate early warning protocols.",
        )
    if runoff > 0.001:
        return (
            "Surface runoff exceeds absorption capacity.",
            "Improve infiltration and runoff control systems.",
        )
    if prob > 0.6:
        return (
            "Image pattern and risk model indicate flood-like behavior.",
            "Increase monitoring and prepare mitigation actions.",
        )
    return ("Hydrological indicators look stable.", "No immediate intervention required.")


def temporal_status_label(status: str | None) -> str:
    mapping = {
        "ok": "Temporal forecast available.",
        "temporal_csv_missing_filename": "No matching weather time-series row for this image filename.",
        "temporal_csv_ambiguous_filename": "More than one weather row matched this filename.",
        "temporal_csv_invalid": "Weather CSV format is invalid.",
        "temporal_csv_empty": "Weather CSV has no usable rows.",
        "temporal_model_missing": "Temporal model is not loaded.",
        "missing_filename": "Image filename is missing, temporal lookup cannot run.",
        "feature_schema_mismatch": "Temporal model/features mismatch.",
        "prediction_failed": "Temporal model inference failed.",
        "not_supported_for_sensor": "Temporal forecast is currently enabled for S1 only.",
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
    }
    if code is None:
        return "ETA unavailable."
    text = str(code)
    if text.startswith("possible_flood_window_"):
        return text
    return mapping.get(text, text.replace("_", " "))


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
    ch = int(arr.shape[-1])
    if ch >= project.SENSOR_CHANNELS["S2"]:
        return "S2"
    if ch >= project.SENSOR_CHANNELS["S1"]:
        return "S1"
    if ch == 1:
        return "S1"
    return None


def load_uploaded_image(raw: bytes, required_channels: int) -> np.ndarray:
    arr = np.asarray(tifffile.imread(BytesIO(raw)))
    arr = to_channels_last(arr)
    if arr.shape[-1] < required_channels:
        if required_channels == 2 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 2, axis=-1)
        else:
            raise ValueError(f"Uploaded image has {arr.shape[-1]} channels but needs {required_channels}")
    arr = arr[..., :required_channels].astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_channel(channel: np.ndarray) -> np.ndarray:
    x = channel.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def normalize_prob(prob: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(prob.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.quantile(p, 0.02))
    hi = float(np.quantile(p, 0.98))
    if hi - lo < 1e-6:
        lo = float(np.min(p))
        hi = float(np.max(p))
    if hi - lo < 1e-6:
        return np.zeros_like(p, dtype=np.float32)
    return np.clip((p - lo) / (hi - lo), 0.0, 1.0)


def build_overlay(overview: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.dstack([overview, overview, overview])
    overlay = rgb.copy()
    overlay[mask > 0] = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    return np.clip(overlay, 0.0, 1.0)


def choose_risk_models(resources: dict[str, Any]) -> tuple[Any | None, Any | None, str]:
    w = resources["risk_pipeline"]["with_weather"]
    n = resources["risk_pipeline"]["no_weather"]
    return w, n, project.PIPELINE_V3_BACKEND_ID


def run_satellite_inference(
    *,
    resources: dict[str, Any],
    raw: bytes,
    filename: str,
    sensor_choice: str,
    threshold: float,
    lat: float,
    lon: float,
    include_weather: bool,
    manual_weather: dict[str, float] | None,
) -> dict[str, Any]:
    sensor_auto = detect_sensor_from_raw(raw)
    sensor = sensor_auto if sensor_choice == "Auto" else sensor_choice
    if sensor not in project.SENSOR_CHANNELS:
        raise ValueError("Could not determine sensor from image channels.")

    backend_state = project.resolve_prediction_backend(
        requested_backend="auto",
        output_dir=resources["output_dir"],
        sensor=sensor,
    )
    seg_backend = str(backend_state.get("segmentation_backend", project.PIPELINE_V3_BACKEND_ID))
    x_img = load_uploaded_image(raw, project.SENSOR_CHANNELS[sensor])

    bundle = resources["pipeline_seg"].get(sensor)
    if bundle is None:
        raise FileNotFoundError(f"No Pipeline V3 segmentation model available for sensor {sensor}.")
    import segmentation_pipeline

    pred_mask, pred_prob, infer_meta = segmentation_pipeline.predict_pipeline_mask_auto(
        model=bundle["model"],
        x_img=x_img,
        threshold=float(threshold),
        patch_size=int(bundle.get("patch_size", 256)),
        stride=int(bundle.get("stride", 192)),
        batch_size=8,
        device=bundle["device"],
        model_kind=str(bundle.get("model_kind", "small_unet")),
    )

    pred_feats = project.summarize_prediction_features(pred_mask, pred_prob)
    risk_w, risk_n, risk_backend_used = choose_risk_models(resources)
    weather_values = None
    weather_status = "not_used"
    if include_weather:
        if manual_weather:
            weather_values = manual_weather
            weather_status = "manual_weather"
        else:
            auto_values, auto_warn = project.try_resolve_weather_values(
                filename=filename,
                manual_weather=None,
                csv_path=resources["weather_csv_path"],
            )
            weather_values = auto_values if auto_values else None
            weather_status = auto_warn if auto_warn else "auto_csv"

    risk = project.route_risk_prediction(
        pred_feats=pred_feats,
        weather_values=weather_values,
        risk_with_weather_model=risk_w,
        risk_no_weather_model=risk_n,
        risk_threshold=project.resolve_risk_threshold(DEFAULT_RISK_PROFILE, None),
        risk_threshold_profile=DEFAULT_RISK_PROFILE,
        sensor=sensor,
    )
    temporal_risk = project.predict_temporal_risk(
        image_filename=filename,
        sensor=sensor,
        pred_feats=pred_feats,
        csv_path=resources["weather_csv_path"],
        temporal_model=resources["risk_pipeline"].get("temporal"),
        risk_threshold=project.resolve_risk_threshold(DEFAULT_RISK_PROFILE, None),
    )
    prediction_eta = project.build_prediction_eta(
        detection_label=risk.get("detection_label"),
        prediction_label=risk.get("prediction_label"),
        temporal_payload=temporal_risk,
    )
    reason, solution = flood_reason_solution(float(pred_feats["pred_flood_ratio"]), float(risk.get("risk_score", 0.0)))

    return {
        "sensor": sensor,
        "sensor_auto": sensor_auto,
        "seg_backend": seg_backend,
        "risk_backend": risk_backend_used,
        "infer_meta": infer_meta,
        "pred_mask": pred_mask,
        "pred_prob": pred_prob,
        "pred_feats": pred_feats,
        "risk": risk,
        "reason": reason,
        "solution": solution,
        "weather_status": weather_status,
        "weather_values": weather_values,
        "temporal_risk": temporal_risk,
        "prediction_eta": prediction_eta,
        "lat": lat,
        "lon": lon,
        "overview": normalize_channel(x_img[..., 0]),
    }


def render_status(resources: dict[str, Any]) -> None:
    st.sidebar.markdown("### Model Status")
    st.sidebar.write(f"S1 PIPELINE: {'OK' if resources['pipeline_seg']['S1'] is not None else 'MISSING'}")
    st.sidebar.write(f"S2 PIPELINE: {'OK' if resources['pipeline_seg']['S2'] is not None else 'MISSING'}")
    st.sidebar.write(
        f"Weather CSV: {'OK' if resources['weather_csv_path'].exists() else 'MISSING'}\n\n`{resources['weather_csv_path']}`"
    )


def app() -> None:
    inject_styles()
    resources = load_resources()
    st.caption("Build: Detection + Prediction split enabled.")

    st.sidebar.markdown("<h1 style='color:#00d4ff;'>HYDROVISION</h1>", unsafe_allow_html=True)
    render_status(resources)

    st.sidebar.markdown("### Target Area Coordinates")
    lat = st.sidebar.number_input("Latitude", value=30.8000, format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=30.9900, format="%.6f")
    mode = st.sidebar.radio(
        "System Modules",
        ["Satellite Radar Analysis", "Manual Sensor Diagnostic", "Geospatial Risk Map"],
    )

    if mode == "Satellite Radar Analysis":
        st.markdown("<h1 class='main-title'>SATELLITE RADAR ANALYSIS</h1>", unsafe_allow_html=True)
        file = st.file_uploader("Upload SAR Image (.tif)", type=["tif", "tiff"])
        c0, c1, c2 = st.columns([1.2, 1.0, 1.0])
        with c0:
            sensor_choice = st.selectbox("Sensor", ["Auto", "S1", "S2"], index=0)
        with c1:
            threshold = st.slider("Mask Threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01)
        with c2:
            include_weather = st.checkbox("Use weather for risk", value=False)

        manual_weather: dict[str, float] | None = None
        if include_weather:
            weather_mode = st.radio("Weather source", ["Auto CSV by filename", "Manual"], horizontal=True)
            if weather_mode == "Manual":
                c1, c2, c3 = st.columns(3)
                temp = c1.number_input("Temperature", value=18.0)
                rain = c2.number_input("tp (precipitation)", value=0.002, format="%.6f")
                runoff = c3.number_input("runoff", value=0.0003, format="%.8f")
                manual_weather = {
                    "Temperature_mean": float(temp),
                    "Temperature_min": float(temp),
                    "Temperature_max": float(temp),
                    "tp_mean": float(rain),
                    "tp_max": float(rain),
                    "tp_sum": float(rain),
                    "runoff_mean": float(runoff),
                    "runoff_max": float(runoff),
                    "runoff_sum": float(runoff),
                    "lat_grid_mean": float(lat),
                    "lon_grid_mean": float(lon),
                }

        if file is not None:
            raw = file.getvalue()
            run = st.button("RUN HYDROVISION AI", use_container_width=True)
            if run:
                try:
                    result = run_satellite_inference(
                        resources=resources,
                        raw=raw,
                        filename=file.name,
                        sensor_choice=sensor_choice,
                        threshold=float(threshold),
                        lat=float(lat),
                        lon=float(lon),
                        include_weather=bool(include_weather),
                        manual_weather=manual_weather,
                    )
                except Exception as ex:
                    st.error(f"Inference failed: {ex}")
                    return

                overlay = build_overlay(result["overview"], result["pred_mask"])
                prob_view = normalize_prob(result["pred_prob"])
                mask_view = (result["pred_mask"] > 0).astype(np.uint8) * 255
                col1, col2, col3 = st.columns(3)
                col1.image(result["overview"], caption="Input Overview", use_container_width=True, clamp=True)
                col2.image(prob_view, caption="Flood Probability Map", use_container_width=True, clamp=True)
                col3.image(mask_view, caption="Detected Flood Mask (Current Scene)", use_container_width=True, clamp=True)
                st.image(overlay, caption="Flood Overlay", use_container_width=True, clamp=True)

                pred_feats = result["pred_feats"]
                risk = result["risk"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Flood Ratio %", f"{float(pred_feats['pred_flood_ratio']) * 100:.2f}%")
                c2.metric("Risk Probability %", f"{float(risk.get('risk_score', 0.0)) * 100:.2f}%")
                c3.metric("Risk Label", str(risk.get("risk_label", "N/A")))
                c4.metric("Sensor", result["sensor"])
                det_label = risk.get("detection_label")
                det_text = "Flood Detected" if det_label == 1 else ("No Flood" if det_label == 0 else "Unknown")
                pred_label = risk.get("prediction_label")
                pred_status = str(risk.get("prediction_status", "unavailable"))
                if pred_status == "suppressed_due_to_detection":
                    pred_text = "N/A (Flood Already Detected)"
                else:
                    pred_text = "Flood Risk Predicted" if pred_label == 1 else ("No Flood Risk Predicted" if pred_label == 0 else "Unknown")
                c5, c6, c7 = st.columns(3)
                c5.metric("Backend", str(result.get("seg_backend", project.PIPELINE_V3_BACKEND_ID)).upper())
                c6.metric("Detection", det_text)
                pred_card_label = "Forecast" if pred_status == "suppressed_due_to_detection" else "Prediction"
                c7.metric(pred_card_label, pred_text)
                st.caption(
                    f"Backend seg={result['seg_backend']} risk={result['risk_backend']} | "
                    f"Risk path={risk.get('risk_model_used', 'none')} | Weather={result['weather_status']}"
                )
                if pred_status == "suppressed_due_to_detection":
                    st.caption(f"Decision split: Detection={det_text} | Forecast disabled (flood already detected).")
                else:
                    st.caption(f"Decision split: Detection={det_text} | Prediction={pred_text}")
                eta = result.get("prediction_eta", {}) if isinstance(result.get("prediction_eta"), dict) else {}
                temporal_payload = result.get("temporal_risk", {}) if isinstance(result.get("temporal_risk"), dict) else {}
                temporal_status = temporal_payload.get("temporal_status")
                if temporal_status not in (None, "ok"):
                    st.caption(temporal_status_label(str(temporal_status)))
                if det_label == 1:
                    st.caption("ETA: Not applicable because flood is already detected.")
                elif eta.get("prediction_eta_start_utc") and eta.get("prediction_eta_end_utc"):
                    st.caption(f"ETA window: {eta.get('prediction_eta_start_utc')} -> {eta.get('prediction_eta_end_utc')}")
                elif eta.get("prediction_eta_text"):
                    st.caption(f"ETA: {eta_text_label(str(eta.get('prediction_eta_text')))}")

                box_cls = "danger-zone" if float(pred_feats["pred_flood_ratio"]) > 0.10 else "success-zone"
                title = "FLOOD INDICATION" if box_cls == "danger-zone" else "NO FLOOD"
                st.markdown(
                    f"<div class='{box_cls}'><h3>{title}</h3>"
                    f"<p><b>Reason:</b> {result['reason']}</p>"
                    f"<p><b>Solution:</b> {result['solution']}</p></div>",
                    unsafe_allow_html=True,
                )

    elif mode == "Manual Sensor Diagnostic":
        st.markdown("<h1 class='main-title'>MANUAL HYDROLOGICAL DIAGNOSTIC</h1>", unsafe_allow_html=True)
        st.markdown("<div class='report-box'>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        temp = c1.number_input("Temperature", value=18.0)
        rain = c2.number_input("Precipitation (tp)", value=0.002, format="%.6f")
        runoff = c3.number_input("Surface Runoff", value=0.0003, format="%.8f")
        c4, c5, c6 = st.columns(3)
        flood_ratio = c4.slider("Estimated Flood Ratio", min_value=0.0, max_value=1.0, value=0.08, step=0.01)
        prob_mean = c5.slider("Estimated Mean Probability", min_value=0.0, max_value=1.0, value=0.12, step=0.01)
        prob_p90 = c6.slider("Estimated P90 Probability", min_value=0.0, max_value=1.0, value=0.20, step=0.01)

        if st.button("RUN HYDROVISION AI", use_container_width=True):
            pred_feats = {
                "pred_flood_ratio": float(flood_ratio),
                "pred_prob_mean": float(prob_mean),
                "pred_prob_p90": float(prob_p90),
            }
            weather_values = {
                "Temperature_mean": float(temp),
                "Temperature_min": float(temp),
                "Temperature_max": float(temp),
                "tp_mean": float(rain),
                "tp_max": float(rain),
                "tp_sum": float(rain),
                "runoff_mean": float(runoff),
                "runoff_max": float(runoff),
                "runoff_sum": float(runoff),
                "lat_grid_mean": float(lat),
                "lon_grid_mean": float(lon),
            }
            risk_w, risk_n, risk_backend = choose_risk_models(resources)
            risk = project.route_risk_prediction(
                pred_feats=pred_feats,
                weather_values=weather_values,
                risk_with_weather_model=risk_w,
                risk_no_weather_model=risk_n,
                risk_threshold=project.resolve_risk_threshold(DEFAULT_RISK_PROFILE, None),
                risk_threshold_profile=DEFAULT_RISK_PROFILE,
                sensor=None,
            )
            reason, solution = flood_reason_solution(runoff=float(runoff), prob=float(risk.get("risk_score", 0.0)))
            st.info(f"Coordinates: {lat}, {lon}")
            if float(risk.get("risk_score", 0.0)) > 0.5:
                st.markdown(
                    f"<div class='danger-zone'><h3>FLOOD RISK ({float(risk.get('risk_score', 0.0)):.2%})</h3>"
                    f"<p><b>Reason:</b> {reason}</p>"
                    f"<p><b>Solution:</b> {solution}</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='success-zone'><h3>SAFE ({float(risk.get('risk_score', 0.0)):.2%})</h3>"
                    f"<p><b>Reason:</b> {reason}</p>"
                    f"<p><b>Solution:</b> {solution}</p></div>",
                    unsafe_allow_html=True,
                )
            st.caption(f"Risk backend: {risk_backend} | path: {risk.get('risk_model_used', 'none')}")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("<h1 class='main-title'>GEOSPATIAL FLOOD RISK MAP</h1>", unsafe_allow_html=True)
        if folium is None or st_folium is None:
            st.warning("Map dependencies are missing. Install: pip install folium streamlit-folium")
            return
        weather_df = resources["weather_df"]
        if weather_df.empty:
            st.warning(f"No weather data loaded from: {resources['weather_csv_path']}")
            return
        m = folium.Map(location=[lat, lon], zoom_start=8, control_scale=True)
        df = weather_df.head(250).copy()
        runoff_ref = float(df["runoff_mean"].mean()) if "runoff_mean" in df.columns else 0.0
        np.random.seed(42)
        for _, row in df.iterrows():
            if "lat_grid_mean" in row and "lon_grid_mean" in row and pd.notna(row["lat_grid_mean"]) and pd.notna(row["lon_grid_mean"]):
                lat_i = float(row["lat_grid_mean"])
                lon_i = float(row["lon_grid_mean"])
            else:
                lat_i = float(lat + np.random.uniform(-0.06, 0.06))
                lon_i = float(lon + np.random.uniform(-0.06, 0.06))
            runoff_v = float(row.get("runoff_mean", 0.0))
            tp_v = float(row.get("tp_sum", 0.0))
            score = runoff_v * 1e4 + tp_v * 10.0
            if runoff_v > runoff_ref * 1.2 or score > np.percentile(df.get("tp_sum", pd.Series([0.0])), 75):
                color, status, rad = "#ff3131", "HIGH RISK", 10
            elif runoff_v > runoff_ref * 0.8:
                color, status, rad = "#f5a524", "MEDIUM RISK", 8
            else:
                color, status, rad = "#39ff14", "LOW RISK", 6
            folium.CircleMarker(
                [lat_i, lon_i],
                radius=rad,
                color=color,
                fill=True,
                fill_opacity=0.75,
                popup=(
                    f"<b>Status:</b> {status}<br>"
                    f"<b>Filename:</b> {row.get('filename', 'N/A')}<br>"
                    f"<b>Runoff mean:</b> {runoff_v:.8f}<br>"
                    f"<b>tp sum:</b> {tp_v:.8f}"
                ),
            ).add_to(m)
        st_folium(m, width=1200, height=600)

    st.sidebar.caption("HYDROVISION | Connected to existing project models")


if __name__ == "__main__":
    app()
