from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

try:
    from env_utils import load_dotenv, resolve_env_path
except Exception:
    def load_dotenv(path: Path, *, override: bool = False) -> dict[str, str]:
        _ = (path, override)
        return {}

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


OUTPUT_DIR_ENV_VAR = "FLOOD_OUTPUT_DIR"
DEFAULT_OUTPUT_DIR_FALLBACK = Path("outputs")


def ensure_streamlit() -> None:
    required_imports = [
        "numpy",
        "pandas",
        "sklearn",
        "joblib",
        "tifffile",
        "matplotlib",
        "streamlit",
        "pyproj",
    ]
    pipeline_required_imports = [
        "torch",
        "torchvision",
        "albumentations",
        "cv2",
        "segmentation_models_pytorch",
        "timm",
        "scipy",
    ]
    missing_core = [
        name for name in required_imports if importlib.util.find_spec(name) is None
    ]
    missing_pipeline = [
        name for name in pipeline_required_imports if importlib.util.find_spec(name) is None
    ]
    if not missing_core and not missing_pipeline:
        return

    if missing_core:
        req_file = Path(__file__).with_name("requirements.txt")
        if req_file.exists():
            print(
                f"[setup] missing core packages ({', '.join(missing_core)}). "
                f"Installing from {req_file.name} ..."
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
            )
        else:
            print(
                f"[setup] missing core packages ({', '.join(missing_core)}). Installing minimal set ..."
            )
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "joblib",
                    "tifffile",
                    "matplotlib",
                    "streamlit",
                    "pyproj",
                ]
            )

    if missing_pipeline:
        pipeline_req_file = Path(__file__).with_name("requirements-pipeline.txt")
        if pipeline_req_file.exists():
            print(
                f"[setup] missing Pipeline V3 packages ({', '.join(missing_pipeline)}). "
                f"Installing from {pipeline_req_file.name} ..."
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", str(pipeline_req_file)]
            )
        else:
            print(
                f"[setup] missing Pipeline V3 packages ({', '.join(missing_pipeline)}). Installing minimal Pipeline V3 set ..."
            )
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "albumentations",
                    "opencv-python-headless",
                    "segmentation-models-pytorch",
                    "timm",
                    "scipy",
                ]
            )


def preflight_project_files(base_dir: Path) -> None:
    required_files = [
        base_dir / "project.py",
        base_dir / "app.py",
        base_dir / "segmentation_pipeline.py",
        base_dir / "env_utils.py",
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required project files are missing. Please keep all core files together. Missing: "
            + ", ".join(missing)
        )


def _has_model_artifacts(folder: Path) -> bool:
    required_any = [
        folder / "unet_model_s1.pth",
        folder / "unet_model_s2.pth",
        folder / "risk_model_no_weather_global_unet.joblib",
        folder / "risk_model_with_weather_s1_unet.joblib",
        folder / "risk_model_temporal_gb_s1_unet.joblib",
    ]
    return any(path.exists() for path in required_any)


def resolve_launcher_output_dir(base_dir: Path) -> tuple[Path, str]:
    configured = resolve_env_path(
        OUTPUT_DIR_ENV_VAR,
        base_dir=base_dir,
        default_relative=DEFAULT_OUTPUT_DIR_FALLBACK,
    ).resolve()
    if _has_model_artifacts(configured):
        return configured, "configured"

    candidates = sorted(
        [
            path.resolve()
            for path in base_dir.iterdir()
            if path.is_dir() and path.name.lower().startswith("outputs")
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if _has_model_artifacts(candidate):
            return candidate, "auto_latest_artifacts"
    return configured, "configured_missing_artifacts"


def _load_json_if_exists(path: Path) -> dict | None:
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


def _format_algorithm_name(raw_name: str | None) -> str:
    name = str(raw_name or "").strip().lower()
    if not name:
        return "Unknown"
    simple_map = {
        "small_unet": "Small Pipeline V3",
        "fcn_resnet50": "FCN ResNet50",
        "deeplabv3_resnet50": "DeepLabV3 ResNet50",
        "maskrcnn_resnet50": "Mask R-CNN ResNet50",
        "segformer_b0": "SegFormer B0",
        "segformer_b2": "SegFormer B2",
        "adaboost": "AdaBoost",
        "gradient_boosting": "Gradient Boosting",
        "hist_gradient_boosting": "Hist Gradient Boosting",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "logistic_regression": "Logistic Regression",
        "gaussian_nb": "Gaussian NB",
        "knn": "KNN",
        "mlp": "MLP",
        "svm_rbf": "SVM RBF",
        "lstm": "LSTM",
        "gru": "GRU",
        "bilstm": "BiLSTM",
        "tcn": "TCN",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
    }
    if name in simple_map:
        return simple_map[name]
    if name.startswith("smp_unet_"):
        encoder = name.removeprefix("smp_unet_").replace("-", " ").replace("_", " ")
        formatted = encoder.title()
        formatted = formatted.replace("Resnet", "ResNet")
        formatted = formatted.replace("Efficientnet", "EfficientNet")
        formatted = formatted.replace("Inceptionresnetv2", "InceptionResNetV2")
        formatted = formatted.replace("Vgg", "VGG")
        return f"SMP Pipeline V3 {formatted}"
    if name.startswith("smp_deeplabv3plus_"):
        encoder = (
            name.removeprefix("smp_deeplabv3plus_")
            .replace("-", " ")
            .replace("_", " ")
        )
        formatted = encoder.title()
        formatted = formatted.replace("Resnet", "ResNet")
        formatted = formatted.replace("Efficientnet", "EfficientNet")
        formatted = formatted.replace("Inceptionresnetv2", "InceptionResNetV2")
        formatted = formatted.replace("Vgg", "VGG")
        return f"SMP DeepLabV3+ {formatted}"
    return name.replace("_", " ").replace("-", " ").title()


def _detect_segmentation_algorithm(output_dir: Path) -> str:
    report = _load_json_if_exists(output_dir / "unet_train_report.json") or {}
    config = report.get("config", {}) if isinstance(report, dict) else {}
    model_kind = config.get("model_kind") if isinstance(config, dict) else None
    if model_kind:
        return _format_algorithm_name(str(model_kind))
    return "Pipeline V3"


def _describe_estimator_algorithm(model: object) -> str:
    if isinstance(model, dict):
        model_name = model.get("model_name")
        if model_name:
            return _format_algorithm_name(str(model_name))
        inner = model.get("model")
        if inner is not None:
            return _describe_estimator_algorithm(inner)
        return "Unknown"

    class_name = type(model).__name__
    if class_name == "CalibratedClassifierCV":
        base = getattr(model, "estimator", None) or getattr(model, "base_estimator", None)
        base_name = _describe_estimator_algorithm(base) if base is not None else "Classifier"
        method = str(getattr(model, "method", "") or "").strip()
        if method:
            return f"{base_name} + {_format_algorithm_name(method)} Calibration"
        return f"{base_name} + Calibration"
    if class_name == "Pipeline":
        steps = getattr(model, "steps", None) or []
        if steps:
            return _describe_estimator_algorithm(steps[-1][1])
        return "Pipeline"

    direct_map = {
        "LogisticRegression": "Logistic Regression",
        "AdaBoostClassifier": "AdaBoost",
        "GradientBoostingClassifier": "Gradient Boosting",
        "HistGradientBoostingClassifier": "Hist Gradient Boosting",
        "RandomForestClassifier": "Random Forest",
        "ExtraTreesClassifier": "Extra Trees",
        "GaussianNB": "Gaussian NB",
        "KNeighborsClassifier": "KNN",
        "MLPClassifier": "MLP",
        "SVC": "SVM RBF",
        "SGDClassifier": "SGD Classifier",
    }
    if class_name in direct_map:
        return direct_map[class_name]
    return class_name.replace("_", " ").strip() or "Unknown"


def _detect_joblib_algorithm(path: Path, *, default_name: str = "Unknown") -> str:
    if not path.exists():
        return default_name
    try:
        import joblib

        payload = joblib.load(path)
    except Exception:
        return default_name
    return _describe_estimator_algorithm(payload)


def preflight_models(output_dir: Path) -> dict[str, bool]:
    paths = {
        "seg_pipeline_s1": output_dir / "unet_model_s1.pth",
        "seg_pipeline_s2": output_dir / "unet_model_s2.pth",
        "risk_with_weather_pipeline": output_dir / "risk_model_with_weather_s1_unet.joblib",
        "risk_no_weather_pipeline": output_dir / "risk_model_no_weather_global_unet.joblib",
        "risk_temporal_pipeline": output_dir / "risk_model_temporal_gb_s1_unet.joblib",
    }
    status = {k: v.exists() for k, v in paths.items()}
    seg_algo = _detect_segmentation_algorithm(output_dir)
    risk_with_weather_algo = _detect_joblib_algorithm(
        paths["risk_with_weather_pipeline"], default_name="Logistic Regression"
    )
    risk_no_weather_algo = _detect_joblib_algorithm(
        paths["risk_no_weather_pipeline"], default_name="Logistic Regression"
    )
    risk_temporal_algo = _detect_joblib_algorithm(
        paths["risk_temporal_pipeline"], default_name="Temporal Model"
    )
    runtime_rows = [
        ("S1 segmentation pipeline", seg_algo, status["seg_pipeline_s1"]),
        ("S2 segmentation pipeline", seg_algo, status["seg_pipeline_s2"]),
        ("Risk with weather", risk_with_weather_algo, status["risk_with_weather_pipeline"]),
        ("Risk fallback", risk_no_weather_algo, status["risk_no_weather_pipeline"]),
        ("Temporal risk", risk_temporal_algo, status["risk_temporal_pipeline"]),
    ]

    print("[preflight] model status:")
    print(f"  - artifacts_dir: {output_dir}")
    print("  - active_runtime:")
    for label, algo, exists in runtime_rows:
        print(f"    * {label} ({algo}): {'OK' if exists else 'MISSING'}")

    has_pipeline_seg = status["seg_pipeline_s1"] or status["seg_pipeline_s2"]
    has_with_weather = status["risk_with_weather_pipeline"]
    has_no_weather = status["risk_no_weather_pipeline"]

    if not has_pipeline_seg:
        cmd = (
            "python project.py train-pipeline --data-roots \"dataset\" "
            "--csv-path \"dataset/Final_Full_Data_Matched.csv\" "
            f"--output-dir \"{output_dir}\""
        )
        print("[preflight] Pipeline V3 segmentation models missing.")
        print(f"[preflight] run this command first:\n  {cmd}")
    elif not has_with_weather:
        print("[preflight] warning: with-weather risk model missing (fallback will still work if no-weather model exists).")

    if not has_no_weather:
        print("[preflight] warning: no-weather fallback risk model missing.")
    if not status["risk_temporal_pipeline"]:
        print("[preflight] warning: temporal risk model missing (short-term forecast will be unavailable).")

    return status


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    if port < 1 or port > 65535:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def pick_available_port(preferred_port: int, search_range: int = 50) -> int:
    max_steps = max(0, int(search_range))
    for step in range(max_steps + 1):
        candidate = preferred_port + step
        if candidate > 65535:
            break
        if is_port_available(candidate):
            return candidate
    raise RuntimeError(
        f"No free port found in range {preferred_port}..{min(65535, preferred_port + max_steps)}. "
        "Use --port with another value."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command launcher for Flood Predictor GUI.")
    parser.add_argument("--check", action="store_true", help="Validate setup without starting server.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--port-search-range", type=int, default=50, help="How many next ports to probe if selected port is busy.")
    parser.add_argument("--env-file", default=".env", help="Path to dotenv file (default: .env).")
    args = parser.parse_args()

    app_path = Path(__file__).with_name("app.py")
    if not app_path.exists():
        raise FileNotFoundError(f"app.py not found: {app_path}")

    env_path = Path(args.env_file)
    loaded = load_dotenv(env_path, override=True)
    if loaded:
        keys = ", ".join(sorted(loaded.keys()))
        print(f"[env] loaded from {env_path}: {keys}")
    else:
        print(f"[env] no dotenv values loaded from {env_path} (file missing or empty).")

    weather_csv = os.getenv("WEATHER_CSV_PATH", "").strip()
    if weather_csv:
        print(f"[env] WEATHER_CSV_PATH={weather_csv}")

    ensure_streamlit()
    base_dir = Path(__file__).parent.resolve()
    preflight_project_files(base_dir)
    output_dir, output_dir_reason = resolve_launcher_output_dir(base_dir)
    os.environ[OUTPUT_DIR_ENV_VAR] = str(output_dir)
    print(f"[env] {OUTPUT_DIR_ENV_VAR}={output_dir} ({output_dir_reason})")
    preflight_models(output_dir)

    if args.check:
        print("[ok] GUI prerequisites are ready.")
        print(f"[ok] App path: {app_path}")
        return

    selected_port = pick_available_port(args.port, search_range=args.port_search_range)
    if selected_port != args.port:
        print(f"[run] requested port {args.port} is busy, using {selected_port} instead.")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(selected_port),
    ]
    print(f"[run] app file: {app_path}")
    print(f"[run] starting GUI on http://localhost:{selected_port} ...")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
