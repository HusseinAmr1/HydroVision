from __future__ import annotations

import copy
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import project

# Avoid noisy online version-check warnings in offline/slow-network training runs.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

try:
    import albumentations as A
except Exception:  # pragma: no cover - optional dependency
    A = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None
    DataLoader = object
    Dataset = object

try:
    from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
except Exception:  # pragma: no cover - optional dependency
    deeplabv3_resnet50 = None
    fcn_resnet50 = None

try:
    from torchvision.models.detection import maskrcnn_resnet50_fpn
except Exception:  # pragma: no cover - optional dependency
    maskrcnn_resnet50_fpn = None

try:
    import segmentation_models_pytorch as smp
except Exception:  # pragma: no cover - optional dependency
    smp = None

try:
    import timm
except Exception:  # pragma: no cover - optional dependency
    timm = None

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - optional dependency
    ndi = None


PIPELINE_MODEL_S1_NAME = "unet_model_s1.pth"
PIPELINE_MODEL_S2_NAME = "unet_model_s2.pth"
PIPELINE_VAL_S1_NAME = "unet_val_metrics_s1.json"
PIPELINE_VAL_S2_NAME = "unet_val_metrics_s2.json"
PIPELINE_VAL_GLOBAL_NAME = "unet_val_metrics_global.json"
PIPELINE_TRAIN_REPORT_NAME = "unet_train_report.json"
PIPELINE_TRAIN_PROGRESS_NAME = "unet_train_progress.json"
PIPELINE_PERIODIC_CHECKPOINT_DIR = "checkpoints"

RISK_WITH_WEATHER_PIPELINE_NAME = "risk_model_with_weather_s1_unet.joblib"
RISK_NO_WEATHER_PIPELINE_NAME = "risk_model_no_weather_global_unet.joblib"
RISK_WITH_WEATHER_PIPELINE_METRICS = "risk_with_weather_cv_metrics_unet.json"
RISK_NO_WEATHER_PIPELINE_METRICS = "risk_no_weather_cv_metrics_unet.json"
RISK_WITH_WEATHER_PIPELINE_TABLE = "risk_with_weather_training_table_unet.csv"
RISK_NO_WEATHER_PIPELINE_TABLE = "risk_no_weather_training_table_unet.csv"
SEGMENTATION_MASK_FILTER_REPORT_NAME = "segmentation_mask_filter_report.json"
SEGMENTATION_MASK_SYNC_POLICY_CHOICES = ("strict", "event-window", "all")
SEGMENTATION_SOURCE_GROUPS_BY_POLICY: dict[str, tuple[str, ...] | None] = {
    # Strongest label assumption: image date and mask date are the same source pair.
    "strict": ("original",),
    # Useful when the downloaded image is intentionally close to the disaster window.
    "event-window": ("original", "flood_event_windows_s1", "flood_event_windows_s2"),
    # Legacy behavior: every discovered pair can train segmentation.
    "all": None,
}

SMP_MODEL_KINDS: tuple[str, ...] = (
    "smp_unet_efficientnet-b0",
    "smp_unet_efficientnet-b2",
    "smp_unet_inceptionresnetv2",
    "smp_unet_resnet18",
    "smp_unet_resnet34",
    "smp_unet_resnet50",
    "smp_unet_resnet101",
    "smp_unet_vgg16",
)
SMP_DEEPLABV3PLUS_MODEL_KINDS: tuple[str, ...] = (
    "smp_deeplabv3plus_resnet18",
    "smp_deeplabv3plus_resnet34",
    "smp_deeplabv3plus_resnet50",
)
SEGFORMER_MODEL_KINDS: tuple[str, ...] = (
    "segformer_b0",
    "segformer_b2",
)
MASKRCNN_MODEL_KINDS: tuple[str, ...] = ("maskrcnn_resnet50",)
MODEL_KIND_CHOICES = (
    "small_unet",
    "fcn_resnet50",
    "deeplabv3_resnet50",
    *MASKRCNN_MODEL_KINDS,
    *SEGFORMER_MODEL_KINDS,
    *SMP_DEEPLABV3PLUS_MODEL_KINDS,
    *SMP_MODEL_KINDS,
)
BN_SENSITIVE_MODEL_KINDS = tuple(kind for kind in MODEL_KIND_CHOICES if kind != "small_unet")


def _env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    raw = str(os.getenv(name, "")).strip()
    try:
        value = int(raw) if raw else int(default)
    except Exception:
        value = int(default)
    if min_value is not None:
        value = max(int(min_value), int(value))
    return int(value)


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def _env_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    raw = str(os.getenv(name, "")).strip()
    try:
        value = float(raw) if raw else float(default)
    except Exception:
        value = float(default)
    if min_value is not None:
        value = max(float(min_value), float(value))
    if max_value is not None:
        value = min(float(max_value), float(value))
    return float(value)


def _resolve_loader_perf(*, device: Any) -> dict[str, Any]:
    cpu_count = int(os.cpu_count() or 4)
    is_windows = bool(os.name == "nt")
    is_cuda = bool(getattr(device, "type", "cpu") == "cuda")
    # Windows + CUDA with many worker processes can hit WinError 1455
    # (paging file too small) while importing torch/cublas in child workers.
    default_workers = 0 if (is_windows and is_cuda) else max(0, cpu_count - 1)
    workers = _env_int("PIPELINE_DATALOADER_WORKERS", default_workers, min_value=0)
    pin_memory_default = bool(is_cuda and workers > 0)
    pin_memory = _env_flag("PIPELINE_PIN_MEMORY", pin_memory_default)
    persistent_workers_default = bool(workers > 0 and not (is_windows and is_cuda))
    persistent_workers = _env_flag("PIPELINE_PERSISTENT_WORKERS", persistent_workers_default)
    prefetch_default = 2 if (is_windows and is_cuda) else 4
    prefetch_factor = _env_int("PIPELINE_PREFETCH_FACTOR", prefetch_default, min_value=2)
    return {
        "num_workers": int(workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers and workers > 0),
        "prefetch_factor": int(prefetch_factor),
    }


def _resolve_bn_sensitive_batch_cap() -> int:
    # Some encoder-heavy models are unstable at batch=1.
    # Keep a configurable lower bound while allowing higher VRAM usage when requested.
    return int(_env_int("PIPELINE_BN_SENSITIVE_BATCH_CAP", 2, min_value=2))


def _resolve_periodic_checkpoint_every() -> int:
    return int(_env_int("PIPELINE_SAVE_EVERY_EPOCHS", 3, min_value=0))


def _resolve_patch_sampling_config() -> dict[str, Any]:
    # Make patch sampling flood-aware by default to reduce severe class imbalance
    # in segmentation batches while keeping configurability via env vars.
    min_positive_ratio = _env_float(
        "PIPELINE_PATCH_MIN_POSITIVE_RATIO",
        0.50,
        min_value=0.0,
        max_value=1.0,
    )
    min_positive_patches = _env_int(
        "PIPELINE_PATCH_MIN_POSITIVE_PATCHES",
        1,
        min_value=0,
    )
    medium_positive_threshold = _env_float(
        "PIPELINE_PATCH_MEDIUM_POSITIVE_THRESHOLD",
        0.01,
        min_value=0.0,
        max_value=0.50,
    )
    strong_positive_threshold = _env_float(
        "PIPELINE_PATCH_STRONG_POSITIVE_THRESHOLD",
        0.05,
        min_value=0.0,
        max_value=0.80,
    )
    if strong_positive_threshold < medium_positive_threshold:
        strong_positive_threshold = medium_positive_threshold
    hard_negative_dilate = _env_int(
        "PIPELINE_PATCH_HARD_NEGATIVE_DILATE",
        24,
        min_value=0,
    )
    hard_negative_ratio = _env_float(
        "PIPELINE_PATCH_HARD_NEGATIVE_RATIO",
        0.70,
        min_value=0.0,
        max_value=1.0,
    )
    return {
        "min_positive_ratio": float(min_positive_ratio),
        "min_positive_patches": int(min_positive_patches),
        "medium_positive_threshold": float(medium_positive_threshold),
        "strong_positive_threshold": float(strong_positive_threshold),
        "hard_negative_dilate": int(hard_negative_dilate),
        "hard_negative_ratio": float(hard_negative_ratio),
    }


def _resolve_input_norm_config() -> dict[str, Any]:
    enabled = _env_flag("PIPELINE_INPUT_NORMALIZATION", True)
    clip_low_pct = _env_float(
        "PIPELINE_NORM_CLIP_LOW_PCT",
        2.0,
        min_value=0.0,
        max_value=25.0,
    )
    clip_high_pct = _env_float(
        "PIPELINE_NORM_CLIP_HIGH_PCT",
        98.0,
        min_value=75.0,
        max_value=100.0,
    )
    if clip_high_pct <= clip_low_pct:
        clip_low_pct = 2.0
        clip_high_pct = 98.0
    max_pixels_per_image = _env_int(
        "PIPELINE_NORM_MAX_PIXELS_PER_IMAGE",
        12000,
        min_value=512,
    )
    max_total_pixels = _env_int(
        "PIPELINE_NORM_MAX_TOTAL_PIXELS",
        400000,
        min_value=4096,
    )
    return {
        "enabled": bool(enabled),
        "clip_low_pct": float(clip_low_pct),
        "clip_high_pct": float(clip_high_pct),
        "max_pixels_per_image": int(max_pixels_per_image),
        "max_total_pixels": int(max_total_pixels),
    }


def _resolve_selection_config() -> dict[str, Any]:
    metric = str(os.getenv("PIPELINE_SELECTION_METRIC", "f1")).strip().lower()
    if metric not in {"f1", "iou"}:
        metric = "f1"
    image_eval_every = _env_int("PIPELINE_IMAGE_VAL_EVERY_EPOCHS", 1, min_value=1)
    return {
        "metric": metric,
        "image_eval_every": int(image_eval_every),
    }


def _write_train_progress(output_dir: Path, payload: dict[str, Any]) -> None:
    try:
        # Progress files are rewritten very frequently, so keep them compact to
        # reduce disk churn and monitor read contention.
        project.save_json(output_dir / PIPELINE_TRAIN_PROGRESS_NAME, payload, compact=True)
    except Exception:
        # Progress tracking is best-effort and must not break training.
        pass


def dependencies_available() -> tuple[bool, list[str]]:
    missing: list[str] = []
    if torch is None:
        missing.append("torch")
    if A is None:
        missing.append("albumentations")
    return len(missing) == 0, missing


def ensure_dependencies() -> None:
    ok, missing = dependencies_available()
    if not ok:
        raise RuntimeError(
            "Pipeline V3 dependencies are missing. Install in current Python environment (Python 3.13): "
            "pip install torch torchvision albumentations opencv-python-headless"
            f" (missing: {', '.join(missing)})"
        )


def ensure_segmentation_model_support(model_kind: str) -> None:
    ensure_dependencies()
    if model_kind in {"fcn_resnet50", "deeplabv3_resnet50"}:
        if deeplabv3_resnet50 is None or fcn_resnet50 is None:
            raise RuntimeError(
                "torchvision segmentation models are unavailable in this environment. "
                "Install/upgrade torchvision to use ResNet-based segmentation backbones."
            )
    if (
        model_kind.startswith("smp_unet_")
        or model_kind in SMP_DEEPLABV3PLUS_MODEL_KINDS
        or model_kind in SEGFORMER_MODEL_KINDS
    ):
        if smp is None:
            raise RuntimeError(
                "segmentation_models_pytorch is required for this model_kind. "
                "Install with: pip install segmentation-models-pytorch timm"
            )
        if timm is None:
            raise RuntimeError(
                "timm is required for SMP-based model kinds in this project. "
                "Install with: pip install timm"
            )
    if model_kind in MASKRCNN_MODEL_KINDS:
        if maskrcnn_resnet50_fpn is None:
            raise RuntimeError(
                "torchvision detection models are unavailable. "
                "Install/upgrade torchvision to use maskrcnn_resnet50."
            )
        if ndi is None:
            raise RuntimeError(
                "scipy is required for Mask R-CNN instance target extraction. "
                "Install with: pip install scipy"
            )


def normalize_model_kind(model_kind: str | None) -> str:
    kind = str(model_kind or "small_unet").strip().lower()
    if kind not in MODEL_KIND_CHOICES:
        raise ValueError(f"Unsupported model_kind '{model_kind}'. Supported: {', '.join(MODEL_KIND_CHOICES)}")
    return kind


def pipeline_model_path(sensor: str, output_dir: Path) -> Path:
    if sensor == "S1":
        return (output_dir / PIPELINE_MODEL_S1_NAME).resolve()
    if sensor == "S2":
        return (output_dir / PIPELINE_MODEL_S2_NAME).resolve()
    raise ValueError(f"unknown sensor: {sensor}")


def _resolve_pretrained_encoder_flag(model_kind: str) -> bool:
    kind = normalize_model_kind(model_kind)
    if (
        kind not in SMP_MODEL_KINDS
        and kind not in SMP_DEEPLABV3PLUS_MODEL_KINDS
        and kind not in SEGFORMER_MODEL_KINDS
    ):
        return False
    return _env_flag("PIPELINE_USE_PRETRAINED_ENCODER", True)


def _sample_image_pixels_for_stats(
    *,
    image: np.ndarray,
    per_image_limit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    flat = np.asarray(image, dtype=np.float32).reshape(-1, image.shape[-1])
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    if flat.shape[0] <= int(per_image_limit):
        return flat
    idx = rng.choice(flat.shape[0], size=int(per_image_limit), replace=False)
    return flat[idx]


def compute_input_normalization_stats(
    *,
    pairs: list[project.PairRecord],
    sensor_channels: int,
    seed: int,
    progress_callback: Any | None = None,
) -> dict[str, Any] | None:
    cfg = _resolve_input_norm_config()
    if not cfg["enabled"] or not pairs:
        return None

    per_image_limit = max(
        512,
        min(
            int(cfg["max_pixels_per_image"]),
            int(cfg["max_total_pixels"]) // max(1, len(pairs)),
        ),
    )
    rng = np.random.default_rng(int(seed))
    sampled_chunks: list[np.ndarray] = []
    total_pairs = int(len(pairs))
    if callable(progress_callback):
        try:
            progress_callback(done=0, total=total_pairs)
        except Exception:
            pass
    for idx, pair in enumerate(pairs, start=1):
        try:
            image = project.load_image(pair.image_path, sensor_channels)
        except Exception:
            if callable(progress_callback) and (
                idx <= 1 or idx >= total_pairs or idx % 10 == 0
            ):
                try:
                    progress_callback(done=int(idx), total=total_pairs)
                except Exception:
                    pass
            continue
        sampled = _sample_image_pixels_for_stats(
            image=image,
            per_image_limit=per_image_limit,
            rng=rng,
        )
        if sampled.size > 0:
            sampled_chunks.append(sampled)
        if callable(progress_callback) and (
            idx <= 1 or idx >= total_pairs or idx % 10 == 0
        ):
            try:
                progress_callback(done=int(idx), total=total_pairs)
            except Exception:
                pass

    if not sampled_chunks:
        return None

    sampled_all = np.concatenate(sampled_chunks, axis=0)
    low = np.percentile(sampled_all, float(cfg["clip_low_pct"]), axis=0).astype(np.float32)
    high = np.percentile(sampled_all, float(cfg["clip_high_pct"]), axis=0).astype(np.float32)
    high = np.maximum(high, low + 1e-6).astype(np.float32)
    clipped = np.clip(sampled_all, low, high)
    mean = clipped.mean(axis=0).astype(np.float32)
    std = clipped.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return {
        "enabled": True,
        "clip_low_pct": float(cfg["clip_low_pct"]),
        "clip_high_pct": float(cfg["clip_high_pct"]),
        "clip_low": [float(x) for x in low],
        "clip_high": [float(x) for x in high],
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "pixels_sampled": int(sampled_all.shape[0]),
        "pairs_sampled": int(len(sampled_chunks)),
    }


def normalize_input_image(
    image: np.ndarray,
    normalization_stats: dict[str, Any] | None,
) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(image), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not normalization_stats or not normalization_stats.get("enabled", True):
        return arr
    try:
        low = np.asarray(normalization_stats.get("clip_low", []), dtype=np.float32)
        high = np.asarray(normalization_stats.get("clip_high", []), dtype=np.float32)
        mean = np.asarray(normalization_stats.get("mean", []), dtype=np.float32)
        std = np.asarray(normalization_stats.get("std", []), dtype=np.float32)
        channels = int(arr.shape[-1]) if arr.ndim == 3 else 1
        if (
            low.size != channels
            or high.size != channels
            or mean.size != channels
            or std.size != channels
        ):
            return arr
        arr = np.clip(arr, low.reshape(1, 1, -1), high.reshape(1, 1, -1))
        arr = (arr - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        return arr.astype(np.float32)
    except Exception:
        return arr


def _metric_value(metrics: dict[str, Any] | None, key: str) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    try:
        return float(metrics.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def build_train_augment(*, sensor: str | None = None) -> Any:
    ensure_dependencies()
    sensor_key = str(sensor or "").strip().upper()
    aug_rotate_limit = _env_float(
        "PIPELINE_AUG_ROTATE_LIMIT", 45.0, min_value=0.0, max_value=180.0
    )
    aug_scale_min = _env_float(
        "PIPELINE_AUG_SCALE_MIN", 0.80, min_value=0.50, max_value=1.50
    )
    aug_scale_max = _env_float(
        "PIPELINE_AUG_SCALE_MAX", 1.20, min_value=0.50, max_value=1.80
    )
    if aug_scale_min > aug_scale_max:
        aug_scale_min, aug_scale_max = aug_scale_max, aug_scale_min
    aug_translate = _env_float(
        "PIPELINE_AUG_TRANSLATE_PERCENT", 0.08, min_value=0.0, max_value=0.25
    )
    aug_affine_p = _env_float(
        "PIPELINE_AUG_AFFINE_P", 0.75, min_value=0.0, max_value=1.0
    )
    ops: list[Any] = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={
                "x": (-float(aug_translate), float(aug_translate)),
                "y": (-float(aug_translate), float(aug_translate)),
            },
            # Wider zoom/rotation range improves invariance to orientation/scale shifts.
            scale=(float(aug_scale_min), float(aug_scale_max)),
            rotate=(-float(aug_rotate_limit), float(aug_rotate_limit)),
            border_mode=0,
            p=float(aug_affine_p),
        ),
        # Adds smooth non-rigid shape variation to reduce overfitting to one flood contour style.
        A.ElasticTransform(p=0.2),
    ]
    if sensor_key == "S2":
        # Optical imagery benefits more from intensity perturbation than radar-like S1 inputs.
        ops.append(A.RandomBrightnessContrast(p=0.3))
    ops.append(A.GaussNoise(p=0.2))
    return A.Compose(ops)


def build_eval_augment() -> Any:
    ensure_dependencies()
    return A.Compose([])


def seed_everything(seed: int) -> None:
    ensure_dependencies()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass


def conv_block(in_ch: int, out_ch: int) -> Any:
    ensure_dependencies()
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, in_channels: int) -> None:
        ensure_dependencies()
        super().__init__()
        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: Any) -> Any:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.head(d1)


class ResNetSegWrapper(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, *, in_channels: int, model_kind: str) -> None:
        ensure_segmentation_model_support(model_kind)
        super().__init__()
        if in_channels == 3:
            self.input_adapter = nn.Identity()
        else:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
            )
        if model_kind == "fcn_resnet50":
            self.model = fcn_resnet50(weights=None, weights_backbone=None, num_classes=1, aux_loss=None)
        elif model_kind == "deeplabv3_resnet50":
            self.model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=1, aux_loss=None)
        else:  # pragma: no cover - validated by caller
            raise ValueError(f"Unsupported model_kind for ResNet wrapper: {model_kind}")

    def forward(self, x: Any) -> Any:
        x = self.input_adapter(x)
        out = self.model(x)
        if isinstance(out, dict):
            out = out.get("out")
        return out


class MaskRCNNSegWrapper(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, *, in_channels: int) -> None:
        ensure_segmentation_model_support("maskrcnn_resnet50")
        super().__init__()
        if in_channels == 3:
            self.input_adapter = nn.Identity()
        else:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
            )
        self.model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2)

    def forward(self, x: Any) -> Any:  # pragma: no cover - not used for loss directly
        x = self.input_adapter(x)
        if self.training:
            # Training should call self.model(images, targets) directly.
            return self.model(x)
        return self.model(x)


class EncoderInputAdapterWrapper(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, *, in_channels: int, inner_model: Any) -> None:
        ensure_dependencies()
        super().__init__()
        if int(in_channels) == 3:
            self.input_adapter = nn.Identity()
        else:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(int(in_channels), 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
            )
        self.model = inner_model

    def forward(self, x: Any) -> Any:
        return self.model(self.input_adapter(x))


def build_segmentation_model(
    model_kind: str,
    in_channels: int,
    *,
    use_pretrained_encoder: bool = False,
) -> Any:
    kind = normalize_model_kind(model_kind)
    if kind == "small_unet":
        return SmallUNet(in_channels=in_channels)
    if kind.startswith("smp_unet_"):
        ensure_segmentation_model_support(kind)
        encoder_name = kind.replace("smp_unet_", "", 1)
        try:
            if use_pretrained_encoder:
                base_model = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
                return EncoderInputAdapterWrapper(
                    in_channels=in_channels,
                    inner_model=base_model,
                )
            return smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=in_channels,
                classes=1,
                activation=None,
            )
        except Exception as ex:
            raise RuntimeError(
                f"Failed to build SMP Pipeline V3 with encoder '{encoder_name}'. "
                "Check installed timm/segmentation_models_pytorch versions."
            ) from ex
    if kind in SMP_DEEPLABV3PLUS_MODEL_KINDS:
        ensure_segmentation_model_support(kind)
        encoder_name = kind.replace("smp_deeplabv3plus_", "", 1)
        try:
            if use_pretrained_encoder:
                base_model = smp.DeepLabV3Plus(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
                return EncoderInputAdapterWrapper(
                    in_channels=in_channels,
                    inner_model=base_model,
                )
            return smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=in_channels,
                classes=1,
                activation=None,
            )
        except Exception as ex:
            raise RuntimeError(
                f"Failed to build SMP DeepLabV3Plus with encoder '{encoder_name}'. "
                "Check installed timm/segmentation_models_pytorch versions."
            ) from ex
    if kind in SEGFORMER_MODEL_KINDS:
        ensure_segmentation_model_support(kind)
        encoder_name = "mit_b0" if kind == "segformer_b0" else "mit_b2"
        try:
            # SegFormer-style transformer encoder through SMP FPN head.
            if use_pretrained_encoder:
                base_model = smp.FPN(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
                return EncoderInputAdapterWrapper(
                    in_channels=in_channels,
                    inner_model=base_model,
                )
            return smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=in_channels,
                classes=1,
                activation=None,
            )
        except Exception as ex:
            raise RuntimeError(
                f"Failed to build SegFormer-style model with encoder '{encoder_name}'."
            ) from ex
    if kind in MASKRCNN_MODEL_KINDS:
        return MaskRCNNSegWrapper(in_channels=in_channels)
    return ResNetSegWrapper(in_channels=in_channels, model_kind=kind)


class FloodSegLoss(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(
        self,
        *,
        smooth: float = 1e-6,
        pos_weight: float = 3.0,
        focal_gamma: float = 2.0,
        bce_weight: float = 0.40,
        focal_weight: float = 0.30,
        tversky_weight: float = 0.30,
        fp_penalty: float = 0.65,
        fn_penalty: float = 0.35,
    ) -> None:
        ensure_dependencies()
        super().__init__()
        self.smooth = float(smooth)
        self.focal_gamma = float(focal_gamma)
        self.bce_weight = float(bce_weight)
        self.focal_weight = float(focal_weight)
        self.tversky_weight = float(tversky_weight)
        self.fp_penalty = float(fp_penalty)
        self.fn_penalty = float(fn_penalty)
        self.pos_weight = float(max(1.0, pos_weight))

    def forward(self, logits: Any, target: Any) -> Any:
        pos_weight = torch.tensor(
            self.pos_weight,
            dtype=logits.dtype,
            device=logits.device,
        )
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, probs, 1.0 - probs).clamp_(1e-6, 1.0 - 1e-6)
        focal = ((1.0 - pt) ** self.focal_gamma) * (-torch.log(pt))
        focal = focal.mean()

        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        tp = (probs_flat * target_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - target_flat)).sum(dim=1)
        fn = ((1.0 - probs_flat) * target_flat).sum(dim=1)
        tversky = (tp + self.smooth) / (
            tp + self.fp_penalty * fp + self.fn_penalty * fn + self.smooth
        )
        tversky_loss = 1.0 - tversky.mean()
        return (
            self.bce_weight * bce
            + self.focal_weight * focal
            + self.tversky_weight * tversky_loss
        )


@dataclass(frozen=True)
class PatchRecord:
    pair: project.PairRecord
    y0: int
    x0: int


def _starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, max(1, length - patch_size + 1), max(1, stride)))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _sample_patch_coords(
    h: int,
    w: int,
    *,
    patch_size: int,
    stride: int,
    max_patches: int,
    rng: np.random.Generator,
    mask_binary: np.ndarray | None = None,
    min_positive_ratio: float = 0.0,
    min_positive_patches: int = 0,
    medium_positive_threshold: float = 0.01,
    strong_positive_threshold: float = 0.05,
    hard_negative_dilate: int = 0,
    hard_negative_ratio: float = 0.70,
) -> list[tuple[int, int]]:
    ys = _starts(h, patch_size, stride)
    xs = _starts(w, patch_size, stride)
    coords = [(y, x) for y in ys for x in xs]
    if len(coords) <= max_patches:
        return coords

    # Fallback to previous behavior when mask is unavailable.
    if mask_binary is None:
        idx = rng.choice(len(coords), size=max_patches, replace=False)
        return [coords[int(i)] for i in idx]

    mask = np.asarray(mask_binary)
    if mask.ndim != 2 or mask.shape[0] != h or mask.shape[1] != w:
        idx = rng.choice(len(coords), size=max_patches, replace=False)
        return [coords[int(i)] for i in idx]

    mask_u8 = (mask > 0).astype(np.uint8, copy=False)
    # Summed-area table with one-pixel padding for O(1) patch flood checks.
    sat = np.pad(mask_u8, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)

    near_sat = None
    if ndi is not None and int(hard_negative_dilate) > 0:
        near_mask = ndi.binary_dilation(mask_u8.astype(bool), iterations=int(hard_negative_dilate))
        near_sat = np.pad(near_mask.astype(np.uint8), ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)

    weak_positive: list[tuple[int, int]] = []
    medium_positive: list[tuple[int, int]] = []
    strong_positive: list[tuple[int, int]] = []
    hard_negative: list[tuple[int, int]] = []
    easy_negative: list[tuple[int, int]] = []
    for y0, x0 in coords:
        y1 = min(h, y0 + patch_size)
        x1 = min(w, x0 + patch_size)
        flood_sum = int(sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0])
        if flood_sum > 0:
            patch_area = max(1, int((y1 - y0) * (x1 - x0)))
            flood_ratio = float(flood_sum) / float(patch_area)
            if flood_ratio >= float(strong_positive_threshold):
                strong_positive.append((y0, x0))
            elif flood_ratio >= float(medium_positive_threshold):
                medium_positive.append((y0, x0))
            else:
                weak_positive.append((y0, x0))
        else:
            is_hard_negative = False
            if near_sat is not None:
                near_sum = int(
                    near_sat[y1, x1] - near_sat[y0, x1] - near_sat[y1, x0] + near_sat[y0, x0]
                )
                is_hard_negative = bool(near_sum > 0)
            if is_hard_negative:
                hard_negative.append((y0, x0))
            else:
                easy_negative.append((y0, x0))

    # If no class diversity is available at patch level, keep random sampling.
    positive = weak_positive + medium_positive + strong_positive
    negative = hard_negative + easy_negative
    if not positive or not negative:
        idx = rng.choice(len(coords), size=max_patches, replace=False)
        return [coords[int(i)] for i in idx]

    def _take(pool: list[tuple[int, int]], take_n: int) -> list[tuple[int, int]]:
        if take_n <= 0 or not pool:
            return []
        if take_n >= len(pool):
            return list(pool)
        idx = rng.choice(len(pool), size=int(take_n), replace=False)
        return [pool[int(i)] for i in idx]

    ratio = float(np.clip(float(min_positive_ratio), 0.0, 1.0))
    min_pos = max(0, int(min_positive_patches))
    target_pos = int(round(float(max_patches) * ratio))
    target_pos = max(min_pos, target_pos)
    target_pos = min(target_pos, int(max_patches), len(positive))
    target_neg = int(max_patches) - target_pos
    if target_neg > len(negative):
        target_neg = len(negative)
        target_pos = min(len(positive), int(max_patches) - target_neg)

    chosen: list[tuple[int, int]] = []
    chosen_set: set[tuple[int, int]] = set()

    def _extend_unique(items: list[tuple[int, int]]) -> None:
        for item in items:
            if item in chosen_set:
                continue
            chosen.append(item)
            chosen_set.add(item)

    if target_pos > 0:
        target_strong = int(round(float(target_pos) * 0.45))
        target_medium = int(round(float(target_pos) * 0.35))
        target_weak = int(target_pos) - target_strong - target_medium
        _extend_unique(_take(strong_positive, target_strong))
        _extend_unique(_take(medium_positive, target_medium))
        _extend_unique(_take(weak_positive, target_weak))
        if len(chosen) < target_pos:
            remain_pos = [c for c in positive if c not in chosen_set]
            _extend_unique(_take(remain_pos, int(target_pos) - len(chosen)))
    if target_neg > 0:
        target_hard_neg = int(round(float(target_neg) * float(hard_negative_ratio)))
        _extend_unique(_take(hard_negative, target_hard_neg))
        if len(chosen) < target_pos + target_neg:
            remain_neg = [c for c in negative if c not in chosen_set]
            _extend_unique(_take(remain_neg, int(target_pos + target_neg) - len(chosen)))

    if len(chosen) < max_patches:
        remaining = [c for c in coords if c not in chosen_set]
        need = int(max_patches) - len(chosen)
        if remaining:
            add_idx = rng.choice(len(remaining), size=min(need, len(remaining)), replace=False)
            _extend_unique([remaining[int(i)] for i in add_idx])

    rng.shuffle(chosen)
    return chosen[: int(max_patches)]


def _pad_patch_image(image: np.ndarray, y0: int, x0: int, patch_size: int) -> np.ndarray:
    patch = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
    ph, pw = patch.shape[:2]
    if ph == patch_size and pw == patch_size:
        return patch
    out = np.zeros((patch_size, patch_size, image.shape[-1]), dtype=image.dtype)
    out[:ph, :pw] = patch
    return out


def _pad_patch_mask(mask: np.ndarray, y0: int, x0: int, patch_size: int) -> np.ndarray:
    patch = mask[y0 : y0 + patch_size, x0 : x0 + patch_size]
    ph, pw = patch.shape[:2]
    if ph == patch_size and pw == patch_size:
        return patch
    out = np.zeros((patch_size, patch_size), dtype=mask.dtype)
    out[:ph, :pw] = patch
    return out


class PatchDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        *,
        records: list[PatchRecord],
        sensor_channels: int,
        patch_size: int,
        augment: Any,
        normalization_stats: dict[str, Any] | None = None,
    ) -> None:
        ensure_dependencies()
        self.records = records
        self.sensor_channels = sensor_channels
        self.patch_size = patch_size
        self.augment = augment
        self.normalization_stats = normalization_stats
        self._cache_image_path: Path | None = None
        self._cache_mask_path: Path | None = None
        self._cache_image: np.ndarray | None = None
        self._cache_mask: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _load_pair(self, rec: PatchRecord) -> tuple[np.ndarray, np.ndarray]:
        if self._cache_image_path != rec.pair.image_path:
            self._cache_image = project.load_image(rec.pair.image_path, self.sensor_channels)
            self._cache_image_path = rec.pair.image_path
        if self._cache_mask_path != rec.pair.mask_path:
            mask = project.to_binary_mask(
                project.load_mask(rec.pair.mask_path),
                mask_path=rec.pair.mask_path,
            ).astype(np.float32)
            self._cache_mask = mask
            self._cache_mask_path = rec.pair.mask_path
        assert self._cache_image is not None
        assert self._cache_mask is not None
        return self._cache_image, self._cache_mask

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        rec = self.records[idx]
        image, mask = self._load_pair(rec)
        x_patch = _pad_patch_image(image, rec.y0, rec.x0, self.patch_size)
        y_patch = _pad_patch_mask(mask, rec.y0, rec.x0, self.patch_size)
        x_patch = normalize_input_image(x_patch, self.normalization_stats)
        y_patch = np.nan_to_num(y_patch, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if self.augment is not None:
            aug = self.augment(image=x_patch, mask=y_patch)
            x_patch = aug["image"]
            y_patch = aug["mask"]

        x_tensor = torch.from_numpy(np.moveaxis(x_patch, -1, 0).copy()).float()
        y_tensor = torch.from_numpy(y_patch[None, ...].copy()).float()
        return x_tensor, y_tensor


def build_patch_records(
    *,
    pairs: list[project.PairRecord],
    sensor_channels: int,
    patch_size: int,
    stride: int,
    max_patches_per_image: int,
    seed: int,
    progress_callback: Any | None = None,
) -> tuple[list[PatchRecord], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    patch_sampling_cfg = _resolve_patch_sampling_config()
    records: list[PatchRecord] = []
    issues: list[dict[str, Any]] = []
    total_pairs = int(len(pairs))
    if callable(progress_callback):
        try:
            progress_callback(done=0, total=total_pairs)
        except Exception:
            pass
    for idx, pair in enumerate(pairs, start=1):
        try:
            mask = project.to_binary_mask(
                project.load_mask(pair.mask_path),
                mask_path=pair.mask_path,
            ).astype(np.uint8)
            # Patch sampling only needs spatial dimensions plus the binary flood mask.
            # Avoid loading the full source image here; PatchDataset loads pixels later.
            h, w = mask.shape[:2]
            coords = _sample_patch_coords(
                h,
                w,
                patch_size=patch_size,
                stride=stride,
                max_patches=max(1, int(max_patches_per_image)),
                rng=rng,
                mask_binary=mask,
                min_positive_ratio=float(patch_sampling_cfg["min_positive_ratio"]),
                min_positive_patches=int(patch_sampling_cfg["min_positive_patches"]),
                medium_positive_threshold=float(patch_sampling_cfg["medium_positive_threshold"]),
                strong_positive_threshold=float(patch_sampling_cfg["strong_positive_threshold"]),
                hard_negative_dilate=int(patch_sampling_cfg["hard_negative_dilate"]),
                hard_negative_ratio=float(patch_sampling_cfg["hard_negative_ratio"]),
            )
            for y0, x0 in coords:
                records.append(PatchRecord(pair=pair, y0=int(y0), x0=int(x0)))
        except Exception as ex:
            issues.append(
                project.make_issue(
                    "unet_records",
                    "record_build_failed",
                    sensor=pair.sensor,
                    filename=pair.filename,
                    image_path=pair.image_path,
                    mask_path=pair.mask_path,
                    details=str(ex),
                )
            )
        if callable(progress_callback) and (
            idx <= 1 or idx >= total_pairs or idx % 10 == 0
        ):
            try:
                progress_callback(done=int(idx), total=total_pairs)
            except Exception:
                pass
    records.sort(key=lambda r: (str(r.pair.image_path).lower(), r.y0, r.x0))
    return records, issues


def _normalize_segmentation_balance_policy(value: str | None) -> str:
    candidate = str(value or "none").strip().lower().replace("_", "-")
    if candidate in {"equal", "equal-flood-non-flood", "balanced"}:
        return "equal-flood-non-flood"
    if candidate in {"off", "disabled", "false"}:
        return "none"
    if candidate not in {"none", "equal-flood-non-flood"}:
        return "none"
    return candidate


def _balance_segmentation_train_pairs(
    pairs: list[project.PairRecord],
    *,
    policy: str,
    min_flood_ratio: float,
    seed: int,
) -> tuple[list[project.PairRecord], dict[str, Any]]:
    normalized_policy = _normalize_segmentation_balance_policy(policy)
    threshold = float(np.clip(float(min_flood_ratio), 0.0, 1.0))
    report: dict[str, Any] = {
        "policy": normalized_policy,
        "status": "disabled" if normalized_policy == "none" else "pending",
        "min_flood_ratio": float(threshold),
        "before_total": int(len(pairs)),
        "before_flood": 0,
        "before_non_flood": 0,
        "label_failures": 0,
        "after_total": int(len(pairs)),
        "after_flood": None,
        "after_non_flood": None,
        "excluded_pairs": 0,
        "reason": None,
    }
    if normalized_policy == "none" or not pairs:
        return list(pairs), report

    flood: list[tuple[int, float]] = []
    non_flood: list[tuple[int, float]] = []
    failures: list[tuple[int, str]] = []
    for idx, pair in enumerate(pairs):
        try:
            mask = project.to_binary_mask(
                project.load_mask(pair.mask_path),
                mask_path=pair.mask_path,
            )
            flood_ratio = float(np.mean(mask > 0))
            if flood_ratio >= threshold:
                flood.append((idx, flood_ratio))
            else:
                non_flood.append((idx, flood_ratio))
        except Exception as ex:
            failures.append((idx, str(ex)))

    report["before_flood"] = int(len(flood))
    report["before_non_flood"] = int(len(non_flood))
    report["label_failures"] = int(len(failures))
    if failures:
        report["failure_examples"] = [
            {
                "filename": str(pairs[idx].filename),
                "mask_path": str(pairs[idx].mask_path),
                "error": str(error),
            }
            for idx, error in failures[:5]
        ]

    if not flood or not non_flood:
        report.update(
            {
                "status": "skipped",
                "reason": "single_class_after_mask_labeling",
                "after_total": int(len(pairs)),
                "after_flood": int(len(flood)),
                "after_non_flood": int(len(non_flood)),
            }
        )
        return list(pairs), report

    target_each = min(len(flood), len(non_flood))
    rng = np.random.default_rng(seed)

    def _sample_indices(group: list[tuple[int, float]]) -> list[int]:
        if len(group) <= target_each:
            return [idx for idx, _ in group]
        chosen = rng.choice(len(group), size=target_each, replace=False)
        return [group[int(i)][0] for i in chosen]

    selected_indices = set(_sample_indices(flood) + _sample_indices(non_flood))
    selected_pairs = [
        pairs[i] for i in sorted(selected_indices, key=lambda k: pairs[k].filename)
    ]
    after_flood = sum(1 for idx, _ in flood if idx in selected_indices)
    after_non_flood = sum(1 for idx, _ in non_flood if idx in selected_indices)
    report.update(
        {
            "status": "applied",
            "after_total": int(len(selected_pairs)),
            "after_flood": int(after_flood),
            "after_non_flood": int(after_non_flood),
            "excluded_pairs": int(len(pairs) - len(selected_pairs)),
            "target_each_class": int(target_each),
            "reason": "majority_class_downsampled",
        }
    )
    return selected_pairs, report


def _loader_metrics(
    model: Any,
    loader: Any,
    *,
    threshold: float,
    device: Any,
    progress_callback: Any | None = None,
) -> dict[str, float]:
    model.eval()
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    total_batches = int(len(loader)) if hasattr(loader, "__len__") else 0
    if callable(progress_callback):
        try:
            progress_callback(done=0, total=total_batches)
        except Exception:
            pass
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)
            pred = (prob >= float(threshold)).float()
            y_true_all.append(y.detach().cpu().numpy().reshape(-1).astype(np.uint8))
            y_pred_all.append(pred.detach().cpu().numpy().reshape(-1).astype(np.uint8))
            if callable(progress_callback) and (
                batch_idx <= 1
                or batch_idx >= total_batches
                or (total_batches > 0 and batch_idx % max(1, total_batches // 10) == 0)
            ):
                try:
                    progress_callback(done=int(batch_idx), total=total_batches)
                except Exception:
                    pass
    if not y_true_all:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return project.compute_metrics(y_true, y_pred)


def _run_epoch(
    *,
    model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
    progress_callback: Any | None = None,
) -> float:
    model.train()
    losses: list[float] = []
    total_batches = int(len(loader)) if hasattr(loader, "__len__") else 0
    if callable(progress_callback):
        try:
            progress_callback(done=0, total=total_batches, last_loss=None)
        except Exception:
            pass
    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if callable(progress_callback) and (
            batch_idx <= 1
            or batch_idx >= total_batches
            or (total_batches > 0 and batch_idx % max(1, total_batches // 10) == 0)
        ):
            try:
                progress_callback(
                    done=int(batch_idx),
                    total=total_batches,
                    last_loss=float(loss_value),
                )
            except Exception:
                pass
    if not losses:
        return 0.0
    return float(np.mean(losses))


def _mask_to_instances(mask: np.ndarray, image_idx: int) -> dict[str, Any]:
    # Build instance targets from a binary mask so Mask R-CNN can train on our segmentation labels.
    if ndi is None:
        raise RuntimeError(
            "scipy is required for Mask R-CNN training (_mask_to_instances). "
            "Install with: pip install scipy"
        )
    mask_u8 = project.to_binary_mask(mask).astype(np.uint8)
    h, w = mask_u8.shape
    boxes: list[list[float]] = []
    masks: list[np.ndarray] = []
    labeled, n_comp = ndi.label(mask_u8)
    for comp in range(1, int(n_comp) + 1):
        comp_mask = (labeled == comp).astype(np.uint8)
        ys, xs = np.where(comp_mask > 0)
        if ys.size == 0:
            continue
        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max() + 1)
        y2 = float(ys.max() + 1)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        masks.append(comp_mask)

    if len(boxes) == 0:
        t_boxes = torch.zeros((0, 4), dtype=torch.float32)
        t_labels = torch.zeros((0,), dtype=torch.int64)
        t_masks = torch.zeros((0, h, w), dtype=torch.uint8)
        t_area = torch.zeros((0,), dtype=torch.float32)
        t_iscrowd = torch.zeros((0,), dtype=torch.int64)
    else:
        t_boxes = torch.tensor(boxes, dtype=torch.float32)
        t_labels = torch.ones((len(boxes),), dtype=torch.int64)
        t_masks = torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8)
        t_area = (t_boxes[:, 2] - t_boxes[:, 0]) * (t_boxes[:, 3] - t_boxes[:, 1])
        t_iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

    return {
        "boxes": t_boxes,
        "labels": t_labels,
        "masks": t_masks,
        "image_id": torch.tensor([int(image_idx)], dtype=torch.int64),
        "area": t_area,
        "iscrowd": t_iscrowd,
    }


def _run_epoch_maskrcnn(
    *,
    model: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    progress_callback: Any | None = None,
) -> float:
    model.train()
    losses: list[float] = []
    total_batches = int(len(loader)) if hasattr(loader, "__len__") else 0
    if callable(progress_callback):
        try:
            progress_callback(done=0, total=total_batches, last_loss=None)
        except Exception:
            pass
    for batch_idx, (x, y) in enumerate(loader, start=1):
        images = [model.input_adapter(img.to(device)) for img in x]
        targets: list[dict[str, Any]] = []
        for i in range(y.shape[0]):
            target = _mask_to_instances(y[i, 0].cpu().numpy(), image_idx=(batch_idx - 1) * y.shape[0] + i)
            targets.append({k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()})
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.model(images, targets)
        loss = sum(v for v in loss_dict.values())
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if callable(progress_callback) and (
            batch_idx <= 1
            or batch_idx >= total_batches
            or (total_batches > 0 and batch_idx % max(1, total_batches // 10) == 0)
        ):
            try:
                progress_callback(
                    done=int(batch_idx),
                    total=total_batches,
                    last_loss=float(loss_value),
                )
            except Exception:
                pass
    if not losses:
        return 0.0
    return float(np.mean(losses))


def _loader_metrics_maskrcnn(
    model: Any,
    loader: Any,
    *,
    threshold: float,
    device: Any,
    progress_callback: Any | None = None,
) -> dict[str, float]:
    model.eval()
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    total_batches = int(len(loader)) if hasattr(loader, "__len__") else 0
    if callable(progress_callback):
        try:
            progress_callback(done=0, total=total_batches)
        except Exception:
            pass
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader, start=1):
            images = [model.input_adapter(img.to(device)) for img in x]
            outputs = model.model(images)
            for i, out in enumerate(outputs):
                h = int(y[i, 0].shape[0])
                w = int(y[i, 0].shape[1])
                prob = np.zeros((h, w), dtype=np.float32)
                masks = out.get("masks")
                scores = out.get("scores")
                if masks is not None and masks.numel() > 0:
                    m = masks[:, 0].detach().cpu().numpy().astype(np.float32)
                    if scores is None:
                        s = np.ones((m.shape[0],), dtype=np.float32)
                    else:
                        s = scores.detach().cpu().numpy().astype(np.float32)
                    keep = s >= 0.05
                    if np.any(keep):
                        m_keep = m[keep]
                        s_keep = s[keep][:, None, None]
                        prob = np.max(m_keep * s_keep, axis=0)
                pred = (prob >= float(threshold)).astype(np.uint8)
                y_true_all.append(y[i, 0].detach().cpu().numpy().reshape(-1).astype(np.uint8))
                y_pred_all.append(pred.reshape(-1).astype(np.uint8))
            if callable(progress_callback) and (
                batch_idx <= 1
                or batch_idx >= total_batches
                or (total_batches > 0 and batch_idx % max(1, total_batches // 10) == 0)
            ):
                try:
                    progress_callback(done=int(batch_idx), total=total_batches)
                except Exception:
                    pass

    if not y_true_all:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return project.compute_metrics(y_true, y_pred)


def predict_pipeline_mask_auto(
    *,
    model: Any,
    x_img: np.ndarray,
    threshold: float,
    patch_size: int,
    stride: int,
    batch_size: int,
    device: Any,
    model_kind: str = "small_unet",
    normalization_stats: dict[str, Any] | None = None,
    postprocess_mask: bool = True,
    postprocess_min_region_scene_ratio: float = 0.0005,
    postprocess_min_region_pixels: int | None = None,
    postprocess_max_regions: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # Sliding-window inference for large geospatial images.
    # Each patch is normalized exactly like training, predicted independently, then
    # blended back into a full-size probability map by averaging overlaps.
    ensure_dependencies()
    model_kind = normalize_model_kind(model_kind)
    h, w, _ = x_img.shape
    ys = _starts(h, patch_size, stride)
    xs = _starts(w, patch_size, stride)
    prob_sum = np.zeros((h, w), dtype=np.float32)
    prob_count = np.zeros((h, w), dtype=np.float32)
    patches: list[np.ndarray] = []
    locs: list[tuple[int, int, int, int]] = []

    def flush() -> None:
        # Run one micro-batch of prepared tiles, then scatter the probabilities
        # back into the full-resolution accumulators.
        if not patches:
            return
        if model_kind in MASKRCNN_MODEL_KINDS:
            images = [torch.from_numpy(np.moveaxis(p, -1, 0)).float().to(device) for p in patches]
            images = [model.input_adapter(img) for img in images]
            with torch.no_grad():
                outputs = model.model(images)
            for idx, (y0, y1, x0, x1) in enumerate(locs):
                ph = y1 - y0
                pw = x1 - x0
                patch_prob = np.zeros((patch_size, patch_size), dtype=np.float32)
                masks = outputs[idx].get("masks")
                scores = outputs[idx].get("scores")
                if masks is not None and masks.numel() > 0:
                    m = masks[:, 0].detach().cpu().numpy().astype(np.float32)
                    if scores is None:
                        s = np.ones((m.shape[0],), dtype=np.float32)
                    else:
                        s = scores.detach().cpu().numpy().astype(np.float32)
                    keep = s >= 0.05
                    if np.any(keep):
                        patch_prob = np.max(m[keep] * s[keep][:, None, None], axis=0)
                prob_sum[y0:y1, x0:x1] += patch_prob[:ph, :pw]
                prob_count[y0:y1, x0:x1] += 1.0
        else:
            batch = np.stack(patches, axis=0)
            tensor = torch.from_numpy(np.moveaxis(batch, -1, 1)).float().to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0, :, :]
            for idx, (y0, y1, x0, x1) in enumerate(locs):
                patch_prob = probs[idx]
                ph = y1 - y0
                pw = x1 - x0
                prob_sum[y0:y1, x0:x1] += patch_prob[:ph, :pw]
                prob_count[y0:y1, x0:x1] += 1.0
        patches.clear()
        locs.clear()

    # Tile the image deterministically so training/evaluation/prediction all use
    # the same patch geometry when given the same patch_size/stride.
    for y0 in ys:
        for x0 in xs:
            y1 = min(h, y0 + patch_size)
            x1 = min(w, x0 + patch_size)
            patch = _pad_patch_image(x_img, y0, x0, patch_size)
            patches.append(normalize_input_image(patch, normalization_stats))
            locs.append((y0, y1, x0, x1))
            if len(patches) >= max(1, int(batch_size)):
                flush()
    flush()
    prob_count = np.where(prob_count <= 0, 1.0, prob_count)
    pred_prob = (prob_sum / prob_count).astype(np.float32)
    pred_mask = (pred_prob >= float(threshold)).astype(np.uint8)
    pred_mask, post_meta = project.postprocess_segmentation_mask(
        pred_mask,
        enabled=bool(postprocess_mask),
        min_region_pixels=postprocess_min_region_pixels,
        min_region_scene_ratio=float(postprocess_min_region_scene_ratio),
        max_regions=postprocess_max_regions,
    )
    infer_meta = {
        "mode": "unet_patch_blend",
        "patch_size": int(patch_size),
        "stride": int(stride),
        "batch_size": int(batch_size),
        "tiles_y": int(len(ys)),
        "tiles_x": int(len(xs)),
        "mask_postprocess": post_meta,
    }
    return pred_mask, pred_prob, infer_meta


def evaluate_pairs(
    *,
    model: Any,
    pairs: list[project.PairRecord],
    sensor_channels: int,
    threshold: float,
    patch_size: int,
    stride: int,
    batch_size: int,
    device: Any,
    model_kind: str = "small_unet",
    normalization_stats: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    total_pairs = int(len(pairs))
    if callable(progress_callback):
        try:
            progress_callback(done=0, total=total_pairs)
        except Exception:
            pass
    for idx, pair in enumerate(pairs, start=1):
        try:
            x_img = project.load_image(pair.image_path, sensor_channels)
            y_true = project.to_binary_mask(
                project.load_mask(pair.mask_path),
                mask_path=pair.mask_path,
            )
            y_pred, _, _ = predict_pipeline_mask_auto(
                model=model,
                x_img=x_img,
                threshold=threshold,
                patch_size=patch_size,
                stride=stride,
                batch_size=batch_size,
                device=device,
                model_kind=model_kind,
                normalization_stats=normalization_stats,
            )
            yt = y_true.reshape(-1)
            yp = y_pred.reshape(-1)
            metrics = project.compute_metrics(yt, yp)
            row = {
                "sensor": pair.sensor,
                "filename": pair.filename,
                "image_path": str(pair.image_path),
                "mask_path": str(pair.mask_path),
                "pixels": int(yt.size),
            }
            row.update(metrics)
            rows.append(row)
            y_true_all.append(yt)
            y_pred_all.append(yp)
        except Exception as ex:
            issues.append(
                project.make_issue(
                    "unet_validation",
                    "pair_eval_failed",
                    sensor=pair.sensor,
                    filename=pair.filename,
                    image_path=pair.image_path,
                    mask_path=pair.mask_path,
                    details=str(ex),
                )
            )
        if callable(progress_callback) and (
            idx <= 1 or idx >= total_pairs or idx % max(1, total_pairs // 10) == 0
        ):
            try:
                progress_callback(done=int(idx), total=total_pairs)
            except Exception:
                pass
    if not y_true_all:
        return (
            {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0},
            rows,
            issues,
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.uint8),
        )
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return project.compute_metrics(y_true, y_pred), rows, issues, y_true, y_pred


def collect_pair_probability_arrays(
    *,
    model: Any,
    pairs: list[project.PairRecord],
    sensor_channels: int,
    patch_size: int,
    stride: int,
    batch_size: int,
    device: Any,
    model_kind: str = "small_unet",
    normalization_stats: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []
    for pair in pairs:
        try:
            x_img = project.load_image(pair.image_path, sensor_channels)
            y_true = project.to_binary_mask(
                project.load_mask(pair.mask_path),
                mask_path=pair.mask_path,
            )
            _, pred_prob, _ = predict_pipeline_mask_auto(
                model=model,
                x_img=x_img,
                threshold=0.5,
                patch_size=patch_size,
                stride=stride,
                batch_size=batch_size,
                device=device,
                model_kind=model_kind,
                normalization_stats=normalization_stats,
            )
            y_true_all.append(y_true.reshape(-1).astype(np.uint8))
            y_prob_all.append(pred_prob.reshape(-1).astype(np.float32))
        except Exception as ex:
            issues.append(
                project.make_issue(
                    "unet_validation_probability",
                    "pair_probability_eval_failed",
                    sensor=pair.sensor,
                    filename=pair.filename,
                    image_path=pair.image_path,
                    mask_path=pair.mask_path,
                    details=str(ex),
                )
            )
    if not y_true_all:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.float32), issues
    return np.concatenate(y_true_all), np.concatenate(y_prob_all), issues


def _fit_logistic_with_cv(
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seed: int,
    model_path: Path,
) -> tuple[Any | None, dict[str, Any]]:
    if df.empty:
        return None, {"status": "skipped", "reason": "empty_training_table"}
    y = df[target_col].astype(np.uint8).to_numpy()
    x = df[feature_cols].to_numpy(dtype=np.float32)
    class_counts = np.bincount(y, minlength=2)
    if np.any(class_counts == 0):
        return None, {"status": "skipped", "reason": "single_class", "class_counts": class_counts.tolist()}

    n_splits = min(5, int(np.min(class_counts)))
    fold_metrics: list[dict[str, float]] = []
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (tr, te) in enumerate(skf.split(x, y), start=1):
            model = project.build_meta_model(seed + fold_idx)
            model.fit(x[tr], y[tr])
            pred = model.predict(x[te]).astype(np.uint8)
            prob = model.predict_proba(x[te])[:, 1].astype(np.float32)
            m = project.compute_metrics(y[te], pred)
            m["roc_auc"] = float(roc_auc_score(y[te], prob))
            m["brier_score"] = float(brier_score_loss(y[te], prob))
            m["fold"] = float(fold_idx)
            fold_metrics.append(m)

    min_class_count = int(np.min(class_counts))
    calibration_cv = min(3, min_class_count)
    if calibration_cv >= 2:
        base = project.build_meta_model(seed)
        final_model = CalibratedClassifierCV(base, method="sigmoid", cv=calibration_cv)
    else:
        final_model = project.build_meta_model(seed)
    final_model.fit(x, y)
    joblib.dump(final_model, model_path)

    summary: dict[str, Any] = {
        "status": "ok",
        "n_samples": int(len(y)),
        "class_counts": class_counts.tolist(),
        "n_splits": int(n_splits),
        "feature_columns": feature_cols,
        "fold_metrics": fold_metrics,
        "calibration": {
            "used": bool(calibration_cv >= 2),
            "method": "sigmoid" if calibration_cv >= 2 else None,
            "cv": int(calibration_cv) if calibration_cv >= 2 else None,
        },
    }
    if fold_metrics:
        keys = [k for k in fold_metrics[0].keys() if k != "fold"]
        summary["cv_mean"] = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
        summary["cv_std"] = {k: float(np.std([m[k] for m in fold_metrics])) for k in keys}
    return final_model, summary

def _train_pipeline_risk_models(
    *,
    output_dir: Path,
    csv_path: Path,
    temporal_csv_path: Path,
    temporal_bridge_csv_path: Path,
    discovery: project.DiscoveryResult,
    no_flood_roots: list[Path] | None,
    seg_models: dict[str, Any],
    normalization_stats_by_sensor: dict[str, dict[str, Any] | None],
    decision_thresholds_by_sensor: dict[str, float] | None,
    threshold: float,
    patch_size: int,
    stride: int,
    infer_batch_size: int,
    device: Any,
    seed: int,
    model_kind: str,
    temporal_model_type: str,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    # Train the non-segmentation models that depend on segmentation output:
    # 1) no-weather risk model from predicted flood features
    # 2) with-weather risk model for S1 rows after weather joins
    # 3) temporal risk model from the with-weather table
    def _emit_progress(
        sub_progress: float,
        sub_stage: str,
        **extra: Any,
    ) -> None:
        if not callable(progress_callback):
            return
        payload: dict[str, Any] = {
            "risk_subprogress": float(np.clip(sub_progress, 0.0, 1.0)),
            "risk_substage": str(sub_stage),
        }
        for key, value in extra.items():
            if value is None:
                continue
            payload[str(key)] = value
        try:
            progress_callback(payload)
        except Exception:
            pass

    weather_rows: list[dict[str, Any]] = []
    no_weather_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    csv_agg, csv_issues = project.aggregate_csv_features(csv_path)
    issues.extend(csv_issues)
    s1_pairs = discovery.pairs_by_sensor.get("S1", [])
    weather_source_counts: dict[str, int] = defaultdict(int)
    weather_lookup_status_counts: dict[str, int] = defaultdict(int)
    weather_missing_examples: list[str] = []
    geo_meta_cache: dict[str, dict[str, Any] | None] = {}
    total_pair_predictions = int(
        sum(
            len(pairs)
            for sensor, pairs in discovery.pairs_by_sensor.items()
            if seg_models.get(sensor) is not None
        )
    )
    pair_predictions_done = 0
    _emit_progress(
        0.01,
        "risk_predict_pairs_start",
        risk_pairs_done=int(pair_predictions_done),
        risk_pairs_total=int(total_pair_predictions),
    )

    for sensor, pairs in discovery.pairs_by_sensor.items():
        model = seg_models.get(sensor)
        sensor_norm_stats = normalization_stats_by_sensor.get(sensor)
        sensor_threshold = float(
            (decision_thresholds_by_sensor or {}).get(sensor, threshold)
        )
        if model is None:
            continue
        # First pass: convert every labeled flood pair into image-level prediction
        # features. These rows are the shared base table for all downstream risk models.
        for pair in pairs:
            try:
                x_img = project.load_image(pair.image_path, project.SENSOR_CHANNELS[sensor])
                y_true = project.to_binary_mask(
                    project.load_mask(pair.mask_path),
                    mask_path=pair.mask_path,
                )
                pred_mask, pred_prob, _ = predict_pipeline_mask_auto(
                    model=model,
                    x_img=x_img,
                    threshold=sensor_threshold,
                    patch_size=patch_size,
                    stride=stride,
                    batch_size=infer_batch_size,
                    device=device,
                    model_kind=model_kind,
                    normalization_stats=sensor_norm_stats,
                )
                pred_feats = project.summarize_prediction_features(pred_mask, pred_prob)
                true_ratio = float(np.mean(y_true))
                no_weather_rows.append(
                    {
                        "sensor": sensor,
                        "filename": pair.filename,
                        "image_path": str(pair.image_path),
                        "sample_source": "flood_pair",
                        **pred_feats,
                        "true_flood_ratio": true_ratio,
                        "y_tile": int(true_ratio >= 0.02),
                    }
                )
            except Exception as ex:
                issues.append(
                    project.make_issue(
                        "unet_risk_no_weather",
                        "pair_prediction_failed",
                        sensor=sensor,
                        filename=pair.filename,
                        image_path=pair.image_path,
                        mask_path=pair.mask_path,
                        details=str(ex),
                    )
                )
            pair_predictions_done += 1
            if pair_predictions_done % 200 == 0 or pair_predictions_done >= total_pair_predictions:
                pair_frac = float(pair_predictions_done) / float(
                    max(1, total_pair_predictions)
                )
                _emit_progress(
                    0.05 + (0.45 * pair_frac),
                    "risk_predict_pairs",
                    risk_pairs_done=int(pair_predictions_done),
                    risk_pairs_total=int(total_pair_predictions),
                    risk_sensor=str(sensor),
                    risk_last_file=str(pair.filename),
                )

    no_flood_added = 0
    known_flood_paths = {p.image_path.resolve() for p in discovery.pairs}
    if no_flood_roots:
        # Optional no-flood roots widen the negative class beyond the labeled flood
        # dataset. This is especially useful when most labeled pairs contain flood.
        no_flood_images, nf_issues, _ = project.discover_no_flood_images(no_flood_roots)
        issues.extend(nf_issues)
        total_no_flood = int(
            sum(
                len(image_paths)
                for sensor, image_paths in no_flood_images.items()
                if seg_models.get(sensor) is not None
            )
        )
        no_flood_done = 0
        _emit_progress(
            0.52,
            "risk_no_flood_start",
            risk_no_flood_done=int(no_flood_done),
            risk_no_flood_total=int(total_no_flood),
        )
        for sensor, image_paths in no_flood_images.items():
            model = seg_models.get(sensor)
            sensor_norm_stats = normalization_stats_by_sensor.get(sensor)
            sensor_threshold = float(
                (decision_thresholds_by_sensor or {}).get(sensor, threshold)
            )
            if model is None:
                continue
            for image_path in image_paths:
                if image_path.resolve() in known_flood_paths:
                    issues.append(
                        project.make_issue(
                            "unet_risk_no_weather",
                            "no_flood_path_conflicts_with_flood_pair",
                            sensor=sensor,
                            filename=image_path.name,
                            image_path=image_path,
                        )
                    )
                    continue
                try:
                    x_img = project.load_image(image_path, project.SENSOR_CHANNELS[sensor])
                    pred_mask, pred_prob, _ = predict_pipeline_mask_auto(
                        model=model,
                        x_img=x_img,
                        threshold=sensor_threshold,
                        patch_size=patch_size,
                        stride=stride,
                        batch_size=infer_batch_size,
                        device=device,
                        model_kind=model_kind,
                        normalization_stats=sensor_norm_stats,
                    )
                    pred_feats = project.summarize_prediction_features(pred_mask, pred_prob)
                    no_weather_rows.append(
                        {
                            "sensor": sensor,
                            "filename": image_path.name,
                            "image_path": str(image_path),
                            "sample_source": "no_flood_root",
                            **pred_feats,
                            "true_flood_ratio": 0.0,
                            "y_tile": 0,
                        }
                    )
                    no_flood_added += 1
                except Exception as ex:
                    issues.append(
                        project.make_issue(
                            "unet_risk_no_weather",
                            "no_flood_prediction_failed",
                            sensor=sensor,
                            filename=image_path.name,
                            image_path=image_path,
                            details=str(ex),
                        )
                    )
                no_flood_done += 1
                if no_flood_done % 200 == 0 or no_flood_done >= total_no_flood:
                    no_flood_frac = float(no_flood_done) / float(max(1, total_no_flood))
                    _emit_progress(
                        0.52 + (0.10 * no_flood_frac),
                        "risk_no_flood_predict",
                        risk_no_flood_done=int(no_flood_done),
                        risk_no_flood_total=int(total_no_flood),
                        risk_sensor=str(sensor),
                        risk_last_file=str(image_path.name),
                    )

    # Build with-weather rows from all S1 prediction rows.
    # Priority order:
    # 1) direct filename match in weather CSV aggregation
    # 2) ERA5 anchor lookup from image geospatial metadata
    s1_rows_for_weather = [
        row
        for row in no_weather_rows
        if str(row.get("sensor", "")).strip().upper() == "S1"
    ]
    total_weather_candidates = int(len(s1_rows_for_weather))
    _emit_progress(
        0.63,
        "risk_weather_lookup_start",
        risk_weather_done=0,
        risk_weather_total=int(total_weather_candidates),
    )
    csv_agg_df = csv_agg if isinstance(csv_agg, pd.DataFrame) else pd.DataFrame()
    for idx_weather, base_row in enumerate(s1_rows_for_weather, start=1):
        filename = str(base_row.get("filename", "")).strip()
        image_path_raw = str(base_row.get("image_path", "")).strip()
        sample_source = str(base_row.get("sample_source", "")).strip() or "flood_pair"
        try:
            pred_flood_ratio = float(base_row.get("pred_flood_ratio", 0.0))
            pred_prob_mean = float(base_row.get("pred_prob_mean", 0.0))
            pred_prob_p90 = float(base_row.get("pred_prob_p90", 0.0))
            true_ratio = float(base_row.get("true_flood_ratio", 0.0))
            y_tile = int(base_row.get("y_tile", 0))
        except Exception:
            continue

        weather_values: dict[str, float] | None = None
        weather_source = "none"

        if not csv_agg_df.empty:
            rec, csv_status = project.find_weather_feature_record(csv_agg_df, filename)
            if rec is not None:
                parsed: dict[str, float] = {}
                for name in project.WEATHER_FEATURE_NAMES:
                    try:
                        parsed[name] = float(rec.get(name, 0.0))
                    except Exception:
                        parsed[name] = 0.0
                weather_values = parsed
                weather_source = "csv_filename_agg"
                weather_source_counts[weather_source] += 1
                weather_lookup_status_counts["ok_csv"] += 1
            elif csv_status:
                weather_lookup_status_counts[str(csv_status)] += 1

        if weather_values is None:
            image_path = Path(image_path_raw) if image_path_raw else None
            geo_meta: dict[str, Any] | None = None
            if image_path is not None:
                cache_key = str(image_path.resolve()) if image_path.exists() else image_path_raw
                if cache_key in geo_meta_cache:
                    geo_meta = geo_meta_cache[cache_key]
                else:
                    try:
                        geo_meta = project.inspect_geospatial_metadata(image_path)
                    except Exception:
                        geo_meta = None
                    geo_meta_cache[cache_key] = geo_meta
            anchor_values, anchor_status, _ = (
                project.lookup_weather_features_for_image_from_temporal(
                    csv_path=temporal_csv_path,
                    image_filename=filename if filename else None,
                    image_path=image_path,
                    geo_meta=geo_meta,
                    bridge_csv_path=temporal_bridge_csv_path,
                )
            )
            if anchor_status is None and anchor_values:
                weather_values = {
                    name: float(anchor_values.get(name, 0.0))
                    for name in project.WEATHER_FEATURE_NAMES
                }
                weather_source = "era5_anchor"
                weather_source_counts[weather_source] += 1
                weather_lookup_status_counts["ok_era5_anchor"] += 1
            else:
                status_text = str(anchor_status or "weather_lookup_unavailable")
                weather_lookup_status_counts[status_text] += 1
                if len(weather_missing_examples) < 20:
                    weather_missing_examples.append(
                        filename or image_path_raw or "<unknown>"
                    )
                issues.append(
                    project.make_issue(
                        "unet_risk_with_weather",
                        "weather_lookup_unavailable",
                        sensor="S1",
                        filename=(filename or None),
                        image_path=(Path(image_path_raw) if image_path_raw else None),
                        details=status_text,
                    )
                )
                continue

        weather_rows.append(
            {
                "sensor": "S1",
                "filename": filename,
                "image_path": image_path_raw,
                "sample_source": sample_source,
                "weather_source": weather_source,
                **weather_values,
                "pred_flood_ratio": pred_flood_ratio,
                "pred_prob_mean": pred_prob_mean,
                "pred_prob_p90": pred_prob_p90,
                "true_flood_ratio": true_ratio,
                "y_tile": y_tile,
            }
        )
        if idx_weather % 200 == 0 or idx_weather >= total_weather_candidates:
            weather_frac = float(idx_weather) / float(max(1, total_weather_candidates))
            _emit_progress(
                0.63 + (0.20 * weather_frac),
                "risk_weather_lookup",
                risk_weather_done=int(idx_weather),
                risk_weather_total=int(total_weather_candidates),
                risk_weather_ok_rows=int(len(weather_rows)),
                risk_last_file=str(filename or image_path_raw),
            )

    weather_df = pd.DataFrame(weather_rows)
    no_weather_df = pd.DataFrame(no_weather_rows)
    project.write_csv(output_dir / RISK_WITH_WEATHER_PIPELINE_TABLE, weather_rows)
    project.write_csv(output_dir / RISK_NO_WEATHER_PIPELINE_TABLE, no_weather_rows)
    _emit_progress(
        0.84,
        "risk_fit_no_weather_start",
        risk_no_weather_rows=int(len(no_weather_rows)),
        risk_with_weather_rows=int(len(weather_rows)),
    )

    # Fit two tabular risk heads:
    # - with-weather: S1 only, where weather features are available
    # - no-weather: all rows, using only image-derived prediction features
    weather_feature_cols = list(project.WEATHER_FEATURE_NAMES) + list(project.IMAGE_FEATURE_NAMES)
    no_weather_feature_cols = list(project.IMAGE_FEATURE_NAMES)
    _, with_weather_metrics = _fit_logistic_with_cv(
        df=weather_df,
        feature_cols=weather_feature_cols,
        target_col="y_tile",
        seed=seed,
        model_path=output_dir / RISK_WITH_WEATHER_PIPELINE_NAME,
    )
    _, no_weather_metrics = _fit_logistic_with_cv(
        df=no_weather_df,
        feature_cols=no_weather_feature_cols,
        target_col="y_tile",
        seed=seed,
        model_path=output_dir / RISK_NO_WEATHER_PIPELINE_NAME,
    )
    _emit_progress(
        0.90,
        "risk_fit_no_weather_done",
        risk_no_weather_status=str(no_weather_metrics.get("status", "unknown"))
        if isinstance(no_weather_metrics, dict)
        else "unknown",
        risk_no_weather_auc=(
            float(no_weather_metrics.get("cv_mean", {}).get("roc_auc"))
            if isinstance(no_weather_metrics, dict)
            and isinstance(no_weather_metrics.get("cv_mean"), dict)
            and no_weather_metrics.get("cv_mean", {}).get("roc_auc") is not None
            else None
        ),
    )
    if isinstance(no_weather_metrics, dict):
        no_weather_metrics["no_flood_samples_added"] = int(no_flood_added)
        if not no_weather_df.empty and "true_flood_ratio" in no_weather_df.columns:
            any_flood_ratio = float(np.mean(no_weather_df["true_flood_ratio"].to_numpy(dtype=np.float32) > 0.0))
            no_weather_metrics["any_flood_sample_ratio"] = any_flood_ratio
            if any_flood_ratio >= 0.999:
                no_weather_metrics["data_warning"] = "all_training_samples_have_some_flood_pixels_consider_no_flood_roots"
    if isinstance(with_weather_metrics, dict):
        with_weather_metrics["csv_agg_rows"] = int(len(csv_agg))
        with_weather_metrics["s1_pairs_available"] = int(len(s1_pairs))
        with_weather_metrics["s1_rows_considered"] = int(len(s1_rows_for_weather))
        with_weather_metrics["weather_training_rows"] = int(len(weather_rows))
        with_weather_metrics["weather_source_counts"] = {
            str(k): int(v) for k, v in sorted(weather_source_counts.items())
        }
        with_weather_metrics["weather_lookup_status_counts"] = {
            str(k): int(v) for k, v in sorted(weather_lookup_status_counts.items())
        }
        with_weather_metrics["csv_rows_matched_pairs"] = int(
            weather_source_counts.get("csv_filename_agg", 0)
        )
        with_weather_metrics["csv_rows_unmatched_pairs"] = int(
            max(
                0,
                int(len(s1_rows_for_weather))
                - int(weather_source_counts.get("csv_filename_agg", 0)),
            )
        )
        if weather_missing_examples:
            with_weather_metrics["weather_missing_examples"] = list(
                weather_missing_examples[:15]
            )
        if len(s1_rows_for_weather) > 0 and len(weather_rows) == 0:
            issues.append(
                project.make_issue(
                    "unet_risk_with_weather",
                    "weather_lookup_failed_all_rows",
                    sensor="S1",
                    details=(
                        f"s1_rows={len(s1_rows_for_weather)} "
                        f"lookup_status={dict(weather_lookup_status_counts)}"
                    ),
                )
            )
    _emit_progress(
        0.94,
        "risk_fit_with_weather_done",
        risk_with_weather_status=str(with_weather_metrics.get("status", "unknown"))
        if isinstance(with_weather_metrics, dict)
        else "unknown",
        risk_with_weather_auc=(
            float(with_weather_metrics.get("cv_mean", {}).get("roc_auc"))
            if isinstance(with_weather_metrics, dict)
            and isinstance(with_weather_metrics.get("cv_mean"), dict)
            and with_weather_metrics.get("cv_mean", {}).get("roc_auc") is not None
            else None
        ),
        risk_weather_ok_rows=int(len(weather_rows)),
    )
    project.save_json(output_dir / RISK_WITH_WEATHER_PIPELINE_METRICS, with_weather_metrics)
    project.save_json(output_dir / RISK_NO_WEATHER_PIPELINE_METRICS, no_weather_metrics)

    # Temporal model is trained last because it depends on the with-weather table
    # built above as its base image-level feature source.
    _emit_progress(
        0.95,
        "risk_fit_temporal_start",
        risk_temporal_model_type=str(temporal_model_type),
        risk_weather_rows_for_temporal=int(len(weather_rows)),
    )
    temporal_metrics = project.train_temporal_risk_model_from_rows(
        csv_path=temporal_csv_path,
        base_rows_df=weather_df,
        model_path=output_dir / project.RISK_TEMPORAL_PIPELINE_NAME,
        metrics_path=output_dir / project.RISK_TEMPORAL_METRICS_PIPELINE_NAME,
        training_table_path=output_dir / project.RISK_TEMPORAL_TABLE_PIPELINE_NAME,
        seed=seed,
        backend_tag=project.PIPELINE_V3_BACKEND_ID,
        model_type=str(temporal_model_type),
        bridge_csv_path=temporal_bridge_csv_path,
    )
    _emit_progress(
        0.99,
        "risk_fit_temporal_done",
        risk_temporal_status=str(temporal_metrics.get("status", "unknown"))
        if isinstance(temporal_metrics, dict)
        else "unknown",
        risk_temporal_auc=(
            float(temporal_metrics.get("cv_mean", {}).get("roc_auc"))
            if isinstance(temporal_metrics, dict)
            and isinstance(temporal_metrics.get("cv_mean"), dict)
            and temporal_metrics.get("cv_mean", {}).get("roc_auc") is not None
            else None
        ),
    )

    return {
        "with_weather_model_path": str((output_dir / RISK_WITH_WEATHER_PIPELINE_NAME).resolve()),
        "with_weather_metrics": with_weather_metrics,
        "no_weather_model_path": str((output_dir / RISK_NO_WEATHER_PIPELINE_NAME).resolve()),
        "no_weather_metrics": no_weather_metrics,
        "temporal_model_path": str((output_dir / project.RISK_TEMPORAL_PIPELINE_NAME).resolve()),
        "temporal_metrics": temporal_metrics,
        "issues": issues,
    }


def save_active_backend(
    *,
    output_dir: Path,
    pipeline_iou_global: float | None,
) -> dict[str, Any]:
    reason = (
        "pipeline_v3_runtime_default"
        if pipeline_iou_global is not None
        else "pipeline_v3_iou_missing"
    )
    payload = {
        "segmentation_backend": project.PIPELINE_V3_BACKEND_ID,
        "risk_backend": project.PIPELINE_V3_BACKEND_ID,
        "promotion_reason": reason,
        "metrics_snapshot": {
            "pipeline_v3_iou_global": pipeline_iou_global,
        },
    }
    project.save_json(output_dir / "active_backend.json", payload)
    return payload


def _build_pipeline_input_profile(
    sensor_normalization_stats: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    sensors: dict[str, dict[str, Any]] = {}
    for sensor, stats in sorted(sensor_normalization_stats.items()):
        if not isinstance(stats, dict):
            sensors[str(sensor)] = {"status": "missing"}
            continue
        mean_vals = stats.get("mean")
        std_vals = stats.get("std")
        if not isinstance(mean_vals, list) or not isinstance(std_vals, list):
            sensors[str(sensor)] = {"status": "missing"}
            continue
        sensors[str(sensor)] = {
            "status": "ok",
            "n_samples": int(stats.get("pixels_sampled", 0)),
            "n_channels": int(len(mean_vals)),
            "mean": [float(x) for x in mean_vals],
            "std": [float(x) for x in std_vals],
            "min": [float(x) for x in (stats.get("clip_low") or [])],
            "max": [float(x) for x in (stats.get("clip_high") or [])],
            "clip_low_pct": float(stats.get("clip_low_pct", 0.0)),
            "clip_high_pct": float(stats.get("clip_high_pct", 100.0)),
        }
    return {
        "created_at_utc": project.utc_now_iso(),
        "source": "pipeline_normalization_stats",
        "sensors": sensors,
    }


def _build_pipeline_model_registry(
    *,
    output_dir: Path,
    data_roots: list[Path],
    discovery: project.DiscoveryResult,
    no_flood_roots: list[Path],
    csv_path: Path,
    temporal_csv_path: Path,
    temporal_bridge_csv_path: Path,
    threshold: float,
    val_ratio: float,
    seed: int,
    patch_size: int,
    stride: int,
    model_kind: str,
    temporal_model_type: str,
    loader_perf: dict[str, Any],
    promotion_state: dict[str, Any],
    segmentation_mask_filter: dict[str, Any] | None = None,
    segmentation_balance_policy: str = "none",
    segmentation_balance_min_flood_ratio: float = 0.02,
) -> dict[str, Any]:
    configured_roots = [str(Path(p).resolve()) for p in (data_roots or [])]
    discovered_roots = sorted({str(pair.root.resolve()) for pair in discovery.pairs})
    return {
        "registry_version": 1,
        "created_at_utc": project.utc_now_iso(),
        "run_id": str(project.uuid4()),
        "runtime": project.PIPELINE_V3_BACKEND_ID,
        "artifacts": {
            "pipeline_model_s1": str((output_dir / PIPELINE_MODEL_S1_NAME).resolve()),
            "pipeline_model_s2": str((output_dir / PIPELINE_MODEL_S2_NAME).resolve()),
            "risk_model_with_weather_s1_pipeline": str(
                (output_dir / RISK_WITH_WEATHER_PIPELINE_NAME).resolve()
            ),
            "risk_model_no_weather_global_pipeline": str(
                (output_dir / RISK_NO_WEATHER_PIPELINE_NAME).resolve()
            ),
            "risk_model_temporal_gb_s1_pipeline": str(
                project.get_temporal_model_path(
                    output_dir=output_dir,
                    backend=project.PIPELINE_V3_BACKEND_ID,
                )
            ),
            "risk_temporal_metrics_pipeline": str(
                (output_dir / project.RISK_TEMPORAL_METRICS_PIPELINE_NAME).resolve()
            ),
            "input_profile": str((output_dir / "input_profile.json").resolve()),
            "dataset_metadata_csv": str(
                (output_dir / project.DATASET_METADATA_CSV_NAME).resolve()
            ),
            "dataset_metadata_summary": str(
                (output_dir / project.DATASET_METADATA_SUMMARY_NAME).resolve()
            ),
            "unet_train_report": str((output_dir / PIPELINE_TRAIN_REPORT_NAME).resolve()),
            "unet_val_metrics_global": str(
                (output_dir / PIPELINE_VAL_GLOBAL_NAME).resolve()
            ),
            "active_backend": str(
                (output_dir / project.ACTIVE_BACKEND_NAME).resolve()
            ),
        },
        "config": {
            "data_roots": configured_roots,
            "discovered_data_roots": discovered_roots,
            "no_flood_roots": [
                str(Path(p).resolve()) for p in (no_flood_roots or [])
            ],
            "csv_path": str(csv_path.resolve()),
            "temporal_csv_path": str(temporal_csv_path.resolve()),
            "temporal_bridge_csv_path": str(temporal_bridge_csv_path.resolve()),
            "threshold": float(threshold),
            "mask_flood_policy": str(project.ACTIVE_MASK_FLOOD_POLICY),
            "val_ratio": float(val_ratio),
            "seed": int(seed),
            "patch_size": int(patch_size),
            "stride": int(stride),
            "model_kind": str(model_kind),
            "temporal_model_type": str(temporal_model_type),
            "segmentation_mask_sync_policy": str(
                (segmentation_mask_filter or {}).get("policy", "unknown")
            ),
            "segmentation_source_groups": (
                segmentation_mask_filter or {}
            ).get("allowed_source_groups"),
            "segmentation_balance_policy": str(segmentation_balance_policy),
            "segmentation_balance_min_flood_ratio": float(
                segmentation_balance_min_flood_ratio
            ),
            "loader_num_workers": int(loader_perf.get("num_workers", 0)),
            "loader_pin_memory": bool(loader_perf.get("pin_memory", False)),
            "loader_prefetch_factor": int(loader_perf.get("prefetch_factor", 0)),
        },
        "promotion": promotion_state,
    }


def load_pipeline_bundle(path: Path, *, device: str = "cpu") -> dict[str, Any]:
    ensure_dependencies()
    if not path.exists():
        raise FileNotFoundError(f"Pipeline V3 model file not found: {path}")
    target_device = torch.device(device)
    ckpt = torch.load(path, map_location=target_device)
    in_channels = int(ckpt.get("in_channels", 2))
    model_kind = normalize_model_kind(ckpt.get("model_kind", "small_unet"))
    use_pretrained_encoder = bool(ckpt.get("use_pretrained_encoder", False))
    model = build_segmentation_model(
        model_kind=model_kind,
        in_channels=in_channels,
        use_pretrained_encoder=use_pretrained_encoder,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(target_device)
    model.eval()
    return {
        "model": model,
        "device": target_device,
        "in_channels": in_channels,
        "sensor": ckpt.get("sensor"),
        "patch_size": int(ckpt.get("patch_size", 256)),
        "stride": int(ckpt.get("stride", 192)),
        "epoch": int(ckpt.get("epoch", 0)),
        "model_kind": model_kind,
        "use_pretrained_encoder": use_pretrained_encoder,
        "normalization_stats": ckpt.get("normalization_stats"),
        "decision_threshold": ckpt.get("decision_threshold"),
        "threshold_tuning": ckpt.get("threshold_tuning"),
    }


def _save_periodic_pipeline_checkpoint(
    *,
    output_dir: Path,
    sensor: str,
    epoch: int,
    model: Any,
    optimizer: Any,
    patch_size: int,
    stride: int,
    model_kind: str,
    use_pretrained_encoder: bool,
    normalization_stats: dict[str, Any] | None,
    selection_metric: str,
    selection_score: float,
    best_patch_val_iou: float,
    train_loss: float,
    val_metrics_patch: dict[str, Any],
    val_metrics_image_level: dict[str, Any] | None,
) -> Path:
    checkpoint_dir = (
        output_dir / PIPELINE_PERIODIC_CHECKPOINT_DIR / sensor.lower()
    ).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        checkpoint_dir / f"unet_model_{sensor.lower()}_epoch_{int(epoch):03d}.pth"
    )
    payload = {
        "checkpoint_type": "periodic",
        "sensor": sensor,
        "epoch": int(epoch),
        "state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "in_channels": project.SENSOR_CHANNELS[sensor],
        "patch_size": int(patch_size),
        "stride": int(stride),
        "model_kind": str(model_kind),
        "use_pretrained_encoder": bool(use_pretrained_encoder),
        "normalization_stats": normalization_stats,
        "selection_metric": str(selection_metric),
        "selection_score": float(selection_score),
        "best_patch_val_iou": float(best_patch_val_iou),
        "train_loss": float(train_loss),
        "val_metrics_patch": val_metrics_patch,
        "val_metrics_image_level": val_metrics_image_level,
        "saved_at_utc": project.utc_now_iso(),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _normalize_segmentation_mask_sync_policy(value: str | None) -> str:
    policy = str(value or "strict").strip().lower().replace("_", "-")
    if policy not in SEGMENTATION_MASK_SYNC_POLICY_CHOICES:
        return "strict"
    return policy


def _normalize_source_groups(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not values:
        return ()
    groups = [str(v).strip() for v in values if str(v).strip()]
    return tuple(dict.fromkeys(groups))


def _candidate_organized_manifest_paths(data_roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in data_roots:
        resolved = Path(root).resolve()
        candidates = [resolved, *list(resolved.parents)[:5]]
        for base in candidates:
            candidate = (base / "metadata" / "organized_manifest.csv").resolve()
            if candidate in seen or not candidate.exists():
                continue
            seen.add(candidate)
            paths.append(candidate)
    return paths


def _load_organized_manifest_for_segmentation_filter(
    data_roots: list[Path],
) -> tuple[pd.DataFrame, list[Path]]:
    manifest_paths = _candidate_organized_manifest_paths(data_roots)
    frames: list[pd.DataFrame] = []
    required = {"sensor", "source_group", "sample_type", "source_tile_name", "output_name"}
    for path in manifest_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not required.issubset(set(df.columns)):
            continue
        df = df.copy()
        df["__manifest_path"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame(), manifest_paths
    merged = pd.concat(frames, ignore_index=True)
    merged["sensor"] = merged["sensor"].astype(str).str.upper().str.strip()
    merged["output_name"] = merged["output_name"].astype(str).str.lower().str.strip()
    merged["source_group"] = merged["source_group"].astype(str).str.strip()
    merged["sample_type"] = merged["sample_type"].astype(str).str.strip()
    merged = merged.drop_duplicates(subset=["sensor", "output_name"], keep="first")
    return merged, manifest_paths


def _resolve_segmentation_allowed_source_groups(
    *,
    mask_sync_policy: str,
    source_groups: list[str] | tuple[str, ...] | None,
) -> tuple[str, tuple[str, ...] | None]:
    custom_groups = _normalize_source_groups(source_groups)
    if custom_groups:
        return "custom", custom_groups
    policy = _normalize_segmentation_mask_sync_policy(mask_sync_policy)
    return policy, SEGMENTATION_SOURCE_GROUPS_BY_POLICY[policy]


def _build_segmentation_training_pairs_by_sensor(
    *,
    discovery: project.DiscoveryResult,
    data_roots: list[Path],
    mask_sync_policy: str,
    source_groups: list[str] | tuple[str, ...] | None,
) -> tuple[dict[str, list[project.PairRecord]], dict[str, Any]]:
    effective_policy, allowed_groups = _resolve_segmentation_allowed_source_groups(
        mask_sync_policy=mask_sync_policy,
        source_groups=source_groups,
    )
    pairs_by_sensor = {
        sensor: list(discovery.pairs_by_sensor.get(sensor, [])) for sensor in ("S1", "S2")
    }
    manifest_df, manifest_paths = _load_organized_manifest_for_segmentation_filter(
        data_roots
    )
    report: dict[str, Any] = {
        "status": "not_applied",
        "policy": str(effective_policy),
        "allowed_source_groups": (
            "all" if allowed_groups is None else [str(g) for g in allowed_groups]
        ),
        "manifest_paths": [str(p) for p in manifest_paths],
        "note": (
            "Segmentation should use masks synchronized with the image date. "
            "Temporal/same-area samples remain available to the downstream risk and "
            "temporal stages, but are not automatically trusted as segmentation labels."
        ),
        "sensors": {},
    }
    if allowed_groups is None:
        report["status"] = "disabled_all_groups_allowed"
        for sensor, pairs in pairs_by_sensor.items():
            report["sensors"][sensor] = {
                "before_pairs": int(len(pairs)),
                "after_pairs": int(len(pairs)),
                "excluded_pairs": 0,
                "excluded_by_source_group": {},
                "kept_by_source_group": {"all": int(len(pairs))},
                "missing_manifest_pairs_kept": 0,
            }
        return pairs_by_sensor, report
    if manifest_df.empty:
        report["status"] = "not_applied_manifest_missing"
        for sensor, pairs in pairs_by_sensor.items():
            report["sensors"][sensor] = {
                "before_pairs": int(len(pairs)),
                "after_pairs": int(len(pairs)),
                "excluded_pairs": 0,
                "excluded_by_source_group": {},
                "kept_by_source_group": {"manifest_missing": int(len(pairs))},
                "missing_manifest_pairs_kept": int(len(pairs)),
            }
        return pairs_by_sensor, report

    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in manifest_df.itertuples(index=False):
        sensor = str(getattr(row, "sensor", "")).upper().strip()
        output_name = str(getattr(row, "output_name", "")).lower().strip()
        if not sensor or not output_name:
            continue
        lookup[(sensor, output_name)] = {
            "source_group": str(getattr(row, "source_group", "")).strip(),
            "sample_type": str(getattr(row, "sample_type", "")).strip(),
            "source_tile_name": str(getattr(row, "source_tile_name", "")).strip(),
        }

    allowed_set = {str(g).strip() for g in allowed_groups if str(g).strip()}
    filtered: dict[str, list[project.PairRecord]] = {}
    total_excluded = 0
    for sensor, pairs in pairs_by_sensor.items():
        kept: list[project.PairRecord] = []
        excluded_by_group: dict[str, int] = defaultdict(int)
        kept_by_group: dict[str, int] = defaultdict(int)
        missing_manifest_kept = 0
        for pair in pairs:
            candidate_names = [
                pair.image_path.name.lower(),
                pair.mask_path.name.lower(),
                str(pair.filename).lower(),
            ]
            meta = None
            for name in candidate_names:
                meta = lookup.get((sensor, name))
                if meta is not None:
                    break
            if meta is None:
                kept.append(pair)
                missing_manifest_kept += 1
                kept_by_group["manifest_missing"] += 1
                continue
            group = str(meta.get("source_group") or "unknown").strip() or "unknown"
            if group in allowed_set:
                kept.append(pair)
                kept_by_group[group] += 1
            else:
                excluded_by_group[group] += 1
        filtered[sensor] = kept
        excluded_count = int(sum(excluded_by_group.values()))
        total_excluded += excluded_count
        report["sensors"][sensor] = {
            "before_pairs": int(len(pairs)),
            "after_pairs": int(len(kept)),
            "excluded_pairs": excluded_count,
            "excluded_by_source_group": dict(sorted(excluded_by_group.items())),
            "kept_by_source_group": dict(sorted(kept_by_group.items())),
            "missing_manifest_pairs_kept": int(missing_manifest_kept),
        }
    report["status"] = "applied" if total_excluded > 0 else "applied_no_exclusions"
    report["excluded_pairs_total"] = int(total_excluded)
    report["before_pairs_total"] = int(sum(len(v) for v in pairs_by_sensor.values()))
    report["after_pairs_total"] = int(sum(len(v) for v in filtered.values()))
    return filtered, report


def train_pipeline_models(
    *,
    data_roots: list[Path],
    no_flood_roots: list[Path] | None,
    csv_path: Path,
    temporal_csv_path: Path | None,
    temporal_bridge_csv_path: Path | None,
    output_dir: Path,
    test_images: list[str],
    val_ratio: float,
    seed: int,
    threshold: float,
    patch_size: int,
    stride: int,
    epochs: int,
    early_stopping_patience: int,
    batch_size_s1: int,
    batch_size_s2: int,
    lr: float,
    weight_decay: float,
    max_patches_per_image: int,
    infer_batch_size: int,
    model_kind: str = "small_unet",
    temporal_model_type: str = "gradient_boosting",
    segmentation_mask_sync_policy: str = "strict",
    segmentation_source_groups: list[str] | tuple[str, ...] | None = None,
    segmentation_balance_policy: str = "equal-flood-non-flood",
    segmentation_balance_min_flood_ratio: float = 0.02,
) -> dict[str, Any]:
    # Master training pipeline:
    # 1) dataset discovery + metadata export
    # 2) per-sensor segmentation training/validation/threshold tuning
    # 3) risk model training on top of segmentation output
    # 4) runtime artifact + report export
    ensure_dependencies()
    model_kind = normalize_model_kind(model_kind)
    ensure_segmentation_model_support(model_kind)
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = project.utc_now_iso()
    _write_train_progress(
        output_dir,
        {
            "status": "running",
            "stage": "discovery",
            "current_sensor": None,
            "epoch_current": None,
            "epoch_total": int(epochs),
            "completed_units": 0,
            "total_units": None,
            "progress_percent": 0.0,
            "run_started_at_utc": run_started_at,
            "updated_at_utc": project.utc_now_iso(),
            "output_dir": str(output_dir.resolve()),
        },
    )
    temporal_csv_resolved = temporal_csv_path.resolve() if temporal_csv_path is not None else csv_path.resolve()
    temporal_bridge_resolved = (
        temporal_bridge_csv_path.resolve() if temporal_bridge_csv_path is not None else csv_path.resolve()
    )
    discovery = project.discover_dataset(data_roots)
    segmentation_pairs_by_sensor, segmentation_filter_report = (
        _build_segmentation_training_pairs_by_sensor(
            discovery=discovery,
            data_roots=data_roots,
            mask_sync_policy=segmentation_mask_sync_policy,
            source_groups=segmentation_source_groups,
        )
    )
    project.save_json(
        output_dir / SEGMENTATION_MASK_FILTER_REPORT_NAME,
        segmentation_filter_report,
    )
    sensors_with_pairs = [
        s for s in ("S1", "S2") if segmentation_pairs_by_sensor.get(s, [])
    ]
    total_units = max(1, int(epochs) * max(1, len(sensors_with_pairs)) + 3)
    completed_units = 1

    def _emit_metadata_export_progress(
        *,
        done: int,
        total: int,
        sensor: str | None = None,
        filename: str | None = None,
        ) -> None:
        # Metadata export can take a while on large TIFF collections, so it reports
        # its own sub-progress before model training starts.
        total_safe = max(1, int(total))
        subprogress = min(0.99, max(0.0, float(done) / float(total_safe)))
        _write_train_progress(
            output_dir,
            {
                "status": "running",
                "stage": "metadata_export",
                "current_sensor": None,
                "epoch_current": None,
                "epoch_total": int(epochs),
                "completed_units": int(completed_units),
                "total_units": int(total_units),
                "progress_percent": round(
                    100.0 * (float(completed_units) + float(subprogress)) / float(total_units),
                    2,
                ),
                "metadata_images_done": int(done),
                "metadata_images_total": int(total),
                "metadata_sensor": str(sensor or ""),
                "metadata_last_file": str(filename or ""),
                "run_started_at_utc": run_started_at,
                "updated_at_utc": project.utc_now_iso(),
                "output_dir": str(output_dir.resolve()),
            },
        )

    metadata_report = project.export_dataset_image_metadata(
        output_dir=output_dir,
        discovery=discovery,
        no_flood_roots=no_flood_roots,
        progress_callback=_emit_metadata_export_progress,
    )
    test_paths, test_issues = project.resolve_test_images(test_images, discovery.image_index)
    test_filenames = {p.name for p in test_paths}

    eval_aug = build_eval_augment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_perf = _resolve_loader_perf(device=device)
    bn_sensitive_batch_cap = _resolve_bn_sensitive_batch_cap()
    periodic_checkpoint_every = _resolve_periodic_checkpoint_every()
    patch_sampling_cfg = _resolve_patch_sampling_config()
    input_norm_cfg = _resolve_input_norm_config()
    selection_cfg = _resolve_selection_config()
    use_pretrained_encoder = _resolve_pretrained_encoder_flag(model_kind)

    sensor_reports: dict[str, Any] = {}
    seg_models: dict[str, Any] = {}
    sensor_normalization_stats: dict[str, dict[str, Any] | None] = {}
    sensor_decision_thresholds: dict[str, float] = {}
    global_true: list[np.ndarray] = []
    global_pred: list[np.ndarray] = []
    all_issues: list[dict[str, Any]] = list(discovery.issues) + list(test_issues)

    for sensor in ("S1", "S2"):
        train_aug = build_train_augment(sensor=sensor)
        pairs = segmentation_pairs_by_sensor.get(sensor, [])
        if not pairs:
            sensor_reports[sensor] = {
                "status": "skipped",
                "reason": "no_pairs_after_segmentation_mask_filter",
                "segmentation_mask_filter": segmentation_filter_report.get(
                    "sensors", {}
                ).get(sensor, {}),
            }
            continue

        def _emit_sensor_prepare_progress(
            *,
            step: str,
            done: int,
            total: int,
            step_idx: int,
            step_count: int = 3,
        ) -> None:
            # Preparation spans multiple expensive pre-training steps. Report them
            # separately so the live monitor does not look stalled before epoch 1.
            total_safe = max(1, int(total))
            step_fraction = min(0.99, max(0.0, float(done) / float(total_safe)))
            overall_fraction = min(
                0.99,
                (
                    (float(step_idx) + float(step_fraction))
                    / max(1.0, float(step_count))
                ),
            )
            _write_train_progress(
                output_dir,
                {
                    "status": "running",
                    "stage": "segmentation_prepare",
                    "current_sensor": sensor,
                    "epoch_current": 0,
                    "epoch_total": int(epochs),
                    "completed_units": int(completed_units),
                    "total_units": int(total_units),
                    "progress_percent": round(
                        100.0
                        * (float(completed_units) + float(overall_fraction))
                        / float(total_units),
                        2,
                    ),
                    "prep_step": str(step),
                    "prep_done": int(done),
                    "prep_total": int(total),
                    "run_started_at_utc": run_started_at,
                    "updated_at_utc": project.utc_now_iso(),
                    "output_dir": str(output_dir.resolve()),
                },
            )

        try:
            train_pairs, val_pairs = project.split_pairs_for_sensor(
                pairs=pairs,
                test_filenames=test_filenames,
                val_ratio=val_ratio,
                seed=seed,
            )
            train_pairs_before_balance = int(len(train_pairs))
            train_pairs, train_balance_report = _balance_segmentation_train_pairs(
                train_pairs,
                policy=segmentation_balance_policy,
                min_flood_ratio=float(segmentation_balance_min_flood_ratio),
                seed=seed + (101 if sensor == "S2" else 0),
            )
        except Exception as ex:
            sensor_reports[sensor] = {
                "status": "skipped",
                "reason": str(ex),
                "segmentation_mask_filter": segmentation_filter_report.get(
                    "sensors", {}
                ).get(sensor, {}),
            }
            continue
        if not train_pairs:
            sensor_reports[sensor] = {
                "status": "skipped",
                "reason": "no_train_pairs_after_image_balance",
                "segmentation_mask_filter": segmentation_filter_report.get(
                    "sensors", {}
                ).get(sensor, {}),
                "train_balance": train_balance_report,
            }
            continue

        sensor_norm_stats = compute_input_normalization_stats(
            pairs=train_pairs,
            sensor_channels=project.SENSOR_CHANNELS[sensor],
            seed=seed + (11 if sensor == "S2" else 0),
            progress_callback=lambda done, total: _emit_sensor_prepare_progress(
                step="input_norm",
                done=int(done),
                total=int(total),
                step_idx=0,
            ),
        )
        sensor_normalization_stats[sensor] = sensor_norm_stats

        train_records, train_issues = build_patch_records(
            pairs=train_pairs,
            sensor_channels=project.SENSOR_CHANNELS[sensor],
            patch_size=patch_size,
            stride=stride,
            max_patches_per_image=max_patches_per_image,
            seed=seed,
            progress_callback=lambda done, total: _emit_sensor_prepare_progress(
                step="train_records",
                done=int(done),
                total=int(total),
                step_idx=1,
            ),
        )
        val_records, val_issues = build_patch_records(
            pairs=val_pairs,
            sensor_channels=project.SENSOR_CHANNELS[sensor],
            patch_size=patch_size,
            stride=stride,
            max_patches_per_image=max(4, max_patches_per_image // 2),
            seed=seed + 13,
            progress_callback=lambda done, total: _emit_sensor_prepare_progress(
                step="val_records",
                done=int(done),
                total=int(total),
                step_idx=2,
            ),
        )
        all_issues.extend(train_issues)
        all_issues.extend(val_issues)
        if not train_records:
            sensor_reports[sensor] = {
                "status": "skipped",
                "reason": "no_train_records",
                "segmentation_mask_filter": segmentation_filter_report.get(
                    "sensors", {}
                ).get(sensor, {}),
            }
            continue

        bs = batch_size_s1 if sensor == "S1" else batch_size_s2
        if model_kind in BN_SENSITIVE_MODEL_KINDS:
            # Encoder/decoder variants with BatchNorm are unstable with batch=1 in train mode.
            bs = max(2, min(int(bs), int(bn_sensitive_batch_cap)))
        # Dataset + DataLoader creation happens after sampling so each epoch works
        # over a fixed patch list rather than rescanning raw TIFFs every iteration.
        train_ds = PatchDataset(
            records=train_records,
            sensor_channels=project.SENSOR_CHANNELS[sensor],
            patch_size=patch_size,
            augment=train_aug,
            normalization_stats=sensor_norm_stats,
        )
        val_ds = PatchDataset(
            records=val_records,
            sensor_channels=project.SENSOR_CHANNELS[sensor],
            patch_size=patch_size,
            augment=eval_aug,
            normalization_stats=sensor_norm_stats,
        )
        train_loader_kwargs: dict[str, Any] = {
            "batch_size": int(bs),
            "shuffle": True,
            "num_workers": int(loader_perf["num_workers"]),
            "pin_memory": bool(loader_perf["pin_memory"]),
            "drop_last": bool(model_kind in BN_SENSITIVE_MODEL_KINDS),
        }
        val_loader_kwargs: dict[str, Any] = {
            "batch_size": int(bs),
            "shuffle": False,
            "num_workers": int(loader_perf["num_workers"]),
            "pin_memory": bool(loader_perf["pin_memory"]),
        }
        if int(loader_perf["num_workers"]) > 0:
            train_loader_kwargs["persistent_workers"] = bool(loader_perf["persistent_workers"])
            train_loader_kwargs["prefetch_factor"] = int(loader_perf["prefetch_factor"])
            val_loader_kwargs["persistent_workers"] = bool(loader_perf["persistent_workers"])
            val_loader_kwargs["prefetch_factor"] = int(loader_perf["prefetch_factor"])
        train_loader = DataLoader(train_ds, **train_loader_kwargs)
        val_loader = DataLoader(val_ds, **val_loader_kwargs)

        model = build_segmentation_model(
            model_kind=model_kind,
            in_channels=project.SENSOR_CHANNELS[sensor],
            use_pretrained_encoder=use_pretrained_encoder,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        criterion = FloodSegLoss().to(device)

        best_state: dict[str, Any] | None = None
        best_patch_iou = -1.0
        best_selection_score = -1.0
        patience = 0
        history: list[dict[str, Any]] = []
        periodic_checkpoints: list[str] = []
        selection_metric_name = str(selection_cfg["metric"])

        for epoch_idx in range(1, int(epochs) + 1):
            # Split one logical epoch into train / patch-validation / image-validation
            # spans so progress reporting can move inside the epoch instead of only
            # once per epoch.
            epoch_unit_base = int(completed_units)
            run_image_eval = bool(
                val_pairs and (int(epoch_idx) % int(selection_cfg["image_eval_every"]) == 0)
            )
            train_span = 0.70
            val_loader_span = 0.15 if run_image_eval else 0.30
            image_eval_span = max(0.0, 1.0 - train_span - val_loader_span)

            def _emit_epoch_stage_progress(
                *,
                stage_name: str,
                unit_fraction: float,
                train_batches_done: int | None = None,
                train_batches_total: int | None = None,
                validate_batches_done: int | None = None,
                validate_batches_total: int | None = None,
                validate_pairs_done: int | None = None,
                validate_pairs_total: int | None = None,
                stage_detail: str | None = None,
                last_train_loss_value: float | None = None,
            ) -> None:
                # Convert fine-grained batch/pair progress inside the epoch into the
                # same "unit" scale used by the live monitor for the whole run.
                payload: dict[str, Any] = {
                    "status": "running",
                    "stage": str(stage_name),
                    "current_sensor": sensor,
                    "epoch_current": int(epoch_idx),
                    "epoch_total": int(epochs),
                    "completed_units": int(epoch_unit_base),
                    "total_units": int(total_units),
                    "progress_percent": round(
                        100.0
                        * (float(epoch_unit_base) + float(np.clip(unit_fraction, 0.0, 0.99)))
                        / float(total_units),
                        2,
                    ),
                    "run_started_at_utc": run_started_at,
                    "updated_at_utc": project.utc_now_iso(),
                    "output_dir": str(output_dir.resolve()),
                }
                if train_batches_done is not None:
                    payload["train_batches_done"] = int(train_batches_done)
                if train_batches_total is not None:
                    payload["train_batches_total"] = int(train_batches_total)
                if validate_batches_done is not None:
                    payload["validate_batches_done"] = int(validate_batches_done)
                if validate_batches_total is not None:
                    payload["validate_batches_total"] = int(validate_batches_total)
                if validate_pairs_done is not None:
                    payload["validate_pairs_done"] = int(validate_pairs_done)
                if validate_pairs_total is not None:
                    payload["validate_pairs_total"] = int(validate_pairs_total)
                if stage_detail:
                    payload["stage_detail"] = str(stage_detail)
                if last_train_loss_value is not None:
                    payload["last_train_loss"] = float(last_train_loss_value)
                _write_train_progress(output_dir, payload)

            if model_kind in MASKRCNN_MODEL_KINDS:
                train_loss = _run_epoch_maskrcnn(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    progress_callback=lambda done, total, last_loss=None: _emit_epoch_stage_progress(
                        stage_name="segmentation_train",
                        unit_fraction=(train_span * (float(done) / float(max(1, total)))),
                        train_batches_done=int(done),
                        train_batches_total=int(total),
                        stage_detail="train_batches",
                        last_train_loss_value=(
                            float(last_loss) if last_loss is not None else None
                        ),
                    ),
                )
                val_metrics = _loader_metrics_maskrcnn(
                    model=model,
                    loader=val_loader,
                    threshold=threshold,
                    device=device,
                    progress_callback=lambda done, total: _emit_epoch_stage_progress(
                        stage_name="segmentation_validate",
                        unit_fraction=(
                            train_span
                            + val_loader_span * (float(done) / float(max(1, total)))
                        ),
                        validate_batches_done=int(done),
                        validate_batches_total=int(total),
                        stage_detail="val_loader_metrics",
                        last_train_loss_value=float(train_loss),
                    ),
                )
            else:
                train_loss = _run_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    progress_callback=lambda done, total, last_loss=None: _emit_epoch_stage_progress(
                        stage_name="segmentation_train",
                        unit_fraction=(train_span * (float(done) / float(max(1, total)))),
                        train_batches_done=int(done),
                        train_batches_total=int(total),
                        stage_detail="train_batches",
                        last_train_loss_value=(
                            float(last_loss) if last_loss is not None else None
                        ),
                    ),
                )
                val_metrics = _loader_metrics(
                    model=model,
                    loader=val_loader,
                    threshold=threshold,
                    device=device,
                    progress_callback=lambda done, total: _emit_epoch_stage_progress(
                        stage_name="segmentation_validate",
                        unit_fraction=(
                            train_span
                            + val_loader_span * (float(done) / float(max(1, total)))
                        ),
                        validate_batches_done=int(done),
                        validate_batches_total=int(total),
                        stage_detail="val_loader_metrics",
                        last_train_loss_value=float(train_loss),
                    ),
                )
            patch_val_iou = _metric_value(val_metrics, "iou")
            image_val_metrics: dict[str, Any] | None = None
            if run_image_eval:
                image_val_metrics, _, image_eval_issues, _, _ = evaluate_pairs(
                    model=model,
                    pairs=val_pairs,
                    sensor_channels=project.SENSOR_CHANNELS[sensor],
                    threshold=threshold,
                    patch_size=patch_size,
                    stride=stride,
                    batch_size=infer_batch_size,
                    device=device,
                    model_kind=model_kind,
                    normalization_stats=sensor_norm_stats,
                    progress_callback=lambda done, total: _emit_epoch_stage_progress(
                        stage_name="segmentation_validate",
                        unit_fraction=(
                            train_span
                            + val_loader_span
                            + image_eval_span * (float(done) / float(max(1, total)))
                        ),
                        validate_pairs_done=int(done),
                        validate_pairs_total=int(total),
                        stage_detail="image_level_eval",
                        last_train_loss_value=float(train_loss),
                    ),
                )
                all_issues.extend(image_eval_issues)
            selection_metrics = image_val_metrics if image_val_metrics else val_metrics
            selection_score = _metric_value(selection_metrics, selection_metric_name)
            history_entry: dict[str, Any] = {
                "epoch": int(epoch_idx),
                "train_loss": float(train_loss),
                "val_metrics_patch": val_metrics,
            }
            if image_val_metrics is not None:
                history_entry["val_metrics_image_level"] = image_val_metrics
            history.append(history_entry)
            completed_units = min(total_units - 1, completed_units + 1)
            _write_train_progress(
                output_dir,
                {
                    "status": "running",
                    "stage": "segmentation_train",
                    "current_sensor": sensor,
                    "epoch_current": int(epoch_idx),
                    "epoch_total": int(epochs),
                    "last_train_loss": float(train_loss),
                    "last_val_patch_iou": float(patch_val_iou),
                    "last_val_image_iou": _metric_value(image_val_metrics, "iou"),
                    "last_val_image_f1": _metric_value(image_val_metrics, "f1"),
                    "selection_metric": selection_metric_name,
                    "selection_score": float(selection_score),
                    "completed_units": int(completed_units),
                    "total_units": int(total_units),
                    "progress_percent": round(100.0 * float(completed_units) / float(total_units), 2),
                    "run_started_at_utc": run_started_at,
                    "updated_at_utc": project.utc_now_iso(),
                    "output_dir": str(output_dir.resolve()),
                },
            )
            print(
                f"[train-pipeline] sensor={sensor} epoch={epoch_idx}/{int(epochs)} "
                f"train_loss={float(train_loss):.6f} "
                f"patch_iou={float(patch_val_iou):.6f} "
                f"image_{selection_metric_name}={float(selection_score):.6f}",
                flush=True,
            )
            if int(periodic_checkpoint_every) > 0 and (int(epoch_idx) % int(periodic_checkpoint_every) == 0):
                checkpoint_path = _save_periodic_pipeline_checkpoint(
                    output_dir=output_dir,
                    sensor=sensor,
                    epoch=int(epoch_idx),
                    model=model,
                    optimizer=optimizer,
                    patch_size=patch_size,
                    stride=stride,
                    model_kind=model_kind,
                    use_pretrained_encoder=use_pretrained_encoder,
                    normalization_stats=sensor_norm_stats,
                    selection_metric=selection_metric_name,
                    selection_score=float(selection_score),
                    best_patch_val_iou=float(best_patch_iou),
                    train_loss=float(train_loss),
                    val_metrics_patch=val_metrics,
                    val_metrics_image_level=image_val_metrics,
                )
                periodic_checkpoints.append(str(checkpoint_path))
                _write_train_progress(
                    output_dir,
                    {
                        "status": "running",
                        "stage": "segmentation_train",
                        "current_sensor": sensor,
                        "epoch_current": int(epoch_idx),
                        "epoch_total": int(epochs),
                        "last_periodic_checkpoint_epoch": int(epoch_idx),
                        "last_periodic_checkpoint_path": str(checkpoint_path),
                        "run_started_at_utc": run_started_at,
                        "updated_at_utc": project.utc_now_iso(),
                        "output_dir": str(output_dir.resolve()),
                    },
                )
                print(
                    f"[train-pipeline] sensor={sensor} saved periodic checkpoint: {checkpoint_path}",
                    flush=True,
                )
            if patch_val_iou > best_patch_iou:
                best_patch_iou = float(patch_val_iou)
            if selection_score > best_selection_score:
                # Selection uses image-level F1/IoU when available because that is
                # closer to the real inference task than patch-only validation.
                best_selection_score = float(selection_score)
                patience = 0
                best_state = {
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "sensor": sensor,
                    "in_channels": project.SENSOR_CHANNELS[sensor],
                    "patch_size": int(patch_size),
                    "stride": int(stride),
                    "epoch": int(epoch_idx),
                    "model_kind": model_kind,
                    "use_pretrained_encoder": bool(use_pretrained_encoder),
                    "normalization_stats": sensor_norm_stats,
                    "selection_metric": selection_metric_name,
                    "selection_score": float(selection_score),
                    "best_patch_val_iou": float(patch_val_iou),
                    "best_image_val_metrics": image_val_metrics,
                }
            else:
                patience += 1
            if patience >= int(early_stopping_patience):
                break

        if best_state is None:
            best_state = {
                "state_dict": copy.deepcopy(model.state_dict()),
                "sensor": sensor,
                "in_channels": project.SENSOR_CHANNELS[sensor],
                "patch_size": int(patch_size),
                "stride": int(stride),
                "epoch": int(epochs),
                "model_kind": model_kind,
                "use_pretrained_encoder": bool(use_pretrained_encoder),
                "normalization_stats": sensor_norm_stats,
                "selection_metric": selection_metric_name,
                "selection_score": float(best_selection_score),
                "best_patch_val_iou": float(best_patch_iou),
                "best_image_val_metrics": None,
            }
        # Reload the selected weights before any threshold tuning or final export.
        model.load_state_dict(best_state["state_dict"])
        tuned_threshold = float(threshold)
        threshold_tuning: dict[str, Any] = {
            "used": False,
            "metric": "f1",
            "best_threshold": float(threshold),
            "default_threshold": float(threshold),
            "thresholds_evaluated": 0,
            "default_metrics": None,
            "best_metrics": None,
            "improvement": None,
            "reason": "val_pairs_unavailable",
            "source": "not_run",
        }
        if val_pairs:
            # Threshold tuning runs on image-level probabilities from validation
            # pairs, then the tuned threshold is embedded into the saved bundle.
            y_true_prob, y_prob, tuning_issues = collect_pair_probability_arrays(
                model=model,
                pairs=val_pairs,
                sensor_channels=project.SENSOR_CHANNELS[sensor],
                patch_size=patch_size,
                stride=stride,
                batch_size=infer_batch_size,
                device=device,
                model_kind=model_kind,
                normalization_stats=sensor_norm_stats,
            )
            all_issues.extend(tuning_issues)
            if y_true_prob.size > 0 and y_prob.size > 0:
                threshold_tuning = project._tune_probability_threshold(
                    y_true_prob,
                    y_prob,
                    metric_name="f1",
                )
                threshold_tuning["source"] = "val_pairs_image_level"
                if bool(threshold_tuning.get("used")):
                    tuned_threshold = float(
                        np.clip(
                            threshold_tuning.get("best_threshold", threshold),
                            0.05,
                            0.95,
                        )
                    )
        best_state["decision_threshold"] = float(tuned_threshold)
        best_state["threshold_tuning"] = threshold_tuning
        sensor_decision_thresholds[sensor] = float(tuned_threshold)

        model_path = pipeline_model_path(sensor, output_dir)
        torch.save(best_state, model_path)
        seg_models[sensor] = model

        # Final validation/export uses the tuned threshold, not the CLI default,
        # so saved metrics reflect the threshold the runtime will actually use.
        val_metrics_full, val_rows, eval_issues, y_true, y_pred = evaluate_pairs(
            model=model,
            pairs=val_pairs,
            sensor_channels=project.SENSOR_CHANNELS[sensor],
            threshold=float(tuned_threshold),
            patch_size=patch_size,
            stride=stride,
            batch_size=infer_batch_size,
            device=device,
            model_kind=model_kind,
            normalization_stats=sensor_norm_stats,
        )
        all_issues.extend(eval_issues)
        if y_true.size > 0:
            global_true.append(y_true)
            global_pred.append(y_pred)

        preview_info: dict[str, Any] = {"train_preview_status": "skipped_no_candidate"}
        preview_pair = val_pairs[0] if val_pairs else (train_pairs[0] if train_pairs else None)
        if preview_pair is not None:
            # Export one qualitative preview per sensor so the run folder always
            # includes a human-readable segmentation example alongside metrics.
            preview_path = (output_dir / f"train_preview_{sensor.lower()}.png").resolve()
            try:
                x_preview = project.load_image(
                    preview_pair.image_path,
                    project.SENSOR_CHANNELS[sensor],
                )
                preview_mask, preview_prob, _ = predict_pipeline_mask_auto(
                    model=model,
                    x_img=x_preview,
                    threshold=float(tuned_threshold),
                    patch_size=patch_size,
                    stride=stride,
                    batch_size=infer_batch_size,
                    device=device,
                    model_kind=model_kind,
                    normalization_stats=sensor_norm_stats,
                )
                project.save_preview(preview_path, x_preview, preview_prob, preview_mask)
                preview_info = {
                    "train_preview_status": "ok",
                    "train_preview_path": str(preview_path),
                    "train_preview_source_image_path": str(preview_pair.image_path.resolve()),
                    "train_preview_source_mask_path": str(preview_pair.mask_path.resolve()),
                }
                print(
                    f"[train-pipeline] sensor={sensor} preview={preview_path}",
                    flush=True,
                )
            except Exception as ex:
                preview_info = {
                    "train_preview_status": "failed",
                    "train_preview_error": str(ex),
                }
                all_issues.append(
                    project.make_issue(
                        "unet_preview",
                        "preview_generation_failed",
                        sensor=sensor,
                        filename=preview_pair.filename,
                        image_path=preview_pair.image_path,
                        mask_path=preview_pair.mask_path,
                        details=str(ex),
                    )
                )

        project.write_csv(output_dir / f"unet_val_report_{sensor.lower()}.csv", val_rows)
        project.save_json(output_dir / (PIPELINE_VAL_S1_NAME if sensor == "S1" else PIPELINE_VAL_S2_NAME), val_metrics_full)
        sensor_reports[sensor] = {
            "status": "ok",
            "segmentation_mask_filter": segmentation_filter_report.get(
                "sensors", {}
            ).get(sensor, {}),
            "train_balance": train_balance_report,
            "train_pairs_before_balance": int(train_pairs_before_balance),
            "train_pairs": int(len(train_pairs)),
            "val_pairs": int(len(val_pairs)),
            "train_records": int(len(train_records)),
            "val_records": int(len(val_records)),
            "model_path": str(model_path),
            "best_epoch": int(best_state.get("epoch", 0)),
            "best_iou_patch": float(best_patch_iou),
            "selection_metric": selection_metric_name,
            "selection_score": float(best_state.get("selection_score", 0.0) or 0.0),
            "normalization_stats": sensor_norm_stats,
            "decision_threshold": float(tuned_threshold),
            "threshold_tuning": threshold_tuning,
            "periodic_checkpoint_every_epochs": int(periodic_checkpoint_every),
            "periodic_checkpoint_count": int(len(periodic_checkpoints)),
            "periodic_checkpoint_paths": periodic_checkpoints,
            "use_pretrained_encoder": bool(use_pretrained_encoder),
            "val_metrics_image_level": val_metrics_full,
            "history": history,
            "history_tail": history[-5:],
            "train_preview_status": str(preview_info.get("train_preview_status", "unknown")),
            "train_preview_path": preview_info.get("train_preview_path"),
            "train_preview_source_image_path": preview_info.get("train_preview_source_image_path"),
            "train_preview_source_mask_path": preview_info.get("train_preview_source_mask_path"),
            "train_preview_error": preview_info.get("train_preview_error"),
        }
        print(
            f"[train-pipeline] sensor={sensor} done best_epoch={int(best_state.get('epoch', 0))} "
            f"best_patch_iou={float(best_patch_iou):.6f} "
            f"best_{selection_metric_name}={float(best_state.get('selection_score', 0.0) or 0.0):.6f} "
            f"decision_threshold={float(tuned_threshold):.4f}",
            flush=True,
        )

    train_preview_paths = {
        sensor: str(payload.get("train_preview_path"))
        for sensor, payload in sensor_reports.items()
        if isinstance(payload, dict) and payload.get("train_preview_status") == "ok" and payload.get("train_preview_path")
    }

    if global_true:
        pipeline_global_metrics = project.compute_metrics(np.concatenate(global_true), np.concatenate(global_pred))
        pipeline_global_metrics["status"] = "ok"
    else:
        pipeline_global_metrics = {"status": "skipped", "reason": "no_validation_outputs", "iou": None}
    project.save_json(output_dir / PIPELINE_VAL_GLOBAL_NAME, pipeline_global_metrics)

    completed_units = min(total_units - 1, completed_units + 1)
    risk_base_completed_units = int(completed_units)

    def _emit_risk_training_progress(payload: dict[str, Any]) -> None:
        risk_subprogress = float(np.clip(payload.get("risk_subprogress", 0.0), 0.0, 0.99))
        progress_payload: dict[str, Any] = {
            "status": "running",
            "stage": "risk_training",
            "current_sensor": None,
            "epoch_current": None,
            "epoch_total": int(epochs),
            "completed_units": int(risk_base_completed_units),
            "total_units": int(total_units),
            "progress_percent": round(
                100.0
                * (float(risk_base_completed_units) + float(risk_subprogress))
                / float(total_units),
                2,
            ),
            "run_started_at_utc": run_started_at,
            "updated_at_utc": project.utc_now_iso(),
            "output_dir": str(output_dir.resolve()),
        }
        for key in [
            "risk_substage",
            "risk_subprogress",
            "risk_pairs_done",
            "risk_pairs_total",
            "risk_no_flood_done",
            "risk_no_flood_total",
            "risk_weather_done",
            "risk_weather_total",
            "risk_weather_ok_rows",
            "risk_no_weather_rows",
            "risk_with_weather_rows",
            "risk_no_weather_status",
            "risk_no_weather_auc",
            "risk_with_weather_status",
            "risk_with_weather_auc",
            "risk_temporal_model_type",
            "risk_temporal_status",
            "risk_temporal_auc",
            "risk_sensor",
            "risk_last_file",
        ]:
            if key in payload:
                progress_payload[key] = payload[key]
        _write_train_progress(output_dir, progress_payload)

    _emit_risk_training_progress(
        {
            "risk_substage": "risk_start",
            "risk_subprogress": 0.0,
        }
    )
    # Risk training consumes the trained segmentation models and their chosen
    # thresholds, then exports all downstream tabular/temporal artifacts.
    risk_report = _train_pipeline_risk_models(
        output_dir=output_dir,
        csv_path=csv_path,
        temporal_csv_path=temporal_csv_resolved,
        temporal_bridge_csv_path=temporal_bridge_resolved,
        discovery=discovery,
        no_flood_roots=no_flood_roots,
        seg_models=seg_models,
        normalization_stats_by_sensor=sensor_normalization_stats,
        decision_thresholds_by_sensor=sensor_decision_thresholds,
        threshold=threshold,
        patch_size=patch_size,
        stride=stride,
        infer_batch_size=infer_batch_size,
        device=device,
        seed=seed,
        model_kind=model_kind,
        temporal_model_type=str(temporal_model_type),
        progress_callback=_emit_risk_training_progress,
    )
    all_issues.extend(risk_report.get("issues", []))

    promotion_state = save_active_backend(
        output_dir=output_dir,
        pipeline_iou_global=(float(pipeline_global_metrics.get("iou")) if pipeline_global_metrics.get("iou") is not None else None),
    )

    input_profile = _build_pipeline_input_profile(sensor_normalization_stats)
    project.save_json(output_dir / "input_profile.json", input_profile)
    model_registry = _build_pipeline_model_registry(
        output_dir=output_dir,
        data_roots=data_roots,
        discovery=discovery,
        no_flood_roots=list(no_flood_roots or []),
        csv_path=csv_path,
        temporal_csv_path=temporal_csv_resolved,
        temporal_bridge_csv_path=temporal_bridge_resolved,
        threshold=threshold,
        val_ratio=val_ratio,
        seed=seed,
        patch_size=patch_size,
        stride=stride,
        model_kind=model_kind,
        temporal_model_type=str(temporal_model_type),
        loader_perf=loader_perf,
        promotion_state=promotion_state,
        segmentation_mask_filter=segmentation_filter_report,
        segmentation_balance_policy=str(segmentation_balance_policy),
        segmentation_balance_min_flood_ratio=float(
            segmentation_balance_min_flood_ratio
        ),
    )
    project.save_json(output_dir / "model_registry.json", model_registry)

    project.write_csv(
        output_dir / "dataset_issues_unet.csv",
        all_issues,
        fieldnames=["stage", "issue_type", "sensor", "root", "filename", "image_path", "mask_path", "details", "candidate_count", "candidates"],
    )
    # Final top-level report is the contract consumed by charts, comparisons, and
    # runtime status screens.
    report = {
        "status": "ok",
        "device": str(device),
        "config": {
            "no_flood_roots": [str(Path(p).resolve()) for p in (no_flood_roots or [])],
            "patch_size": int(patch_size),
            "stride": int(stride),
            "epochs": int(epochs),
            "early_stopping_patience": int(early_stopping_patience),
            "batch_size_s1": int(batch_size_s1),
            "batch_size_s2": int(batch_size_s2),
            "bn_sensitive_batch_cap": int(bn_sensitive_batch_cap),
            "effective_batch_size_s1": int(
                max(2, min(int(batch_size_s1), int(bn_sensitive_batch_cap)))
                if model_kind in BN_SENSITIVE_MODEL_KINDS
                else int(batch_size_s1)
            ),
            "effective_batch_size_s2": int(
                max(2, min(int(batch_size_s2), int(bn_sensitive_batch_cap)))
                if model_kind in BN_SENSITIVE_MODEL_KINDS
                else int(batch_size_s2)
            ),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "max_patches_per_image": int(max_patches_per_image),
            "infer_batch_size": int(infer_batch_size),
            "loader_num_workers": int(loader_perf.get("num_workers", 0)),
            "loader_pin_memory": bool(loader_perf.get("pin_memory", False)),
            "loader_persistent_workers": bool(loader_perf.get("persistent_workers", False)),
            "loader_prefetch_factor": int(loader_perf.get("prefetch_factor", 0)),
            "periodic_checkpoint_every_epochs": int(periodic_checkpoint_every),
            "patch_min_positive_ratio": float(
                patch_sampling_cfg.get("min_positive_ratio", 0.0)
            ),
            "patch_min_positive_patches": int(
                patch_sampling_cfg.get("min_positive_patches", 0)
            ),
            "patch_medium_positive_threshold": float(
                patch_sampling_cfg.get("medium_positive_threshold", 0.0)
            ),
            "patch_strong_positive_threshold": float(
                patch_sampling_cfg.get("strong_positive_threshold", 0.0)
            ),
            "patch_hard_negative_dilate": int(
                patch_sampling_cfg.get("hard_negative_dilate", 0)
            ),
            "patch_hard_negative_ratio": float(
                patch_sampling_cfg.get("hard_negative_ratio", 0.0)
            ),
            "input_normalization": input_norm_cfg,
            "selection_metric": str(selection_cfg.get("metric", "f1")),
            "image_val_every_epochs": int(selection_cfg.get("image_eval_every", 1)),
            "use_pretrained_encoder": bool(use_pretrained_encoder),
            "val_ratio": float(val_ratio),
            "threshold": float(threshold),
            "model_kind": model_kind,
            "temporal_model_type": str(temporal_model_type),
            "segmentation_mask_sync_policy": str(
                segmentation_filter_report.get("policy", segmentation_mask_sync_policy)
            ),
            "segmentation_source_groups": segmentation_filter_report.get(
                "allowed_source_groups"
            ),
            "segmentation_balance_policy": str(
                _normalize_segmentation_balance_policy(segmentation_balance_policy)
            ),
            "segmentation_balance_min_flood_ratio": float(
                np.clip(float(segmentation_balance_min_flood_ratio), 0.0, 1.0)
            ),
            "temporal_csv_path": str(temporal_csv_resolved),
            "temporal_bridge_csv_path": str(temporal_bridge_resolved),
        },
        "sensors": sensor_reports,
        "segmentation_mask_filter": segmentation_filter_report,
        "decision_thresholds_by_sensor": {
            str(k): float(v) for k, v in sorted(sensor_decision_thresholds.items())
        },
        "global_val_metrics": pipeline_global_metrics,
        "train_preview_paths": train_preview_paths,
        "train_preview_count": int(len(train_preview_paths)),
        "risk_models": risk_report,
        "dataset_metadata": metadata_report,
        "promotion_state": promotion_state,
        "issues_count": int(len(all_issues)),
    }
    project.save_json(output_dir / PIPELINE_TRAIN_REPORT_NAME, report)
    project.save_json(
        output_dir / "submission_model_report.json",
        project.build_submission_model_report(output_dir),
    )
    _write_train_progress(
        output_dir,
        {
            "status": "completed",
            "stage": "completed",
            "current_sensor": None,
            "epoch_current": int(epochs),
            "epoch_total": int(epochs),
            "completed_units": int(total_units),
            "total_units": int(total_units),
            "progress_percent": 100.0,
            "run_started_at_utc": run_started_at,
            "updated_at_utc": project.utc_now_iso(),
            "output_dir": str(output_dir.resolve()),
            "report_path": str((output_dir / PIPELINE_TRAIN_REPORT_NAME).resolve()),
        },
    )
    return report
