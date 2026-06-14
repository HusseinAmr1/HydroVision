from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def _first_existing_dir(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p.resolve()
    return candidates[0].resolve()


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = _first_existing_dir(
    [PROJECT_DIR / "Dataset", PROJECT_DIR / "dataset"]
)
DEFAULT_RAW_DIR = DEFAULT_DATASET_DIR / "Sen1Floods_Raw"


@dataclass
class PrepareStats:
    copied_pairs: int = 0
    copied_images: int = 0
    copied_masks: int = 0
    skipped_missing_label: int = 0
    skipped_name_mismatch: int = 0
    skipped_existing_target: int = 0


def _ensure_dirs(paths: list[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _extract_base_name(name: str, sensor_suffix: str) -> str | None:
    if not name.endswith(".tif"):
        return None
    if not name.endswith(sensor_suffix):
        return None
    return name[: -len(sensor_suffix)]


def _copy_pair(
    *,
    image_path: Path,
    label_path: Path,
    target_image_path: Path,
    target_mask_path: Path,
    overwrite: bool,
    dry_run: bool,
) -> bool:
    if (target_image_path.exists() or target_mask_path.exists()) and not overwrite:
        return False
    if dry_run:
        return True
    shutil.copy2(image_path, target_image_path)
    shutil.copy2(label_path, target_mask_path)
    return True


def _prepare_sensor(
    *,
    sensor_name: str,
    source_dir: Path,
    labels_dir: Path,
    sensor_suffix: str,
    label_suffix: str,
    target_image_dir: Path,
    target_mask_dir: Path,
    overwrite: bool,
    dry_run: bool,
    log: Callable[[str], None],
) -> PrepareStats:
    stats = PrepareStats()
    if not source_dir.exists():
        log(f"[warn] source dir not found for {sensor_name}: {source_dir}")
        return stats

    tif_files = sorted(source_dir.glob("*.tif"), key=lambda p: p.name.lower())
    for image_file in tif_files:
        base_name = _extract_base_name(image_file.name, sensor_suffix=sensor_suffix)
        if base_name is None:
            stats.skipped_name_mismatch += 1
            continue

        label_file = labels_dir / f"{base_name}{label_suffix}"
        if not label_file.exists():
            stats.skipped_missing_label += 1
            continue

        target_name = f"{base_name}.tif"
        target_image = target_image_dir / target_name
        target_mask = target_mask_dir / target_name

        copied = _copy_pair(
            image_path=image_file,
            label_path=label_file,
            target_image_path=target_image,
            target_mask_path=target_mask,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        if copied:
            stats.copied_pairs += 1
            stats.copied_images += 1
            stats.copied_masks += 1
        else:
            stats.skipped_existing_target += 1
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Sen1Floods11 HandLabeled data into project-compatible roots "
            "(same filename for image/mask, with Floodmaps folder)."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Path to Sen1Floods_Raw root (contains data/flood_events/HandLabeled).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Project dataset root where prepared folders will be created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without copying files.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="",
        help="Optional path to save a JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()

    hand_labeled_dir = raw_dir / "data" / "flood_events" / "HandLabeled"
    labels_dir = hand_labeled_dir / "LabelHand"
    s1_raw_dir = hand_labeled_dir / "S1Hand"
    s2_raw_dir = hand_labeled_dir / "S2Hand"

    s1_new_root = dataset_dir / "Sen1Floods_S1_Global"
    s1_img_dir = s1_new_root / "S1"
    s1_mask_dir = s1_new_root / "Floodmaps"

    s2_new_root = dataset_dir / "Sen1Floods_S2_Global"
    s2_img_dir = s2_new_root / "S2"
    s2_mask_dir = s2_new_root / "Floodmaps"

    _ensure_dirs([s1_img_dir, s1_mask_dir, s2_img_dir, s2_mask_dir])

    if not hand_labeled_dir.exists():
        raise FileNotFoundError(
            f"HandLabeled folder not found: {hand_labeled_dir}\n"
            "Expected layout: Sen1Floods_Raw/data/flood_events/HandLabeled"
        )
    if not labels_dir.exists():
        raise FileNotFoundError(f"LabelHand folder not found: {labels_dir}")

    print(f"[info] raw_dir      : {raw_dir}")
    print(f"[info] dataset_dir  : {dataset_dir}")
    print(f"[info] overwrite    : {bool(args.overwrite)}")
    print(f"[info] dry_run      : {bool(args.dry_run)}")
    print(f"[info] labels_dir   : {labels_dir}")

    print("\n[step] Preparing S1...")
    s1_stats = _prepare_sensor(
        sensor_name="S1",
        source_dir=s1_raw_dir,
        labels_dir=labels_dir,
        sensor_suffix="_S1Hand.tif",
        label_suffix="_LabelHand.tif",
        target_image_dir=s1_img_dir,
        target_mask_dir=s1_mask_dir,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        log=print,
    )

    print("\n[step] Preparing S2...")
    s2_stats = _prepare_sensor(
        sensor_name="S2",
        source_dir=s2_raw_dir,
        labels_dir=labels_dir,
        sensor_suffix="_S2Hand.tif",
        label_suffix="_LabelHand.tif",
        target_image_dir=s2_img_dir,
        target_mask_dir=s2_mask_dir,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        log=print,
    )

    report = {
        "status": "ok",
        "raw_dir": str(raw_dir),
        "dataset_dir": str(dataset_dir),
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "outputs": {
            "s1_root": str(s1_new_root),
            "s2_root": str(s2_new_root),
        },
        "s1": s1_stats.__dict__,
        "s2": s2_stats.__dict__,
        "total_pairs": int(s1_stats.copied_pairs + s2_stats.copied_pairs),
    }

    print("\n[done] Preparation finished.")
    print(
        "S1 pairs prepared: "
        f"{s1_stats.copied_pairs} "
        f"(missing label: {s1_stats.skipped_missing_label}, "
        f"existing target skipped: {s1_stats.skipped_existing_target})"
    )
    print(
        "S2 pairs prepared: "
        f"{s2_stats.copied_pairs} "
        f"(missing label: {s2_stats.skipped_missing_label}, "
        f"existing target skipped: {s2_stats.skipped_existing_target})"
    )
    print(f"Total prepared pairs: {report['total_pairs']}")

    report_path: Path
    if str(args.report_json).strip():
        report_path = Path(args.report_json).resolve()
    else:
        report_path = dataset_dir / "Sen1Floods_prepare_report.json"
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[report] {report_path}")
    else:
        print("[report] dry-run enabled; report file not written.")


if __name__ == "__main__":
    main()
