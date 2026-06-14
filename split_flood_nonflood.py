from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split paired image/mask samples into flood vs no-flood groups "
            "based on mask pixels."
        )
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "Dataset/S1, France",
            "Dataset/S2, France",
            "Dataset/Sen1Floods_S1_Global",
        ],
        help="Dataset roots to scan.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/flood_split",
        help="Output directory for CSV/JSON reports.",
    )
    parser.add_argument(
        "--flood-ratio-threshold",
        type=float,
        default=0.0,
        help="Image is labeled flood only if flood_ratio > threshold.",
    )
    parser.add_argument(
        "--mask-flood-policy",
        type=str,
        choices=project.MASK_FLOOD_POLICY_CHOICES,
        default=project.DEFAULT_MASK_FLOOD_POLICY,
        help="Mask-to-binary conversion policy used before counting flood/no-flood.",
    )
    return parser.parse_args()


def _safe_label_from_root(root: Path) -> str:
    parts = [p for p in root.parts if p.lower() not in {"dataset", "new folder"}]
    if not parts:
        return root.name or str(root)
    return parts[-1]


def main() -> None:
    args = parse_args()
    roots = [Path(x).resolve() for x in args.roots]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    flood_ratio_threshold = float(max(0.0, args.flood_ratio_threshold))
    project.set_active_mask_flood_policy(getattr(args, "mask_flood_policy", None))

    discovery = project.discover_dataset(roots)

    rows_all: list[dict[str, Any]] = []
    rows_flood: list[dict[str, Any]] = []
    rows_no_flood: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = list(discovery.issues)

    counts_by_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "flood": 0, "no_flood": 0}
    )
    counts_by_sensor: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "flood": 0, "no_flood": 0}
    )

    for pair in sorted(
        discovery.pairs, key=lambda p: (p.sensor, str(p.root), p.filename)
    ):
        try:
            y = project.to_binary_mask(
                project.load_mask(pair.mask_path),
                mask_path=pair.mask_path,
            )
            total_pixels = int(y.size)
            flood_pixels = int(np.sum(y > 0))
            flood_ratio = float(flood_pixels / max(1, total_pixels))
            label = "flood" if flood_ratio > flood_ratio_threshold else "no_flood"
        except Exception as ex:
            issues.append(
                project.make_issue(
                    "split_flood_nonflood",
                    "mask_read_failed",
                    sensor=pair.sensor,
                    root=pair.root,
                    filename=pair.filename,
                    image_path=pair.image_path,
                    mask_path=pair.mask_path,
                    details=str(ex),
                )
            )
            continue

        source_label = _safe_label_from_root(pair.root)
        row = {
            "source_root": str(pair.root),
            "source_label": source_label,
            "sensor": pair.sensor,
            "filename": pair.filename,
            "image_path": str(pair.image_path),
            "mask_path": str(pair.mask_path),
            "flood_label": label,
            "flood_pixels": flood_pixels,
            "total_pixels": total_pixels,
            "flood_ratio": flood_ratio,
        }
        rows_all.append(row)
        if label == "flood":
            rows_flood.append(row)
        else:
            rows_no_flood.append(row)

        counts_by_source[source_label]["total"] += 1
        counts_by_source[source_label][label] += 1
        counts_by_sensor[pair.sensor]["total"] += 1
        counts_by_sensor[pair.sensor][label] += 1

    all_csv = output_dir / "all_pairs_flood_split.csv"
    flood_csv = output_dir / "flood_images.csv"
    no_flood_csv = output_dir / "no_flood_images.csv"
    issues_csv = output_dir / "split_issues.csv"
    summary_json = output_dir / "split_summary.json"

    fieldnames = [
        "source_root",
        "source_label",
        "sensor",
        "filename",
        "image_path",
        "mask_path",
        "flood_label",
        "flood_pixels",
        "total_pixels",
        "flood_ratio",
    ]
    project.write_csv(all_csv, rows_all, fieldnames=fieldnames)
    project.write_csv(flood_csv, rows_flood, fieldnames=fieldnames)
    project.write_csv(no_flood_csv, rows_no_flood, fieldnames=fieldnames)
    project.write_csv(
        issues_csv,
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

    summary = {
        "status": "ok",
        "roots": [str(r) for r in roots],
        "total_pairs": int(len(rows_all)),
        "flood_count": int(len(rows_flood)),
        "no_flood_count": int(len(rows_no_flood)),
        "flood_ratio_threshold": float(flood_ratio_threshold),
        "mask_flood_policy": str(project.ACTIVE_MASK_FLOOD_POLICY),
        "issues_count": int(len(issues)),
        "counts_by_source": {k: dict(v) for k, v in sorted(counts_by_source.items())},
        "counts_by_sensor": {k: dict(v) for k, v in sorted(counts_by_sensor.items())},
        "outputs": {
            "all_pairs_csv": str(all_csv),
            "flood_csv": str(flood_csv),
            "no_flood_csv": str(no_flood_csv),
            "issues_csv": str(issues_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
