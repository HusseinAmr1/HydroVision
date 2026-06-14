from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


TRUSTED_MODEL_EXTENSIONS = {".joblib", ".pkl"}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except Exception:
        return False


def _default_trusted_roots() -> list[Path]:
    base_dir = Path(__file__).resolve().parent
    roots = [
        base_dir.resolve(),
        (base_dir / "outputs").resolve(),
        Path.cwd().resolve(),
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def validate_trusted_model_path(path: Path, *, allowed_roots: list[Path] | None = None) -> Path:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"model file not found: {p}")
    if p.suffix.lower() not in TRUSTED_MODEL_EXTENSIONS:
        exts = ", ".join(sorted(TRUSTED_MODEL_EXTENSIONS))
        raise ValueError(f"untrusted model extension: {p.suffix} (allowed: {exts})")

    roots = [Path(x).resolve() for x in (allowed_roots or _default_trusted_roots())]
    if not any(_is_relative_to(p, root) for root in roots):
        raise ValueError(
            "refusing to load model from an untrusted location. "
            f"path={p}, trusted_roots={[str(r) for r in roots]}"
        )
    return p


def safe_joblib_load(
    path: str | Path,
    *,
    allowed_roots: list[Path] | None = None,
    allow_untrusted: bool = False,
) -> Any:
    p = Path(path).resolve()
    if allow_untrusted:
        if not p.exists():
            raise FileNotFoundError(f"model file not found: {p}")
        return joblib.load(p)
    trusted_path = validate_trusted_model_path(p, allowed_roots=allowed_roots)
    return joblib.load(trusted_path)

