"""Backend settings. Every value can be overridden with an EAGLE_* environment variable.

Model paths and YOLO defaults come from the project's root config.py (single
source of truth); only API-specific limits live here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import config as project_config  # root config.py (not this module)


def _env(name: str, default, cast=str):
    raw = os.environ.get(f"EAGLE_{name}")
    return default if raw is None or raw == "" else cast(raw)


@dataclass
class Settings:
    # ---- models (from the root project config) ---------------------------
    yolo_weights: Path = field(default_factory=lambda: Path(_env("YOLO_WEIGHTS", project_config.YOLO_WEIGHTS, Path)))
    siamese_weights: Path = field(default_factory=lambda: Path(_env("SIAMESE_WEIGHTS", project_config.SIAMESE_WEIGHTS, Path)))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))          # auto | cpu | cuda | cuda:N
    # imgsz 640 / conf 0.25 are what the models were trained and evaluated at; 640 also keeps
    # YOLO well inside 4 GB VRAM. Larger sizes are allowed per request but capped below.
    default_conf: float = project_config.YOLO_CONF
    default_imgsz: int = project_config.YOLO_IMGSZ
    max_imgsz: int = field(default_factory=lambda: _env("MAX_IMGSZ", 1280, int))

    # ---- uploads ----------------------------------------------------------
    allowed_extensions: tuple = (".jpg", ".jpeg", ".png")
    max_upload_bytes: int = field(default_factory=lambda: _env("MAX_UPLOAD_MB", 25, int) * 1024 * 1024)
    min_side_px: int = field(default_factory=lambda: _env("MIN_SIDE_PX", 64, int))
    max_side_px: int = field(default_factory=lambda: _env("MAX_SIDE_PX", 8192, int))
    max_pixels: int = field(default_factory=lambda: _env("MAX_PIXELS", 4096 * 4096, int))

    # ---- storage ----------------------------------------------------------
    storage_dir: Path = field(default_factory=lambda: Path(_env("STORAGE_DIR", project_config.OUTPUTS_DIR / "analyses", Path)))

    # ---- reports ----------------------------------------------------------
    reports_dir: Path = field(default_factory=lambda: Path(_env("REPORTS_DIR", project_config.OUTPUTS_DIR / "reports", Path)))
    # Optional LLM explanation layer. OFF unless EAGLE_LLM_ENABLED=1 *and* ANTHROPIC_API_KEY is set. When on, only
    # aggregate counts/ids are sent (no images, filenames or user labels). Any failure falls back to the template.
    llm_enabled: bool = field(default_factory=lambda: _env("LLM_ENABLED", "0") == "1")
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "claude-sonnet-5-5"))
    llm_timeout_s: float = field(default_factory=lambda: _env("LLM_TIMEOUT_S", 30.0, float))

    # ---- concurrency ------------------------------------------------------
    # One inference at a time (one GPU, ultralytics predictors are not thread-safe);
    # up to max_pending requests may wait. Beyond that the API answers 503 + Retry-After.
    max_pending: int = field(default_factory=lambda: _env("MAX_PENDING", 4, int))

    # ---- misc -------------------------------------------------------------
    cors_origins: tuple = field(default_factory=lambda: tuple(
        o.strip() for o in _env("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173").split(",") if o.strip()))
    min_confidence_default: float = 0.0


def get_settings() -> Settings:
    return Settings()
