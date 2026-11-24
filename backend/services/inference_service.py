"""Model lifecycle and inference. Reuses inference.pipeline.DamagePipeline unchanged.

* Models are loaded ONCE (application startup), never per request.
* One inference runs at a time (single GPU; ultralytics predictors are not
  thread-safe). Up to `max_pending` requests may queue; beyond that -> 503.
* GPU memory: YOLO runs at imgsz 640 by default and the Siamese crops are
  classified in fp16 batches of 64; the CUDA cache is released after each request.
  A CUDA out-of-memory error is reported as 503 instead of crashing the server.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
from pathlib import Path

import torch

from backend.config import Settings
from backend.errors import ApiError
from inference.pipeline import DamagePipeline, InferenceError, PipelineConfig

log = logging.getLogger("eagle.inference")


class InferenceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline: DamagePipeline | None = None
        self.load_error: str | None = None
        self._gpu_lock = threading.Lock()
        self._count_lock = threading.Lock()
        self._inflight = 0                      # running + waiting analyses
        self._capacity = 1 + settings.max_pending

    # ---- lifecycle --------------------------------------------------------
    def load(self) -> None:
        s = self.settings
        try:
            self.pipeline = DamagePipeline(PipelineConfig(
                yolo_weights=s.yolo_weights, siamese_weights=s.siamese_weights,
                conf=s.default_conf, imgsz=s.default_imgsz, device=s.device))
            self.load_error = None
            log.info("Models loaded on %s", self.pipeline.device)
        except Exception as exc:      # keep the API up so /api/health can explain what failed
            self.pipeline, self.load_error = None, f"{type(exc).__name__}: {exc}"
            log.error("Model loading failed: %s", self.load_error)

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def status(self) -> dict:
        s = self.settings
        cuda = torch.cuda.is_available()
        gpu = None
        if cuda:
            free, total = torch.cuda.mem_get_info()
            gpu = {"name": torch.cuda.get_device_name(0), "memory_free_mb": free // 2**20,
                   "memory_total_mb": total // 2**20}
        return {
            "yolo": {"file": Path(s.yolo_weights).name, "exists": Path(s.yolo_weights).is_file(),
                     "loaded": self.ready},
            "siamese": {"file": Path(s.siamese_weights).name, "exists": Path(s.siamese_weights).is_file(),
                        "loaded": self.ready},
            "load_error": self.load_error,
            "device_in_use": str(self.pipeline.device) if self.ready else None,
            "cuda_available": cuda,
            "gpu": gpu,
        }

    # ---- admission control --------------------------------------------------
    def check_available(self) -> None:
        """Cheap early refusal, called before a large upload is read."""
        if not self.ready:
            raise ApiError(503, "models_unavailable", "Models are not loaded; see GET /api/health.",
                           {"load_error": self.load_error})
        if self._inflight >= self._capacity:
            raise ApiError(503, "server_busy", "Too many analyses are queued; retry shortly.",
                           headers={"Retry-After": "15"})

    def _enter(self) -> None:
        with self._count_lock:
            if self._inflight >= self._capacity:
                raise ApiError(503, "server_busy", "Too many analyses are queued; retry shortly.",
                               headers={"Retry-After": "15"})
            self._inflight += 1

    def _leave(self) -> None:
        with self._count_lock:
            self._inflight -= 1

    # ---- inference --------------------------------------------------------
    def analyze(self, before_path: Path, after_path: Path, conf: float, imgsz: int, alignment: str) -> tuple[dict, dict]:
        self.check_available()
        self._enter()
        try:
            with self._gpu_lock:
                base = self.pipeline.cfg
                # per-request settings; the shared pipeline object is only touched under the lock
                self.pipeline.cfg = dataclasses.replace(base, conf=conf, imgsz=imgsz, alignment=alignment)
                try:
                    return self.pipeline.run(before_path, after_path)
                finally:
                    self.pipeline.cfg = base
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        except InferenceError as exc:
            raise ApiError(422, "inference_input_error", str(exc)) from exc
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise ApiError(503, "gpu_out_of_memory",
                           "GPU memory exhausted. Retry with a smaller imgsz or use EAGLE_DEVICE=cpu.") from exc
        finally:
            self._leave()


def sanitize_report(report: dict, analysis_id: str, before_file: str, after_file: str) -> dict:
    """Remove server filesystem paths from a Phase 1 report before it is stored or served."""
    out = dict(report)
    out["analysis_id"] = analysis_id
    out["inputs"] = {**report["inputs"], "before_image": before_file, "after_image": after_file}
    models = {}
    for name, info in report["models"].items():
        info = dict(info)
        info["path"] = Path(info["path"]).name
        models[name] = info
    out["models"] = models
    out.pop("outputs", None)
    return out
