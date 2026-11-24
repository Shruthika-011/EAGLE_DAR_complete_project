from __future__ import annotations

import os

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from backend import API_VERSION
from backend.deps import get_inference, get_settings, get_store
from backend.services.inference_service import InferenceService

router = APIRouter(tags=["health"])


@router.get("/health", summary="Backend status, model availability and CUDA")
def health(inference: InferenceService = Depends(get_inference), settings=Depends(get_settings),
           store=Depends(get_store)):
    """Returns 200 when both models are loaded and storage is writable, otherwise 503 with the reason."""
    st = inference.status()
    storage_ok = os.access(store.root, os.W_OK)
    ok = inference.ready and storage_ok
    body = {
        "status": "ok" if ok else "degraded",
        "api_version": API_VERSION,
        "models": {"yolo": st["yolo"], "siamese": st["siamese"], "load_error": st["load_error"]},
        "device": {"in_use": st["device_in_use"], "requested": settings.device,
                   "cuda_available": st["cuda_available"], "gpu": st["gpu"]},
        "storage": {"writable": storage_ok},
        "reports": {"pdf": True, "llm_enabled": settings.llm_enabled,
                    "llm_key_present": bool(os.environ.get("ANTHROPIC_API_KEY")),   # presence only, never the value
                    "explanation": "llm (verified) with template fallback" if settings.llm_enabled else "deterministic template"},
        "limits": {"max_upload_mb": settings.max_upload_bytes // (1024 * 1024),
                   "max_pixels": settings.max_pixels, "max_pending_requests": settings.max_pending},
    }
    return JSONResponse(body, status_code=200 if ok else 503)
