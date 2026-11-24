from __future__ import annotations

import time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile
from fastapi.responses import FileResponse

from backend.config import Settings
from backend.deps import get_inference, get_settings, get_store
from backend.errors import ApiError
from backend.schemas.analysis import AnalysisResponse, ErrorResponse
from backend.services import evidence_service, spatial_service
from backend.services.inference_service import InferenceService, sanitize_report
from backend.utils import image_validation as iv
from backend.utils.storage import AnalysisStore, utcnow, write_json_atomic
from inference.pipeline import render_comparison, write_image
from reasoning.damage_map import render_damage_map

router = APIRouter(tags=["analysis"])

ERRORS = {413: {"model": ErrorResponse, "description": "File too large"},
          415: {"model": ErrorResponse, "description": "Unsupported or mismatched file type"},
          422: {"model": ErrorResponse, "description": "Invalid image, dimensions or parameters"},
          503: {"model": ErrorResponse, "description": "Models unavailable, server busy or GPU out of memory"}}
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Unknown analysis id"}}


def clean_label(text: str | None) -> str | None:
    """Trim, drop control characters, collapse whitespace; empty -> None."""
    if text is None:
        return None
    cleaned = " ".join("".join(ch for ch in text if ch.isprintable()).split())
    return cleaned or None


@router.post("/analyze", response_model=AnalysisResponse, status_code=201, responses=ERRORS,
             summary="Analyse a before/after aerial image pair")
def analyze(
    before: Annotated[UploadFile, File(description="Pre-disaster image (JPG, JPEG or PNG)")],
    after: Annotated[UploadFile, File(description="Post-disaster image (JPG, JPEG or PNG), same size as `before`")],
    conf: Annotated[float | None, Form(gt=0, lt=1, description="YOLO confidence threshold (default 0.25)")] = None,
    imgsz: Annotated[int | None, Form(description="YOLO inference size, multiple of 32 (default 640)")] = None,
    alignment: Annotated[Literal["check", "register"], Form(
        description="`check` measures and reports the pre/post offset; `register` also translates "
                    "the before image onto the after image when an offset is found.")] = "check",
    disaster_name: Annotated[str | None, Form(max_length=100, description="Optional label (informational only)")] = None,
    district: Annotated[str | None, Form(max_length=100, description="Optional, as provided by the user")] = None,
    state: Annotated[str | None, Form(max_length=100, description="Optional, as provided by the user")] = None,
    settings: Settings = Depends(get_settings),
    store: AnalysisStore = Depends(get_store),
    inference: InferenceService = Depends(get_inference),
):
    """Runs YOLOv8-seg building segmentation on the **after** image and the Siamese ResNet18 damage classifier on
    each building's before/after crop. Returns damage statistics, per-building confidence scores and links to the
    generated visualisations. Models are loaded once at startup; requests are serialised on the GPU."""
    inference.check_available()                      # refuse early, before reading large uploads
    conf = settings.default_conf if conf is None else conf
    imgsz = settings.default_imgsz if imgsz is None else imgsz
    if imgsz % 32 or not 32 <= imgsz <= settings.max_imgsz:
        raise ApiError(422, "invalid_imgsz", f"imgsz must be a multiple of 32 between 32 and {settings.max_imgsz}.")

    imgs = {}
    for role, up in (("before", before), ("after", after)):
        data = iv.read_limited(up.file, settings.max_upload_bytes, role)
        imgs[role] = iv.validate_upload(data, up.filename, role, settings)
    iv.validate_pair(imgs["before"], imgs["after"])

    analysis_id, d = store.new_dir()          # fresh uuid directory; never overwrites an earlier analysis
    try:
        files = {role: f"{role}{v.extension}" for role, v in imgs.items()}
        for role, v in imgs.items():
            (d / files[role]).write_bytes(v.data)

        t0 = time.perf_counter()
        report, arrays = inference.analyze(d / files["before"], d / files["after"], conf, imgsz, alignment)
        seconds = round(time.perf_counter() - t0, 3)

        write_json_atomic(d / "predictions.json", sanitize_report(report, analysis_id, files["before"], files["after"]))
        scene = spatial_service.load_scene_from_dir(d, analysis_id)
        dmap = render_damage_map(arrays["post"], scene)
        write_image(d / "damage_map.png", dmap)
        write_image(d / "comparison.jpg", render_comparison(arrays["pre_used"], arrays["post"], dmap))

        write_json_atomic(d / "meta.json", {            # written last: its presence marks the analysis complete
            "analysis_id": analysis_id, "created_utc": utcnow(), "status": "complete",
            # user-supplied labels: stored verbatim (cleaned), never derived from the imagery
            "context": {"disaster_name": clean_label(disaster_name), "district": clean_label(district),
                        "state": clean_label(state)},
            "image": {"width": imgs["after"].width, "height": imgs["after"].height,
                      **{role: {"display_name": v.display_name, "bytes": len(v.data), "sha256": v.sha256}
                         for role, v in imgs.items()}},
            "parameters": {"conf": conf, "imgsz": imgsz, "alignment": alignment,
                           "device": report["parameters"]["device"], "inference_seconds": seconds},
        })
    except BaseException:
        store.discard(d)                              # no half-written analyses
        raise
    return spatial_service.build_analysis_view(store, analysis_id)


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse, responses=NOT_FOUND,
            summary="Retrieve a saved analysis")
def get_analysis(analysis_id: str,
                 include_polygons: Annotated[bool, Query(description="Include each building's polygon.")] = False,
                 store: AnalysisStore = Depends(get_store)):
    return spatial_service.build_analysis_view(store, analysis_id, include_polygons)


@router.get("/analysis/{analysis_id}/map", responses={**NOT_FOUND, 422: {"model": ErrorResponse}},
            response_class=Response, summary="Five-class colour-coded damage map (PNG)")
def get_map(analysis_id: str,
            evidence: Annotated[str | None, Query(description="Outline an evidence region: x1,y1,x2,y2")] = None,
            highlight: Annotated[str | None, Query(description="Comma-separated building ids to emphasise")] = None,
            legend: Annotated[bool, Query(description="Include the legend panel")] = True,
            store: AnalysisStore = Depends(get_store)):
    """Without parameters this returns the map saved at analysis time. With `evidence`/`highlight` the map is
    re-rendered (not stored) so any answer's evidence region and supporting buildings can be shown."""
    d = store.path(analysis_id)
    if evidence is None and highlight is None and legend:
        return FileResponse(d / "damage_map.png", media_type="image/png")
    scene = spatial_service.load_scene(store, analysis_id)
    png = evidence_service.render_map_png(
        evidence_service.load_after_image(store, analysis_id), scene,
        evidence_service.parse_evidence_param(evidence, scene.width, scene.height),
        evidence_service.parse_highlight_param(highlight, scene), legend)
    return Response(png, media_type="image/png")


@router.get("/analysis/{analysis_id}/image/{role}", responses=NOT_FOUND, response_class=Response,
            summary="The uploaded before or after image, exactly as stored")
def get_image(analysis_id: str, role: Literal["before", "after"], store: AnalysisStore = Depends(get_store)):
    path = store.image_path(analysis_id, role)
    return FileResponse(path, media_type="image/png" if path.suffix == ".png" else "image/jpeg")


@router.get("/analysis/{analysis_id}/comparison", responses=NOT_FOUND, response_class=Response,
            summary="Before | after | damage map strip (JPEG)")
def get_comparison(analysis_id: str, store: AnalysisStore = Depends(get_store)):
    return FileResponse(store.path(analysis_id) / "comparison.jpg", media_type="image/jpeg")
