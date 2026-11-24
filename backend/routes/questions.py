from __future__ import annotations

import json
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Response

from backend.deps import get_store
from backend.errors import ApiError
from backend.schemas.questions import QuestionsRequest, QuestionsResponse, ErrorResponse
from backend.services import evidence_service, spatial_service
from backend.utils.storage import AnalysisStore, utcnow
from reasoning.official import validate_official

router = APIRouter(tags=["questions"])
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Unknown analysis id"}}


@router.post("/questions", response_model=QuestionsResponse, responses=NOT_FOUND,
             summary="Answer official-style questions about an analysis")
def ask(req: QuestionsRequest, store: AnalysisStore = Depends(get_store)):
    """Supports the four PS-04 question types, either as natural language (official templates) or as a structured
    object. Answers are computed by the deterministic spatial reasoning engine from the stored building predictions;
    no language model is involved.

    Each result separates the **official** submission record (`id`, `answer`, `region` only) from **internal**
    metadata (supporting building ids, evidence, tie flags)."""
    d = store.path(req.analysis_id)                     # 404 for unknown ids
    scene = spatial_service.load_scene(store, req.analysis_id)

    ids = [q.id for q in req.questions if q.id]
    if len(ids) != len(set(ids)):
        raise ApiError(422, "duplicate_question_id", "Question ids must be unique within a request.")

    results, saved = [], {}
    for item in req.questions:
        qid = item.id or f"q{uuid.uuid4().hex[:8]}"
        query, ans, record, text = spatial_service.answer_item(scene, item, qid, req.min_confidence, req.region_format)
        internal = {
            "question_type": spatial_service.LETTER_TO_TYPE[ans.qtype], "question_text": text,
            "parsed": {"level": query.level, "scope": [query.scope_kind, query.scope_name],
                       "halves": list(query.halves) if query.halves else None, "reason": query.reason},
            "details": ans.details,
            "evidence": evidence_service.evidence_payload(req.analysis_id, scene, ans),
        }
        results.append({"official": record, "internal": internal})
        saved[qid] = {"official": record, "internal": internal, "question_text": text, "answered_utc": utcnow(),
                      "min_confidence": req.min_confidence}

    with store.locked(req.analysis_id):                 # read-modify-write is serialised per analysis
        records = store.questions(req.analysis_id)
        records.update(saved)                           # same id -> latest answer replaces the earlier one
        store.save_questions(req.analysis_id, records)
    return {"analysis_id": req.analysis_id, "results": results}


@router.get("/analysis/{analysis_id}/questions", responses=NOT_FOUND,
            summary="Questions answered so far, with their internal evidence (not the official format)")
def list_questions(analysis_id: str, store: AnalysisStore = Depends(get_store)):
    """Official record plus supporting building ids per question, so a UI can show which questions cite a building."""
    store.path(analysis_id)
    return {"analysis_id": analysis_id, "questions": [
        {"id": qid, "question_text": r.get("question_text"), "official": r["official"],
         "supporting_building_ids": r["internal"]["evidence"]["supporting_building_ids"],
         "evidence_region": r["internal"]["evidence"]["region"], "answered_utc": r.get("answered_utc")}
        for qid, r in store.questions(analysis_id).items()]}


@router.get("/analysis/{analysis_id}/export", responses=NOT_FOUND, summary="Official PS-04 answers for this analysis",
            response_class=Response)
def export(analysis_id: str, format: Annotated[Literal["json", "jsonl"], Query()] = "json",
           store: AnalysisStore = Depends(get_store)):
    """Returns the official submission records for every question asked so far, in the order first asked:
    exactly `{"id", "answer", "region"}` per record. `format=json` gives an array, `format=jsonl` one record per
    line (the PS-04 submission format). An analysis with no questions yet exports an empty list."""
    store.path(analysis_id)
    meta = store.meta(analysis_id)
    records = []
    for entry in store.questions(analysis_id).values():
        rec = entry["official"]
        validate_official(rec, meta["image"]["width"], meta["image"]["height"])   # compliance re-checked on export
        records.append(rec)
    headers = {"X-Analysis-Id": analysis_id, "X-Record-Count": str(len(records))}
    if format == "jsonl":
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        return Response(body, media_type="application/x-ndjson", headers=headers)
    return Response(json.dumps(records, ensure_ascii=False), media_type="application/json", headers=headers)
