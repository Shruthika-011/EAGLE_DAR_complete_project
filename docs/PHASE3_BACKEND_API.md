# Phase 3 — FastAPI backend

Connects the existing modules through REST: `inference.pipeline` (YOLOv8-seg + Siamese ResNet18),
`reasoning` (deterministic spatial engine, evidence, five-class map, report facts). The backend adds no
inference, counting or reasoning logic of its own.

```
backend/
  main.py                    app factory, lifespan (models loaded once), CORS, routers under /api
  config.py                  limits and EAGLE_* env overrides (model paths come from the root config.py)
  errors.py                  single error envelope + handlers
  deps.py                    access to app-level settings / store / inference service
  routes/    health.py  analysis.py  questions.py  reports.py
  services/  inference_service.py  spatial_service.py  evidence_service.py  report_service.py
  schemas/   analysis.py  questions.py  reports.py
  utils/     image_validation.py  storage.py
  requirements.txt
```

## Run

Use the **base Python 3.11** (it has ultralytics, torch+CUDA, FastAPI). The project `.venv` has neither
ultralytics nor FastAPI; install `backend/requirements.txt` there first if you prefer it.

```powershell
cd C:\Users\shrut\Desktop\EAGLE-DAR\EAGLE_DAR_complete_project
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

* Swagger UI: http://127.0.0.1:8000/docs   ·   ReDoc: /redoc   ·   OpenAPI: /openapi.json
* Run from the project root (the backend imports `config`, `inference`, `reasoning`).
* Use **one** uvicorn worker. Each worker loads its own copy of the models and has its own GPU lock.
* Startup takes ~8 s (model load). `GET /api/health` returns 200 once both models are loaded, 503 otherwise.

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `EAGLE_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N` (CPU fallback is automatic when CUDA is absent) |
| `EAGLE_YOLO_WEIGHTS` / `EAGLE_SIAMESE_WEIGHTS` | from `config.py` | model files |
| `EAGLE_STORAGE_DIR` | `outputs/analyses` | where analyses are saved |
| `EAGLE_MAX_UPLOAD_MB` | 25 | per image |
| `EAGLE_MAX_PIXELS` | 16 777 216 | per image (4096²) |
| `EAGLE_MIN_SIDE_PX` / `EAGLE_MAX_SIDE_PX` | 64 / 8192 | side length range |
| `EAGLE_MAX_IMGSZ` | 1280 | largest YOLO `imgsz` a request may ask for |
| `EAGLE_MAX_PENDING` | 4 | analyses allowed to wait behind the running one |
| `EAGLE_CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | comma-separated |

## Endpoints

| Method / path | Purpose |
|---|---|
| `GET /api/health` | status, model availability, CUDA / GPU memory, limits |
| `POST /api/analyze` | multipart `before`, `after` (+ optional `conf`, `imgsz`, `alignment`, and user labels `disaster_name`, `district`, `state` stored as typed in `context`) → 201 analysis |
| `GET /api/analysis/{id}` | saved analysis (`?include_polygons=true` adds polygons) |
| `GET /api/analysis/{id}/map` | five-class damage map PNG; `?evidence=x1,y1,x2,y2&highlight=1,2,3&legend=false` re-renders |
| `GET /api/analysis/{id}/image/{before\|after}` | the uploaded image exactly as stored (added for the frontend) |
| `GET /api/analysis/{id}/comparison` | before │ after │ damage map strip (JPEG) |
| `POST /api/questions` | natural-language or structured questions → official records + internal evidence |
| `GET /api/analysis/{id}/export` | official PS-04 records asked so far (`?format=json` array or `jsonl`) |
| `POST /api/analysis/{id}/reports` | **start generating the PDF report** (202; returns a report id to poll). Always creates a new report; an in-flight one is returned instead of a duplicate (200) |
| `GET /api/analysis/{id}/reports` | reports generated for this analysis, newest first |
| `GET /api/reports/{report_id}` | status: `queued/generating/complete/failed`, real `stage` and `progress`, pages, size, sha256, explanation source, error |
| `GET /api/reports/{report_id}/download` | the PDF (`?inline=true` to show in the browser); 409 `report_not_ready` until complete |
| `GET /api/analysis/{id}/questions` | answered questions with supporting building ids (internal view, not the official format) |
| `GET /api/analysis/{id}/report` | quick HTML/JSON view with the same findings and recommendations (`?format=json`, `?refresh=true`) |

Real captured request/response pairs are in [`docs/examples/`](examples/): `analyze_response.json`
(buildings trimmed to 2), `questions_request.json`, `questions_response.json`, `export_response.json`,
`report_response.json` (facts trimmed), `health.json`, `damage_map.png`, `damage_map_evidence.png`.

### Examples

```powershell
curl.exe -s -F before=@dataset_source/practice/images/pb67aa0fe_pre.jpg `
            -F after=@dataset_source/practice/images/pb67aa0fe_post.jpg `
            -F conf=0.25 -F imgsz=640 http://127.0.0.1:8000/api/analyze
```

```json
POST /api/questions
{"analysis_id": "<id>", "questions": [
  {"id": "q1", "question": "Count the destroyed buildings in the image."},
  {"id": "q2", "structured": {"type": 3, "halves": ["left", "right"]}},
  {"id": "q3", "question": "Which building in the image is a hospital?"}]}
```

Each result separates what may be submitted from what is internal:

```json
{"official": {"id": "q1", "answer": 4, "region": [155, 0, 737, 839]},
 "internal": {"question_type": 1, "question_text": "...", "parsed": {...}, "details": {...},
              "evidence": {"region": [155,0,737,839], "supporting_building_ids": [11,21,39,53],
                           "supporting_buildings": [...], "map_url": "/api/analysis/<id>/map?evidence=...&highlight=11,21,39,53"}}}
```

Structured questions: `{"type":1,"level":"destroyed","scope":{"kind":"image"}}` ·
`{"type":1,"level":"no-damage","scope":{"kind":"quadrant","name":"top-left"}}` ·
`{"type":2,"level":"damaged"}` (or `"destroyed"`) · `{"type":3,"halves":["bottom","top"]}` · `{"type":4}`.
Levels: `no-damage, major-damage, destroyed, damaged (= major+destroyed)`. Minor damage is merged into major damage; questions naming minor or un-classified buildings answer "cannot be determined".

Official records are exactly `{"id","answer","region"}`; every record is validated by
`reasoning.official.validate_official` before it is returned and again on export. Spatial rules, tie handling and the
open questions about the official key are in [PHASE2_SPATIAL_RULES.md](PHASE2_SPATIAL_RULES.md).
Re-using a question id replaces the earlier answer; omitted ids are generated (`q` + 8 hex).

## Errors

Every error uses one envelope: `{"error": {"code": "...", "message": "...", "details": {...}}}`.

| HTTP | `code` | Cause |
|---|---|---|
| 413 | `file_too_large` / `image_too_large` | over the byte limit / pixel limit |
| 415 | `unsupported_file_type` / `extension_mismatch` | not JPG/JPEG/PNG by extension or magic bytes / extension disagrees with content |
| 422 | `corrupted_image` / `empty_file` | truncated or undecodable / zero bytes |
| 422 | `invalid_dimensions` / `dimension_mismatch` | side outside limits / before and after differ in size |
| 422 | `invalid_imgsz` / `validation_error` / `duplicate_question_id` | bad parameters or request body (`details.errors` lists fields) |
| 422 | `invalid_evidence` / `invalid_highlight` | bad map query parameters |
| 422 | `inference_input_error` | the pipeline rejected the images |
| 404 | `analysis_not_found` | unknown, malformed or traversal-style id (also generic `not_found` for unknown routes) |
| 405 | `method_not_allowed` | wrong HTTP method |
| 503 | `models_unavailable` | weights missing or failed to load (reason in `details.load_error`) |
| 503 | `server_busy` | queue full; has `Retry-After: 15` |
| 503 | `gpu_out_of_memory` | CUDA OOM; retry with smaller `imgsz` or `EAGLE_DEVICE=cpu` |
| 500 | `internal_error` / `corrupt_analysis` | unexpected error (details only in the server log) / stored data unreadable |

## Image handling and storage

* Accepted: JPG, JPEG, PNG. Validated in order: size (streamed, stops early) → extension → magic bytes →
  header-only dimension read (refuses decompression bombs before decoding) → full decode → dimension limits →
  before/after dimensions equal. The client filename is sanitised for display only and **never used as a path**;
  stored names are fixed (`before.jpg`, `after.png`, ...) and chosen from the detected content type.
* Each analysis gets a random `uuid4` hex id and its own directory created with `exist_ok=False`, so a new
  analysis can never overwrite an old one. `meta.json` is written last; a failed request deletes its directory.
* Ids in URLs must match `^[0-9a-f]{32}$` and resolve inside the storage root.
* Stored `predictions.json` has server filesystem paths removed.
* Analyses are kept until you delete the folder; there is no authentication or retention policy yet.

## Performance and the 4 GB GPU

* Models load **once** in the FastAPI lifespan and are reused; a test asserts the same objects serve every request.
* One inference at a time (single GPU; ultralytics predictors are not thread-safe). Up to `EAGLE_MAX_PENDING` further
  requests may wait; the next gets 503 `server_busy`. Upload parsing and rendering run in parallel with inference.
* Defaults match training/evaluation: YOLO `imgsz 640`, `conf 0.25`. Siamese crops are classified in fp16 autocast
  batches of 64; the CUDA cache is released after each request. YOLO + Siamese use roughly 0.5 GB on the RTX 3050.
* Measured here (RTX 3050 Laptop, 1024×1024 pair, GPU shared with a training job for part of the run): the
  inference part of `POST /api/analyze` is reported in `parameters.inference_seconds`.

## Tests

```powershell
python -m pytest tests -q          # 90 Phase 2 tests + 35 backend tests (real models, real practice images)
```

The backend tests assert, among others: models loaded once; API results equal an independent `DamagePipeline` run;
per-building confidence = detection × damage probability; every id returns its own data; concurrent requests are
safe and deterministic; invalid uploads store nothing; hostile filenames and ids are harmless; export is official
and isolated per analysis.

## Known limitations

* The report is **template-based over computed facts** (no language model). It reads measured limitations from
  `outputs/evaluation/*.json` and states none if those files are missing.
* Detection recall of the YOLO model is low (≈0.12 building recall at IoU 0.5 on the practice scenes), so counts
  are strong under-estimates. This is a model limitation, not an API one, and the report says so.
* `confidence_level` thresholds (0.60 / 0.35) are heuristic; with the current models most buildings are "low".
* Questions are matched against the official templates; anything else becomes type 4 ("cannot be determined").
* No authentication, rate limiting per client, or automatic deletion of stored analyses.

## PDF reports (Phase 5)

`POST /api/analysis/{id}/reports` renders the *Disaster Damage Assessment Report* in one background worker (ReportLab) from the stored
predictions, the deterministic reasoning engine and the evaluation files, and stores it as `outputs/reports/<report_id>/report.pdf`
(git-ignored; written to a temp name and renamed when complete; interrupted jobs are marked failed on restart). Sections: cover,
executive summary, before/after, classification, building table (every building), five-class map, spatial assessment, evidence and
question answers with evidence crops, confidence and reliability, key findings, recommendations (labelled AI-assisted, require human
verification), methodology, limitations. Text is tagged COMPUTED / MODEL PREDICTION / AI EXPLANATION.

Explanation layer: a deterministic template by default. `EAGLE_LLM_ENABLED=1` plus `ANTHROPIC_API_KEY` lets an LLM rephrase three
paragraphs; it receives only aggregate counts and ids, and its text is discarded (template used, reason recorded in
`fallback_reason`) if it contains any number not present in those facts, is not valid JSON, or the call fails. The LLM path has not been
exercised against the real service in this repository's tests; the guard and the fallback have.

New error codes: `report_not_found` (404), `report_not_ready` (409). A failed PDF returns `status: failed` with a safe message
(exception type only, no paths); the analysis is unaffected and generation can be retried.
