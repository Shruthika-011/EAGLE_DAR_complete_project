# EAGLE-DAR final test report (Phase 5)

All results below were produced by running the tests in this repository against the real models, the real practice images,
the running backend and a real Chrome browser. Nothing was mocked except where a test deliberately simulates a failure
(aborted network, failing PDF generator, LLM call that raises); those are named. Model quality is a separate matter: see
[EVALUATION_REPORT.md](EVALUATION_REPORT.md).

## Summary

| Layer | Command | Result |
|---|---|---|
| Python (reasoning, PS-04 oracle, backend API, PDF reports, integrity) | `python -m pytest tests -q` | **231 passed** |
| Frontend unit tests | `cd frontend; npm test` | **15 passed** |
| Frontend type check and production build | `npm run typecheck`, `npm run build` | pass |
| Real-browser end-to-end (demo sequence + failure cases) | `cd frontend; npm run e2e` | **73 checks, 0 failed, no console errors; 3 consecutive runs** |

Python tests by file: `test_spatial_reasoning` 90, `test_ps04_oracle` 54, `test_backend_api` 38, `test_reports` 26, `test_repo_integrity` 20, `test_consistency` 3.

## Bug found and fixed in this phase

The browser test caught an intermittent failure: on Windows `os.replace` raised `PermissionError` when the UI's status poll had the
report's `meta.json` open at the instant the worker rewrote it, which failed the whole report. Fixed three ways (store lock around reads,
retrying rename, progress writes made best-effort) with two regression tests (HTTP status endpoint hammered by 4 threads during generation;
500 atomic writes against polling readers). Three consecutive browser runs passed afterwards. An intermittent bug can still hide; this one was
only found because the browser test polls the way a real user does.

## Backend endpoints tested (real models, real images)

`GET /api/health` · `POST /api/analyze` · `GET /api/analysis/{id}` · `.../map` (default, evidence, highlight, no legend) · `.../comparison` ·
`.../image/{before|after}` · `POST /api/questions` · `GET .../questions` · `GET .../export` (json, jsonl) · `GET .../report` (html, json) ·
`POST .../reports` · `GET .../reports` · `GET /api/reports/{id}` · `GET /api/reports/{id}/download` (attachment, inline) · Swagger/OpenAPI.
Error paths: corrupt, empty, wrong-extension, wrong-type, oversize, tiny, mismatched-size images; bad parameters; unknown/malformed/traversal ids;
duplicate and invalid question ids; model unavailable (503, saved results still readable, reports still work); server busy (503 + Retry-After);
report not ready (409); PDF failure; LLM unavailable.

## Frontend pages tested (browser)

Dashboard (empty state, every KPI against the backend JSON, doughnut, slider, insights, location, Action Center) · Upload (missing before/after,
invalid format, corrupt, size mismatch, real upload, double click = one analysis, progress) · View Results · Damage Map (polygon count/coordinates/colours,
viewBox, hover, selection, zoom, pan, class filter, low-confidence highlight, evidence references) · Spatial questions (5 types vs the engine) ·
Analytics · Reports (generate, real progress, status, PDF preview, download, regenerate keeps old report, history) · Settings · About (loaded in the
navigation sweep) · offline backend · unknown analysis id · image with no detections · tablet and phone layouts.

## Spatial reasoning and PS-04 compliance

* **Independent oracle** (`tests/test_ps04_oracle.py`): a separate naive implementation compared with the engine on 480 random scenes
  (sizes 1024², 800x600, 1025x513, 64², 0-120 buildings, a quarter of the centres exactly on or next to the centre lines) x 63 question forms
  (about 30,000 comparisons of answer **and** evidence region): all equal. Official records validated each time (exactly `id`, `answer`, `region`).
* Each building is counted once across quadrants, across halves, and across the five classes; un-classified is never damaged; empty regions; strict
  `yes` rule; type 4 is exactly `cannot be determined` with region `null`.
* **Real model predictions**: all 50 practice questions on the 30 predicted scenes equal the oracle.
* Official practice key: with ground-truth buildings 47/50 answers match (3 documented key discrepancies, pinned in tests); with predicted buildings 16/50.
  The difference is the models. Tie handling (quadrant ties, equal halves) has no official example and is implemented and documented conservatively.

## PDF report tests

Generated through the API from real analyses and read back with pypdf/PyMuPDF: all 12 numbered sections in order; analysis ID, disaster name, percentages,
five class names, inference time; **every building appears in the table** (its bounding box and confidence checked individually for all 77);
per-class counts equal the analysis; provenance tags and the recommendations label present; no accuracy-target claim; at least 10 embedded images including a
near-full-resolution damage map; limitations quote the measured evaluation files; evidence region and official answers match the engine; zero-building
analysis still produces a valid report; **layout** (every text line inside the page margins, no overlapping lines, no blank pages) checked automatically;
cover and pages inspected by eye (15-page sample at `outputs/demo/sample_report.pdf`). Download: `%PDF` header, byte count and SHA-256 equal the stored metadata,
page count equals metadata, server-built safe filename. Repeated generation creates new reports and never overwrites; an in-flight report is reused;
failure leaves no partial file and no leaked paths; retry works; interrupted jobs are marked failed after a restart.

## Performance (measured; full tables in [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md))

RTX 3050 Laptop 4 GB, 1024x1024 pairs: median **1.05 s** per analysis end to end (0.79 s model inference); peak GPU memory **714 MB** (448 MB above idle);
8 simultaneous requests: 5 completed, 3 refused with 503 (queue limit working as designed); CPU fallback median **4.6 s**; PDF median **2.0 s**; read endpoints 4-36 ms.

## Security and integrity (`tests/test_repo_integrity.py` and API tests)

* Uploads: type by extension **and** magic bytes, size streamed with early stop, header-only dimension read before decode (decompression bombs), full decode, dimension limits; client filenames never used as paths.
* Unique random ids; directories created with `exist_ok=False`; analyses and reports never overwritten; ids validated by pattern and resolved inside the storage root (traversal blocked).
* Downloads: server-built filenames; 404/409 for unknown/not-ready; stored predictions contain no server paths; PDF errors expose exception type only.
* Model weights are byte-identical to the verified backup (SHA-256); `dataset_source` has no modifications; no weights/PDF/.env tracked by Git; reports, analyses, models, runs, processed data, dataset, `.env`, node_modules and demo output are git-ignored; no hard-coded secrets; the API key is read only from the environment and never returned.
* Temporary files: failed analyses delete their directory; PDF temp files are removed on failure and on restart recovery.

## Not tested, not done, and known issues

* **The optional LLM explanation has not been run against the real service** (no key in this environment). Its number guard, JSON handling and fallback were tested with stand-in call functions. The default (template) path is what all report tests exercise.
* The browser test runs on Chrome only. PNG upload through the UI, upload cancel, slider keyboard control and advanced upload settings are not covered by the browser test (the API tests cover PNG and parameters).
* The PDF preview depends on the browser's PDF viewer (verified in Chrome).
* PDF uses standard Latin-1 fonts: non-Latin characters in user-typed labels print as `?`.
* One PDF renders at a time (one worker); reports queue.
* **Model quality is poor** (detector mask recall 0.21; classifier never predicts minor damage; accuracy below the majority-class baseline). Counts under-report buildings and minor/major damage, and over-report destroyed. No retraining was done.
* The PS-04 86 % target is not evaluated and not claimed.
* `dataset_source/` (CC BY-NC-SA data, 1,300+ files) was already tracked in Git before this work; keep the repository private or remove it from history before publishing.
* No authentication; analyses and reports are kept until deleted by hand.
