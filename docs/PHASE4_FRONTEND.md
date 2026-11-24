# Phase 4 — React frontend

Vite + React 18 + TypeScript, React Router, Recharts, lucide-react. Plain CSS (no UI framework).
Everything shown comes from the Phase 3 backend; there is no sample or fallback data anywhere in the UI.

## Run

```powershell
# terminal 1 — backend (project root, base Python 3.11)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# terminal 2 — frontend
cd frontend
npm install          # first time only
npm run dev          # http://localhost:5173
```

* Backend URL: `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`; copy `frontend/.env.example`).
* The origin must be in the backend's CORS list (`EAGLE_CORS_ORIGINS`); `localhost`/`127.0.0.1` on ports 3000 and 5173 are allowed by default.
* `npm run build` produces `frontend/dist`; `npm run typecheck`; `npm test` (15 unit tests for the derivation and zoom/pan maths).

## Pages and the backend calls they use

| Route | Page | Backend |
|---|---|---|
| `/` | Dashboard (hero, 4 KPIs, before/after slider, donut, damage map, insights, location, actions) | `GET /api/health`, `GET /api/analysis/{id}?include_polygons=true`, `…/image/{before,after}` |
| `/upload` | Upload Satellite Images | `POST /api/analyze` (XHR for upload progress), `GET /api/health` limits |
| `/results` | View Results: images, class table, sortable/filterable building table, inspector, metadata | same analysis record |
| `/map` | Full-page damage map, class filters, details, spatial questions | `POST /api/questions` |
| `/analytics` | Distribution, total vs affected, class comparison, confidence histogram, quadrant stats | same analysis record |
| `/reports` | Findings, rule-based follow-up, full backend report, downloads | `GET …/report?format=json|html`, `…/export` |
| `/settings`, `/about` | Backend/model status, display preferences; project, pipeline, novelties | `GET /api/health` |

The selected analysis id lives in the URL (`?a=<id>`) and in `localStorage`, so it survives navigation and reloads and links are shareable. Recent analyses (id, label, time) are kept in this browser only.

## How the spec's requirements are met

* **No hardcoded numbers.** KPIs, donut, insights and charts are derived in `src/lib/derive.ts` from the backend record. Missing values render `—`; an empty analysis gives `null` shares, not 0 or NaN. A browser test compares every displayed KPI with the backend's own JSON.
* **Affected = major + destroyed** (minor merged into major; no un-classified class) (`AFFECTED_CLASSES`, unit-tested).
* **Damage map.** The after-image is drawn with an SVG overlay whose `viewBox` is the image size, so the backend's polygon coordinates are used unchanged and scale exactly with the image (tested: points equal the backend's, viewBox equals the image size). Hover/click show ID, class and confidence; wheel zoom, drag pan and zoom buttons; per-class toggles; five fixed class colours identical to the backend map. The UI states that these are image pixels, not geographic coordinates.
* **Before/after slider** with a draggable (and keyboard) divider, shared zoom/pan, loading and error states, original aspect ratio kept.
* **Upload.** Client-side checks (JPG/JPEG/PNG, size, dimensions, before = after size), previews, file name and dimensions, Replace/Remove, upload progress then an indeterminate "processing" bar, cancel, backend errors shown with code, message and details. A ref guard prevents duplicate requests (a double click sends one request — tested). On success the analysis id is shown, then the dashboard opens.
* **Honest states.** Offline backend, degraded models, unknown analysis id (404), network failures and empty states each have an explicit UI and a retry/clear action; the "report generated" message appears only after the backend answers 200.

## Where the backend could not supply a requested feature (decisions)

| Requested | Decision |
|---|---|
| Before/after images | Added `GET /api/analysis/{id}/image/{before\|after}` to the backend (they were not served). |
| Disaster name / district / state | Added optional form fields to `POST /api/analyze`; stored verbatim (cleaned) in the analysis and shown as "as typed, not derived from the imagery". |
| PDF download | The backend has no PDF generator. The UI offers **Print / Save as PDF** (browser print of the real report) and **Download report (HTML)**. No fake PDF button. |
| GeoJSON | Not offered: the images are plain JPG/PNG with no georeferencing. Location Summary says "Geographic location unavailable". |
| Affected area, historical comparison | Not shown (no data). Analytics states there is nothing to trend with one analysis. |
| "Analysis time" | The backend reports model inference time only; the label says so. |
| Recommendations | The backend has none. The Reports page shows rule-based follow-up (triggered by the numbers) in a separate card labelled "not a model prediction". |
| Model reliability | The backend report now quotes the measured Siamese validation recall per class (minor damage 0.00) and detector recall, read from `outputs/evaluation/`. |
| Government emblem / ministry text / NIC footer in the mockup | Not reproduced, to avoid implying endorsement: a labelled **LOGO placeholder** and "research prototype · not an official government system". |
| Hero image | A crop of the project's own xBD practice imagery with attribution (CC BY-NC-SA 4.0). |

Header subtitle follows the brief ("Earthquake & Disaster Assessment using AI for Rapid Response"); the About page uses the project's full name (Evidence-Aware Geospatial Learning and Explanation for Disaster Assessment and Reasoning). Dark theme is not implemented; display preferences (overlay opacity, building IDs, compact spacing) are.

## Verification performed

* `npm run typecheck`, `npm test` (15), `npm run build` pass.
* A real-browser run (Chrome via Playwright, outside the repo) against the live backend: ~60 checks including a real upload→analysis, KPI values equal to the backend JSON, polygon/viewBox fidelity, tooltip, class toggle, zoom, slider drag, table/inspector, a question answer equal to the API's, report iframe, HTML download, offline backend, unknown analysis id, mobile drawer, no horizontal overflow, and **no console errors**.
* Python: 128 tests pass (90 reasoning + 38 backend, including the new image/context/report tests).

## Known limitations

* Recharts 2.x is used (React 18 compatible); npm prints a deprecation notice. The bundle is ~670 kB (190 kB gzipped); no code-splitting yet.
* The dashboard's small map and slider use Ctrl+scroll to zoom so the page still scrolls; the full map page uses plain scroll.
* Recent analyses are browser-local; there is no server-side list/delete endpoint, no authentication.
* The model outputs themselves are weak (see the report): building recall ≈ 0.12–0.21, minor-damage recall 0. The UI displays them faithfully and says so; it cannot fix them.
