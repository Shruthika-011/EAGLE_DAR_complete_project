# EAGLE-DAR hackathon demonstration

About 6 minutes. Everything shown is produced live by the real models; nothing is pre-baked. The same sequence is
automated in `frontend/e2e/e2e.mjs` (run it before the demo as a rehearsal: `npm run e2e`).

## Before you go on stage

1. `scripts\start.ps1`, wait ~10 s, open http://localhost:5173. The hero must say **System Online**
   (Settings page: both models *Loaded*, device `cuda`).
2. Have the demo pair ready: `dataset_source/practice/images/pb67aa0fe_pre.jpg` (before) and `pb67aa0fe_post.jpg` (after).
   A dense coastal settlement; the models find 77 buildings, 4 predicted destroyed.
3. Warm up once (upload the pair, delete nothing). The first analysis after a restart is a little slower.
4. Backup plan if the live run fails: the dashboard keeps previous analyses (Settings > Recent analyses), and
   `outputs/demo/` holds screenshots from the last end-to-end run.

## The sequence (with what to say)

| # | Do | Say |
|---|---|---|
| 1 | Open the dashboard | "EAGLE-DAR compares before and after satellite images to assess disaster damage building by building. The status badge is a live backend health check." |
| 2-3 | **Upload** > choose the before image, then the after image | "Both are validated: JPG/PNG, same size, size limits. Wrong files are rejected with a clear message." |
| 4 | Type a disaster name, press **Start Disaster Analysis** | "One click. The backend runs YOLO to find building footprints and a Siamese network to grade each one." |
| 5 | Show the progress bar | "Real upload progress, then processing on the GPU." |
| 6 | Dashboard opens with the analysis ID | "77 buildings detected in about two seconds of model time (measured, shown on the card). Affected means major damage plus destroyed (minor damage is merged into major)." |
| 7 | Point at the doughnut and the slider | "Five-class distribution from the real predictions. Drag the divider to compare before and after." |
| 8 | **Damage Map** | **Novelty 2:** "Every detected building is drawn from its real polygon, coloured green, yellow, orange, red or grey. Zoom, pan, and filter by class. Dashed outlines mark low-confidence buildings." |
| 9 | Click a red building | **Novelty 1:** "Each building has its own confidence: detection score times class probability. It is a model score, not a guarantee, and it is not calibrated. The inspector shows both crops and all five class probabilities." |
| 10 | Ask: *Count the destroyed buildings in the image.* | "Questions are answered by deterministic code, never by a language model." |
| 11 | Show answer, dashed evidence box, highlighted buildings; expand *Official record* | "The official answer has exactly id, answer and region. The evidence region and the supporting building IDs are shown for traceability, kept out of the official JSON." Then ask *Which building is a hospital?* > "cannot be determined". |
| 12 | **Reports** > **Generate Disaster Assessment Report** | **Novelty 3:** "The backend builds a PDF from this analysis. The progress is real: it is reading images, drawing the map, answering standard questions, writing sections." |
| 13-14 | Scroll the preview: cover, before/after, annotated map | "Original before and after images, the annotated five-class map, all from this analysis." |
| 15 | Building table, confidence histogram, low-confidence list | "Every building is listed with class, confidence, centre and evidence. Nothing is omitted." |
| 16 | Spatial section, question evidence crops, recommendations | "Quadrant and half statistics under the official PS-04 rules. Each text block is tagged COMPUTED, MODEL PREDICTION, or AI EXPLANATION. Recommendations are labelled as requiring human verification." |
| 17 | **Download PDF** | "A 13-15 page report, ready to print. It works without any external LLM; if one is enabled, its text is rejected unless every number is in the computed facts." |

## Talk track: why the three novelties matter

* **Building-level confidence** tells responders where to look first: the 66 low-confidence buildings in this scene need a human check before anyone relies on their class.
* **The five-class map** turns thousands of building decisions into one picture a decision-maker can read in seconds, identical in the dashboard and the report.
* **The evidence-backed report** lets a human audit every statement: each count traces to buildings, each answer to an evidence region, and the wording cannot add numbers.

## Be upfront about limits (judges respect this)

* The current models are weak: they find about one building in five, and the classifier never predicts *minor damage*. The report quotes these measured figures on its limitations page. Show the confusion matrix from `docs/EVALUATION_REPORT.md` if asked.
* The 86 % target has not been measured and is not claimed; on the 50 official practice questions the end-to-end answer match is 32 %, versus 94 % when the reasoning engine is fed ground-truth buildings. That isolates the gap to the models.
* No geographic output: the images are not georeferenced.

## If something goes wrong

| Symptom | Fix |
|---|---|
| Badge says Backend Offline | start the backend; the UI re-checks every 20 s |
| Badge says Degraded | models not loaded: Settings shows which file is missing |
| Report shows Failed | the message names the cause; press Regenerate (previous reports are kept) |
| Server busy (503) | another analysis is running; retry in a few seconds |
