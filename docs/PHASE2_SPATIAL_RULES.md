# Phase 2 — Spatial reasoning and evidence engine

Models detect and classify buildings (Phase 1). Everything here is deterministic
Python. No LLM counts, compares or classifies anything.

```
Phase 1 predictions JSON ──► reasoning.loaders ──► Scene(buildings)
question text ──► reasoning.questions.parse_question ──► Query (A/B/C/D)
Scene + Query ──► reasoning.engine.answer_query ──► Answer(answer, region, supporting ids, details)
Answer ──► reasoning.official.to_official ──► {"id","answer","region"}   (submission)
Answer ──► reasoning.official.trace_record ──► internal trace             (never submitted)
Scene ──► reasoning.damage_map / reasoning.report_facts                   (novelties 2 and 3)
```

## Rules

| Rule | Implementation | Basis |
|---|---|---|
| Split at the actual image centre `(w/2, h/2)` | `spatial.quadrant_of` | templates; works for non-1024 sizes |
| A building belongs to the part containing its **centre** = polygon centroid | `loaders` recompute the centroid from the polygon | templates; 17/17 official counts that don't touch the discrepancy below |
| Parts are half-open `[lo, hi)`; the exact centre line goes to **right / bottom** | `spatial` | **assumption** — templates give closed overlapping rectangles; only matters for integer/synthetic centres |
| Every building is in exactly one quadrant and one horizontal + one vertical half | tested over a boundary grid | spec |
| Damaged = major + destroyed (minor merged into major; no un-classified class) | `LEVELS["damaged"] = {1,2}` | templates |
| Duplicate building ids are rejected; buildings are never counted twice | `Scene.__post_init__` | spec |

## Answers and evidence (verified against the 50 official practice answers)

| Type | Answer | Region (`[x1,y1,x2,y2]`) |
|---|---|---|
| 1 / A count | number of buildings of the level in scope (image or quadrant; halves also accepted) | union bbox of the counted buildings, `floor` min / `ceil` max, clipped to the image (16/17 exact; may extend beyond the quadrant). **Count 0 → the queried quadrant rectangle** (official example `[0,0,512,512]`); for a zero count over the whole image the full image is used by analogy (no official example) |
| 2 / B quadrant | quadrant with most damaged (or destroyed) buildings | rectangle of the winning quadrant |
| 3 / C halves | `yes` only if the first-named half has **strictly more** damaged buildings; else `no` | rectangle of the half **with more** damaged — not necessarily the first named. Verified 10/10 |
| 4 / D | exactly `cannot be determined` | `null` |

`question_templates.json` rule: no buildings at all → type B/C answer `cannot be determined`.

## Not covered by any official example (handled conservatively, flagged in the trace)

1. **Type 2 tie** (two or more quadrants share the maximum, including 0 damaged everywhere):
   answer `cannot be determined`, region `null`, `details.tie_unresolved = true`. No quadrant
   is picked by an invented tie-break.
2. **Type 3 equal counts**: answer `no` (strictly-more rule from the task brief). Region has no
   precedent: we return the union of both compared halves, `details.tie = true`.
3. **Exact centre-line placement** (above).
4. **Region format**: README allows a box or a quadrant/half name; default is the box
   (identical to `answers.jsonl`). `--region-format name` switches types 2/3 to names.

## Open discrepancies in the official practice set (not "fixed" with invented rules)

| id | key | our rules | note |
|---|---|---|---|
| `q084c5b9` | 50 undamaged in top-left | 51 | the key's region `[0,0,518,531]` also excludes building `b0121` (centroid y = 509.7, bbox bottom 539.3). Centroid, bbox-centre, vertex-mean and area-majority all fail to explain it |
| `qb62381d`, `q17b7814` | `cannot be determined` | `no` / `bottom-right` | scene `p6adc4d5d` has 144 buildings and a unique winner (damaged per quadrant 5/1/1/8). The pair is visually usable, but alignment cannot be verified (phase-corr. response 0.017, 2 ORB inliers). Hypothesis only: the organisers may mark unalignable pairs unanswerable. **No rule implemented.** Ask the organisers |

These three are pinned in `tests/test_spatial_reasoning.py::KNOWN_UNEXPLAINED`.

## Results

| Run | Buildings come from | Answer match | Notes |
|---|---|---|---|
| Logic check | official labels | **47 / 50** (type 1 19/20, type 2 10/10, type 3 10/10, type 4 8/10) | validates the engine only; mean region IoU 1.000 |
| End to end | trained models (current `siamese_damage.pth`) | **16 / 50 = 32 %** (type 1 2/20, type 2 1/10, type 3 4/10, type 4 9/10); mean region IoU 0.449 | practice set only; provisional until Siamese training finishes |

The end-to-end figure is limited by Phase 1: the YOLO segmenter's building recall is ≈ 0.12
at IoU 0.5, so counts are strongly under-estimated (e.g. 17 detected vs 166 true). Part of the
type-4 score comes from empty or unparseable cases and is not evidence of understanding.
**The 86 % target is not met and no claim is made about it.** This is a practice-set
measurement, not the official hidden-test evaluation, and the scoring rule for regions is unknown.

## Commands

```powershell
# tests (core is standard-library only)
python -m pytest tests -q

# 1. model predictions for every pair in a questions file (base Python has ultralytics)
python -m inference.predict_batch --questions dataset_source/practice/questions.jsonl --images-dir dataset_source/practice

# 2. official submission + internal trace (+ scoring against a key when one exists)
python -m reasoning.answer_questions --questions dataset_source/practice/questions.jsonl `
    --source predictions --answers dataset_source/practice/answers.jsonl `
    --out outputs/phase2/practice_model_submission.jsonl

# logic-only check with ground-truth buildings
python -m reasoning.answer_questions --questions dataset_source/practice/questions.jsonl `
    --source ground-truth --labels-dir dataset_source/practice/labels `
    --answers dataset_source/practice/answers.jsonl --out outputs/phase2/practice_groundtruth_logic_check.jsonl

# five-class map with an evidence rectangle, and report facts
python -m reasoning.damage_map --predictions outputs/predictions/<id>_predictions.json `
    --image dataset_source/practice/images/<id>_post.jpg --out outputs/phase2/<id>_damage_map.png --evidence 0,0,1024,512
python -m reasoning.report_facts --predictions outputs/predictions/<id>_predictions.json --out outputs/phase2/<id>_report_facts.json
```

## The three novelties

1. **Building-level confidence** — `building_confidence = detection × damage` (Phase 1), carried on every
   `Building`, filterable with `--min-confidence`, shown as outline weight on the map, summarised in report facts.
2. **Five-class colour-coded map** — `reasoning/damage_map.py`: green / yellow / orange / red / grey,
   one fixed colour per class, identical to `config.DAMAGE_COLORS_BGR` (tested).
3. **Assessment report data** — `reasoning/report_facts.py` emits counts, per-quadrant/half statistics,
   confidence summary, priority buildings (destroyed, then major) with evidence regions, data-quality
   warnings, measured model limitations read from evaluation files, and explicit LLM guardrails.
   The narrative report itself is Phase 3.

## Changes to existing files

* `inference/pipeline.py`: centre rounded to 3 decimals and polygon to 2 (1-decimal rounding could move a
  centre across the quadrant line, e.g. 511.96 → 512.0). Phase 2 additionally recomputes the centre from the polygon.
