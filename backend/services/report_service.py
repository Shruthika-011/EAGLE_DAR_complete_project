"""Disaster assessment report (Novelty 3).

Every number comes from `reasoning.report_facts.build_report_facts`; the prose is
filled in by deterministic templates. No language model is called, so nothing in
the report can be invented. (A narrative LLM could later be given only the facts
block and its guardrails.)
"""
from __future__ import annotations

import base64
import html
from datetime import datetime, timezone

import cv2
import numpy as np

from backend.errors import ApiError
from backend.services import report_content, spatial_service
from backend.utils.storage import AnalysisStore
from inference import alignment as align
from inference.pipeline import render_comparison
from reasoning.damage_map import CLASS_COLORS_BGR, render_damage_map
from reasoning.report_facts import build_report_facts
from reasoning.schema import CLASS_NAMES

GENERATION = ("Deterministic templates over computed facts; no language model was used, "
              "so no figure in this report is generated text.")
QUAD_ORDER = ("top-left", "top-right", "bottom-left", "bottom-right")


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _hex(name: str) -> str:
    b, g, r = CLASS_COLORS_BGR[name]
    return f"#{r:02x}{g:02x}{b:02x}"


# --------------------------------------------------------------------------
# text sections (each sentence is a template over facts)
# --------------------------------------------------------------------------

def build_sections(facts: dict, questions: dict) -> list[dict]:
    total = facts["scene"]["total_buildings"]
    c = facts["damage_counts"]
    sections = []

    # ---- summary: the same computed findings the PDF uses --------------------
    sections.append({"title": "Summary", "paragraphs": [f["text"] for f in report_content.build_findings(facts)]})

    # ---- confidence -------------------------------------------------------
    conf = facts["confidence"]
    p = []
    if conf["mean"] is not None:
        lv = conf["levels"]
        p.append(f"Building-level confidence (detection score x damage-class probability): mean {conf['mean']:.3f}, "
                 f"minimum {conf['min']:.3f}, maximum {conf['max']:.3f}. Confidence levels: {lv['high']} high, "
                 f"{lv['medium']} medium, {lv['low']} low.")
        n_low = len(conf["low_confidence_building_ids"])
        p.append(f"{n_low} building(s) fall below the low-confidence threshold of {conf['low_confidence_threshold']}.")
    else:
        p.append("No building-level confidence values are available.")
    p.append("These values are uncalibrated model scores; they rank how strongly the model supports a prediction "
             "and are not probabilities that the prediction is correct.")
    sections.append({"title": "Confidence", "paragraphs": p})

    # ---- data quality and limits -----------------------------------------
    dq = facts["data_quality"]
    al, ctx = dq.get("alignment") or {}, dq.get("evaluation_context") or {}
    p = []
    if al:
        p.append(f"Before/after alignment: {al.get('status')} (estimated shift {al.get('shift_px', 0):.1f} px via "
                 f"{al.get('method')}{', registered before classification' if al.get('registered') else ''}). "
                 + (al.get("note") or ""))
    p.extend(f"Warning: {w}" for w in dq.get("warnings", []))
    p.append("Counts are model predictions, not ground truth.")
    yv = ctx.get("yolo_validation")
    if yv:
        p.append(f"On the held-out validation images the building segmentation model reached mask recall "
                 f"{yv['mask_recall']:.2f} and mask precision {yv['mask_precision']:.2f}; buildings it misses are "
                 f"absent from every count above.")
    sv = ctx.get("siamese_validation")
    if sv:
        rec = ", ".join(f"{k} {v:.2f}" for k, v in sv["recall_by_class"].items())
        p.append(f"On {sv['n_val']} held-out validation building pairs the damage classifier reached accuracy "
                 f"{sv['accuracy']:.2f} but macro F1 only {sv['macro_f1']:.2f}. Per-class recall: {rec}. "
                 "Classes with near-zero recall are effectively not detected, so their counts are unreliable.")
    for mode, v in (ctx.get("pipeline_practice") or {}).items():
        p.append(f"On the PS-04 practice scenes (alignment mode '{mode}') the full pipeline's building recall was "
                 f"{v['building_recall_iou50']:.2f} and precision {v['building_precision_iou50']:.2f} at IoU 0.5.")
    if not ctx:
        p.append("No evaluation files were found, so model limitations are not quantified here.")
    p.append("This report has not been validated against the official PS-04 evaluation. Verify visually before operational use.")
    sections.append({"title": "Data quality and limitations", "paragraphs": p})

    # ---- recommendations (same source as the PDF) ----------------------------
    sections.append({"title": "Recommendations (AI-assisted, require human verification)",
                     "paragraphs": [r["text"] for r in report_content.build_recommendations(facts)]})

    # ---- questions --------------------------------------------------------
    if questions:
        p = []
        for qid, rec in questions.items():
            o = rec["official"]
            p.append(f"{qid}: {rec.get('question_text') or ''} -> {o['answer']}"
                     + (f" (region {o['region']})" if o["region"] is not None else ""))
        sections.append({"title": "Questions answered", "paragraphs": p})
    return sections


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------

def _data_uri(img: np.ndarray, quality: int = 85, max_w: int | None = None) -> str:
    if max_w and img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, (max_w, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _crop(img: np.ndarray, bbox, pad: int = 24, size: int = 220) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    c = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
    s = size / max(c.shape[:2])
    return cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_AREA)


def _read(store, analysis_id, role):
    img = cv2.imdecode(np.fromfile(str(store.image_path(analysis_id, role)), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ApiError(500, "corrupt_analysis", f"Stored {role} image could not be decoded.")
    return img


# --------------------------------------------------------------------------
# html
# --------------------------------------------------------------------------

CSS = """
:root{--bg:#fff;--fg:#1d2733;--mut:#5b6775;--line:#d9dee5;--card:#f6f8fa}
@media(prefers-color-scheme:dark){:root{--bg:#14181d;--fg:#e6eaef;--mut:#9aa6b4;--line:#2c343d;--card:#1b2026}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:1.6rem;margin:.2em 0}h2{font-size:1.2rem;margin:1.8em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:.25em}
.meta{color:var(--mut);font-size:.85rem}table{border-collapse:collapse;width:100%;margin:.6em 0;font-size:.92rem}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left}th{background:var(--card)}td.n{text-align:right;font-variant-numeric:tabular-nums}
.sw{display:inline-block;width:.9em;height:.9em;border:1px solid #0006;vertical-align:-1px;margin-right:6px}
img{max-width:100%;height:auto;border:1px solid var(--line)}.pair{display:flex;gap:6px;flex-wrap:wrap}
.warn{background:var(--card);border-left:4px solid #e0a800;padding:8px 12px;margin:.5em 0}
"""


def render_html(analysis_id: str, meta: dict, facts: dict, sections: list[dict], images: dict, priority: list[dict],
                generated: str) -> str:
    e = html.escape
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>EAGLE-DAR assessment {e(analysis_id[:8])}</title><style>{CSS}</style></head><body><main>",
           "<h1>EAGLE-DAR disaster assessment report</h1>",
           f"<div class='meta'>Analysis {e(analysis_id)} &middot; generated {e(generated)} &middot; image "
           f"{facts['scene']['width']}&times;{facts['scene']['height']} px &middot; models: "
           f"{e(', '.join(m for m in ('yolo', 'siamese') if m in meta.get('models', {})) or 'YOLOv8-seg + Siamese ResNet18')}</div>"]

    def block(title):
        s = next((s for s in sections if s["title"] == title), None)
        if s:
            out.append(f"<h2>{e(title)}</h2>" + "".join(
                f"<p class='{'warn' if p.startswith('Warning') else ''}'>{e(p)}</p>" for p in s["paragraphs"]))

    block("Summary")
    out.append("<h2>Visual overview</h2>")
    out.append(f"<img alt='before, after and damage map' src='{images['comparison']}'>")
    out.append("<h2>Damage map (three classes)</h2>")
    out.append(f"<img alt='three-class damage map' src='{images['map']}'>")

    out.append("<h2>Damage by class</h2><table><tr><th>Class</th><th>Colour</th><th>Buildings</th><th>Share</th></tr>")
    for n in CLASS_NAMES:
        out.append(f"<tr><td>{n}</td><td><span class='sw' style='background:{_hex(n)}'></span></td>"
                   f"<td class='n'>{facts['damage_counts'][n]}</td><td class='n'>{_pct((facts['class_fractions'] or {}).get(n))}</td></tr>")
    out.append("</table>")

    out.append("<h2>Damage by quadrant</h2><table><tr><th>Quadrant</th>" + "".join(f"<th>{n}</th>" for n in CLASS_NAMES)
               + "<th>damaged</th></tr>")
    for q in QUAD_ORDER:
        r = facts["quadrants"][q]
        out.append(f"<tr><td>{q}</td>" + "".join(f"<td class='n'>{r[n]}</td>" for n in CLASS_NAMES)
                   + f"<td class='n'><b>{r['damaged']}</b></td></tr>")
    out.append("</table>")

    block("Confidence")

    out.append("<h2>Priority buildings</h2>")
    if priority:
        out.append("<p class='meta'>Destroyed first, then major damage; ordered by confidence. Left: before, right: after.</p>"
                   "<table><tr><th>ID</th><th>Damage</th><th>Confidence</th><th>Centre (x, y)</th><th>Before / after</th></tr>")
        for p in priority:
            out.append(f"<tr><td class='n'>{p['building_id']}</td><td><span class='sw' style='background:{_hex(p['damage'])}'></span>"
                       f"{e(p['damage'])}</td><td class='n'>{p['confidence']:.3f}</td>"
                       f"<td class='n'>{p['centre'][0]:.0f}, {p['centre'][1]:.0f}</td>"
                       f"<td><div class='pair'><img alt='before' src='{p['before']}'><img alt='after' src='{p['after']}'></div></td></tr>")
        out.append("</table>")
    else:
        out.append("<p>No buildings were classified as destroyed or major damage.</p>")

    block("Recommendations (AI-assisted, require human verification)")
    block("Data quality and limitations")
    block("Questions answered")
    out.append("<h2>Method</h2><p>" + e(GENERATION) + " Buildings are detected and segmented by a YOLOv8 model on the "
               "post-disaster image and classified by a Siamese ResNet18 comparing before/after crops. Counts, quadrants "
               "and comparisons are computed by deterministic code (quadrants split at the image centre, a building belongs "
               "to the part containing its centre; damaged = major + destroyed).</p></main></body></html>")
    return "".join(out)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def build_report(store: AnalysisStore, analysis_id: str) -> tuple[dict, str]:
    """-> (json document, html)."""
    meta = store.meta(analysis_id)
    preds = store.predictions(analysis_id)
    scene = spatial_service.load_scene(store, analysis_id)
    facts = build_report_facts(scene, preds["alignment"], preds["warnings"])
    questions = store.questions(analysis_id)
    sections = build_sections(facts, questions)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    pre, post = _read(store, analysis_id, "before"), _read(store, analysis_id, "after")
    al = preds["alignment"]
    pre_used = align.register_pre_to_post(pre, al["dx"], al["dy"]) if al.get("registered") else pre
    dmap = render_damage_map(post, scene)
    images = {"comparison": _data_uri(render_comparison(pre_used, post, dmap, width=1800), 82, 1400),
              "map": _data_uri(dmap, 85, 1100)}
    priority = [{"building_id": p["building_id"], "damage": p["damage"], "confidence": p["confidence"] or 0.0,
                 "centre": p["centre"], "before": _data_uri(_crop(pre_used, p["bbox"])),
                 "after": _data_uri(_crop(post, p["bbox"]))} for p in facts["priority_buildings"]]

    document = {"analysis_id": analysis_id, "generated_utc": generated, "generation": GENERATION,
                "sections": [{"title": s["title"], "paragraphs": s["paragraphs"]} for s in sections],
                "facts": facts}
    return document, render_html(analysis_id, preds, facts, sections, images, priority, generated)
