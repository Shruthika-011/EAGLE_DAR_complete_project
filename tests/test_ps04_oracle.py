"""PS-04 compliance against an INDEPENDENT oracle.

The oracle below is a separate, deliberately naive implementation (plain loops, its own centroid, its own region
arithmetic). It shares no code with reasoning/ except the data classes used to feed both. Engine answers and official JSON
records must equal the oracle's on random scenes (boundary values included) and on real model predictions.
"""
import json
import math
import random
from pathlib import Path

import pytest

from reasoning.engine import answer_query
from reasoning.official import to_official, validate_official
from reasoning.questions import LEVELS, parse_question
from reasoning.schema import CLASS_NAMES, Building, Scene

ROOT = Path(__file__).resolve().parent.parent
CBD = "cannot be determined"
DAMAGED = (1, 2)            # major-damage (includes former minor) + destroyed


# ----------------------------------------------------------------------------- the oracle

def o_part(cx, cy, w, h):
    return ("bottom" if cy >= h / 2 else "top") + "-" + ("right" if cx >= w / 2 else "left")


def o_in_half(cx, cy, w, h, half):
    return {"left": cx < w / 2, "right": cx >= w / 2, "top": cy < h / 2, "bottom": cy >= h / 2}[half]


def o_rect(name, w, h):
    cx, cy = w / 2, h / 2
    x_lo, x_hi, y_lo, y_hi = math.ceil(cx), math.floor(cx), math.ceil(cy), math.floor(cy)
    return {"image": (0, 0, w, h), "left": (0, 0, x_lo, h), "right": (x_hi, 0, w, h), "top": (0, 0, w, y_lo), "bottom": (0, y_hi, w, h),
            "top-left": (0, 0, x_lo, y_lo), "top-right": (x_hi, 0, w, y_lo), "bottom-left": (0, y_hi, x_lo, h),
            "bottom-right": (x_hi, y_hi, w, h)}[name]


def o_union(boxes, w, h):
    return (max(0, min(b[0] for b in boxes)), max(0, min(b[1] for b in boxes)), min(w, max(b[2] for b in boxes)), min(h, max(b[3] for b in boxes)))


def o_in_scope(b, kind, name, w, h):
    cx, cy = b["centre"]
    return kind == "image" or (kind == "quadrant" and o_part(cx, cy, w, h) == name) or (kind == "half" and o_in_half(cx, cy, w, h, name))


def oracle(blds, w, h, q):
    """blds: list of dicts {id, cls, centre, bbox}. Returns (answer, region) per the documented PS-04 conventions."""
    if q.qtype == "D":
        return CBD, None
    if q.qtype == "A":
        want = LEVELS[q.level]
        sel = [b for b in blds if b["cls"] in want and o_in_scope(b, q.scope_kind, q.scope_name, w, h)]
        return len(sel), (o_union([b["bbox"] for b in sel], w, h) if sel else o_rect(q.scope_name or "image", w, h))
    if not blds:
        return CBD, None
    if q.qtype == "B":
        want = LEVELS[q.level]
        per = {p: 0 for p in ("top-left", "top-right", "bottom-left", "bottom-right")}
        for b in blds:
            if b["cls"] in want:
                per[o_part(*b["centre"], w, h)] += 1
        top = max(per.values())
        winners = [p for p, n in per.items() if n == top]
        return (winners[0], o_rect(winners[0], w, h)) if len(winners) == 1 else (CBD, None)
    a, b2 = q.halves
    ca = sum(1 for b in blds if b["cls"] in DAMAGED and o_in_half(*b["centre"], w, h, a))
    cb = sum(1 for b in blds if b["cls"] in DAMAGED and o_in_half(*b["centre"], w, h, b2))
    if ca > cb:
        return "yes", o_rect(a, w, h)
    if cb > ca:
        return "no", o_rect(b2, w, h)
    return "no", o_union([o_rect(a, w, h), o_rect(b2, w, h)], w, h)


def to_scene(blds, w, h):
    return Scene(w, h, tuple(Building(b["id"], b["cls"], b["centre"], b["bbox"]) for b in blds))


# ----------------------------------------------------------------------------- questions

LEVEL_TEXT = {"no-damage": "undamaged buildings", "major-damage": "buildings with major damage",
              "destroyed": "destroyed buildings", "damaged": "damaged buildings (major or destroyed)"}
# Levels that no longer exist: the engine must answer "cannot be determined" (type D) for them.
REMOVED_LEVEL_TEXT = {"minor-damage": "buildings with minor damage", "un-classified": "unclassified buildings"}
QUADS = ["top-left", "top-right", "bottom-left", "bottom-right"]


def all_questions():
    qs = []
    for lvl, txt in LEVEL_TEXT.items():
        qs.append(f"How many {txt} are there in the image?")
        for q in QUADS:
            qs.append(f"Count the {txt} in the {q} quadrant.")
        for h in ("left", "right", "top", "bottom"):
            qs.append(f"How many {txt} are there in the {h} half?")
    for txt in REMOVED_LEVEL_TEXT.values():
        qs += [f"How many {txt} are there in the image?", f"Count the {txt} in the top-left quadrant."]
    qs += ["Which quadrant has the most damaged buildings?", "Which quadrant has the most destroyed buildings?"]
    for a, b in (("left", "right"), ("right", "left"), ("top", "bottom"), ("bottom", "top")):
        qs.append(f"Is the {a} half of the image more damaged than the {b} half?")
    qs += ["Which building in the image is a hospital?", "How many people lived in the destroyed buildings?", "What date was the post-disaster image taken?"]
    return qs


def random_scene(rng):
    w, h = rng.choice([(1024, 1024), (800, 600), (1025, 513), (64, 64)])
    n = rng.choice([0, 1, 2, 3, 5, 12, 40, 120])
    xs = [0, w / 2, w, w / 2 - 0.001, w / 2 + 0.001]
    ys = [0, h / 2, h, h / 2 - 0.001, h / 2 + 0.001]
    blds = []
    for i in range(n):
        cx = rng.choice(xs) if rng.random() < 0.25 else rng.uniform(0, w)       # a quarter sit exactly on / next to boundaries
        cy = rng.choice(ys) if rng.random() < 0.25 else rng.uniform(0, h)
        r = rng.uniform(2, 40)
        bbox = (max(0, math.floor(cx - r)), max(0, math.floor(cy - r)), min(w, math.ceil(cx + r)), min(h, math.ceil(cy + r)))
        cls = rng.choices(range(3), weights=rng.choice([[1, 1, 1], [8, 1, 1], [0, 0, 1], [1, 0, 3]]))[0]
        blds.append({"id": i, "cls": cls, "centre": (cx, cy), "bbox": bbox})
    return w, h, blds


# ----------------------------------------------------------------------------- tests

QUESTIONS = all_questions()
PARSED = [parse_question(q) for q in QUESTIONS]


def test_question_set_covers_all_four_types():
    assert {p.qtype for p in PARSED} == {"A", "B", "C", "D"}


@pytest.mark.parametrize("seed", range(40))
def test_engine_equals_oracle_on_random_scenes(seed):
    rng = random.Random(seed)
    for _ in range(12):
        w, h, blds = random_scene(rng)
        scene = to_scene(blds, w, h)
        for text, q in zip(QUESTIONS, PARSED):
            ans = answer_query(scene, q)
            exp_answer, exp_region = oracle(blds, w, h, q)
            assert ans.answer == exp_answer, (text, w, h, len(blds))
            assert (list(ans.region) if ans.region else None) == (list(exp_region) if exp_region else None), (text, w, h)
            rec = to_official(f"q{seed}", ans)
            validate_official(rec, w, h)
            assert list(rec) == ["id", "answer", "region"]


@pytest.mark.parametrize("seed", range(10))
def test_each_building_is_counted_in_exactly_one_quadrant_and_one_half_per_axis(seed):
    w, h, blds = random_scene(random.Random(1000 + seed))
    scene = to_scene(blds, w, h)
    total = len(blds)
    for lvl in LEVELS:
        parts = sum(answer_query(scene, parse_question(f"Count the {LEVEL_TEXT[lvl]} in the {q} quadrant.")).answer for q in QUADS)
        lr = sum(answer_query(scene, parse_question(f"How many {LEVEL_TEXT[lvl]} are there in the {h_} half?")).answer for h_ in ("left", "right"))
        tb = sum(answer_query(scene, parse_question(f"How many {LEVEL_TEXT[lvl]} are there in the {h_} half?")).answer for h_ in ("top", "bottom"))
        whole = answer_query(scene, parse_question(f"How many {LEVEL_TEXT[lvl]} are there in the image?")).answer
        assert parts == lr == tb == whole                           # no double counting, nothing dropped
    assert sum(answer_query(scene, parse_question(f"How many {LEVEL_TEXT[l]} are there in the image?")).answer
               for l in CLASS_NAMES) == total   # the three classes partition


def test_boundary_and_special_cases_explicitly():
    w = h = 1024
    on_line = [{"id": 0, "cls": 2, "centre": (512, 512), "bbox": (500, 500, 524, 524)}]
    scene = to_scene(on_line, w, h)
    assert answer_query(scene, parse_question("Count the destroyed buildings in the bottom-right quadrant.")).answer == 1
    assert answer_query(scene, parse_question("Count the destroyed buildings in the top-left quadrant.")).answer == 0
    undamaged_only = to_scene([{"id": 0, "cls": 0, "centre": (10, 10), "bbox": (0, 0, 20, 20)}], w, h)
    assert answer_query(undamaged_only, parse_question("Count the damaged buildings (major or destroyed) in the image.")).answer == 0
    assert answer_query(undamaged_only, parse_question("Is the left half of the image more damaged than the right half?")).answer == "no"
    for removed in ("Count the buildings with minor damage in the image.", "Count the unclassified buildings in the image."):
        r = answer_query(undamaged_only, parse_question(removed))
        assert r.answer == CBD and r.region is None, removed
    assert answer_query(to_scene([], w, h), parse_question("Which quadrant has the most damaged buildings?")).answer == CBD
    tie = to_scene([{"id": 0, "cls": 2, "centre": (10, 10), "bbox": (0, 0, 20, 20)}, {"id": 1, "cls": 2, "centre": (900, 900), "bbox": (880, 880, 920, 920)}], w, h)
    a = answer_query(tie, parse_question("Which quadrant has the most damaged buildings?"))
    assert a.answer == CBD and a.region is None
    assert answer_query(tie, parse_question("Is the top half of the image more damaged than the bottom half?")).answer == "no"


# ---- real model predictions vs the oracle (the 30 practice scenes predicted in Phase 2) ----------------

PRACTICE = ROOT / "dataset_source" / "practice"
PRED_DIR = ROOT / "outputs" / "predictions"


def polygon_centroid(pts):
    a = cx = cy = 0.0
    for i in range(len(pts)):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % len(pts)]
        cr = x0 * y1 - x1 * y0
        a += cr; cx += (x0 + x1) * cr; cy += (y0 + y1) * cr
    return (cx / (3 * a), cy / (3 * a)) if abs(a) > 1e-9 else (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


@pytest.mark.skipif(not (PRACTICE / "questions.jsonl").exists() or not PRED_DIR.exists(), reason="practice data / predictions missing")
def test_real_predictions_answers_equal_oracle_for_all_practice_questions():
    from config import SUBTYPE_TO_APP_NAME
    from reasoning.loaders import scene_from_prediction_json
    qs = [json.loads(l) for l in (PRACTICE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    n = 0
    for k in qs:
        f = PRED_DIR / f"{k['pair_id']}_predictions.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        w, h = data["inputs"]["width"], data["inputs"]["height"]
        blds = []
        for b in data["buildings"]:
            # Stored files may be legacy 5-class output: minor -> major-damage, un-classified buildings are dropped.
            name = SUBTYPE_TO_APP_NAME.get(b["predicted_damage"])
            if name is None:
                continue
            pts = b["polygon_xy"]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            blds.append({"id": b["building_id"], "cls": CLASS_NAMES.index(name), "centre": polygon_centroid(pts),
                         "bbox": (max(0, math.floor(min(xs))), max(0, math.floor(min(ys))), min(w, math.ceil(max(xs))), min(h, math.ceil(max(ys))))})
        q = parse_question(k["question"])
        ans = answer_query(scene_from_prediction_json(f), q)
        exp_a, exp_r = oracle(blds, w, h, q)
        assert ans.answer == exp_a, (k["id"], k["question"])
        assert (list(ans.region) if ans.region else None) == (list(exp_r) if exp_r else None), k["id"]
        rec = to_official(k["id"], ans)
        validate_official(rec, w, h)
        assert set(rec) == {"id", "answer", "region"}
        n += 1
    assert n >= 40


def test_type4_answer_is_exactly_the_official_string_and_region_null():
    for q in ("Which building in the image is a hospital?", "How many buildings will need to be rebuilt next year?"):
        rec = to_official("q", answer_query(to_scene([], 1024, 1024), parse_question(q)))
        assert rec == {"id": "q", "answer": "cannot be determined", "region": None}
        assert json.dumps(rec) == '{"id": "q", "answer": "cannot be determined", "region": null}'
