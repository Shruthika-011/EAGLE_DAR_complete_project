"""Phase 2 tests: spatial rules, question parsing, answers, evidence, official JSON.

Run:  python -m pytest tests -q
The core under test needs only the standard library; practice-data and
rendering tests skip themselves when their inputs/dependencies are missing.
"""
import json
import math
from pathlib import Path

import pytest

from reasoning import spatial
from reasoning.engine import answer_query, count_by_quadrant
from reasoning.official import OFFICIAL_KEYS, to_official, validate_official
from reasoning.questions import LEVELS, parse_question
from reasoning.schema import CANNOT_BE_DETERMINED, CLASS_NAMES, Building, Scene

ROOT = Path(__file__).resolve().parent.parent
PRACTICE = ROOT / "dataset_source" / "practice"
TEMPLATES = ROOT / "dataset_source" / "question_templates.json"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def bld(bid, cls, cx, cy, half=5, w=1024, h=1024, conf=None):
    x1, y1 = max(0, math.floor(cx - half)), max(0, math.floor(cy - half))
    x2, y2 = min(w, math.ceil(cx + half)), min(h, math.ceil(cy + half))
    return Building(bid, cls, (cx, cy), (x1, y1, x2, y2), confidence=conf)


def scene(buildings, w=1024, h=1024):
    return Scene(w, h, tuple(buildings))


def ask(sc, text, **kw):
    return answer_query(sc, parse_question(text), **kw)


# quadrant centres used throughout (1024x1024)
TL, TR, BL, BR = (256, 256), (768, 256), (256, 768), (768, 768)


# --------------------------------------------------------------------------
# quadrant assignment, boundaries, image centre
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pt,expected", [
    ((0, 0), "top-left"), ((100, 100), "top-left"), ((511.999, 511.999), "top-left"),
    ((512, 0), "top-right"), ((1024, 0), "top-right"), ((900, 100), "top-right"),
    ((0, 512), "bottom-left"), ((100, 900), "bottom-left"), ((511.999, 1024), "bottom-left"),
    ((512, 512), "bottom-right"), ((1024, 1024), "bottom-right"), ((600, 600), "bottom-right"),
])
def test_quadrant_assignment_and_boundaries(pt, expected):
    assert spatial.quadrant_of(*pt, 1024, 1024) == expected


def test_exact_centre_line_goes_right_and_bottom():
    assert spatial.quadrant_of(512, 100, 1024, 1024) == "top-right"
    assert spatial.quadrant_of(100, 512, 1024, 1024) == "bottom-left"
    assert not spatial.in_half(512, 5, 1024, 1024, "left")
    assert spatial.in_half(512, 5, 1024, 1024, "right")


def test_split_uses_actual_image_centre_not_512():
    # 800x600: centre is (400, 300)
    assert spatial.quadrant_of(399, 299, 800, 600) == "top-left"
    assert spatial.quadrant_of(400, 299, 800, 600) == "top-right"
    assert spatial.quadrant_of(399, 300, 800, 600) == "bottom-left"
    assert spatial.quadrant_of(450, 350, 800, 600) == "bottom-right"
    # a point that would be top-left in a 1024 image is bottom-right here
    assert spatial.quadrant_of(450, 350, 800, 600) != spatial.quadrant_of(450, 350, 1024, 1024)


def test_odd_image_size_centre():
    assert spatial.quadrant_of(512.4, 10, 1025, 1025) == "top-left"
    assert spatial.quadrant_of(512.5, 10, 1025, 1025) == "top-right"


def test_centre_outside_image_is_rejected():
    for pt in [(-1, 5), (5, -0.1), (1024.1, 5), (5, 1025)]:
        with pytest.raises(ValueError):
            spatial.quadrant_of(*pt, 1024, 1024)


def test_every_point_belongs_to_exactly_one_quadrant():
    for w, h in [(1024, 1024), (800, 600), (1025, 513)]:
        for x in [0, 1, w / 2 - 0.001, w / 2, w / 2 + 0.001, w - 1, w]:
            for y in [0, 1, h / 2 - 0.001, h / 2, h / 2 + 0.001, h - 1, h]:
                inside = [q for q in spatial.QUADRANTS if spatial.quadrant_of(x, y, w, h) == q]
                assert len(inside) == 1
                lr = [hn for hn in ("left", "right") if spatial.in_half(x, y, w, h, hn)]
                tb = [hn for hn in ("top", "bottom") if spatial.in_half(x, y, w, h, hn)]
                assert len(lr) == 1 and len(tb) == 1
                assert set(inside[0].split("-")) == {tb[0], lr[0]}   # quadrant agrees with halves


def test_region_rectangles_match_official_templates():
    t = json.loads(TEMPLATES.read_text(encoding="utf-8")) if TEMPLATES.exists() else None
    if t is None:
        pytest.skip("question_templates.json not present")
    for name, rect in {**t["quadrants"], **t["halves"]}.items():
        assert list(spatial.region_rect(name, 1024, 1024)) == rect, name


def test_region_rectangles_scale_with_image_size():
    assert spatial.region_rect("bottom-right", 800, 600) == (400, 300, 800, 600)
    assert spatial.region_rect("left", 800, 600) == (0, 0, 400, 600)
    assert spatial.region_rect("image", 800, 600) == (0, 0, 800, 600)


def test_polygon_centroid_matches_shapely():
    shapely_geom = pytest.importorskip("shapely.geometry")
    pts = [(10, 10), (60, 12), (55, 50), (20, 70), (8, 40)]
    c = spatial.polygon_centroid(pts)
    ref = shapely_geom.Polygon(pts).centroid
    assert c == pytest.approx((ref.x, ref.y))


def test_polygon_bbox_is_floor_ceil_and_clipped():
    assert spatial.polygon_bbox([(10.2, 20.7), (30.1, 40.0)], 1024, 1024) == (10, 20, 31, 40)
    assert spatial.polygon_bbox([(-5, -5), (2000, 2000)], 1024, 1024) == (0, 0, 1024, 1024)


# --------------------------------------------------------------------------
# type A: counts, the three classes, removed levels, empty regions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls,name,text", [
    (0, "no-damage", "How many undamaged buildings are there in the image?"),
    (1, "major-damage", "Count the buildings with major damage in the image."),
    (2, "destroyed", "Count the destroyed buildings in the image."),
])
def test_all_three_damage_classes_count_only_themselves(cls, name, text):
    assert CLASS_NAMES[cls] == name
    # 1 building of class 0, 2 of class 1, 3 of class 2 -> distinguishable counts
    bs, i = [], 0
    for c, n in enumerate([1, 2, 3]):
        for _ in range(n):
            bs.append(bld(i, c, 100 + i * 20, 100)); i += 1
    ans = ask(scene(bs), text)
    assert ans.qtype == "A" and ans.answer == cls + 1
    assert len(ans.supporting_ids) == cls + 1


@pytest.mark.parametrize("text", [
    "How many buildings with minor damage are there in the image?",
    "Count the minor-damage buildings in the top-left quadrant.",
    "Count the unclassified buildings in the image.",
    "How many un-classified buildings are there in the bottom-right quadrant?",
    "Which quadrant has the most unclassified buildings?",
])
def test_removed_levels_minor_and_unclassified_are_cannot_be_determined(text):
    a = ask(scene([bld(0, 0, *TL), bld(1, 1, *TL), bld(2, 2, *BR)]), text)
    assert a.qtype == "D" and a.answer == CANNOT_BE_DETERMINED and a.region is None
    assert "reason" in a.details


def test_removed_levels_are_not_in_levels_and_there_are_three_classes():
    assert set(LEVELS) == {"no-damage", "major-damage", "destroyed", "damaged"}
    assert CLASS_NAMES == ("no-damage", "major-damage", "destroyed")


def test_damaged_is_major_plus_destroyed_and_excludes_nodamage():
    bs = [bld(0, 0, *TL), bld(1, 1, *TL), bld(2, 2, *TL), bld(3, 0, *TL), bld(4, 2, *TL)]
    ans = ask(scene(bs), "Count the damaged buildings (major or destroyed) in the image.")
    assert ans.answer == 3
    assert set(ans.supporting_ids) == {1, 2, 4}


def test_no_damage_buildings_never_count_as_damaged_in_comparisons_and_quadrants():
    bs = [bld(i, 0, *TL) for i in range(10)] + [bld(100, 2, *BR)]
    sc = scene(bs)
    assert ask(sc, "Which quadrant has the most damaged buildings?").answer == "bottom-right"
    assert ask(sc, "Is the left half of the image more damaged than the right half?").answer == "no"
    assert ask(sc, "Is the right half of the image more damaged than the left half?").answer == "yes"


def test_quadrant_count_scope():
    bs = [bld(0, 2, *TL), bld(1, 2, *TL), bld(2, 2, *TR), bld(3, 0, *TL)]
    sc = scene(bs)
    assert ask(sc, "Count the destroyed buildings in the top-left quadrant.").answer == 2
    assert ask(sc, "Count the destroyed buildings in the top-right quadrant.").answer == 1
    assert ask(sc, "Count the destroyed buildings in the bottom-left quadrant.").answer == 0
    assert ask(sc, "Count the destroyed buildings in the image.").answer == 3


def test_empty_region_count_is_zero_with_scope_rectangle_as_evidence():
    sc = scene([bld(0, 2, *TL)])
    a = ask(sc, "How many buildings with major damage are there in the top-left quadrant?")
    assert (a.answer, a.region, a.supporting_ids) == (0, (0, 0, 512, 512), ())   # official: [0,0,512,512]
    a = ask(sc, "Count the destroyed buildings in the bottom-right quadrant.")
    assert (a.answer, a.region) == (0, (512, 512, 1024, 1024))
    a = ask(scene([]), "Count the destroyed buildings in the image.")
    assert (a.answer, a.region) == (0, (0, 0, 1024, 1024))


def test_building_on_the_centre_line_is_counted_once_in_one_quadrant():
    sc = scene([bld(0, 2, 512, 512)])           # exactly on both centre lines
    counts = {q: ask(sc, f"Count the destroyed buildings in the {q} quadrant.").answer for q in spatial.QUADRANTS}
    assert sum(counts.values()) == 1 and counts["bottom-right"] == 1


# --------------------------------------------------------------------------
# type B: most damaged quadrant
# --------------------------------------------------------------------------

def test_most_damaged_quadrant_each_quadrant_can_win():
    for centre, name in [(TL, "top-left"), (TR, "top-right"), (BL, "bottom-left"), (BR, "bottom-right")]:
        sc = scene([bld(0, 1, *centre), bld(1, 2, *centre), bld(2, 2, *TL if name != "top-left" else TR)])
        a = ask(sc, "Which quadrant of the image contains the largest number of damaged buildings "
                    "(major or destroyed)?")
        assert a.qtype == "B" and a.answer == name
        assert a.region == spatial.region_rect(name, 1024, 1024)
        assert a.region_name == name


def test_most_damaged_quadrant_ties_are_not_resolved_by_invented_rule():
    sc = scene([bld(0, 2, *TL), bld(1, 2, *BR)])
    a = ask(sc, "Which quadrant has the most damaged buildings?")
    assert a.answer == CANNOT_BE_DETERMINED and a.region is None
    assert a.details["tie_unresolved"] is True
    assert set(a.details["tied_quadrants"]) == {"top-left", "bottom-right"}


def test_no_damaged_buildings_anywhere_is_a_four_way_tie():
    sc = scene([bld(0, 0, *TL), bld(1, 0, *BR)])
    a = ask(sc, "Which quadrant has the most damaged buildings?")
    assert a.answer == CANNOT_BE_DETERMINED and a.details["tie_unresolved"] is True


def test_quadrant_question_on_image_without_buildings_cannot_be_determined():
    a = ask(scene([]), "Which quadrant has the most damaged buildings?")
    assert a.answer == CANNOT_BE_DETERMINED and a.region is None


def test_most_destroyed_variant_from_templates():
    sc = scene([bld(0, 1, *TL), bld(1, 1, *TL), bld(2, 2, *BL)])
    a = ask(sc, "Which quadrant has the most destroyed buildings?")
    assert a.answer == "bottom-left"            # major damage in TL must not count


# --------------------------------------------------------------------------
# type C: half comparisons, strictly-more rule, equal counts
# --------------------------------------------------------------------------

def test_half_comparison_yes_when_strictly_more():
    sc = scene([bld(0, 2, *TL), bld(1, 1, *BL), bld(2, 1, *TR)])
    a = ask(sc, "Is the left half of the image more damaged than the right half?")
    assert a.answer == "yes" and a.region == (0, 0, 512, 1024) and a.region_name == "left half"


def test_half_comparison_no_when_fewer_and_region_is_the_half_with_more_damage():
    sc = scene([bld(0, 2, *TL), bld(1, 1, *BL), bld(2, 1, *TR)])
    a = ask(sc, "Is the right half of the image more damaged than the left half?")
    assert a.answer == "no"
    assert a.region == (0, 0, 512, 1024)        # official convention: region = the half that has more
    assert set(a.supporting_ids) == {0, 1}


def test_half_comparison_equal_counts_is_no_never_yes():
    sc = scene([bld(0, 2, *TL), bld(1, 2, *TR)])
    for q in ["Is the left half of the image more damaged than the right half?",
              "Does the right half of the image have more damaged buildings than the left half?"]:
        assert ask(sc, q).answer == "no", q
    a = ask(sc, "Is the left half of the image more damaged than the right half?")
    assert a.details["tie"] is True and a.region == (0, 0, 1024, 1024)
    tb = scene([bld(0, 2, *TL), bld(1, 2, *BL)])             # 1 damaged in top, 1 in bottom
    assert ask(tb, "Is the top half of the image more damaged than the bottom half?").answer == "no"
    assert ask(tb, "Is the bottom half of the image more damaged than the top half?").answer == "no"


def test_half_comparison_zero_damage_both_sides_is_no():
    sc = scene([bld(0, 0, *TL), bld(1, 0, *BR), bld(2, 0, *TR)])
    assert ask(sc, "Is the top half of the image more damaged than the bottom half?").answer == "no"


def test_half_comparison_empty_image_cannot_be_determined():
    a = ask(scene([]), "Is the left half of the image more damaged than the right half?")
    assert a.answer == CANNOT_BE_DETERMINED and a.region is None


def test_top_bottom_halves():
    sc = scene([bld(0, 2, *BL), bld(1, 2, *BR), bld(2, 2, *TL)])
    assert ask(sc, "Is the bottom half of the image more damaged than the top half?").answer == "yes"
    assert ask(sc, "Does the top half of the image have more damaged buildings than the bottom half?").answer == "no"


def test_non_opposite_or_single_half_comparison_is_unsupported():
    sc = scene([bld(0, 2, *TL)])
    assert ask(sc, "Is the left half of the image more damaged than the top half?").answer == CANNOT_BE_DETERMINED
    assert ask(sc, "Is the left half of the image more damaged than the whole image?").answer == CANNOT_BE_DETERMINED


# --------------------------------------------------------------------------
# type D
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "How many people lived in the destroyed buildings?",
    "How many buildings will need to be rebuilt next year?",
    "Which building in the image is a hospital?",
    "What date was the post-disaster image taken?",
    "",
    "Count the buildings.",
    "Count the destroyed and undamaged buildings in the image.",
])
def test_type_d_is_exactly_cannot_be_determined(text):
    sc = scene([bld(0, 2, *TL)])
    a = ask(sc, text)
    assert a.qtype == "D" and a.answer == "cannot be determined" and a.region is None
    assert "reason" in a.details


# --------------------------------------------------------------------------
# evidence and traceability
# --------------------------------------------------------------------------

def test_count_evidence_is_union_bbox_of_counted_buildings_only():
    bs = [bld(0, 2, 100, 100, half=10), bld(1, 2, 300, 400, half=10), bld(2, 2, 700, 700, half=10),
          bld(3, 0, 900, 900, half=10)]
    a = ask(scene(bs), "Count the destroyed buildings in the image.")
    assert a.answer == 3 and a.supporting_ids == (0, 1, 2)
    assert a.region == (90, 90, 710, 710)        # the no-damage building at 900 is not evidence


def test_count_evidence_may_exceed_the_quadrant_like_the_official_key_and_is_clipped_to_the_image():
    # centre inside top-left, polygon sticks into the neighbouring quadrants and out of the image
    b = Building(0, 0, (505, 505), (480, 480, 540, 540))
    a = ask(scene([b]), "How many undamaged buildings are there in the top-left quadrant?")
    assert a.region == (480, 480, 540, 540)
    edge = Building(1, 2, (3, 3), (0, 0, 20, 20))
    assert ask(scene([edge]), "Count the destroyed buildings in the image.").region == (0, 0, 20, 20)


def test_every_supporting_id_exists_and_matches_the_answer():
    bs = [bld(i, i % 3, 50 + 90 * i, 60 + 80 * i) for i in range(11)]
    sc = scene(bs)
    by_id = {b.building_id: b for b in sc.buildings}
    for text in ["Count the destroyed buildings in the image.",
                 "Count the damaged buildings (major or destroyed) in the image.",
                 "Which quadrant has the most damaged buildings?",
                 "Is the left half of the image more damaged than the right half?"]:
        a = ask(sc, text)
        assert set(a.supporting_ids) <= set(by_id)
        assert len(set(a.supporting_ids)) == len(a.supporting_ids)       # no building twice
        if a.qtype == "A":
            assert a.answer == len(a.supporting_ids)


def test_quadrant_and_half_evidence_are_image_coordinate_rectangles():
    sc = scene([bld(0, 2, *BR)])
    b = ask(sc, "Which quadrant has the most destroyed buildings?")
    assert b.region == (512, 512, 1024, 1024)
    c = ask(sc, "Is the bottom half of the image more damaged than the top half?")
    assert c.region == (0, 512, 1024, 1024)


def test_min_confidence_filter_removes_low_confidence_buildings_only_when_requested():
    bs = [bld(0, 2, *TL, conf=0.9), bld(1, 2, *TL, conf=0.1), bld(2, 2, *TL, conf=None)]
    sc = scene(bs)
    q = "Count the destroyed buildings in the image."
    assert ask(sc, q).answer == 3
    assert ask(sc, q, min_confidence=0.5).answer == 2        # None (ground truth) is kept
    assert ask(sc, q, min_confidence=0.0).answer == 3


# --------------------------------------------------------------------------
# duplicate prevention / data validation
# --------------------------------------------------------------------------

def test_duplicate_building_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        scene([bld(1, 2, *TL), bld(1, 0, *BR)])


def test_buildings_are_never_double_counted_across_regions():
    bs = [bld(i, i % 3, (i * 97) % 1025, (i * 193) % 1025) for i in range(60)]
    sc = scene(bs)
    for cls_name, level in [("destroyed", "destroyed"), ("undamaged", "no-damage"), ("damaged", "damaged")]:
        per_quadrant = count_by_quadrant(sc, level)
        assert sum(per_quadrant.values()) == sum(1 for b in bs if b.class_id in LEVELS[level])
    parts = spatial.assign_quadrants(sc.buildings, 1024, 1024)
    ids = [b.building_id for v in parts.values() for b in v]
    assert sorted(ids) == sorted(b.building_id for b in bs)


@pytest.mark.parametrize("kwargs", [
    dict(class_id=3), dict(class_id=-1), dict(centre=(2000, 5)), dict(confidence=1.5),
    dict(bbox=(10, 10, 5, 20)),
])
def test_invalid_buildings_are_rejected(kwargs):
    base = dict(building_id=0, class_id=1, centre=(10, 10), bbox=(0, 0, 20, 20), confidence=0.5)
    base.update(kwargs)
    with pytest.raises(ValueError):
        Scene(1024, 1024, (Building(**base),))


# --------------------------------------------------------------------------
# question parsing: every official template phrasing
# --------------------------------------------------------------------------

def _templates():
    if not TEMPLATES.exists():
        pytest.skip("question_templates.json not present")
    return json.loads(TEMPLATES.read_text(encoding="utf-8"))


REMOVED_LEVELS = {"minor-damage", "un-classified"}


def test_every_type1_template_levels_and_scopes_parse():
    t = _templates()
    scopes = [("in the image", "image", None)] + [(f"in the {q} quadrant", "quadrant", q) for q in spatial.QUADRANTS]
    n = 0
    for tmpl in t["type_1"]:
        for lvl_key, phrase in t["levels"].items():
            for scope_text, kind, name in scopes:
                q = parse_question(tmpl.format(lvl=phrase, scope=scope_text))
                if lvl_key in REMOVED_LEVELS:
                    # minor-damage no longer exists as a class: the official phrasing must now be "cannot be determined"
                    assert q.qtype == "D" and q.reason, (tmpl, phrase, scope_text)
                else:
                    assert (q.qtype, q.level, q.scope_kind, q.scope_name) == ("A", lvl_key, kind, name), (tmpl, phrase, scope_text)
                n += 1
    assert n == 2 * len(t["levels"]) * 5


def test_every_type2_template_parses():
    t = _templates()
    for level, texts in t["type_2"].items():
        for text in texts:
            q = parse_question(text)
            assert (q.qtype, q.level) == ("B", level), text


def test_every_type3_template_parses_for_all_four_orderings():
    t = _templates()
    for tmpl in t["type_3"]:
        for a, b in [("left", "right"), ("right", "left"), ("top", "bottom"), ("bottom", "top")]:
            q = parse_question(tmpl.format(a=a, b=b))
            assert (q.qtype, q.halves) == ("C", (a, b)), tmpl


def test_every_type4_example_is_unsupported():
    for text in _templates()["type_4_out_of_scope_examples"]:
        assert parse_question(text).qtype == "D", text


def test_undamaged_is_not_confused_with_damaged_and_parenthetical_is_not_a_second_level():
    assert parse_question("How many undamaged buildings are there in the image?").level == "no-damage"
    # the official template wording still parses as 'damaged' (the parenthesis is a definition)
    q = parse_question("How many damaged buildings (minor, major or destroyed) are there in the image?")
    assert q.level == "damaged"
    q = parse_question("How many damaged buildings (major or destroyed) are there in the image?")
    assert q.level == "damaged"
    assert LEVELS["damaged"] == {1, 2}


# --------------------------------------------------------------------------
# official JSON
# --------------------------------------------------------------------------

def test_official_record_has_exactly_id_answer_region():
    sc = scene([bld(0, 2, *BR)])
    for text, answer in [("Count the destroyed buildings in the image.", 1),
                         ("Which quadrant has the most damaged buildings?", "bottom-right"),
                         ("Is the bottom half of the image more damaged than the top half?", "yes"),
                         ("Which building is a hospital?", "cannot be determined")]:
        rec = to_official("qtest", ask(sc, text))
        assert tuple(rec.keys()) == OFFICIAL_KEYS
        assert rec["answer"] == answer
        validate_official(rec, 1024, 1024)
        json.dumps(rec)


def test_official_answer_types():
    sc = scene([bld(0, 2, *BR)])
    r = to_official("q", ask(sc, "Count the destroyed buildings in the image."))
    assert isinstance(r["answer"], int) and not isinstance(r["answer"], bool)
    r = to_official("q", ask(sc, "Which building is a hospital?"))
    assert r == {"id": "q", "answer": "cannot be determined", "region": None}


def test_region_name_format_is_available_for_type_2_and_3():
    sc = scene([bld(0, 2, *BR)])
    assert to_official("q", ask(sc, "Which quadrant has the most damaged buildings?"), "name")["region"] == "bottom-right"
    assert to_official("q", ask(sc, "Is the bottom half of the image more damaged than the top half?"),
                       "name")["region"] == "bottom half"


@pytest.mark.parametrize("rec", [
    {"id": "q", "answer": 1, "region": [0, 0, 1, 1], "extra": 1},
    {"id": "q", "answer": 1},
    {"id": "q", "answer": True, "region": None},
    {"id": "q", "answer": "maybe", "region": None},
    {"id": "q", "answer": 1.5, "region": None},
    {"id": "q", "answer": "cannot be determined", "region": [0, 0, 1, 1]},
    {"id": "q", "answer": 1, "region": [0, 0, 1]},
    {"id": "q", "answer": 1, "region": [5, 0, 1, 1]},
    {"id": "q", "answer": 1, "region": [0, 0, 2000, 1]},
    {"answer": 1, "id": "q", "region": None},          # key order is part of the contract
])
def test_invalid_official_records_are_rejected(rec):
    with pytest.raises(ValueError):
        validate_official(rec, 1024, 1024)


# --------------------------------------------------------------------------
# loaders: Phase 1 JSON
# --------------------------------------------------------------------------

def _prediction_json(tmp_path, buildings, w=1024, h=1024):
    data = {"schema": "eagle-dar/building-damage/v1", "inputs": {"width": w, "height": h,
            "after_image": "x/t0001_post.jpg"}, "alignment": {"status": "aligned"}, "warnings": [],
            "buildings": buildings}
    p = tmp_path / "p.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_loader_recomputes_centre_from_polygon_and_ignores_stored_quadrant(tmp_path):
    from reasoning.loaders import scene_from_prediction_json
    # polygon centroid is y=511.96 (top); a 1-decimal rounded stored centre would say 512.0 (bottom)
    poly = [[100, 501.96], [120, 501.96], [120, 521.96], [100, 521.96]]
    b = {"building_id": 7, "class_id": 2, "polygon_xy": poly, "centre_xy": [110.0, 512.0],
         "quadrant": "bottom-left", "building_confidence": 0.5}
    sc = scene_from_prediction_json(_prediction_json(tmp_path, [b]))
    assert sc.buildings[0].centre == pytest.approx((110.0, 511.96))
    assert spatial.quadrant_of(*sc.buildings[0].centre, 1024, 1024) == "top-left"
    assert sc.scene_id == "t0001"
    assert sc.buildings[0].confidence == 0.5


def test_loader_maps_legacy_five_class_predictions(tmp_path):
    """Files written by the old 5-class pipeline: minor -> major-damage, un-classified buildings dropped."""
    from reasoning.loaders import scene_from_prediction_json
    sq = lambda x: [[x, 10], [x + 20, 10], [x + 20, 30], [x, 30]]
    legacy = [
        {"building_id": 0, "class_id": 0, "predicted_damage": "no-damage", "polygon_xy": sq(10)},
        {"building_id": 1, "class_id": 1, "predicted_damage": "minor-damage", "polygon_xy": sq(100)},
        {"building_id": 2, "class_id": 2, "predicted_damage": "major-damage", "polygon_xy": sq(200)},
        {"building_id": 3, "class_id": 3, "predicted_damage": "destroyed", "polygon_xy": sq(300)},
        {"building_id": 4, "class_id": 4, "predicted_damage": "un-classified", "polygon_xy": sq(400)},
    ]
    sc = scene_from_prediction_json(_prediction_json(tmp_path, legacy))
    assert [(b.building_id, b.damage_name) for b in sc.buildings] == [
        (0, "no-damage"), (1, "major-damage"), (2, "major-damage"), (3, "destroyed")]       # id 4 dropped
    assert ask(sc, "Count the damaged buildings (major or destroyed) in the image.").answer == 3


def test_loader_ground_truth_maps_minor_to_major_and_drops_unclassified(tmp_path):
    pytest.importorskip("shapely")
    from reasoning.loaders import scene_from_ground_truth
    def feat(uid, subtype, x):
        return {"properties": {"feature_type": "building", "subtype": subtype, "uid": uid},
                "wkt": f"POLYGON (({x} 10, {x + 20} 10, {x + 20} 30, {x} 30, {x} 10))"}
    data = {"features": {"xy": [feat("a", "no-damage", 10), feat("b", "minor-damage", 100), feat("c", "major-damage", 200),
                                feat("d", "destroyed", 300), feat("e", "un-classified", 400)]},
            "metadata": {"width": 1024, "height": 1024}}
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    sc = scene_from_ground_truth(p)
    assert [(b.building_id, b.damage_name) for b in sc.buildings] == [
        ("a", "no-damage"), ("b", "major-damage"), ("c", "major-damage"), ("d", "destroyed")]


def test_collapse_probs_ignores_unclassified_merges_minor_and_normalises():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from inference.pipeline import collapse_probs
    raw = np.array([[0.10, 0.20, 0.10, 0.10, 0.50],      # un-classified dominates -> must be ignored
                    [0.25, 0.25, 0.25, 0.25, 0.00],
                    [1.00, 0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00, 1.00]])     # only un-classified: degenerate, must not produce NaN
    out = collapse_probs(raw)
    assert out.shape == (4, 3)
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx([0.10 / 0.50, 0.30 / 0.50, 0.10 / 0.50])       # (no, minor+major, destroyed) / (1 - p4)
    assert out[1] == pytest.approx([0.25, 0.50, 0.25])
    assert out[2] == pytest.approx([1.0, 0.0, 0.0])
    assert out[3] == pytest.approx([0.0, 0.0, 0.0])
    assert out[:3].sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])
    assert int(out[0].argmax()) == 1                                              # picked major-damage, never a 4th class
    assert collapse_probs(np.zeros((0, 5))).shape == (0, 3)


def test_loader_falls_back_to_bbox_and_rejects_unknown_schema(tmp_path):
    from reasoning.loaders import scene_from_prediction_json
    b = {"building_id": 1, "class_id": 0, "bbox_xyxy": [10, 10, 30, 50], "building_confidence": 0.4}
    sc = scene_from_prediction_json(_prediction_json(tmp_path, [b]))
    assert sc.buildings[0].centre == (20.0, 30.0)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other", "inputs": {}, "buildings": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        scene_from_prediction_json(bad)


# --------------------------------------------------------------------------
# official practice examples (ground-truth buildings: validates the LOGIC only)
# --------------------------------------------------------------------------

# Practice answers our rules do NOT reproduce. Pinned so any change is noticed.
#  q084c5b9: official 50, labels give 51 (building b0121, centroid y=509.7, is excluded by
#            the key's region too; no centre definition we tried explains it)
#  qb62381d / q17b7814: scene p6adc4d5d is clearly answerable (unique winner) but the key
#            says "cannot be determined"; the reason is not derivable from labels or templates
KNOWN_UNEXPLAINED = {"q084c5b9", "qb62381d", "q17b7814"}

# Official practice questions that name a level which no longer exists (minor damage / un-classified). The app
# now answers "cannot be determined" for them, so they intentionally differ from the official key.
# 'damaged' questions still match: the ground-truth loader maps minor -> major-damage, so
# damaged = (minor + major) + destroyed, exactly as the official key counts it (un-classified excluded in both).
REMOVED_LEVEL_QUESTIONS = {"q3dc25a7", "qd318ce2"}
# 'major damage' now includes minor-damage buildings, so the count can exceed the official major-only key.
# qf7ad0b5: official 0 major in the top-left quadrant; the 2 buildings we count there are labelled minor-damage.
MERGED_MAJOR_QUESTIONS = {"qf7ad0b5": 2}


@pytest.fixture(scope="module")
def practice_results():
    pytest.importorskip("shapely")
    if not (PRACTICE / "answers.jsonl").exists():
        pytest.skip("practice dataset not present")
    from reasoning.loaders import scene_from_ground_truth
    key = [json.loads(l) for l in (PRACTICE / "answers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    scenes, out = {}, []
    for k in key:
        sc = scenes.setdefault(k["pair_id"], scene_from_ground_truth(PRACTICE / "labels" / f"{k['pair_id']}.json"))
        out.append((k, answer_query(sc, parse_question(k["question"])), sc))
    return out


def test_practice_all_type2_and_type3_answers_and_regions_reproduced(practice_results):
    n = 0
    for k, a, _ in practice_results:
        if k["type"] in (2, 3) and k["id"] not in KNOWN_UNEXPLAINED:
            assert a.answer == k["answer"], k["id"]
            assert list(a.region) == k["region"], k["id"]
            n += 1
    assert n == 20


def test_practice_type1_counts_and_regions_reproduced(practice_results):
    n = 0
    for k, a, _ in practice_results:
        if k["type"] == 1 and k["id"] not in KNOWN_UNEXPLAINED | REMOVED_LEVEL_QUESTIONS | set(MERGED_MAJOR_QUESTIONS):
            assert a.answer == k["answer"], k["id"]
            assert list(a.region) == k["region"], k["id"]
            n += 1
    assert n == 16


def test_practice_major_damage_counts_now_include_minor_damage(practice_results):
    seen = set()
    for k, a, sc in practice_results:
        if k["id"] in MERGED_MAJOR_QUESTIONS:
            labels = json.loads((PRACTICE / "labels" / f"{k['pair_id']}.json").read_text(encoding="utf-8"))
            subtype = {f["properties"]["uid"]: f["properties"]["subtype"] for f in labels["features"]["xy"]}
            assert a.answer == MERGED_MAJOR_QUESTIONS[k["id"]] == len(a.supporting_ids), k["id"]
            assert {subtype[i] for i in a.supporting_ids} == {"minor-damage"}, k["id"]
            seen.add(k["id"])
    assert seen == set(MERGED_MAJOR_QUESTIONS)


def test_practice_removed_level_questions_are_cannot_be_determined(practice_results):
    seen = set()
    for k, a, _ in practice_results:
        if k["id"] in REMOVED_LEVEL_QUESTIONS:
            assert a.qtype == "D" and a.answer == CANNOT_BE_DETERMINED and a.region is None, k["id"]
            seen.add(k["id"])
    assert seen == REMOVED_LEVEL_QUESTIONS


def test_practice_type4_questions_are_cannot_be_determined(practice_results):
    n = 0
    for k, a, _ in practice_results:
        if k["type"] == 4 and k["id"] not in KNOWN_UNEXPLAINED:
            assert a.answer == CANNOT_BE_DETERMINED and a.region is None, k["id"]
            n += 1
    assert n == 8


def test_practice_mismatches_are_exactly_the_documented_set(practice_results):
    mismatched = {k["id"] for k, a, _ in practice_results if a.answer != k["answer"]}
    assert mismatched == KNOWN_UNEXPLAINED | REMOVED_LEVEL_QUESTIONS | set(MERGED_MAJOR_QUESTIONS)


def test_practice_official_records_validate(practice_results):
    for k, a, sc in practice_results:
        rec = to_official(k["id"], a)
        validate_official(rec, sc.width, sc.height)


# --------------------------------------------------------------------------
# five-class colour map
# --------------------------------------------------------------------------

def test_class_colours_are_consistent_with_config_and_have_the_required_hues():
    pytest.importorskip("cv2")
    from reasoning.damage_map import CLASS_COLORS_BGR, CLASS_COLOR_LABELS
    import config
    assert dict(CLASS_COLORS_BGR) == dict(config.DAMAGE_COLORS_BGR)
    assert tuple(CLASS_COLORS_BGR) == CLASS_NAMES
    assert tuple(config.APP_DAMAGE_NAMES[i] for i in range(3)) == CLASS_NAMES
    assert list(CLASS_COLOR_LABELS.values()) == ["green", "orange", "red"]
    assert len(set(CLASS_COLORS_BGR.values())) == 3
    b, g, r = CLASS_COLORS_BGR["no-damage"];       assert g > r and g > b            # green
    b, g, r = CLASS_COLORS_BGR["major-damage"];    assert r > 200 and 100 < g < 200 and b < 50  # orange
    b, g, r = CLASS_COLORS_BGR["destroyed"];       assert r > 200 and g < 100 and b < 100      # red


def test_damage_map_paints_each_building_in_its_class_colour():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from reasoning.damage_map import CLASS_COLORS_BGR, render_damage_map
    sc = Scene(200, 200, tuple(
        Building(i, i, (20 + 36 * i, 100), (10 + 36 * i, 90, 30 + 36 * i, 110),
                 polygon=((10 + 36 * i, 90), (30 + 36 * i, 90), (30 + 36 * i, 110), (10 + 36 * i, 110)), confidence=0.9)
        for i in range(3)))
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    out = render_damage_map(img, sc, alpha=1.0, draw_ids=False, draw_split_lines=False, legend=False)
    for i, name in enumerate(CLASS_NAMES):
        px = tuple(int(v) for v in out[100, 20 + 36 * i - 4])      # inside the filled square, away from the outline
        assert px == CLASS_COLORS_BGR[name], name
    with pytest.raises(ValueError):
        render_damage_map(np.zeros((100, 100, 3), dtype=np.uint8), sc)


# --------------------------------------------------------------------------
# report facts (Novelty 3 data layer)
# --------------------------------------------------------------------------

def test_report_facts_are_consistent_with_the_engine():
    from reasoning.report_facts import build_report_facts
    bs = [bld(0, 2, *TL, conf=0.9), bld(1, 1, *TL, conf=0.2), bld(2, 1, *BR, conf=0.5),
          bld(3, 0, *TR, conf=0.7), bld(4, 0, *BL, conf=0.6)]
    f = build_report_facts(scene(bs), eval_dir=Path("does-not-exist"))
    assert f["scene"]["total_buildings"] == 5
    assert f["damage_counts"] == {"no-damage": 2, "major-damage": 2, "destroyed": 1}
    assert f["damaged_total"] == 3 and f["damaged_fraction"] == 0.6
    assert f["quadrants"]["top-left"]["damaged"] == 2
    assert sum(q["damaged"] for q in f["quadrants"].values()) == f["damaged_total"]
    assert f["most_damaged_quadrant"]["answer"] == "top-left"
    assert f["confidence"]["low_confidence_building_ids"] == [1]
    assert [p["building_id"] for p in f["priority_buildings"]] == [0, 2, 1]    # destroyed first, then major by confidence
    assert f["data_quality"]["evaluation_context"] == {}                       # nothing fabricated
    assert f["llm_guardrails"]
    json.dumps(f)
