"""Backend tests. They run the REAL models on REAL practice images (no mocks).

    python -m pytest tests/test_backend_api.py -q

Skipped automatically when the model weights, ultralytics or the practice images are missing.
"""
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("ultralytics")
pytest.importorskip("httpx")

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.errors import ApiError
from backend.main import create_app
from backend.utils import image_validation as iv
from backend.utils.storage import AnalysisStore

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "dataset_source" / "practice" / "images"
PAIR_A, PAIR_B = "pb67aa0fe", "p4ca216d8"      # 77 and 6 buildings with the current models

settings_default = Settings()
pytestmark = pytest.mark.skipif(
    not (settings_default.yolo_weights.exists() and settings_default.siamese_weights.exists() and IMG.exists()),
    reason="model weights or practice images not present")


def pair(pid):
    return (IMG / f"{pid}_pre.jpg").read_bytes(), (IMG / f"{pid}_post.jpg").read_bytes()


def post_pair(client, pid, **form):
    b, a = pair(pid)
    return client.post("/api/analyze", files={"before": (f"{pid}_pre.jpg", b), "after": (f"{pid}_post.jpg", a)}, data=form)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    s = Settings(storage_dir=tmp_path_factory.mktemp("analyses"), max_pending=1)
    with TestClient(create_app(s)) as c:              # lifespan: models loaded once for the whole module
        yield c


@pytest.fixture(scope="module")
def analysis_a(client):
    r = post_pair(client, PAIR_A)
    assert r.status_code == 201, r.text
    return r.json()


def err(resp, status, code):
    assert resp.status_code == status, resp.text
    assert resp.json()["error"]["code"] == code, resp.text


# ---------------------------------------------------------------- health / docs

def test_health_reports_models_and_cuda(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    h = r.json()
    assert h["status"] == "ok"
    assert h["models"]["yolo"]["loaded"] and h["models"]["siamese"]["loaded"]
    assert h["models"]["yolo"]["exists"] and h["models"]["siamese"]["exists"]
    import torch
    assert h["device"]["cuda_available"] == torch.cuda.is_available()
    assert h["device"]["in_use"] in ("cuda", "cuda:0", "cpu")


def test_swagger_and_openapi_list_every_required_endpoint(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for p, m in [("/api/health", "get"), ("/api/analyze", "post"), ("/api/questions", "post"),
                 ("/api/analysis/{analysis_id}", "get"), ("/api/analysis/{analysis_id}/map", "get"),
                 ("/api/analysis/{analysis_id}/report", "get"), ("/api/analysis/{analysis_id}/export", "get"),
                 ("/api/analysis/{analysis_id}/reports", "post"), ("/api/reports/{report_id}", "get"), ("/api/reports/{report_id}/download", "get")]:
        assert m in paths[p], (p, m)


def test_models_are_loaded_once_not_per_request(client):
    pipe = client.app.state.inference.pipeline
    yolo_id, siamese_id = id(pipe.yolo), id(pipe.siamese)
    assert post_pair(client, PAIR_B).status_code == 201
    pipe2 = client.app.state.inference.pipeline
    assert (id(pipe2), id(pipe2.yolo), id(pipe2.siamese)) == (id(pipe), yolo_id, siamese_id)


# ---------------------------------------------------------------- analyze

def test_analyze_returns_real_statistics_and_confidences(client, analysis_a):
    j = analysis_a
    n = j["statistics"]["scene"]["total_buildings"]
    assert n == len(j["buildings"]) > 0
    assert sum(j["statistics"]["damage_counts"].values()) == n
    for b in j["buildings"]:
        assert 0 < b["building_confidence"] <= 1
        assert b["building_confidence"] == pytest.approx(b["detection_confidence"] * b["damage_confidence"], abs=2e-3)
        assert sum(b["class_probabilities"].values()) == pytest.approx(1.0, abs=2e-3)
    for key in ("map", "comparison", "report", "export", "self"):
        assert j["links"][key].startswith("/api/")


def test_analyze_matches_an_independent_pipeline_run(client, analysis_a):
    """Same models, separate code path (direct DamagePipeline call): identical counts and confidences."""
    from inference.pipeline import DamagePipeline, PipelineConfig
    s = client.app.state.settings
    direct = DamagePipeline(PipelineConfig(yolo_weights=s.yolo_weights, siamese_weights=s.siamese_weights,
                                           conf=s.default_conf, imgsz=s.default_imgsz))
    rep, _ = direct.run(IMG / f"{PAIR_A}_pre.jpg", IMG / f"{PAIR_A}_post.jpg")
    assert analysis_a["statistics"]["damage_counts"] == rep["summary"]["damage_counts"]
    assert [b["building_confidence"] for b in analysis_a["buildings"]] == pytest.approx(
        [b["building_confidence"] for b in rep["buildings"]], abs=1e-3)


def test_every_analysis_id_returns_its_own_results(client, analysis_a):
    rb = post_pair(client, PAIR_B).json()
    assert rb["analysis_id"] != analysis_a["analysis_id"]
    assert rb["statistics"]["scene"]["total_buildings"] != analysis_a["statistics"]["scene"]["total_buildings"]
    for j in (analysis_a, rb):
        got = client.get(f"/api/analysis/{j['analysis_id']}").json()
        assert got["analysis_id"] == j["analysis_id"] and got["statistics"] == j["statistics"]
        assert got["buildings"] == j["buildings"]


def test_same_input_twice_creates_two_analyses_and_never_overwrites(client):
    r1, r2 = post_pair(client, PAIR_B).json(), post_pair(client, PAIR_B).json()
    assert r1["analysis_id"] != r2["analysis_id"]
    assert client.get(f"/api/analysis/{r1['analysis_id']}").status_code == 200
    assert r1["statistics"]["damage_counts"] == r2["statistics"]["damage_counts"]


def test_png_upload_and_parameters(client):
    b, a = pair(PAIR_B)
    png = lambda raw: cv2.imencode(".png", cv2.imdecode(np.frombuffer(raw, np.uint8), 1))[1].tobytes()
    r = client.post("/api/analyze", files={"before": ("x.png", png(b)), "after": ("y.PNG", png(a))},
                    data={"conf": "0.4", "imgsz": "512", "alignment": "register"})
    assert r.status_code == 201, r.text
    assert r.json()["parameters"]["conf"] == 0.4 and r.json()["parameters"]["imgsz"] == 512


def test_higher_confidence_threshold_never_adds_buildings(client):
    low = post_pair(client, PAIR_A, conf="0.15").json()["statistics"]["scene"]["total_buildings"]
    high = post_pair(client, PAIR_A, conf="0.5").json()["statistics"]["scene"]["total_buildings"]
    assert high <= low


def test_stored_predictions_contain_no_server_paths(client, analysis_a):
    d = client.app.state.store.path(analysis_a["analysis_id"])
    text = (d / "predictions.json").read_text(encoding="utf-8")
    assert str(ROOT) not in text and "C:\\" not in text and "Users" not in text


@pytest.mark.parametrize("mutate,status,code", [
    (lambda b, a: (b[:3000], a), 422, "corrupted_image"),
    (lambda b, a: (b"not an image at all", a), 415, "unsupported_file_type"),
    (lambda b, a: (b"", a), 422, "empty_file"),
    (lambda b, a: (b, cv2.imencode(".jpg", np.zeros((512, 512, 3), np.uint8))[1].tobytes()), 422, "dimension_mismatch"),
    (lambda b, a: (cv2.imencode(".jpg", np.zeros((20, 20, 3), np.uint8))[1].tobytes(),) * 2, 422, "invalid_dimensions"),
])
def test_invalid_uploads_are_rejected_cleanly(client, mutate, status, code):
    b, a = mutate(*pair(PAIR_B))
    n_before = len(list(client.app.state.store.root.iterdir()))
    r = client.post("/api/analyze", files={"before": ("x.jpg", b), "after": ("y.jpg", a)})
    err(r, status, code)
    assert len(list(client.app.state.store.root.iterdir())) == n_before       # nothing stored for a rejected request


def test_extension_and_content_must_agree(client):
    b, a = pair(PAIR_B)
    err(client.post("/api/analyze", files={"before": ("x.png", b), "after": ("y.jpg", a)}), 415, "extension_mismatch")
    err(client.post("/api/analyze", files={"before": ("x.gif", b), "after": ("y.jpg", a)}), 415, "unsupported_file_type")


def test_oversized_upload_is_refused(client):
    s = client.app.state.settings
    old, s.max_upload_bytes = s.max_upload_bytes, 10_000
    try:
        b, a = pair(PAIR_B)
        err(client.post("/api/analyze", files={"before": ("x.jpg", b), "after": ("y.jpg", a)}), 413, "file_too_large")
    finally:
        s.max_upload_bytes = old


def test_bad_parameters(client):
    err(post_pair(client, PAIR_B, imgsz="650"), 422, "invalid_imgsz")
    err(post_pair(client, PAIR_B, imgsz="4096"), 422, "invalid_imgsz")
    err(post_pair(client, PAIR_B, conf="1.5"), 422, "validation_error")
    err(post_pair(client, PAIR_B, alignment="warp"), 422, "validation_error")


def test_hostile_filename_is_never_used_as_a_path(client):
    b, a = pair(PAIR_B)
    r = client.post("/api/analyze", files={"before": ("../../../evil.jpg", b), "after": ("..\\..\\evil2.jpg", a)})
    assert r.status_code == 201
    names = r.json()["image"]
    assert "/" not in names["before"]["display_name"] and "\\" not in names["after"]["display_name"]
    d = client.app.state.store.path(r.json()["analysis_id"])
    assert sorted(p.name for p in d.iterdir()) == sorted(
        ["after.jpg", "before.jpg", "comparison.jpg", "damage_map.png", "meta.json", "predictions.json"])


def test_unknown_and_malicious_analysis_ids(client):
    for bad in ("0" * 32, "G" * 32, "a" * 31, "..", "%2e%2e", "../../etc/passwd"):
        assert client.get(f"/api/analysis/{bad}").status_code == 404
    err(client.get("/api/analysis/" + "0" * 32), 404, "analysis_not_found")
    err(client.post("/api/questions", json={"analysis_id": "0" * 32, "questions": [{"question": "x"}]}), 404, "analysis_not_found")


# ---------------------------------------------------------------- concurrency

def test_concurrent_requests_are_safe_and_deterministic(client, analysis_a):
    with ThreadPoolExecutor(max_workers=4) as ex:
        rs = list(ex.map(lambda _: post_pair(client, PAIR_A), range(4)))
    assert {r.status_code for r in rs} <= {201, 503}
    ok = [r.json() for r in rs if r.status_code == 201]
    assert ok, "at least one concurrent request must succeed"
    assert len({j["analysis_id"] for j in ok}) == len(ok)
    for j in ok:                                           # no cross-request contamination
        assert j["statistics"]["damage_counts"] == analysis_a["statistics"]["damage_counts"]
    for r in rs:
        if r.status_code == 503:
            err(r, 503, "server_busy")
            assert r.headers["retry-after"]


# ---------------------------------------------------------------- questions / evidence / export

def test_questions_official_records_and_evidence(client, analysis_a):
    aid = analysis_a["analysis_id"]
    qs = [{"id": "qa", "question": "Count the destroyed buildings in the image."},
          {"id": "qb", "question": "Which quadrant has the most damaged buildings?"},
          {"id": "qc", "question": "Is the bottom half of the image more damaged than the top half?"},
          {"id": "qd", "question": "Which building in the image is a hospital?"},
          {"id": "qe", "structured": {"type": 1, "level": "no-damage", "scope": {"kind": "quadrant", "name": "top-left"}}}]
    r = client.post("/api/questions", json={"analysis_id": aid, "questions": qs})
    assert r.status_code == 200, r.text
    res = {x["official"]["id"]: x for x in r.json()["results"]}
    for x in res.values():
        assert list(x["official"].keys()) == ["id", "answer", "region"]
        assert set(x["internal"]) == {"question_type", "question_text", "parsed", "details", "evidence"}
    assert res["qa"]["official"]["answer"] == analysis_a["statistics"]["damage_counts"]["destroyed"]
    assert res["qd"]["official"] == {"id": "qd", "answer": "cannot be determined", "region": None}
    assert res["qb"]["official"]["answer"] == analysis_a["statistics"]["most_damaged_quadrant"]["answer"]
    ids = res["qa"]["internal"]["evidence"]["supporting_building_ids"]
    assert len(ids) == res["qa"]["official"]["answer"]
    by_id = {b["building_id"]: b for b in analysis_a["buildings"]}
    assert all(by_id[i]["predicted_damage"] == "destroyed" for i in ids)
    assert res["qe"]["official"]["answer"] == sum(
        1 for b in analysis_a["buildings"] if b["predicted_damage"] == "no-damage" and b["quadrant"] == "top-left")


def test_structured_and_natural_language_agree(client, analysis_a):
    aid = analysis_a["analysis_id"]
    nl = client.post("/api/questions", json={"analysis_id": aid, "questions": [
        {"id": "n1", "question": "Is the left half of the image more damaged than the right half?"}]}).json()
    st = client.post("/api/questions", json={"analysis_id": aid, "questions": [
        {"id": "n1", "structured": {"type": 3, "halves": ["left", "right"]}}]}).json()
    assert nl["results"][0]["official"] == st["results"][0]["official"]


def test_question_validation_errors(client, analysis_a):
    aid = analysis_a["analysis_id"]
    post = lambda qs, **kw: client.post("/api/questions", json={"analysis_id": aid, "questions": qs, **kw})
    err(post([{"question": "x", "structured": {"type": 4}}]), 422, "validation_error")
    err(post([{}]), 422, "validation_error")
    err(post([{"structured": {"type": 3, "halves": ["left", "top"]}}]), 422, "validation_error")
    err(post([{"structured": {"type": 2, "level": "no-damage"}}]), 422, "validation_error")
    err(post([{"id": "../x", "question": "x"}]), 422, "validation_error")
    err(post([{"id": "a", "question": "x"}, {"id": "a", "question": "y"}]), 422, "duplicate_question_id")
    err(post([{"question": "x"}], min_confidence=2), 422, "validation_error")


def test_min_confidence_changes_counts_consistently(client, analysis_a):
    aid = analysis_a["analysis_id"]
    q = [{"id": "m", "question": "How many undamaged buildings are there in the image?"}]
    all_n = client.post("/api/questions", json={"analysis_id": aid, "questions": q}).json()["results"][0]["official"]["answer"]
    thr = 0.3
    some = client.post("/api/questions", json={"analysis_id": aid, "questions": q, "min_confidence": thr}).json()
    expected = sum(1 for b in analysis_a["buildings"] if b["predicted_damage"] == "no-damage" and b["building_confidence"] >= thr)
    assert some["results"][0]["official"]["answer"] == expected < all_n


def test_export_is_official_and_isolated_per_analysis(client, analysis_a):
    other = post_pair(client, PAIR_B).json()["analysis_id"]
    assert client.get(f"/api/analysis/{other}/export").json() == []
    aid = analysis_a["analysis_id"]
    client.post("/api/questions", json={"analysis_id": aid, "questions": [
        {"id": "exp1", "question": "Count the destroyed buildings in the image."}]})
    r = client.get(f"/api/analysis/{aid}/export")
    recs = r.json()
    assert r.headers["x-analysis-id"] == aid and int(r.headers["x-record-count"]) == len(recs)
    assert all(list(x.keys()) == ["id", "answer", "region"] for x in recs)
    assert any(x["id"] == "exp1" for x in recs)
    lines = client.get(f"/api/analysis/{aid}/export", params={"format": "jsonl"}).text.splitlines()
    assert [json.loads(l) for l in lines] == recs
    from reasoning.official import validate_official
    for x in recs:
        validate_official(x, analysis_a["image"]["width"], analysis_a["image"]["height"])


def test_reasked_question_id_replaces_the_earlier_answer(client, analysis_a):
    aid = analysis_a["analysis_id"]
    for text in ("Count the destroyed buildings in the image.", "Count the undamaged buildings in the image."):
        client.post("/api/questions", json={"analysis_id": aid, "questions": [{"id": "same_id", "question": text}]})
    recs = [x for x in client.get(f"/api/analysis/{aid}/export").json() if x["id"] == "same_id"]
    assert len(recs) == 1 and recs[0]["answer"] == analysis_a["statistics"]["damage_counts"]["no-damage"]


# ---------------------------------------------------------------- map / comparison / report

def test_map_default_and_evidence_variants(client, analysis_a):
    aid = analysis_a["analysis_id"]
    r = client.get(f"/api/analysis/{aid}/map")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), 1)
    assert img.shape[0] == analysis_a["image"]["height"] and img.shape[1] > analysis_a["image"]["width"]   # image + legend
    ev = client.get(f"/api/analysis/{aid}/map", params={"evidence": "0,0,512,512", "highlight": "0,1"})
    assert ev.status_code == 200 and ev.content != r.content
    nolegend = client.get(f"/api/analysis/{aid}/map", params={"legend": "false"})
    assert cv2.imdecode(np.frombuffer(nolegend.content, np.uint8), 1).shape[1] == analysis_a["image"]["width"]
    err(client.get(f"/api/analysis/{aid}/map", params={"evidence": "0,0,5000,5000"}), 422, "invalid_evidence")
    err(client.get(f"/api/analysis/{aid}/map", params={"evidence": "a,b"}), 422, "invalid_evidence")
    err(client.get(f"/api/analysis/{aid}/map", params={"highlight": "99999"}), 422, "invalid_highlight")
    assert client.get(f"/api/analysis/{aid}/comparison").headers["content-type"] == "image/jpeg"


def test_map_colours_follow_the_three_class_scheme(client, analysis_a):
    from reasoning.damage_map import CLASS_COLORS_BGR
    aid = analysis_a["analysis_id"]
    r = client.get(f"/api/analysis/{aid}/map", params={"legend": "false"})
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), 1)
    pj = client.get(f"/api/analysis/{aid}", params={"include_polygons": True}).json()
    destroyed = next(b for b in pj["buildings"] if b["predicted_damage"] == "destroyed")
    x1, y1, x2, y2 = destroyed["bbox_xyxy"]
    patch = img[y1:y2, x1:x2].reshape(-1, 3).astype(int)
    target = np.array(CLASS_COLORS_BGR["destroyed"])
    near = (np.abs(patch - target).sum(axis=1) < 150).mean()     # red tint dominates the building footprint
    assert near > 0.05


def test_report_contains_only_computed_numbers_and_images(client, analysis_a):
    aid = analysis_a["analysis_id"]
    r = client.get(f"/api/analysis/{aid}/report")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert r.text.count("data:image/jpeg;base64,") >= 2 + 2        # comparison + map + at least one before/after pair
    doc = client.get(f"/api/analysis/{aid}/report", params={"format": "json"}).json()
    f, n = doc["facts"], analysis_a["statistics"]["scene"]["total_buildings"]
    assert f["scene"]["total_buildings"] == n
    summary = " ".join(next(s for s in doc["sections"] if s["title"] == "Summary")["paragraphs"])
    assert f"{f['damaged_total']} of {n} detected buildings" in summary
    assert "no language model" in doc["generation"]
    assert f["llm_guardrails"]


def test_report_regenerates_after_new_questions(client, analysis_a):
    aid = analysis_a["analysis_id"]
    first = client.get(f"/api/analysis/{aid}/report").text
    client.post("/api/questions", json={"analysis_id": aid, "questions": [
        {"id": "report_q", "question": "Count the damaged buildings (major or destroyed) in the image."}]})
    assert "report_q" in client.get(f"/api/analysis/{aid}/report").text and "report_q" not in first.replace("report_questions", "")


# ---------------------------------------------------------------- frontend support: images, context

def test_uploaded_images_are_served_back_unchanged(client, analysis_a):
    aid = analysis_a["analysis_id"]
    before, after = pair(PAIR_A)
    r = client.get(f"/api/analysis/{aid}/image/before")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content == before
    assert client.get(f"/api/analysis/{aid}/image/after").content == after
    err(client.get(f"/api/analysis/{aid}/image/other"), 422, "validation_error")
    assert analysis_a["links"]["before_image"] == f"/api/analysis/{aid}/image/before"
    err(client.get("/api/analysis/" + "0" * 32 + "/image/before"), 404, "analysis_not_found")


def test_context_labels_are_stored_cleaned_and_default_to_null(client):
    r = post_pair(client, PAIR_B, disaster_name="  Test\u0007  flood \n", district="D", state="")
    assert r.status_code == 201
    assert r.json()["context"] == {"disaster_name": "Test flood", "district": "D", "state": None}
    assert client.get(f"/api/analysis/{r.json()['analysis_id']}").json()["context"] == r.json()["context"]
    assert post_pair(client, PAIR_B).json()["context"] == {"disaster_name": None, "district": None, "state": None}
    err(post_pair(client, PAIR_B, disaster_name="x" * 101), 422, "validation_error")


def test_report_quotes_measured_siamese_limits_when_evaluation_exists(client, analysis_a):
    f = ROOT / "outputs" / "evaluation" / "siamese_damage_metrics.json"
    if not f.exists():
        pytest.skip("Siamese evaluation not run")
    m = json.loads(f.read_text(encoding="utf-8"))
    doc = client.get(f"/api/analysis/{analysis_a['analysis_id']}/report", params={"format": "json"}).json()
    text = " ".join(p for s in doc["sections"] for p in s["paragraphs"])
    from reasoning.report_facts import _collapse_siamese_metrics
    c = _collapse_siamese_metrics(m, f)
    assert f"macro F1 only {c['macro_f1']:.2f}" in text
    assert f"major-damage {c['recall_by_class']['major-damage']:.2f}" in text


# ---------------------------------------------------------------- units (no models needed)

def _settings():
    return Settings()


def _jpeg(w=128, h=128):
    return cv2.imencode(".jpg", np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8))[1].tobytes()


def test_validate_upload_accepts_good_images_and_sanitises_names():
    v = iv.validate_upload(_jpeg(), "..\\..\\a b/c<>.JPG", "before", _settings())
    assert (v.width, v.height, v.extension) == (128, 128, ".jpg")
    assert v.display_name == "c__.JPG" and len(v.sha256) == 64


def test_validate_upload_rejects_decompression_bomb_by_header():
    s = _settings(); s.max_pixels = 1000
    with pytest.raises(ApiError) as e:
        iv.validate_upload(_jpeg(128, 128), "x.jpg", "after", s)
    assert e.value.code == "image_too_large" and e.value.status == 413


def test_read_limited_stops_early():
    with pytest.raises(ApiError) as e:
        iv.read_limited(io.BytesIO(b"x" * 5000), 1000, "before")
    assert e.value.code == "file_too_large"


def test_store_blocks_traversal_and_never_reuses_ids(tmp_path):
    store = AnalysisStore(tmp_path / "s")
    (tmp_path / "secret").mkdir(); (tmp_path / "secret" / "meta.json").write_text("{}")
    for bad in ("../secret", "..", "../../secret", "/etc/passwd", "a" * 32, 5, None):
        with pytest.raises(ApiError):
            store.path(bad)
    ids = {store.new_dir()[0] for _ in range(50)}
    assert len(ids) == 50
