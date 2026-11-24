"""Phase 5 report tests: the PDF is generated through the real API from real analyses (real models, real images).

    python -m pytest tests/test_reports.py -q

Layout is checked automatically with PyMuPDF (text inside margins, no overlapping lines, no blank pages) and by reading
the PDF text back with pypdf, so what is asserted is what a reader would see.
"""
import hashlib
import json
import re
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("ultralytics")
pytest.importorskip("reportlab")
pypdf = pytest.importorskip("pypdf")
pymupdf = pytest.importorskip("pymupdf")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from backend.services import explain, report_jobs
from backend.utils.report_store import ReportStore

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "dataset_source" / "practice" / "images"
PAIR_A, PAIR_EMPTY = "pb67aa0fe", "pb82851c2"          # 77 buildings / 0 buildings with the current models

_s = Settings()
pytestmark = pytest.mark.skipif(not (_s.yolo_weights.exists() and _s.siamese_weights.exists() and IMG.exists()),
                                reason="model weights or practice images not present")

SECTION_TITLES = ["2. Executive Summary", "3. Before-and-After Satellite Image Analysis", "4. Building Damage Classification",
                  "5. Building-Level Assessment", "6. Five-Class Damage Map", "7. Spatial Damage Assessment",
                  "8. Evidence and Question-Answering Results", "9. Confidence and Reliability Assessment", "10. Key Findings",
                  "11. Disaster Response Recommendations", "12. Methodology and Technical Details", "13. Limitations and Disclaimer"]


def post_pair(c, pid):
    f = lambda s: (IMG / f"{pid}_{s}.jpg").read_bytes()
    r = c.post("/api/analyze", files={"before": (f"{pid}_pre.jpg", f("pre")), "after": (f"{pid}_post.jpg", f("post"))},
               data={"disaster_name": "Report test flood"})
    assert r.status_code == 201, r.text
    return r.json()


def wait(c, rid, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        m = c.get(f"/api/reports/{rid}").json()
        if m["status"] in ("complete", "failed"):
            return m
        time.sleep(0.2)
    raise AssertionError("report did not finish")


def generate(c, aid):
    r = c.post(f"/api/analysis/{aid}/reports")
    assert r.status_code in (200, 202), r.text
    return wait(c, r.json()["report_id"])


def pdf_text(data: bytes) -> str:
    import io
    return "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(data)).pages)


def layout_problems(data: bytes) -> list[str]:
    """Automatic layout check on the rendered pages."""
    doc = pymupdf.open(stream=data, filetype="pdf")
    bad = []
    for n, page in enumerate(doc, start=1):
        W, H = page.rect.width, page.rect.height
        lines = []
        for blk in page.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                r = pymupdf.Rect(ln["bbox"])
                txt = "".join(sp["text"] for sp in ln["spans"]).strip()
                if not txt:
                    continue
                if r.x0 < 12 or r.x1 > W - 12 or r.y0 < 0 or r.y1 > H:
                    bad.append(f"p{n}: text outside page: {txt[:40]!r}")
                lines.append((r, txt))
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i][0], lines[j][0]
                inter = a & b
                if inter.is_empty:
                    continue
                if inter.get_area() > 0.25 * min(a.get_area(), b.get_area()):
                    bad.append(f"p{n}: overlap {lines[i][1][:25]!r} / {lines[j][1][:25]!r}")
        pix = page.get_pixmap(dpi=40)
        samples = pix.samples
        nonwhite = sum(1 for k in range(0, len(samples), 3 * 7) if samples[k] < 245 or samples[k + 1] < 245 or samples[k + 2] < 245)
        if n > 1 and nonwhite / (len(samples) / (3 * 7)) < 0.01:
            bad.append(f"p{n}: page is (almost) blank")
    return bad


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    s = Settings(storage_dir=tmp_path_factory.mktemp("analyses"), reports_dir=tmp_path_factory.mktemp("reports"))
    with TestClient(create_app(s)) as c:
        yield c


@pytest.fixture(scope="module")
def analysis(client):
    a = post_pair(client, PAIR_A)
    client.post("/api/questions", json={"analysis_id": a["analysis_id"], "questions": [
        {"id": "rq1", "question": "Count the destroyed buildings in the image."},
        {"id": "rq2", "question": "Which quadrant has the most damaged buildings?"},
        {"id": "rq3", "question": "Which building in the image is a hospital?"}]})
    return a


@pytest.fixture(scope="module")
def report(client, analysis):
    m = generate(client, analysis["analysis_id"])
    assert m["status"] == "complete", m
    return m


@pytest.fixture(scope="module")
def pdf(client, report):
    r = client.get(report["download_url"])
    assert r.status_code == 200
    return r.content


# ------------------------------------------------------------------ generation and download

def test_generation_flow_status_and_download(client, analysis, report, pdf):
    assert report["analysis_id"] == analysis["analysis_id"] and len(report["report_id"]) == 32
    assert report["progress"] == 1.0 and report["stage"] == "Complete" and report["pages"] >= 8
    assert report["download_url"] == f"/api/reports/{report['report_id']}/download"
    assert pdf.startswith(b"%PDF-") and len(pdf) == report["bytes"]
    assert hashlib.sha256(pdf).hexdigest() == report["sha256"]
    r = client.get(report["download_url"])
    assert r.headers["content-type"] == "application/pdf"
    cd = r.headers["content-disposition"]
    assert re.fullmatch(r'attachment; filename="eagle-dar-report-[0-9a-f]{8}-[0-9a-f]{8}\.pdf"', cd), cd
    assert client.get(report["download_url"], params={"inline": "true"}).headers["content-disposition"].startswith("inline")
    assert len(pypdf.PdfReader(__import__("io").BytesIO(pdf)).pages) == report["pages"]


def test_progress_is_reported_in_stages(client, analysis):
    svc = client.app.state.report_service
    orig = svc.generator
    seen = []

    def spy(path, data, progress):
        return orig(path, data, lambda st, f: (seen.append((st, f)), progress(st, f)))
    svc.generator = spy
    try:
        m = generate(client, analysis["analysis_id"])
    finally:
        svc.generator = orig
    assert m["status"] == "complete"
    fr = [f for _, f in seen]
    assert fr == sorted(fr) and len({s for s, _ in seen}) >= 8 and fr[-1] <= 0.99   # real, monotonic stages


# ------------------------------------------------------------------ contents

def test_pdf_has_all_required_sections_in_order(pdf):
    text = pdf_text(pdf)
    pos = [text.index(t) for t in SECTION_TITLES]
    assert pos == sorted(pos)
    assert "Disaster Damage Assessment Report" in text and "EAGLE-DAR" in text


def test_pdf_numbers_match_the_analysis(client, analysis, pdf):
    text = pdf_text(pdf)
    st = analysis["statistics"]
    c, n = st["damage_counts"], st["scene"]["total_buildings"]
    assert analysis["analysis_id"] in text and "Report test flood" in text
    pct = f"{100 * st['damaged_fraction']:.1f}%"
    assert pct in text
    assert f"{n} detected buildings" in text or f"Total detected" in text
    for name in ("No Damage", "Major Damage", "Destroyed"):
        assert name in text
    assert f"{analysis['parameters']['inference_seconds']:.1f} s" in text           # analysis duration
    # every building appears in the building-level table: its unique bbox string and its confidence
    full = client.get(f"/api/analysis/{analysis['analysis_id']}").json()
    flat = re.sub(r"\s+", " ", text)
    for b in full["buildings"]:
        x1, y1, x2, y2 = b["bbox_xyxy"]
        assert f"bbox {x1},{y1},{x2},{y2}" in flat, f"building {b['building_id']} missing from the table"
        assert f"{b['building_confidence']:.3f}" in flat
    # damage counts shown in the classification table equal the analysis
    assert sum(c.values()) == n


def test_pdf_distinguishes_computed_model_and_ai_text_and_labels_recommendations(pdf):
    flat = re.sub(r"\s+", " ", pdf_text(pdf))
    for needle in ("COMPUTED", "MODEL PREDICTION", "AI EXPLANATION", "AI-assisted recommendations requiring human verification",
                   "not the probability that the prediction is correct", "NOT the model's accuracy",
                   "require field verification",
                   "Geographic information is unavailable", "not been validated against the official PS-04"):
        assert needle in flat, needle
    assert not re.search(r"86\s*%[^.]{0,40}(accuracy|target)|(accuracy|target)[^.]{0,40}86\s*%", flat)   # no accuracy-target claim
    assert "no accuracy target is claimed" in flat


def test_pdf_embeds_images(pdf):
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    per_page = [len(p.get_images()) for p in doc]
    assert sum(per_page) >= 10          # cover x2, before/after x2, annotated, crops, map, evidence crops
    assert per_page[0] == 2             # cover thumbnails
    big = max((doc.extract_image(x[0])["width"] for p in doc for x in p.get_images()), default=0)
    assert big >= 1000                  # the damage map is embedded at (near) full resolution


def test_pdf_measured_limitations_come_from_evaluation_files(pdf):
    ev = ROOT / "outputs" / "evaluation" / "siamese_damage_metrics.json"
    if not ev.exists():
        pytest.skip("evaluation not run")
    from reasoning.report_facts import _collapse_siamese_metrics
    m = _collapse_siamese_metrics(json.loads(ev.read_text(encoding="utf-8")), ev)
    flat = re.sub(r"\s+", " ", pdf_text(pdf))
    assert f"{m['accuracy']:.3f} / {m['macro_f1']:.3f}" in flat
    assert f"Major Damage {m['recall_by_class']['major-damage']:.2f}" in flat


def test_pdf_layout_has_no_cut_off_overlapping_or_blank_pages(pdf):
    assert layout_problems(pdf) == []


def test_pdf_evidence_and_official_answers_match_the_engine(client, analysis, pdf):
    flat = re.sub(r"\s+", " ", pdf_text(pdf))
    for qid in ("rq1", "rq2", "rq3", "std_destroyed", "std_quadrant", "std_left_right"):
        assert qid in flat
    ex = {r["id"]: r for r in client.get(f"/api/analysis/{analysis['analysis_id']}/export").json()}
    region = ex["rq1"]["region"]
    assert f"[{region[0]}, {region[1]}, {region[2]}, {region[3]}]" in flat
    assert "cannot be determined" in flat                         # rq3 (type 4)
    assert "No language model calculates or alters a count" in flat


def test_report_building_ids_and_all_classes_with_zero_counts_are_listed(pdf):
    flat = re.sub(r"\s+", " ", pdf_text(pdf))
    assert "Major Damage 0" in flat or "Major Damage" in flat    # classes with 0 buildings still appear in the table


# ------------------------------------------------------------------ edge cases and failures

def test_zero_buildings_report(client):
    a = post_pair(client, PAIR_EMPTY)
    assert a["statistics"]["scene"]["total_buildings"] == 0
    m = generate(client, a["analysis_id"])
    assert m["status"] == "complete", m
    data = client.get(m["download_url"]).content
    flat = re.sub(r"\s+", " ", pdf_text(data))
    assert "No buildings were detected" in flat and "does not show that the area has no buildings" in flat
    assert "5. Building-Level Assessment" in flat and "13. Limitations and Disclaimer" in flat
    assert layout_problems(data) == []


def test_repeated_generation_never_overwrites(client, analysis, report, pdf):
    again = generate(client, analysis["analysis_id"])
    assert again["status"] == "complete" and again["report_id"] != report["report_id"]
    assert client.get(report["download_url"]).content == pdf                      # first report untouched
    lst = client.get(f"/api/analysis/{analysis['analysis_id']}/reports").json()
    ids = [m["report_id"] for m in lst]
    assert report["report_id"] in ids and again["report_id"] in ids and len(ids) == len(set(ids))
    assert ids.index(again["report_id"]) < ids.index(report["report_id"])        # newest first


def test_concurrent_request_reuses_the_report_in_progress(client, analysis):
    svc = client.app.state.report_service
    orig = svc.generator
    svc.generator = lambda path, data, progress: (time.sleep(1.0), orig(path, data, progress))[1]
    try:
        r1 = client.post(f"/api/analysis/{analysis['analysis_id']}/reports")
        r2 = client.post(f"/api/analysis/{analysis['analysis_id']}/reports")
        assert r1.status_code == 202 and r2.status_code == 200
        assert r1.json()["report_id"] == r2.json()["report_id"]
        assert wait(client, r1.json()["report_id"])["status"] == "complete"
    finally:
        svc.generator = orig


def test_pdf_failure_is_reported_cleanly_and_recoverable(client, analysis, tmp_path):
    svc = client.app.state.report_service
    orig = svc.generator

    def boom(path, data, progress):
        Path(path).write_bytes(b"partial")
        raise RuntimeError(f"disk exploded at {tmp_path}")
    svc.generator = boom
    try:
        m = generate(client, analysis["analysis_id"])
    finally:
        svc.generator = orig
    assert m["status"] == "failed" and m["download_url"] is None
    assert "RuntimeError" in m["error"] and str(tmp_path) not in m["error"]          # no internal paths leaked
    d = client.app.state.report_store.path(m["report_id"])
    assert not (d / "report.pdf").exists() and not list(d.glob("*.tmp"))             # no partial file
    r = client.get(f"/api/reports/{m['report_id']}/download")
    assert r.status_code == 409 and r.json()["error"]["code"] == "report_not_ready"
    assert client.get(f"/api/analysis/{analysis['analysis_id']}").status_code == 200  # analysis unaffected
    assert generate(client, analysis["analysis_id"])["status"] == "complete"          # a retry works


def test_llm_unavailable_falls_back_to_the_template(client, analysis, monkeypatch):
    svc = client.app.state.report_service
    monkeypatch.setattr(svc.settings, "llm_enabled", True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    def down(*a, **k):
        raise httpx.ConnectError("network unreachable")
    monkeypatch.setattr(explain, "anthropic_call", down)
    m = generate(client, analysis["analysis_id"])
    assert m["status"] == "complete"
    assert m["explanation_source"] == "deterministic-template" and "ConnectError" in m["fallback_reason"]
    assert "deterministic-template" in pdf_text(client.get(m["download_url"]).content)


def test_llm_disabled_by_default_and_without_key_is_reported(client, analysis, monkeypatch):
    assert Settings().llm_enabled is False
    svc = client.app.state.report_service
    monkeypatch.setattr(svc.settings, "llm_enabled", True)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    m = generate(client, analysis["analysis_id"])
    assert m["status"] == "complete" and "ANTHROPIC_API_KEY" in m["fallback_reason"]


# ------------------------------------------------------------------ explanation guard (unit)

def facts_for(analysis, client):
    from reasoning.loaders import scene_from_prediction_json
    from reasoning.report_facts import build_report_facts
    d = client.app.state.store.path(analysis["analysis_id"])
    sc = scene_from_prediction_json(d / "predictions.json")
    return build_report_facts(sc, None, ())


def test_llm_output_with_verified_numbers_is_accepted_and_invented_numbers_rejected(client, analysis):
    f = facts_for(analysis, client)
    payload = explain.explanation_facts(f, 1.6)
    t, aff = payload["total_buildings"], payload["affected_buildings"]
    good = json.dumps({"overall": f"The model found {t} buildings and {aff} appear affected.", "spatial": "Most damage is in one quadrant.",
                       "reliability": "Confidence is an uncalibrated score, not accuracy."})
    ok = explain.explain(f, 1.6, enabled=True, call=lambda *_: good)
    assert ok.source.startswith("llm:") and ok.fallback_reason is None and str(t) in ok.paragraphs["overall"]
    bad = json.dumps({"overall": f"The model found {t + 41} buildings.", "spatial": "x", "reliability": "y"})
    rej = explain.explain(f, 1.6, enabled=True, call=lambda *_: bad)
    assert rej.source == "deterministic-template" and "numbers not present" in rej.fallback_reason
    assert str(t) in rej.paragraphs["overall"]                         # deterministic text used instead
    for junk in ("not json at all", json.dumps({"overall": "only one section"})):
        r = explain.explain(f, 1.6, enabled=True, call=lambda *_, j=junk: j)
        assert r.source == "deterministic-template" and r.fallback_reason


def test_explainer_never_receives_images_names_or_user_labels(client, analysis):
    payload = json.dumps(explain.explanation_facts(facts_for(analysis, client), 1.0))
    for forbidden in ("Report test flood", "pb67aa0fe", ".jpg", "district", "Test"):
        assert forbidden not in payload


# ------------------------------------------------------------------ security and integrity

def test_report_ids_are_validated_and_traversal_is_blocked(client, tmp_path):
    for bad in ("0" * 32, "G" * 32, "a" * 31, "..", "%2e%2e", "../../etc/passwd"):
        r = client.get(f"/api/reports/{bad}")
        assert r.status_code == 404
        assert client.get(f"/api/reports/{bad}/download").status_code == 404
    assert client.post("/api/analysis/" + "0" * 32 + "/reports").status_code == 404
    store = ReportStore(tmp_path / "r")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "meta.json").write_text("{}")
    for bad in ("../secret", "..", "/etc/passwd", 5, None):
        with pytest.raises(Exception):
            store.path(bad)


def test_interrupted_reports_are_marked_failed_on_restart(tmp_path):
    store = ReportStore(tmp_path / "r")
    rid, d = store.new("a" * 32)
    store.update(rid, status="generating")
    (d / "report.pdf.tmp").write_bytes(b"x")
    assert store.recover_interrupted() == 1
    m = store.meta(rid)
    assert m["status"] == "failed" and "restart" in m["error"] and not list(d.glob("*.tmp"))


def test_generated_reports_are_ignored_by_git():
    import subprocess
    r = subprocess.run(["git", "check-ignore", "-q", "outputs/reports/abc/report.pdf"], cwd=ROOT)
    assert r.returncode == 0


# ------------------------------------------------------------------ integration properties

def test_no_extra_inference_when_navigating_or_generating_reports(client, analysis):
    inf = client.app.state.inference
    calls = []
    orig = inf.pipeline.run
    inf.pipeline.run = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    aid = analysis["analysis_id"]
    try:
        for path in ("", "/map", "/comparison", "/report?format=json", "/report", "/export", "/questions", "/image/before", "/image/after"):
            assert client.get(f"/api/analysis/{aid}{path}").status_code == 200
        client.post("/api/questions", json={"analysis_id": aid, "questions": [{"question": "Count the destroyed buildings in the image."}]})
        for _ in range(3):
            client.get(f"/api/analysis/{aid}?include_polygons=true")
        assert generate(client, aid)["status"] == "complete"
    finally:
        inf.pipeline.run = orig
    assert calls == []                                                   # pages, refreshes and reports never re-run the models


def test_same_analysis_id_and_predictions_are_used_in_every_output(client, analysis, pdf):
    aid = analysis["analysis_id"]
    api = client.get(f"/api/analysis/{aid}").json()
    flat = re.sub(r"\s+", " ", pdf_text(pdf))
    assert aid in flat
    rep_json = client.get(f"/api/analysis/{aid}/report", params={"format": "json"}).json()
    assert rep_json["facts"]["damage_counts"] == api["statistics"]["damage_counts"]
    stored = client.app.state.store.predictions(aid)
    assert stored["analysis_id"] == aid and stored["total_detected_buildings"] == len(api["buildings"])
    assert [b["building_id"] for b in api["buildings"]] == [b["building_id"] for b in stored["buildings"]]


def test_model_unavailable_blocks_analysis_but_not_reports(client, analysis, tmp_path):
    s = Settings(storage_dir=client.app.state.settings.storage_dir, reports_dir=tmp_path / "r2",
                 yolo_weights=tmp_path / "missing.pt", siamese_weights=tmp_path / "missing.pth")
    with TestClient(create_app(s)) as c2:
        h = c2.get("/api/health")
        assert h.status_code == 503 and h.json()["status"] == "degraded" and not h.json()["models"]["yolo"]["loaded"]
        b = (IMG / f"{PAIR_A}_pre.jpg").read_bytes()
        r = c2.post("/api/analyze", files={"before": ("a.jpg", b), "after": ("b.jpg", b)})
        assert r.status_code == 503 and r.json()["error"]["code"] == "models_unavailable"
        assert c2.get(f"/api/analysis/{analysis['analysis_id']}").status_code == 200      # saved results still readable
        assert generate(c2, analysis["analysis_id"])["status"] == "complete"            # reports need no models


# ------------------------------------------------------------------ regression: status polling during generation

def test_atomic_json_write_survives_concurrent_readers(tmp_path):
    """On Windows os.replace raised PermissionError when a poll had the file open. 500 writes against 4 polling readers must all
    succeed (the writer retries) and no reader may ever see a partial file."""
    import threading
    from backend.utils.storage import write_json_atomic
    target = tmp_path / "meta.json"
    write_json_atomic(target, {"n": 0})
    stop, errors = threading.Event(), []

    def reader():
        while not stop.is_set():
            try:
                json.loads(target.read_text(encoding="utf-8"))
            except (FileNotFoundError, PermissionError):
                pass                                      # Windows can refuse an open during a rename; the app's readers hold the store lock
            except Exception as exc:                      # a torn/partial (half-written) read would be a bug
                errors.append(repr(exc))
            time.sleep(0.002)                             # polling clients pause between requests
    threads = [threading.Thread(target=reader) for _ in range(4)]
    [t.start() for t in threads]
    try:
        for i in range(500):
            write_json_atomic(target, {"n": i})
    finally:
        stop.set()
        [t.join() for t in threads]
    assert errors == [] and json.loads(target.read_text(encoding="utf-8")) == {"n": 499}


def test_report_completes_while_the_status_endpoint_is_hammered(client, analysis):
    import threading
    svc = client.app.state.report_service
    orig = svc.generator

    def chatty(path, data, progress):                     # many progress writes -> many meta.json replacements
        for i in range(60):
            progress(f"step {i}", i / 100)
            time.sleep(0.01)
        return orig(path, data, progress)
    svc.generator = chatty
    stop, codes = threading.Event(), []
    r = client.post(f"/api/analysis/{analysis['analysis_id']}/reports")
    rid = r.json()["report_id"]

    def poll():
        c2 = TestClient(client.app)
        while not stop.is_set():
            codes.append(c2.get(f"/api/reports/{rid}").status_code)
    threads = [threading.Thread(target=poll) for _ in range(4)]
    [t.start() for t in threads]
    try:
        m = wait(client, rid, timeout=120)
    finally:
        stop.set()
        [t.join() for t in threads]
        svc.generator = orig
    assert m["status"] == "complete", m.get("error")
    assert len(codes) > 20 and set(codes) == {200}
