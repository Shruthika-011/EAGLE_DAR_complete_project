"""Measure the running platform. Every number in docs/PERFORMANCE_REPORT.md comes from this script's output.

Needs the backend running (GPU) at --api; the CPU-fallback timing loads its own CPU copy of the models in this process.

    python -m evaluation.benchmark_api --api http://127.0.0.1:8000
"""
import argparse
import json
import platform
import statistics as st
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "dataset_source" / "practice" / "images"
PAIRS = ["pb67aa0fe", "pb83bb922", "pee6e1c7b", "paf5c854c", "pc771c488", "p4ca216d8"]


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,name", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        return int(out[0]), int(out[1]), out[2].strip()
    except Exception:
        return None, None, None


class GpuSampler:
    def __init__(self):
        self.samples, self._stop = [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            used, _, _ = gpu_used_mb()
            if used is not None:
                self.samples.append(used)
            time.sleep(0.15)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def post_pair(c, pid):
    f = lambda s: (IMG / f"{pid}_{s}.jpg").read_bytes()
    t = time.perf_counter()
    r = c.post("/api/analyze", files={"before": (f"{pid}_pre.jpg", f("pre")), "after": (f"{pid}_post.jpg", f("post"))})
    return r, time.perf_counter() - t


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "performance" / "benchmark.json"))
    args = ap.parse_args()
    c = httpx.Client(base_url=args.api, timeout=600)
    res = {"measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "machine": {"python": platform.python_version(), "os": platform.platform()}}

    h = c.get("/api/health").json()
    used0, total_mb, gpu_name = gpu_used_mb()
    res["machine"].update({"gpu": gpu_name, "gpu_total_mb": total_mb, "device_in_use": h["device"]["in_use"], "gpu_used_mb_idle_with_models_loaded": used0})
    try:
        import torch
        res["machine"]["torch"] = torch.__version__
    except Exception:
        pass

    # ---- sequential analyses -----------------------------------------------------------------------------------------
    print("sequential analyses...", flush=True)
    seq, ids = [], {}
    with GpuSampler() as g:
        for pid in PAIRS:
            r, wall = post_pair(c, pid)
            assert r.status_code == 201, r.text
            j = r.json()
            ids[pid] = j["analysis_id"]
            seq.append({"pair": pid, "buildings": j["statistics"]["scene"]["total_buildings"], "wall_s": round(wall, 3),
                        "inference_s": j["parameters"]["inference_seconds"]})
    res["sequential"] = {"runs": seq, "wall_median_s": round(st.median(r["wall_s"] for r in seq), 3), "wall_max_s": max(r["wall_s"] for r in seq),
                         "inference_median_s": round(st.median(r["inference_s"] for r in seq), 3),
                         "gpu_peak_used_mb": max(g.samples) if g.samples else None,
                         "gpu_peak_over_idle_mb": (max(g.samples) - used0) if g.samples and used0 is not None else None}

    # ---- concurrent analyses -----------------------------------------------------------------------------------------
    print("concurrent analyses...", flush=True)
    n = 8
    t0 = time.perf_counter()
    with GpuSampler() as g, ThreadPoolExecutor(max_workers=n) as ex:
        outs = list(ex.map(lambda i: post_pair(httpx.Client(base_url=args.api, timeout=600), PAIRS[i % len(PAIRS)]), range(n)))
    total = time.perf_counter() - t0
    codes = [r.status_code for r, _ in outs]
    ok = [w for (r, w) in outs if r.status_code == 201]
    res["concurrent"] = {"requests": n, "status_codes": {str(k): codes.count(k) for k in sorted(set(codes))}, "total_wall_s": round(total, 2),
                         "completed": len(ok), "latency_median_s": round(st.median(ok), 2) if ok else None, "latency_max_s": round(max(ok), 2) if ok else None,
                         "throughput_analyses_per_min": round(60 * len(ok) / total, 1),
                         "gpu_peak_used_mb": max(g.samples) if g.samples else None,
                         "note": "one inference runs at a time; up to EAGLE_MAX_PENDING more may queue; beyond that the API answers 503 with Retry-After"}

    # ---- read endpoints ---------------------------------------------------------------------------------------------
    aid = ids[PAIRS[0]]
    lat = {}
    for name, path in (("GET analysis (with polygons)", f"/api/analysis/{aid}?include_polygons=true"), ("GET damage map", f"/api/analysis/{aid}/map"),
                       ("GET comparison", f"/api/analysis/{aid}/comparison"), ("GET export", f"/api/analysis/{aid}/export"), ("GET health", "/api/health")):
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            assert c.get(path).status_code == 200
            ts.append((time.perf_counter() - t) * 1000)
        lat[name] = {"median_ms": round(st.median(ts), 1), "p95_ms": round(pctl(ts, 95), 1)}
    c.post("/api/questions", json={"analysis_id": aid, "questions": [{"question": "Count the destroyed buildings in the image."}]})
    ts = []
    for _ in range(20):
        t = time.perf_counter()
        assert c.post("/api/questions", json={"analysis_id": aid, "questions": [{"question": "Which quadrant has the most damaged buildings?"}]}).status_code == 200
        ts.append((time.perf_counter() - t) * 1000)
    lat["POST question"] = {"median_ms": round(st.median(ts), 1), "p95_ms": round(pctl(ts, 95), 1)}
    res["read_latency"] = lat

    # ---- PDF generation ---------------------------------------------------------------------------------------------
    print("report generation...", flush=True)
    rep = []
    for pid in PAIRS[:4]:
        t = time.perf_counter()
        m = c.post(f"/api/analysis/{ids[pid]}/reports").json()
        while m["status"] in ("queued", "generating"):
            time.sleep(0.1)
            m = c.get(f"/api/reports/{m['report_id']}").json()
        rep.append({"pair": pid, "status": m["status"], "wall_s": round(time.perf_counter() - t, 2), "generation_s": m["generation_seconds"], "pages": m["pages"], "mb": round((m["bytes"] or 0) / 1048576, 2)})
    res["report_generation"] = {"runs": rep, "wall_median_s": round(st.median(r["wall_s"] for r in rep), 2)}

    # ---- CPU fallback (separate process copy of the models) ----------------------------------------------------------
    print("CPU fallback...", flush=True)
    try:
        from backend.config import Settings
        from inference.pipeline import DamagePipeline, PipelineConfig
        s = Settings()
        pipe = DamagePipeline(PipelineConfig(yolo_weights=s.yolo_weights, siamese_weights=s.siamese_weights, device="cpu"))
        cpu = []
        for pid in PAIRS[:3]:
            t = time.perf_counter()
            rep_, _ = pipe.run(IMG / f"{pid}_pre.jpg", IMG / f"{pid}_post.jpg")
            cpu.append({"pair": pid, "buildings": rep_["total_detected_buildings"], "seconds": round(time.perf_counter() - t, 2)})
        res["cpu_fallback"] = {"device": str(pipe.device), "runs": cpu, "median_s": round(st.median(r["seconds"] for r in cpu), 2)}
    except Exception as exc:
        res["cpu_fallback"] = {"error": f"{type(exc).__name__}: {exc}"}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
