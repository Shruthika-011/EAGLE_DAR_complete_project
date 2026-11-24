"""Render docs/PERFORMANCE_REPORT.md from outputs/performance/benchmark.json (produced by evaluation.benchmark_api)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    d = json.loads((ROOT / "outputs/performance/benchmark.json").read_text(encoding="utf-8"))
    m, seq, con, rg, cpu = d["machine"], d["sequential"], d["concurrent"], d["report_generation"], d["cpu_fallback"]
    L = ["# EAGLE-DAR performance report", "",
         f"Measured {d['measured_utc']} by `python -m evaluation.benchmark_api` against the running backend. All figures below are read from "
         "`outputs/performance/benchmark.json`; nothing is estimated. They describe this one machine and the 1024 x 1024 PS-04 practice images, "
         "and say nothing about real-time performance on other hardware or image sizes.", "",
         "## Machine", "",
         table(["Item", "Value"], [["GPU", f"{m['gpu']} ({m['gpu_total_mb']} MB)"], ["Device used by the backend", m["device_in_use"]],
                                   ["Python / torch", f"{m['python']} / {m.get('torch', 'n/a')}"], ["OS", m["os"]],
                                   ["GPU memory in use, models loaded, idle", f"{m['gpu_used_mb_idle_with_models_loaded']} MB"]]), "",
         "## Analysis latency (POST /api/analyze, sequential, one request at a time)", "",
         table(["Image pair", "Buildings found", "Wall time incl. upload and rendering (s)", "Model inference (s)"],
               [[r["pair"], r["buildings"], r["wall_s"], r["inference_s"]] for r in seq["runs"]]), "",
         f"Median wall time **{seq['wall_median_s']} s**, worst {seq['wall_max_s']} s; median model inference {seq['inference_median_s']} s. "
         "The first request after a start is the slowest (CUDA warm-up). Time grows with the number of buildings (one Siamese pass per building).", "",
         "## GPU memory", "",
         f"Peak GPU memory used during the sequential run: **{seq['gpu_peak_used_mb']} MB** of {m['gpu_total_mb']} MB, i.e. {seq['gpu_peak_over_idle_mb']} MB above the idle level with models loaded "
         f"({m['gpu_used_mb_idle_with_models_loaded']} MB). During the concurrent test the peak was {con['gpu_peak_used_mb']} MB, because inferences are serialised on the GPU. "
         "The 4 GB card is not a constraint at imgsz 640 and Siamese fp16 batches of 64; the CUDA cache is released after each request, and a CUDA out-of-memory error is reported as HTTP 503 instead of crashing the server.", "",
         "## Concurrency", "",
         table(["Simultaneous requests", "Completed (201)", "Refused (503, queue full)", "Total wall time (s)", "Median / max latency of completed (s)", "Throughput (analyses/min)"],
               [[con["requests"], con["status_codes"].get("201", 0), con["status_codes"].get("503", 0), con["total_wall_s"], f"{con['latency_median_s']} / {con['latency_max_s']}", con["throughput_analyses_per_min"]]]), "",
         f"{con['note']}. The refused requests received `server_busy` with a `Retry-After` header and nothing was stored for them. Models are loaded once at startup and shared.", "",
         "## Read endpoints (20 calls each; no inference is run)", "",
         table(["Endpoint", "Median (ms)", "p95 (ms)"], [[k, v["median_ms"], v["p95_ms"]] for k, v in d["read_latency"].items()]), "",
         "## PDF report generation", "",
         table(["Analysis", "Status", "Wall time (s)", "Generation (s)", "Pages", "Size (MB)"], [[r["pair"], r["status"], r["wall_s"], r["generation_s"], r["pages"], r["mb"]] for r in rg["runs"]]), "",
         f"Median wall time {rg['wall_median_s']} s. Rendering runs in one background worker, separate from the GPU lock, and reports the real stage it is in.", "",
         "## CPU fallback", ""]
    if "error" in cpu:
        L += [f"Not measured: {cpu['error']}"]
    else:
        L += [table(["Image pair", "Buildings", "Seconds on CPU"], [[r["pair"], r["buildings"], r["seconds"]] for r in cpu["runs"]]), "",
              f"Median **{cpu['median_s']} s** on `{cpu['device']}`. The same code path runs without a GPU (set `EAGLE_DEVICE=cpu` or run on a machine without CUDA)."]
    L += ["", "## Not measured", "", "* Images larger than 1024 x 1024, other GPUs, long-running memory growth, or more than 8 concurrent clients.", "* Browser rendering performance."]
    (ROOT / "docs/PERFORMANCE_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Wrote docs/PERFORMANCE_REPORT.md")


if __name__ == "__main__":
    main()
