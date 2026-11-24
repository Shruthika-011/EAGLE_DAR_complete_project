# EAGLE-DAR performance report

Measured 2026-11-23T23:26:09Z by `python -m evaluation.benchmark_api` against the running backend. All figures below are read from `outputs/performance/benchmark.json`; nothing is estimated. They describe this one machine and the 1024 x 1024 PS-04 practice images, and say nothing about real-time performance on other hardware or image sizes.

## Machine

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (4096 MB) |
| Device used by the backend | cuda |
| Python / torch | 3.11.9 / 2.12.0+cu130 |
| OS | Windows-10-10.0.26300-SP0 |
| GPU memory in use, models loaded, idle | 266 MB |

## Analysis latency (POST /api/analyze, sequential, one request at a time)

| Image pair | Buildings found | Wall time incl. upload and rendering (s) | Model inference (s) |
|---|---|---|---|
| pb67aa0fe | 77 | 2.324 | 1.83 |
| pb83bb922 | 59 | 1.383 | 1.101 |
| pee6e1c7b | 29 | 0.994 | 0.73 |
| paf5c854c | 17 | 1.042 | 0.799 |
| pc771c488 | 14 | 0.94 | 0.632 |
| p4ca216d8 | 6 | 1.065 | 0.772 |

Median wall time **1.054 s**, worst 2.324 s; median model inference 0.786 s. The first request after a start is the slowest (CUDA warm-up). Time grows with the number of buildings (one Siamese pass per building).

## GPU memory

Peak GPU memory used during the sequential run: **714 MB** of 4096 MB, i.e. 448 MB above the idle level with models loaded (266 MB). During the concurrent test the peak was 752 MB, because inferences are serialised on the GPU. The 4 GB card is not a constraint at imgsz 640 and Siamese fp16 batches of 64; the CUDA cache is released after each request, and a CUDA out-of-memory error is reported as HTTP 503 instead of crashing the server.

## Concurrency

| Simultaneous requests | Completed (201) | Refused (503, queue full) | Total wall time (s) | Median / max latency of completed (s) | Throughput (analyses/min) |
|---|---|---|---|---|---|
| 8 | 5 | 3 | 9.3 | 4.17 / 6.62 | 32.3 |

one inference runs at a time; up to EAGLE_MAX_PENDING more may queue; beyond that the API answers 503 with Retry-After. The refused requests received `server_busy` with a `Retry-After` header and nothing was stored for them. Models are loaded once at startup and shared.

## Read endpoints (20 calls each; no inference is run)

| Endpoint | Median (ms) | p95 (ms) |
|---|---|---|
| GET analysis (with polygons) | 35.1 | 46.5 |
| GET damage map | 20.2 | 22.3 |
| GET comparison | 8.3 | 9.6 |
| GET export | 5.8 | 6.4 |
| GET health | 4.2 | 5.3 |
| POST question | 35.6 | 40.7 |

## PDF report generation

| Analysis | Status | Wall time (s) | Generation (s) | Pages | Size (MB) |
|---|---|---|---|---|---|
| pb67aa0fe | complete | 2.29 | 2.24 | 15 | 2.3 |
| pb83bb922 | complete | 2.23 | 2.1 | 13 | 2.33 |
| pee6e1c7b | complete | 1.74 | 1.7 | 12 | 1.92 |
| paf5c854c | complete | 1.71 | 1.67 | 11 | 1.53 |

Median wall time 1.98 s. Rendering runs in one background worker, separate from the GPU lock, and reports the real stage it is in.

## CPU fallback

| Image pair | Buildings | Seconds on CPU |
|---|---|---|
| pb67aa0fe | 77 | 6.84 |
| pb83bb922 | 59 | 4.62 |
| pee6e1c7b | 29 | 2.25 |

Median **4.62 s** on `cpu`. The same code path runs without a GPU (set `EAGLE_DEVICE=cpu` or run on a machine without CUDA).

## Not measured

* Images larger than 1024 x 1024, other GPUs, long-running memory growth, or more than 8 concurrent clients.
* Browser rendering performance.
