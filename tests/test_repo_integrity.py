"""Security and data-integrity checks on the repository and its artefacts."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode not in (0, 1):
        pytest.skip(f"git unavailable: {r.stderr[:80]}")
    return r


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_trained_model_weights_are_unchanged_since_the_verified_backup():
    manifests = sorted((ROOT / "models" / "backup").glob("*/MANIFEST.json"))
    if not manifests:
        pytest.skip("no backup manifest")
    sources = {"siamese_damage.pth": ROOT / "models" / "siamese_damage.pth", "siamese_damage_last.pth": ROOT / "models" / "siamese_damage_last.pth",
               "yolo_best.pt": ROOT / "runs/eagle_buildings-2/weights/best.pt", "yolo_last.pt": ROOT / "runs/eagle_buildings-2/weights/last.pt"}
    for entry in json.loads(manifests[0].read_text(encoding="utf-8-sig")):
        assert sha(sources[entry["file"]]) == entry["sha256"].lower(), f"{entry['file']} changed since the backup"   # PowerShell writes upper-case hex


def test_no_weights_secrets_reports_or_env_files_are_tracked_by_git():
    tracked = git("ls-files").stdout.splitlines()
    bad = [f for f in tracked if re.search(r"\.(pt|pth|onnx|pdf)$", f) or re.search(r"(^|/)\.env(\..*)?$", f) or "outputs/reports/" in f or "models/backup" in f]
    assert bad == []


@pytest.mark.parametrize("path", ["outputs/reports/x/report.pdf", "outputs/analyses/x/before.jpg", "outputs/demo/a.png", "models/siamese_damage.pth",
                                  "models/backup/x/yolo_best.pt", "runs/x/weights/best.pt", "processed/siamese/crops/a.png", "dataset/raw/train/images/a.jpg",
                                  ".env", ".env.local", ".venv/x", "frontend/node_modules/x", "frontend/dist/index.html", "frontend/.env"])
def test_generated_and_sensitive_paths_are_git_ignored(path):
    assert git("check-ignore", "-q", path).returncode == 0, f"{path} is not ignored"


def test_tracked_dataset_files_are_unmodified():
    r = git("status", "--porcelain", "--", "dataset_source")
    assert r.stdout.strip() == "", "dataset_source has local modifications:\n" + r.stdout[:300]


def test_source_has_no_hard_coded_secrets():
    pat = re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{32,}|AKIA[0-9A-Z]{16}|(api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]", re.I)
    hits = []
    for base in ("backend", "reasoning", "inference", "evaluation", "training", "frontend/src", "frontend/e2e", "scripts"):
        for p in (ROOT / base).rglob("*"):
            if p.is_file() and p.suffix in {".py", ".ts", ".tsx", ".mjs", ".ps1", ".json"} and "node_modules" not in p.parts:
                for m in pat.finditer(p.read_text(encoding="utf-8", errors="ignore")):
                    hits.append(f"{p.relative_to(ROOT)}: {m.group(0)[:20]}")
    assert hits == []


def test_api_key_is_only_read_from_the_environment_and_never_returned():
    src = (ROOT / "backend/services/explain.py").read_text(encoding="utf-8")
    assert 'os.environ.get("ANTHROPIC_API_KEY")' in src
    health = (ROOT / "backend/routes/health.py").read_text(encoding="utf-8")
    assert 'bool(os.environ.get("ANTHROPIC_API_KEY"))' in health and 'os.environ["ANTHROPIC_API_KEY"]' not in health


def test_error_responses_do_not_leak_server_paths():
    for f in (ROOT / "backend").rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert "traceback.format_exc" not in text and "str(exc.__traceback__" not in text, f
