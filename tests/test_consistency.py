"""Cross-component consistency: the damage-class definitions must be identical everywhere (Python and TypeScript)."""
import re
from pathlib import Path

import config
from reasoning.damage_map import CLASS_COLORS_BGR
from reasoning.questions import LEVELS
from reasoning.schema import CLASS_NAMES, DAMAGED_CLASS_IDS

ROOT = Path(__file__).resolve().parent.parent
TS = (ROOT / "frontend" / "src" / "lib" / "damage.ts").read_text(encoding="utf-8")


def test_python_definitions_agree():
    assert tuple(config.APP_DAMAGE_NAMES[i] for i in range(3)) == CLASS_NAMES
    assert config.APP_DAMAGE_CLASSES == {n: i for i, n in enumerate(CLASS_NAMES)}
    # model-level classes are unchanged; minor merges into major, un-classified has no app class
    assert config.DAMAGE_CLASSES["minor-damage"] == 1 and config.DAMAGE_CLASSES["un-classified"] == 4
    assert {config.DAMAGE_NAMES[s]: CLASS_NAMES[d] for s, d in config.MODEL_TO_APP_CLASS.items()} == {
        "no-damage": "no-damage", "minor-damage": "major-damage", "major-damage": "major-damage", "destroyed": "destroyed"}
    assert dict(config.DAMAGE_COLORS_BGR) == dict(CLASS_COLORS_BGR)
    assert set(config.DAMAGED_CLASSES) == {CLASS_NAMES[i] for i in DAMAGED_CLASS_IDS} == {"major-damage", "destroyed"}
    assert LEVELS["damaged"] == DAMAGED_CLASS_IDS


def test_frontend_order_labels_and_colours_match_the_backend():
    order = re.search(r"DAMAGE_ORDER[^=]*=\s*\[([^\]]+)\]", TS).group(1)
    assert tuple(re.findall(r"'([^']+)'", order)) == CLASS_NAMES
    for name, (b, g, r) in CLASS_COLORS_BGR.items():
        m = re.search(rf"'?{re.escape(name)}'?:\s*\{{[^}}]*color:\s*'(#[0-9a-fA-F]{{6}})'", TS)
        assert m, name
        assert m.group(1).lower() == f"#{r:02x}{g:02x}{b:02x}", name
    affected = re.search(r"AFFECTED_CLASSES[^=]*=\s*\[([^\]]+)\]", TS).group(1)
    assert set(re.findall(r"'([^']+)'", affected)) == {"major-damage", "destroyed"}


def test_class_colour_names():
    r = lambda n: CLASS_COLORS_BGR[n][::-1]
    assert r("no-damage")[1] > r("no-damage")[0]                                       # green
    assert r("major-damage")[0] > 200 and 100 < r("major-damage")[1] < 200              # orange
    assert r("destroyed")[0] > 200 and r("destroyed")[1] < 100                          # red
