"""Structured building data. Standard library only."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

CLASS_NAMES = ("no-damage", "major-damage", "destroyed")
NUM_CLASSES = len(CLASS_NAMES)
# 'damaged' means major + destroyed (minor damage is merged into major-damage;
# there is no un-classified class).
DAMAGED_CLASS_IDS = frozenset({1, 2})
CANNOT_BE_DETERMINED = "cannot be determined"

BBox = tuple  # (x1, y1, x2, y2) integers, image coordinates


@dataclass(frozen=True)
class Building:
    building_id: int | str
    class_id: int
    centre: tuple          # (cx, cy) in image pixels; decides the spatial region
    bbox: BBox             # integer xyxy, floor/ceil of the polygon bounds, clipped to the image
    polygon: tuple = ()    # ((x, y), ...) image pixels
    confidence: float | None = None   # building-level confidence (Novelty 1)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def damage_name(self) -> str:
        return CLASS_NAMES[self.class_id]

    @property
    def is_damaged(self) -> bool:
        return self.class_id in DAMAGED_CLASS_IDS

    @property
    def evidence_region(self) -> BBox:
        """The image region that supports any answer this building contributes to."""
        return self.bbox

    def to_dict(self) -> dict:
        return {
            "building_id": self.building_id,
            "class_id": self.class_id,
            "damage": self.damage_name,
            "is_damaged": self.is_damaged,
            "confidence": self.confidence,
            "centre": list(self.centre),
            "bbox": list(self.bbox),
            "evidence_region": list(self.evidence_region),
            "polygon": [list(p) for p in self.polygon],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Scene:
    """One post-disaster image and the buildings found in it."""
    width: int
    height: int
    buildings: tuple
    scene_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (isinstance(self.width, int) and isinstance(self.height, int)
                and self.width > 0 and self.height > 0):
            raise ValueError(f"image size must be positive integers, got {self.width}x{self.height}")
        seen = set()
        for b in self.buildings:
            if b.building_id in seen:
                raise ValueError(f"duplicate building_id {b.building_id!r}")
            seen.add(b.building_id)
            if not (isinstance(b.class_id, int) and 0 <= b.class_id < NUM_CLASSES):
                raise ValueError(f"building {b.building_id!r}: class_id {b.class_id!r} not in 0..{NUM_CLASSES - 1}")
            cx, cy = b.centre
            if not (math.isfinite(cx) and math.isfinite(cy)
                    and 0 <= cx <= self.width and 0 <= cy <= self.height):
                raise ValueError(f"building {b.building_id!r}: centre {b.centre} is outside the "
                                 f"{self.width}x{self.height} image")
            x1, y1, x2, y2 = b.bbox
            if not (0 <= x1 <= x2 <= self.width and 0 <= y1 <= y2 <= self.height):
                raise ValueError(f"building {b.building_id!r}: bbox {b.bbox} invalid for the image")
            if b.confidence is not None and not (0.0 <= b.confidence <= 1.0):
                raise ValueError(f"building {b.building_id!r}: confidence {b.confidence} not in [0, 1]")

    def with_min_confidence(self, threshold: float) -> "Scene":
        """Drop buildings whose confidence is known and below threshold. None is kept."""
        if threshold <= 0:
            return self
        kept = tuple(b for b in self.buildings if b.confidence is None or b.confidence >= threshold)
        return Scene(self.width, self.height, kept, self.scene_id, self.metadata)
