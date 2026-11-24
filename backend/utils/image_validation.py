"""Upload validation. Nothing here trusts the client-supplied filename or content type.

Checks, in order: size, extension, magic bytes, header-only dimension read
(so a decompression bomb is refused before it is decoded), full decode
(catches truncated/corrupt files), dimension limits.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.config import Settings
from backend.errors import ApiError

_SIGNATURES = {"jpeg": (b"\xff\xd8\xff",), "png": (b"\x89PNG\r\n\x1a\n",)}
_EXT_FOR_FORMAT = {"jpeg": ".jpg", "png": ".png"}
_FORMAT_FOR_EXT = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png"}


@dataclass(frozen=True)
class ValidatedImage:
    role: str
    data: bytes
    extension: str          # canonical, derived from the content (".jpg" / ".png")
    width: int
    height: int
    sha256: str
    display_name: str       # sanitised original name, for display only; never used as a path


def safe_display_name(name: str | None) -> str:
    """Strip directories and unsafe characters. For metadata only."""
    base = re.split(r"[\\/]", name or "")[-1]
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return (base or "upload")[:100]


def _bad(role: str, code: str, message: str, status: int = 422, **details) -> ApiError:
    return ApiError(status, code, f"{role} image: {message}", {"role": role, **details} if details else {"role": role})


def read_limited(fileobj, limit: int, role: str) -> bytes:
    """Read an upload in chunks and stop as soon as it exceeds the limit."""
    chunks, total = [], 0
    while True:
        chunk = fileobj.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise _bad(role, "file_too_large", f"exceeds the {limit // (1024 * 1024)} MB limit.", 413,
                       limit_bytes=limit)
        chunks.append(chunk)
    return b"".join(chunks)


def validate_upload(data: bytes, filename: str | None, role: str, settings: Settings) -> ValidatedImage:
    if not data:
        raise _bad(role, "empty_file", "the uploaded file is empty.")

    name = safe_display_name(filename)
    ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    if ext not in settings.allowed_extensions:
        raise _bad(role, "unsupported_file_type",
                   f"extension '{ext or '(none)'}' is not allowed. Use JPG, JPEG or PNG.", 415,
                   allowed=list(settings.allowed_extensions))

    fmt = next((f for f, sigs in _SIGNATURES.items() if any(data.startswith(s) for s in sigs)), None)
    if fmt is None:
        raise _bad(role, "unsupported_file_type", "content is not a JPEG or PNG image.", 415)
    if _FORMAT_FOR_EXT[ext] != fmt:
        raise _bad(role, "extension_mismatch", f"file extension '{ext}' does not match its {fmt.upper()} content.", 415)

    # Header-only read first: refuses decompression bombs before anything is decoded.
    try:
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise _bad(role, "corrupted_image", f"cannot be read ({type(exc).__name__}).") from exc
    if width * height > settings.max_pixels:
        raise _bad(role, "image_too_large", f"{width}x{height} exceeds the {settings.max_pixels:,} pixel limit.", 413,
                   width=width, height=height)
    if min(width, height) < settings.min_side_px or max(width, height) > settings.max_side_px:
        raise _bad(role, "invalid_dimensions",
                   f"{width}x{height} is outside the allowed {settings.min_side_px}..{settings.max_side_px}px side range.",
                   width=width, height=height)

    # Full decode: catches truncated or corrupt pixel data that the header read accepts.
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape[:2] != (height, width):
        raise _bad(role, "corrupted_image", "pixel data could not be decoded (file truncated or corrupt).")

    return ValidatedImage(role, data, _EXT_FOR_FORMAT[fmt], width, height, hashlib.sha256(data).hexdigest(), name)


def validate_pair(before: ValidatedImage, after: ValidatedImage) -> None:
    if (before.width, before.height) != (after.width, after.height):
        raise ApiError(422, "dimension_mismatch",
                       f"before ({before.width}x{before.height}) and after ({after.width}x{after.height}) images "
                       "must have identical dimensions; register or resize them first.",
                       {"before": [before.width, before.height], "after": [after.width, after.height]})
