"""Pre/post image alignment checks.

Training crops were cut from pre and post images at identical pixel
coordinates. That only shows the same building if the two images are
co-registered. A survey of the 440 train+practice pairs (phase correlation,
response >= 0.05, n=313) gave a median global offset of 4.3 px, with 23%
above the 8 px crop padding and 4% above 16 px. Alignment is therefore
measured and reported rather than assumed.

Convention: cv2.phaseCorrelate(pre, post) returns (dx, dy) such that the
post image equals the pre image translated by (+dx, +dy).
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np

PC_MIN_RESPONSE = 0.05      # below this the phase-correlation peak is unreliable
ORB_MIN_INLIERS = 20
ALIGNED_PX = 2.0
DISAGREE_PX = 4.0


@dataclass
class GlobalAlignment:
    status: str                 # aligned | minor_offset | misaligned | unverified
    dx: float
    dy: float
    shift_px: float
    phase_corr_response: float
    orb_inliers: int
    orb_dx: float | None
    orb_dy: float | None
    orb_rotation_deg: float | None
    orb_scale: float | None
    method: str
    registered: bool = False
    note: str = ""

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _phase_corr(a, b):
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy), float(resp)


def _orb_affine(pre_gray_u8, post_gray_u8):
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(pre_gray_u8, None)
    k2, d2 = orb.detectAndCompute(post_gray_u8, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None
    src = np.float32([k1[g.queryIdx].pt for g in good])
    dst = np.float32([k2[g.trainIdx].pt for g in good])
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None
    return {
        "inliers": int(inliers.sum()),
        "dx": float(M[0, 2]),
        "dy": float(M[1, 2]),
        "rotation_deg": float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
        "scale": float(np.hypot(M[0, 0], M[1, 0])),
    }


def estimate_global_alignment(pre, post, tolerance_px):
    """Estimate translation of post relative to pre and classify it."""
    g1, g2 = _gray(pre), _gray(post)
    dx, dy, resp = _phase_corr(g1, g2)
    orb = _orb_affine(g1.astype(np.uint8), g2.astype(np.uint8))

    pc_ok = resp >= PC_MIN_RESPONSE
    orb_ok = (orb is not None and orb["inliers"] >= ORB_MIN_INLIERS
              and abs(orb["rotation_deg"]) < 3 and abs(orb["scale"] - 1) < 0.03)

    note = ""
    if pc_ok:
        method = "phase_correlation"
        if orb_ok and np.hypot(dx - orb["dx"], dy - orb["dy"]) > DISAGREE_PX:
            note = "phase correlation and ORB disagree; estimate is uncertain"
    elif orb_ok:
        method = "orb_affine"
        dx, dy = orb["dx"], orb["dy"]
    else:
        method = "none"

    shift = float(np.hypot(dx, dy))
    if method == "none" or note:
        status = "unverified"
        if not note:
            note = ("no reliable alignment estimate (low texture or large scene change); "
                    "pre/post crops may not show the same building")
    elif shift <= ALIGNED_PX:
        status = "aligned"
    elif shift <= tolerance_px:
        status = "minor_offset"
    else:
        status = "misaligned"
        note = (f"global offset {shift:.1f}px exceeds crop padding {tolerance_px}px; "
                "consider --register")

    if orb_ok and abs(orb["rotation_deg"]) > 0.5:
        note = (note + "; " if note else "") + f"rotation of {orb['rotation_deg']:.2f} deg detected (not corrected)"

    return GlobalAlignment(
        status=status, dx=dx, dy=dy, shift_px=shift, phase_corr_response=resp,
        orb_inliers=0 if orb is None else orb["inliers"],
        orb_dx=None if orb is None else orb["dx"], orb_dy=None if orb is None else orb["dy"],
        orb_rotation_deg=None if orb is None else orb["rotation_deg"],
        orb_scale=None if orb is None else orb["scale"],
        method=method, note=note,
    )


def register_pre_to_post(pre, dx, dy):
    """Translate the pre image by (+dx, +dy) so it overlays the post image."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(pre, M, (pre.shape[1], pre.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def local_shift(pre, post, bbox, margin=24, min_size=48):
    """Phase-correlate a window around one building. Returns (dx, dy, response).

    Low response is expected for heavily damaged buildings (the content really
    changed), so a low response alone is not evidence of misalignment.
    """
    h, w = post.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max((x2 - x1) / 2 + margin, min_size / 2)
    half_h = max((y2 - y1) / 2 + margin, min_size / 2)
    a1, b1 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
    a2, b2 = int(min(w, cx + half_w)), int(min(h, cy + half_h))
    if a2 - a1 < 16 or b2 - b1 < 16:
        return None
    win_pre = _gray(pre[b1:b2, a1:a2])
    win_post = _gray(post[b1:b2, a1:a2])
    if win_pre.std() < 1 or win_post.std() < 1:
        return None
    return _phase_corr(win_pre, win_post)
