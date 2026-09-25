"""Per-session ROI placement by rigid alignment to a reference session (`--<camera>-roi aligned`).

The default ROIs are defined in the reference session's pixel frame. Cameras move a little
between sessions and rigs (typically 0-4 deg and up to ~100 px on Face/Nose), so a fixed box
drifts off the body part it was drawn on. With `--<camera>-roi aligned` each video is aligned
to the bundled reference template (`alignment_reference/<Camera>.png`, see
`video_alignment.py`) and the default ROI's centre is carried through that transform.

The ROI stays axis-aligned with the default's size (facemap needs an axis-aligned crop), so
the rotation is not followed; `max_corner_error_px` records how far the shifted box's
corners sit from the exact rotated box. At the rotations seen on this rig (<5 deg) the
motion-energy trace from the shifted box matches the exact rotated box to r > 0.999.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np

import video_alignment
from facemap_rois import ROI

def map_points(W, pts) -> np.ndarray:
    """Map (N, 2) reference-frame points into session-frame points."""
    W = np.asarray(W, float)
    return np.asarray(pts, float).reshape(-1, 2) @ W[:, :2].T + W[:, 2]


def roi_corners(roi: ROI) -> np.ndarray:
    return np.array([[roi.x, roi.y], [roi.x1, roi.y], [roi.x1, roi.y1], [roi.x, roi.y1]], float)


def shift_roi(roi: ROI, W, frame_w: int, frame_h: int) -> tuple[ROI, dict]:
    """Move `roi` (reference frame) so its centre follows W; keep size; keep inside the frame."""
    cx, cy = map_points(W, [[roi.x + roi.w / 2.0, roi.y + roi.h / 2.0]])[0]
    x, y = int(round(cx - roi.w / 2.0)), int(round(cy - roi.h / 2.0))
    xc, yc = min(max(x, 0), frame_w - roi.w), min(max(y, 0), frame_h - roi.h)
    new = ROI(xc, yc, roi.w, roi.h)
    err = float(np.max(np.linalg.norm(map_points(W, roi_corners(roi)) - roi_corners(new), axis=1)))
    return new, {"max_corner_error_px": err, "pushed_inside_frame": (xc, yc) != (x, y)}


def align_default_roi(camera: str, video_path: pathlib.Path, default: ROI,
                      frame_w: int, frame_h: int,
                      qc_path: pathlib.Path | None = None) -> tuple[ROI | None, dict]:
    """Align `video_path` to the reference for `camera` and return the shifted default ROI.

    Returns (roi, info). roi is None when the alignment is not usable (no reference for the
    camera, frame-size mismatch, or low ECC confidence); the caller then keeps the default.
    """
    info: dict = {"mode": "rigid"}
    try:
        ref_info = video_alignment.load_reference_info()
        cfg = ref_info["cameras"].get(camera)
        if cfg is None:
            return None, info | {"status": "skipped", "reason": f"no reference template for {camera}"}
        if (cfg["frame_width"], cfg["frame_height"]) != (frame_w, frame_h):
            return None, info | {"status": "skipped",
                                 "reason": f"video is {frame_w}x{frame_h}, reference is "
                                           f"{cfg['frame_width']}x{cfg['frame_height']}"}
        res, ref, mov = video_alignment.align_video(str(video_path), camera, return_images=True)
    except Exception as e:  # noqa: BLE001 - alignment is an add-on; never lose the run to it
        return None, info | {"status": "error", "reason": f"{type(e).__name__}: {e}"}

    info |= {"reference_session": ref_info["session"],
             "reference_template": f"alignment_reference/{cfg['template']}",
             **{k: res[k] for k in ("dx", "dy", "rot_deg", "ecc", "warp")},
             "min_ecc": cfg.get("min_ecc", 0.2)}
    roi, extra = shift_roi(default, res["warp"], frame_w, frame_h)
    info |= extra
    if res["low_confidence"]:
        info |= {"status": "rejected", "reason": f"ECC {res['ecc']:.2f} < {info['min_ecc']}"}
    else:
        info |= {"status": "applied", "aligned_roi": roi.as_dict()}
    if qc_path is not None:
        try:
            write_qc_figure(qc_path, ref, mov, np.asarray(res["warp"]), default, roi, info)
            info["qc_figure"] = f"qc/{qc_path.name}"
        except Exception as e:  # noqa: BLE001
            print(f"  [align] WARNING {camera}: could not write QC figure ({e})", flush=True)
    return (roi if info["status"] == "applied" else None), info


def write_qc_figure(path: pathlib.Path, ref: np.ndarray, mov: np.ndarray, W: np.ndarray,
                    default: ROI, roi: ROI, info: dict) -> None:
    """before | after (magenta = reference, green = session) | session with exact (cyan) and
    shifted (yellow) ROI."""
    warped = video_alignment.warp_frame(mov, W, ref.shape)
    before = np.dstack([ref, mov, ref]).copy()
    after = np.dstack([ref, warped, ref]).copy()
    raw = cv2.cvtColor(mov, cv2.COLOR_GRAY2BGR)
    for im in (before, after):
        cv2.rectangle(im, (default.x, default.y), (default.x1, default.y1), (255, 255, 255), 2)
    cv2.polylines(raw, [map_points(W, roi_corners(default)).round().astype(np.int32)], True,
                  (255, 255, 0), 2)
    used = roi if info["status"] == "applied" else default
    cv2.rectangle(raw, (used.x, used.y), (used.x1, used.y1), (0, 255, 255), 2)
    cv2.putText(before, "before (magenta=reference, green=session)", (8, 24), 0, .55, (255, 255, 255), 2)
    cv2.putText(after, f"after rot={info['rot_deg']:.1f} dx={info['dx']:.0f} dy={info['dy']:.0f} "
                       f"ecc={info['ecc']:.2f} [{info['status']}]", (8, 24), 0, .55, (255, 255, 255), 2)
    cv2.putText(raw, "session: cyan=exact ROI, yellow=used ROI", (8, 24), 0, .55, (255, 255, 255), 2)
    sep = np.full((ref.shape[0], 6, 3), 255, np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack([before, sep, after, sep, raw]))
