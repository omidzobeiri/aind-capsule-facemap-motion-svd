"""
build_minimal_proc.py
---------------------
Helper to auto-generate a minimal facemap proc dict from a video file.
Mirrors the exact key structure of an existing proc file; creates a single
'motion SVD' ROI so facemap.process.run() can proceed without
a hand-initialized GUI session.

Also provides ``detect_motion_roi`` which samples frame differences across
the video, computes a mean motion-energy map, and returns the tight bounding
box of the high-motion region.  This is used by the ``auto`` ROI mode to let
facemap work on the part of the frame that actually moves rather than the
whole frame.

Usage
-----
    from build_minimal_proc import build_minimal_proc, detect_motion_roi
    from facemap import process as facemap_process

    yrange, xrange = detect_motion_roi(video_path)
    proc = build_minimal_proc(video_path, save_dir,
                               roi_yrange=yrange, roi_xrange=xrange)
    facemap_process.run(filenames=proc['filenames'], proc=proc,
                        savepath=proc['save_path'])
"""

import cv2
import numpy as np
from pathlib import Path


def detect_motion_roi(
    video_path: str,
    roi_w: int | None = None,
    roi_h: int | None = None,
    n_frames: int = 300,
    threshold_pct: float = 80.0,
    padding: int = 20,
    min_side: int = 32,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Detect the motion-energy ROI from sampled frame differences.

    **Fixed-size centroid mode** (used when ``roi_w`` and ``roi_h`` are given):
    computes the motion-weighted centroid and places a box of exactly
    ``roi_w`` × ``roi_h`` pixels centred on it.  This is the recommended mode
    when you want the same pixel footprint as the hand-tuned defaults while
    letting the position adapt to where the animal actually sits in the frame.

    **Bounding-box mode** (used when ``roi_w``/``roi_h`` are ``None``):
    returns the tight bounding box of above-threshold pixels, expanded by
    *padding* and clipped to the frame.

    Parameters
    ----------
    video_path       : path to the video file
    roi_w, roi_h     : target width / height for fixed-size centroid mode
    n_frames         : number of frame-difference samples to draw
    threshold_pct    : percentile of non-zero motion energy for thresholding
                       (80 = broad mask good for centroid estimation)
    padding          : padding around bounding box (bounding-box mode only)
    min_side         : minimum ROI side in pixels

    Returns
    -------
    ``(roi_yrange, roi_xrange)`` — each a ``(start, end)`` tuple (exclusive end).
    Falls back to the full frame on any error.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    _full = (0, h), (0, w)  # fallback

    if h == 0 or w == 0 or total < 2:
        cap.release()
        print(f"[detect_motion_roi] degenerate video ({w}×{h}, {total} frames) — using full frame")
        return _full

    # Skip the first ~5 % of the video (acquisition preamble) and sample evenly.
    start_idx = max(1, total // 20)
    end_idx = total - 1
    n_sample = min(n_frames + 1, end_idx - start_idx + 1)
    indices = np.linspace(start_idx, end_idx, n_sample, dtype=int)

    motion_map = np.zeros((h, w), dtype=np.float64)
    n_diffs = 0
    prev_gray: np.ndarray | None = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if frame.ndim == 3 else frame.astype(np.float32))
        if prev_gray is not None:
            motion_map += np.abs(gray - prev_gray)
            n_diffs += 1
        prev_gray = gray

    cap.release()

    if n_diffs == 0:
        print("[detect_motion_roi] no frame diffs obtained — using full frame")
        return _full

    nonzero = motion_map[motion_map > 0]
    if len(nonzero) == 0:
        print("[detect_motion_roi] zero motion energy — using full frame")
        return _full

    threshold = float(np.percentile(nonzero, threshold_pct))
    mask = motion_map >= threshold

    if not mask.any():
        print("[detect_motion_roi] no pixels above threshold — using full frame")
        return _full

    # ── Fixed-size centroid mode ──────────────────────────────────────────────
    if roi_w is not None and roi_h is not None:
        # Weighted centroid of above-threshold pixels (weighted by motion energy).
        masked_energy = motion_map * mask
        total_energy = masked_energy.sum()
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cy = int(round((ys * masked_energy).sum() / total_energy))
        cx = int(round((xs * masked_energy).sum() / total_energy))

        # Place the fixed-size box centred on (cy, cx) and clip to frame.
        y0 = max(0, cy - roi_h // 2)
        y1 = y0 + roi_h
        if y1 > h:
            y1 = h
            y0 = max(0, h - roi_h)

        x0 = max(0, cx - roi_w // 2)
        x1 = x0 + roi_w
        if x1 > w:
            x1 = w
            x0 = max(0, w - roi_w)

        print(f"[detect_motion_roi] centroid mode: centroid=({cx},{cy})  "
              f"ROI: y[{y0}:{y1}] x[{x0}:{x1}]  "
              f"({x1-x0}×{y1-y0} px, from {n_diffs} diffs)")
        return (y0, y1), (x0, x1)

    # ── Bounding-box mode ─────────────────────────────────────────────────────
    rows_active = np.any(mask, axis=1)
    cols_active = np.any(mask, axis=0)

    y0 = int(np.where(rows_active)[0][0])
    y1 = int(np.where(rows_active)[0][-1]) + 1
    x0 = int(np.where(cols_active)[0][0])
    x1 = int(np.where(cols_active)[0][-1]) + 1

    y0 = max(0, y0 - padding)
    y1 = min(h, y1 + padding)
    x0 = max(0, x0 - padding)
    x1 = min(w, x1 + padding)

    if y1 - y0 < min_side:
        cy = (y0 + y1) // 2
        y0 = max(0, cy - min_side // 2)
        y1 = min(h, y0 + min_side)
    if x1 - x0 < min_side:
        cx = (x0 + x1) // 2
        x0 = max(0, cx - min_side // 2)
        x1 = min(w, x0 + min_side)

    print(f"[detect_motion_roi] bbox mode: y[{y0}:{y1}] x[{x0}:{x1}]  "
          f"({x1-x0}×{y1-y0} px, from {n_diffs} diffs, threshold={threshold:.1f})")
    return (y0, y1), (x0, x1)


def build_minimal_proc(
    video_path: str,
    save_path: str,
    sbin: int = 4,
    roi_yrange: tuple | None = None,
    roi_xrange: tuple | None = None,
) -> dict:
    """
    Build a minimal facemap proc dict from a video file.

    Parameters
    ----------
    video_path  : full path to the MP4 video file
    save_path   : directory where facemap should write results
    sbin        : spatial binning factor (default 4, matches facemap default)
    roi_yrange  : (y_start, y_end) pixel range; defaults to full frame height
    roi_xrange  : (x_start, x_end) pixel range; defaults to full frame width

    Returns
    -------
    dict ready to pass directly to facemap.process.run()
    """
    video_path = str(video_path)
    save_path  = str(save_path)

    # ── read frame dimensions from the video ─────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    Ly = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Lx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    if Ly == 0 or Lx == 0:
        raise ValueError(f"Video reports zero dimensions ({Ly}x{Lx}): {video_path}")

    # ── binned dims ──────────────────────────────────────────────────────────
    Lybin = Ly // sbin
    Lxbin = Lx // sbin

    # ── ROI pixel ranges (default: full frame) ───────────────────────────────
    y0, y1 = (0, Ly) if roi_yrange is None else roi_yrange
    x0, x1 = (0, Lx) if roi_xrange is None else roi_xrange

    yrange     = np.arange(y0, y1, dtype=np.int64)
    xrange     = np.arange(x0, x1, dtype=np.int64)
    yrange_bin = np.arange(y0 // sbin, y1 // sbin, dtype=np.int64)
    xrange_bin = np.arange(x0 // sbin, x1 // sbin, dtype=np.int64)

    roi = {
        'rind':        1,                  # 1 = motion SVD
        'rtype':       'motion SVD',
        'iROI':        0,
        'ivid':        0,
        'color':       (255.0, 0.0, 0.0),
        'yrange':      yrange,
        'xrange':      xrange,
        'saturation':  255.0,
        'pupil_sigma': 2.0,
        'ellipse':     np.zeros((len(yrange), len(xrange)), dtype=bool),
        'yrange_bin':  yrange_bin,
        'xrange_bin':  xrange_bin,
    }

    proc = {
        'filenames': [[video_path]],
        'save_path': save_path,
        'Ly':        [Ly],
        'Lx':        [Lx],
        'sbin':      sbin,
        'fullSVD':   False,
        'save_mat':  True,
        'Lybin':     np.array([Lybin], dtype=np.int32),
        'Lxbin':     np.array([Lxbin], dtype=np.int32),
        'sybin':     np.array([0],     dtype=np.int64),
        'sxbin':     np.array([0],     dtype=np.int64),
        'LYbin':     np.int64(Lybin),
        'LXbin':     np.int64(Lxbin),
        'sy':        np.array([0],     dtype=np.int64),
        'sx':        np.array([0],     dtype=np.int64),
        'rois':      [roi],
    }

    print(f"[build_minimal_proc] {Path(video_path).name}  "
          f"({Ly}px × {Lx}px, sbin={sbin})  "
          f"ROI: y[{y0}:{y1}] x[{x0}:{x1}]")
    return proc


def save_proc(proc: dict, path: str) -> None:
    """Persist a proc dict as a .npy file for later reuse."""
    np.save(path, proc)
    print(f"[save_proc] saved → {path}")
