"""Rigid (rotation + translation) alignment of behavior-camera videos to a reference session.

Per camera, a median "template" image is built from a few frames spread across the video and
aligned to the stored reference template (alignment_reference/<cam>.png):

  1. edge maps (log intensity -> Sobel magnitude); saturated regions (screens, reflections)
     are masked out, plus an optional top-of-frame crop per camera (Face: objective ring)
  2. coarse rotation grid search (1 deg steps, +/-15 deg) with phase correlation for the
     translation, at 1/4 resolution
  3. ECC (MOTION_EUCLIDEAN) refinement at 1/2 resolution from the top-3 candidates plus the
     identity; the highest ECC score wins

Warp convention: W (2x3) maps REFERENCE pixel coords -> SESSION pixel coords:
    [x_sess, y_sess] = W @ [x_ref, y_ref, 1]
Bring a session frame into reference space with
    cv2.warpAffine(frame, W, (ref_w, ref_h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(HERE, 'alignment_reference')
CAMS = ['Behavior', 'Face', 'Nose']


# ----------------------------------------------------------------------------- reference

def load_reference_info(reference_dir=REFERENCE_DIR):
    with open(os.path.join(reference_dir, 'reference.json')) as f:
        return json.load(f)


def load_reference(cam, reference_dir=REFERENCE_DIR):
    """Reference template (uint8 grayscale) and its per-camera settings."""
    info = load_reference_info(reference_dir)
    cfg = info['cameras'][cam]
    img = cv2.imread(os.path.join(reference_dir, cfg['template']), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'missing reference template for {cam} in {reference_dir}')
    return img, cfg


# ----------------------------------------------------------------------------- templates

def video_template(path, n_frames=9, start=0.05, stop=0.95):
    """Median of n_frames frames evenly spaced between `start` and `stop` (fractions of the video).

    The first/last 5% are skipped by default, which also avoids the white metadata frame and
    exposure-settling frames at the start of MVR recordings.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f'cannot open {path}')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(start, stop, n_frames) * n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, x = cap.read()
        if ok:
            frames.append(x[..., 0] if x.ndim == 3 else x)
    cap.release()
    if not frames:
        raise IOError(f'could not decode frames from {path}')
    return np.median(np.stack(frames), 0).astype(np.uint8)


# ----------------------------------------------------------------------------- alignment

def _ncc(a, b, mask):
    a = a[mask].astype(float); b = b[mask].astype(float)
    if a.size == 0:
        return -1.0
    a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def _edges(im, scale, top_crop=0.0, sat_thresh=240):
    x = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    x = cv2.GaussianBlur(np.log1p(x), (0, 0), 1.0)
    g = cv2.magnitude(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))
    # screen / reflection edges vary session to session -> suppress saturated regions
    sat = cv2.resize((im >= sat_thresh).astype(np.uint8), g.shape[::-1], interpolation=cv2.INTER_NEAREST)
    k = max(3, int(24 * scale) | 1)
    g[cv2.dilate(sat, np.ones((k, k), np.uint8)) > 0] = 0
    if top_crop:
        g[:int(top_crop * g.shape[0])] = 0
    return g / (g.mean() + 1e-6)


def _rot(ang, c):
    return cv2.getRotationMatrix2D(c, ang, 1.0).astype(np.float32)


def rigid_align(ref, mov, top_crop=0.0, angles=np.arange(-15, 15.1, 1.0), s0=0.25, s1=0.5, ntop=3):
    """Euclidean warp W (2x3, full-res px, ref -> mov coords) and its ECC score (0..1)."""
    if ref.shape != mov.shape:
        raise ValueError(f'frame size mismatch: reference {ref.shape} vs session {mov.shape}')
    R = _edges(ref, s0, top_crop); M = _edges(mov, s0, top_crop)
    h, w = R.shape; c = (w / 2, h / 2)
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    ones = np.ones_like(M)
    cands = []
    for a in angles:
        (dx, dy), _ = cv2.phaseCorrelate(R, cv2.warpAffine(M, _rot(a, c), (w, h)), win)
        Rinv = cv2.invertAffineTransform(_rot(a, c))
        W = Rinv.copy(); W[:, 2] += Rinv[:, :2] @ np.array([dx, dy], np.float32)
        wm = cv2.warpAffine(M, W, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        v = cv2.warpAffine(ones, W, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST) > 0.5
        cands.append((_ncc(R, wm, v), W))
    cands.sort(key=lambda x: -x[0])
    starts = [W for _, W in cands[:ntop]] + [np.float32([[1, 0, 0], [0, 1, 0]])]
    R1 = _edges(ref, s1, top_crop); M1 = _edges(mov, s1, top_crop)
    best_cc, best_W = -2.0, None
    for W in starts:
        W = W.copy(); W[:, 2] *= s1 / s0
        try:
            cc, W = cv2.findTransformECC(R1, M1, W, cv2.MOTION_EUCLIDEAN,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 3)
        except cv2.error:
            continue
        if cc > best_cc:
            best_cc, best_W = cc, W
    if best_W is None:  # ECC failed from every start: fall back to the best grid candidate
        best_W = starts[0].copy(); best_W[:, 2] *= s1 / s0; best_cc = float('nan')
    best_W[:, 2] /= s1
    return best_W, float(best_cc)


def describe(W):
    return dict(dx=float(W[0, 2]), dy=float(W[1, 2]),
                rot_deg=float(np.degrees(np.arctan2(W[1, 0], W[0, 0]))),
                warp=np.asarray(W, dtype=float).tolist())


def align_video(video_path, cam, reference_dir=REFERENCE_DIR, n_frames=9, return_images=False):
    """Align one session video to the reference for camera `cam` ('Behavior', 'Face', 'Nose').

    Returns dict(dx, dy, rot_deg, warp, ecc, ...). With return_images=True also returns
    (result, ref_template, session_template).
    """
    ref, cfg = load_reference(cam, reference_dir)
    mov = video_template(video_path, n_frames=n_frames)
    W, cc = rigid_align(ref, mov, top_crop=cfg.get('top_crop', 0.0))
    res = dict(camera=cam, video=os.path.abspath(video_path), ecc=cc,
               low_confidence=bool(not np.isfinite(cc) or cc < cfg.get('min_ecc', 0.2)), **describe(W))
    return (res, ref, mov) if return_images else res


def warp_frame(frame, W, ref_shape):
    """Session frame -> reference space."""
    return cv2.warpAffine(frame, np.asarray(W, np.float32), (ref_shape[1], ref_shape[0]),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
