"""Per-camera QC metrics and pass/fail checks.

Everything here is computed from what the pipeline already has in memory --
facemap's own full-frame `avgframe`/`avgmotion`, the repaired motion trace, the
singular values and the sync alignment -- so QC costs no extra decode pass.

The metric that matters most for this capsule is **ROI placement**. A fixed crop is
what makes sessions comparable, and it is also the thing most likely to be silently
wrong when a rig is adjusted or an animal sits differently. `roi_placement_metrics`
answers three questions from the session's own average motion map: is there more
motion inside the box than outside (`motion_ratio_in_out`), is that motion centred
in the box (`motion_centroid_offset`), and is it pressed up against the edge as if
the animal were being clipped (`edge_motion_enrichment`).
"""
from __future__ import annotations

import numpy as np

# ── thresholds (recorded in processing.json; tune per rig) ──────────────────────
MIN_MOTION_RATIO_IN_OUT = 1.5
"""In-ROI mean motion must exceed out-of-ROI mean motion by this factor."""
MAX_CENTROID_OFFSET = 0.6
"""Motion centroid offset from ROI centre, as a fraction of the ROI half-size."""
MAX_EDGE_ENRICHMENT = 1.4
"""Motion mass in the outer 10% border, relative to that border's area share."""
MAX_SATURATED_FRACTION = 0.02
"""Fraction of ROI pixels at/above 250 in the average frame (blown-out highlights)."""
MAX_FRAME_COUNT_MISMATCH = 2
"""Allowed |container frames - MVR FramesRecorded|."""
MAX_KEYFRAME_FRACTION = 0.05
"""Fraction of motion samples masked as compression-contaminated."""
MIN_MOTION_CV = 0.05
"""Coefficient of variation of the cleaned motion trace; below this the view is dead."""

EDGE_BORDER = 0.10
"""Border width used by `edge_motion_enrichment`, as a fraction of each ROI side."""


def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def roi_placement_metrics(avgmotion: np.ndarray, avgframe: np.ndarray,
                          roi, sbin: int = 1) -> dict:
    """Where the session's motion fell, relative to the ROI box.

    `avgmotion` / `avgframe` are facemap's full-frame averages (binned by `sbin`).
    """
    y0, y1 = int(roi.y / sbin), int(roi.y1 / sbin)
    x0, x1 = int(roi.x / sbin), int(roi.x1 / sbin)
    motion = np.asarray(avgmotion, dtype=float)
    frame = np.asarray(avgframe, dtype=float)
    y1, x1 = min(y1, motion.shape[0]), min(x1, motion.shape[1])
    inside = motion[y0:y1, x0:x1]
    if inside.size == 0:
        return {"error": "empty ROI after clipping to the average-motion map"}

    total_sum, total_n = float(motion.sum()), int(motion.size)
    in_sum, in_n = float(inside.sum()), int(inside.size)
    out_n = max(total_n - in_n, 1)
    out_mean = (total_sum - in_sum) / out_n
    in_mean = in_sum / in_n

    # centroid of in-ROI motion, in units of the ROI half-size (0 = centred)
    h, w = inside.shape
    weights = np.clip(inside, 0, None)
    total_weight = float(weights.sum())
    if total_weight > 0:
        cy = float((weights.sum(axis=1) @ np.arange(h)) / total_weight)
        cx = float((weights.sum(axis=0) @ np.arange(w)) / total_weight)
        offset = float(np.hypot((cx - (w - 1) / 2) / max((w - 1) / 2, 1e-9),
                                (cy - (h - 1) / 2) / max((h - 1) / 2, 1e-9)) / np.sqrt(2))
    else:
        cx = cy = offset = float("nan")

    # motion mass pushed into the outer border, relative to that border's area share
    by, bx = max(int(round(EDGE_BORDER * h)), 1), max(int(round(EDGE_BORDER * w)), 1)
    core = weights[by:h - by, bx:w - bx] if h > 2 * by and w > 2 * bx else weights[:0, :0]
    edge_mass = total_weight - float(core.sum())
    edge_area_share = 1.0 - (core.size / in_n) if in_n else float("nan")
    enrichment = (edge_mass / total_weight / edge_area_share
                  if total_weight > 0 and edge_area_share > 0 else float("nan"))

    frame_in = frame[y0:y1, x0:x1]
    return {
        "motion_mean_in_roi": in_mean,
        "motion_mean_out_of_roi": out_mean,
        "motion_ratio_in_out": float(in_mean / out_mean) if out_mean > 0 else float("inf"),
        "roi_share_of_frame_motion": float(in_sum / total_sum) if total_sum > 0 else float("nan"),
        "roi_share_of_frame_area": float(in_n / total_n),
        "motion_centroid_xy_in_roi": [cx, cy],
        "motion_centroid_offset": offset,
        "edge_motion_enrichment": float(enrichment),
        "brightness_mean_in_roi": float(np.mean(frame_in)),
        "brightness_median_in_roi": float(np.median(frame_in)),
        "saturated_fraction_in_roi": float(np.mean(frame_in >= 250)),
        "black_fraction_in_roi": float(np.mean(frame_in <= 5)),
    }


def motion_metrics(motion_clean: np.ndarray, motion_raw: np.ndarray,
                   keyframe_mask: np.ndarray, fps: float, n_blocks: int = 10) -> dict:
    """Level, spread, drift and dropout of the cleaned motion-energy trace."""
    clean = np.asarray(motion_clean, dtype=float)
    finite = _finite(clean)
    out: dict = {
        "n_samples": int(clean.size),
        "n_nan": int(clean.size - finite.size),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite)) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "p01": float(np.percentile(finite, 1)) if finite.size else float("nan"),
        "p99": float(np.percentile(finite, 99)) if finite.size else float("nan"),
        "max": float(np.max(finite)) if finite.size else float("nan"),
    }
    out["coefficient_of_variation"] = (out["std"] / out["mean"]
                                       if out["mean"] else float("nan"))
    if finite.size:
        mad = float(np.median(np.abs(finite - out["median"]))) or float("nan")
        threshold = out["median"] + 3 * 1.4826 * mad
        out["movement_threshold"] = threshold
        out["fraction_frames_moving"] = float(np.mean(finite > threshold))
    # drift: per-block means over the session
    blocks = [b for b in np.array_split(clean, n_blocks) if _finite(b).size]
    block_means = [float(np.nanmean(b)) for b in blocks]
    out["block_means"] = block_means
    if block_means and min(block_means) > 0:
        out["block_mean_max_over_min"] = float(max(block_means) / min(block_means))
    # what the keyframe pops were worth, before cleaning
    raw = np.asarray(motion_raw, dtype=float)
    mask = np.asarray(keyframe_mask, dtype=bool)
    if mask.any() and (~mask).any():
        masked, unmasked = _finite(raw[mask]), _finite(raw[~mask])
        if masked.size and unmasked.size and np.median(unmasked) > 0:
            out["keyframe_pop_amplitude_ratio"] = float(np.median(masked) / np.median(unmasked))
    out["duration_s"] = float(clean.size / fps) if fps else float("nan")
    return out


def svd_metrics(singular_values: np.ndarray, motsvd: np.ndarray) -> dict:
    """Spectrum summary.

    Two views of "how concentrated is the motion", because each has a caveat:

    * From the **projections** (`motsvd`): the variance of component k over the whole
      session, as a fraction of the variance in the 500 retained components. Always
      available, full-session, but it is a fraction *of the retained subspace*, not
      of total pixel variance.
    * From facemap's **singular values**: the spectrum of the subsampled matrix the
      masks were fit to. facemap only fills these in when the video is long enough
      for more than one subsample chunk (>= 2000 frames), so on short clips they are
      all zero and are reported as unavailable rather than as zeros.
    """
    s = np.asarray(singular_values, dtype=float)
    comps = np.asarray(motsvd, dtype=float)
    out: dict = {"n_components": int(s.size)}

    if comps.size:
        variances = np.nanvar(comps, axis=0)
        total = float(np.nansum(variances))
        out["component_std_first10"] = np.sqrt(variances[:10]).tolist()
        if total > 0:
            cumulative = np.cumsum(variances) / total
            for k in (1, 5, 10, 50, 100):
                if k <= cumulative.size:
                    out[f"variance_fraction_first{k}"] = float(cumulative[k - 1])
            out["variance_fraction_basis"] = "projection variance / retained-subspace variance"

    available = bool(np.any(s != 0))
    out["singular_values_available"] = available
    if available:
        power = s**2
        out["singular_values_first10"] = s[:10].tolist()
        cumulative = np.cumsum(power) / float(power.sum())
        for k in (1, 5, 10, 50, 100):
            if k <= cumulative.size:
                out[f"singular_variance_fraction_first{k}"] = float(cumulative[k - 1])
    else:
        out["singular_values_note"] = (
            "facemap leaves motSv zeroed when the video yields a single subsample "
            "chunk (< ~2000 frames); use the projection-based fractions instead")
    return out


def evaluate(camera_metrics: dict) -> dict:
    """Apply the thresholds above; returns {check_name: {status, value, threshold}}."""
    video = camera_metrics.get("video", {})
    placement = camera_metrics.get("roi_placement", {})
    motion = camera_metrics.get("motion", {})
    keyframes = camera_metrics.get("keyframes", {})
    sync_info = camera_metrics.get("sync", {})
    trim = camera_metrics.get("trim", {})

    def check(name, value, ok, threshold, note=""):
        return {name: {"status": "pass" if ok else "fail", "value": value,
                       "threshold": threshold, "note": note}}

    checks: dict = {}
    missing = sync_info.get("n_frames_without_timestamp")
    if missing is not None:
        checks |= check("sync_covers_all_frames", missing, missing == 0, 0,
                        "every video frame must have a sync pulse")

    recorded, container = video.get("frames_recorded"), video.get("n_frames")
    if recorded and container:
        diff = abs(int(container) - int(recorded))
        checks |= check("frame_count_matches_recording", diff,
                        diff <= MAX_FRAME_COUNT_MISMATCH, MAX_FRAME_COUNT_MISMATCH,
                        "container frame count vs MVR FramesRecorded")
    lost = video.get("frames_lost_count")
    if lost is not None:
        checks |= check("no_frames_lost_at_acquisition", lost, int(lost) == 0, 0)

    ratio = placement.get("motion_ratio_in_out")
    if ratio is not None:
        checks |= check("roi_contains_the_motion", ratio, ratio >= MIN_MOTION_RATIO_IN_OUT,
                        MIN_MOTION_RATIO_IN_OUT, "in-ROI vs out-of-ROI average motion")
    offset = placement.get("motion_centroid_offset")
    if offset is not None and np.isfinite(offset):
        checks |= check("roi_motion_is_centred", offset, offset <= MAX_CENTROID_OFFSET,
                        MAX_CENTROID_OFFSET, "0 = centred, 1 = at the ROI corner")
    edge = placement.get("edge_motion_enrichment")
    if edge is not None and np.isfinite(edge):
        checks |= check("roi_not_clipping_the_animal", edge, edge <= MAX_EDGE_ENRICHMENT,
                        MAX_EDGE_ENRICHMENT, "motion mass concentrated at the ROI border")
    saturated = placement.get("saturated_fraction_in_roi")
    if saturated is not None:
        checks |= check("roi_not_overexposed", saturated, saturated <= MAX_SATURATED_FRACTION,
                        MAX_SATURATED_FRACTION)

    fraction = keyframes.get("masked_fraction")
    if fraction is not None:
        checks |= check("keyframe_contamination_low", fraction,
                        fraction <= MAX_KEYFRAME_FRACTION, MAX_KEYFRAME_FRACTION)

    settle = trim.get("settle_frame")
    if settle is not None:
        dropped = int(trim.get("frames_dropped", 0))
        checks |= check("acquisition_preamble_excluded",
                        {"settle_frame": int(settle), "frames_dropped": dropped},
                        dropped >= int(settle), "frames_dropped >= settle_frame",
                        "the white metadata frame and exposure settling must not reach "
                        "the SVD; see video_trim")

    cv = motion.get("coefficient_of_variation")
    if cv is not None and np.isfinite(cv):
        checks |= check("motion_trace_not_flat", cv, cv >= MIN_MOTION_CV, MIN_MOTION_CV,
                        "a frozen or blank view gives a near-constant trace")
    return checks


def overall_status(checks: dict) -> str:
    if not checks:
        return "pending"
    return "fail" if any(c["status"] == "fail" for c in checks.values()) else "pass"


def thresholds() -> dict:
    return {
        "min_motion_ratio_in_out": MIN_MOTION_RATIO_IN_OUT,
        "max_centroid_offset": MAX_CENTROID_OFFSET,
        "max_edge_enrichment": MAX_EDGE_ENRICHMENT,
        "max_saturated_fraction": MAX_SATURATED_FRACTION,
        "max_frame_count_mismatch": MAX_FRAME_COUNT_MISMATCH,
        "max_keyframe_fraction": MAX_KEYFRAME_FRACTION,
        "min_motion_cv": MIN_MOTION_CV,
        "edge_border_fraction": EDGE_BORDER,
    }
