"""QC figures for the facemap capsule.

Three per-camera figures plus one session dashboard:

* `qc/{camera}_roi_placement.png` -- the figure to look at first. The fixed ROI is
  only trustworthy if this session's motion actually landed inside it, so the box is
  drawn over both the average frame and the average motion map, with the placement
  metrics printed alongside.
* `qc/{camera}_motion_energy.png` -- the motion-energy trace: raw, cleaned, the
  samples flagged as compression pops, a one-minute zoom and the distribution.
* `qc/{camera}_svd.png` -- singular-value spectrum and the leading motion masks, so
  it is visible whether the components describe the animal or the apparatus.
* `facemap_qc.png` -- one row per camera, the reference image for the QC portal.

Palette: categorical slots are taken in fixed order from the reference instance
(blue, orange, aqua, yellow) and used only for line series, which is the adjacent-
pair case that palette is validated for. Magnitude (motion maps) uses one
perceptually uniform sequential ramp; raw/unprocessed traces stay neutral grey so
the processed trace reads as the foreground.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
"""Categorical slots 1-4, fixed order, for the leading SVD components."""
RAW = "#8a8a84"
"""Neutral ink for the raw/uncleaned trace."""
FLAG = "#e34948"
"""Status colour for flagged (compression-contaminated) samples; always labelled."""
MOTION_CMAP = "magma"
GRID = {"color": "#d8d8d2", "linewidth": 0.6, "alpha": 0.9}
DPI = 150


def _style(ax, xlabel=None, ylabel=None, title=None):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b5ae")
    ax.tick_params(colors="#55554f", labelsize=8)
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color="#33332f")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#33332f")
    if title:
        ax.set_title(title, fontsize=10, color="#1a1a19", loc="left")
    return ax


def _roi_box(ax, roi, color="#1baf7a", label=None):
    ax.add_patch(Rectangle((roi.x, roi.y), roi.w, roi.h, fill=False,
                           edgecolor=color, linewidth=1.6, label=label))


def plot_roi_placement(result, placement: dict, checks: dict,
                       out_path: pathlib.Path) -> pathlib.Path:
    """Average frame + average motion with the ROI drawn, and the placement metrics."""
    fig = plt.figure(figsize=(14, 3.9), dpi=DPI, facecolor="white")
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 0.75, 0.85], wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(result.avgframe, cmap="gray")
    _roi_box(ax, result.roi)
    ax.set_title(f"{result.camera}: average frame", fontsize=10, loc="left")
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    vmax = float(np.percentile(result.avgmotion, 99.5)) or None
    im = ax.imshow(result.avgmotion, cmap=MOTION_CMAP, vmax=vmax)
    _roi_box(ax, result.roi)
    ax.set_title("average motion  |f(t) - f(t-1)|", fontsize=10, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[0, 2])
    roi = result.roi
    crop = result.avgmotion[roi.y:roi.y1, roi.x:roi.x1]
    ax.imshow(crop, cmap=MOTION_CMAP, vmax=vmax)
    cx, cy = placement.get("motion_centroid_xy_in_roi", [np.nan, np.nan])
    if np.isfinite(cx):
        ax.plot(cx, cy, marker="+", color="#ffffff", markersize=12, markeredgewidth=1.8)
        ax.plot((crop.shape[1] - 1) / 2, (crop.shape[0] - 1) / 2, marker="x",
                color="#1baf7a", markersize=9, markeredgewidth=1.8)
    ax.set_title("inside the ROI\n(+ motion centroid, x ROI centre)", fontsize=9, loc="left")
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
    rows = [
        ("ROI x,y,w,h", ",".join(str(v) for v in result.roi)),
        ("in/out motion ratio", f"{placement.get('motion_ratio_in_out', float('nan')):.2f}"),
        ("centroid offset", f"{placement.get('motion_centroid_offset', float('nan')):.2f}"),
        ("edge enrichment", f"{placement.get('edge_motion_enrichment', float('nan')):.2f}"),
        ("ROI share of frame motion",
         f"{placement.get('roi_share_of_frame_motion', float('nan')):.1%}"),
        ("ROI share of frame area",
         f"{placement.get('roi_share_of_frame_area', float('nan')):.1%}"),
        ("mean brightness", f"{placement.get('brightness_mean_in_roi', float('nan')):.0f}"),
        ("saturated pixels", f"{placement.get('saturated_fraction_in_roi', float('nan')):.2%}"),
    ]
    y = 1.0
    ax.text(0, y, "placement metrics", fontsize=9.5, weight="bold", color="#1a1a19")
    y -= 0.062
    for label, value in rows:
        ax.text(0, y, label, fontsize=8, color="#55554f")
        ax.text(1.0, y, value, fontsize=8, color="#1a1a19", ha="right")
        y -= 0.05
    y -= 0.015
    ax.text(0, y, "checks", fontsize=9.5, weight="bold", color="#1a1a19")
    y -= 0.062
    for name, check in checks.items():
        ok = check["status"] == "pass"
        ax.text(0, y, name.replace("_", " "), fontsize=7.5, color="#55554f")
        ax.text(1.0, y, "PASS" if ok else "FAIL", fontsize=7.5, ha="right",
                color="#1baf7a" if ok else FLAG, weight="bold")
        y -= 0.046

    fig.suptitle(f"{result.camera} camera -- ROI placement", fontsize=12,
                 weight="bold", x=0.01, ha="left")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_motion_energy(result, motion_clean: np.ndarray, mask: np.ndarray,
                       timestamps: np.ndarray | None, fps: float,
                       out_path: pathlib.Path, zoom_s: float = 60.0) -> pathlib.Path:
    """Full trace, a one-minute zoom, and the distribution."""
    t = (timestamps if timestamps is not None and np.isfinite(timestamps).any()
         else np.arange(result.motion.size) / (fps or 60.0))
    t = np.asarray(t, dtype=float)

    # y limits come from the CLEANED trace: the raw one still carries the keyframe
    # pops (up to ~100x the behavioural range), which would flatten everything else.
    finite_clean = motion_clean[np.isfinite(motion_clean)]
    top = float(np.percentile(finite_clean, 99.9) * 1.25) if finite_clean.size else None
    clipped = int(np.sum(result.motion[np.isfinite(result.motion)] > top)) if top else 0

    fig = plt.figure(figsize=(14, 7.5), dpi=DPI, facecolor="white")
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.3, 1, 1], hspace=0.45, wspace=0.25)

    ax = _style(fig.add_subplot(gs[0, :]), "time (s)", "motion energy\n(mean |diff| per px)",
                "full session"
                + (f"  (y clipped at the cleaned p99.9; {clipped} raw samples above)"
                   if clipped else ""))
    ax.plot(t, result.motion, color=RAW, linewidth=0.3, label="raw")
    ax.plot(t, motion_clean, color=SERIES[0], linewidth=0.4, label="cleaned")
    ax.margins(x=0)
    if top:
        ax.set_ylim(0, top)
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    # zoom on a window containing at least one keyframe pop, if there is one
    window = int((zoom_s) * (fps or 60.0))
    flagged = np.flatnonzero(mask)
    centre = int(flagged[flagged.size // 2]) if flagged.size else result.motion.size // 2
    lo = max(0, min(centre - window // 2, result.motion.size - window))
    hi = min(result.motion.size, lo + window)
    ax = _style(fig.add_subplot(gs[1, :]), "time (s)", "motion energy",
                f"{(hi - lo) / (fps or 60.0):.0f} s zoom -- "
                f"flagged samples span an H.264 keyframe")
    ax.plot(t[lo:hi], result.motion[lo:hi], color=RAW, linewidth=0.8, label="raw")
    ax.plot(t[lo:hi], motion_clean[lo:hi], color=SERIES[0], linewidth=1.0, label="cleaned")
    sel = mask[lo:hi]
    if sel.any():
        ax.plot(t[lo:hi][sel], result.motion[lo:hi][sel], linestyle="none", marker="o",
                markersize=3.5, color=FLAG, label="flagged (keyframe)")
    ax.margins(x=0)
    if top:
        ax.set_ylim(0, top)
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    ax = _style(fig.add_subplot(gs[2, 0]), "motion energy", "frames", "distribution")
    finite = motion_clean[np.isfinite(motion_clean)]
    if finite.size:
        ax.hist(finite, bins=120, color=SERIES[0], edgecolor="none")
        ax.set_yscale("log")

    ax = _style(fig.add_subplot(gs[2, 1:]), "block (session split into 10)",
                "mean motion energy", "drift across the session")
    blocks = [b[np.isfinite(b)] for b in np.array_split(motion_clean, 10)]
    means = [float(b.mean()) if b.size else np.nan for b in blocks]
    ax.bar(np.arange(1, len(means) + 1), means, color=SERIES[0], width=0.7)
    ax.set_xticks(np.arange(1, len(means) + 1))

    fig.suptitle(f"{result.camera} camera -- motion energy", fontsize=12,
                 weight="bold", x=0.01, ha="left")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_svd(result, fps: float, out_path: pathlib.Path,
             n_masks: int = 6, n_traces: int = 4, zoom_s: float = 30.0) -> pathlib.Path:
    """Singular-value spectrum, leading motion masks, and leading component traces."""
    fig = plt.figure(figsize=(14, 7.5), dpi=DPI, facecolor="white")
    gs = GridSpec(3, n_masks, figure=fig, height_ratios=[1.1, 1.0, 1.2],
                  hspace=0.5, wspace=0.18)

    s = result.singular_values
    ax = _style(fig.add_subplot(gs[0, :n_masks // 2]), "component", "singular value",
                "spectrum (subsampled SVD)")
    if np.any(s != 0):
        ax.plot(np.arange(1, s.size + 1), s, color=SERIES[0], linewidth=1.2)
        ax.set_xscale("log"); ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "not computed by facemap\n(single subsample chunk)",
                ha="center", va="center", fontsize=9, color="#55554f",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

    variances = np.nanvar(result.motsvd, axis=0)
    ax = _style(fig.add_subplot(gs[0, n_masks // 2:]), "components",
                "cumulative variance fraction",
                "cumulative variance of the projections (within the 500 retained)")
    if np.nansum(variances) > 0:
        ax.plot(np.arange(1, variances.size + 1),
                np.cumsum(variances) / np.nansum(variances), color=SERIES[0], linewidth=1.2)
    ax.set_xscale("log"); ax.set_ylim(0, 1)

    limit = float(np.percentile(np.abs(result.motmask[:, :, :n_masks]), 99)) or 1.0
    for i in range(min(n_masks, result.motmask.shape[-1])):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(result.motmask[:, :, i], cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"mask {i + 1}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    t = np.arange(result.motsvd.shape[0]) / (fps or 60.0)
    window = int(zoom_s * (fps or 60.0))
    lo = max(0, result.motsvd.shape[0] // 2 - window // 2)
    hi = min(result.motsvd.shape[0], lo + window)
    ax = _style(fig.add_subplot(gs[2, :]), "time (s)", "projection (a.u.)",
                f"leading components, {zoom_s:.0f} s")
    for i in range(min(n_traces, result.motsvd.shape[1])):
        ax.plot(t[lo:hi], result.motsvd[lo:hi, i], color=SERIES[i % len(SERIES)],
                linewidth=0.9, label=f"component {i + 1}")
    ax.margins(x=0)
    ax.legend(fontsize=8, loc="upper right", frameon=False, ncols=n_traces)

    fig.suptitle(f"{result.camera} camera -- motion SVD", fontsize=12,
                 weight="bold", x=0.01, ha="left")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def build_dashboard(results: list, per_camera: dict, out_path: pathlib.Path) -> pathlib.Path:
    """One row per camera: ROI on the motion map, the motion trace, the top mask.

    This is the image referenced by `quality_control.json`, so it has to be readable
    on its own: the header states pass/fail per camera and why.
    """
    n = max(len(results), 1)
    fig = plt.figure(figsize=(15, 1.6 + 3.1 * n), dpi=DPI, facecolor="white")
    gs = GridSpec(n + 1, 4, figure=fig, height_ratios=[0.45] + [1] * n,
                  width_ratios=[1, 2.4, 0.9, 0.9], hspace=0.45, wspace=0.2)

    header = fig.add_subplot(gs[0, :]); header.axis("off")
    lines = []
    for result in results:
        camera = result.camera
        checks = per_camera[camera]["checks"]
        failed = [k for k, v in checks.items() if v["status"] == "fail"]
        lines.append(f"{camera}: {'PASS' if not failed else 'FAIL -- ' + ', '.join(failed)}")
    header.text(0, 0.65, "facemap motion SVD -- session QC", fontsize=13, weight="bold",
                color="#1a1a19")
    header.text(0, 0.1, "     |     ".join(lines) if lines else "no cameras processed",
                fontsize=9.5, color="#55554f")

    for row, result in enumerate(results, start=1):
        camera = result.camera
        entry = per_camera[camera]
        ax = fig.add_subplot(gs[row, 0])
        vmax = float(np.percentile(result.avgmotion, 99.5)) or None
        ax.imshow(result.avgmotion, cmap=MOTION_CMAP, vmax=vmax)
        _roi_box(ax, result.roi)
        ax.set_title(f"{camera}: avg motion + ROI", fontsize=9, loc="left")
        ax.set_xticks([]); ax.set_yticks([])

        clean = entry["motion_clean"]
        t = np.arange(clean.size) / (entry["fps"] or 60.0)
        ax = _style(fig.add_subplot(gs[row, 1]), "time (s)", "motion energy",
                    f"{camera}: motion energy (cleaned)")
        ax.plot(t, clean, color=SERIES[0], linewidth=0.3)
        ax.margins(x=0)

        ax = fig.add_subplot(gs[row, 2])
        limit = float(np.percentile(np.abs(result.motmask[:, :, 0]), 99)) or 1.0
        ax.imshow(result.motmask[:, :, 0], cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title("mask 1", fontsize=9); ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(gs[row, 3]); ax.axis("off")
        motion = entry["metrics"]["motion"]
        svd = entry["metrics"]["svd"]
        summary = [
            f"frames: {result.n_frames}",
            f"ROI: {tuple(result.roi)}",
            f"motion mean: {motion.get('mean', float('nan')):.2f}",
            f"moving: {motion.get('fraction_frames_moving', float('nan')):.1%}",
            ("var. first 10 PC: n/a" if svd.get("variance_fraction_first10") is None
             else f"var. first 10 PC: {svd['variance_fraction_first10']:.1%}"),
            f"keyframes masked: {entry['metrics']['keyframes'].get('masked_fraction', float('nan')):.2%}",
        ]
        for i, text in enumerate(summary):
            ax.text(0, 0.92 - 0.14 * i, text, fontsize=8.5, color="#33332f")

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
