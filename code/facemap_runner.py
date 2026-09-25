"""Run facemap's motion SVD on one camera's video, over a fixed ROI.

`facemap.process.run` is driven headless by handing it a `proc` dict -- the same
structure the GUI writes when you hit "save ROIs" -- so no Qt is involved.

Two facemap behaviours the rest of the capsule depends on:

* With `fullSVD=False`, index 0 of `motion` / `motSVD` / `motMask` is an empty
  placeholder for the (not computed) multivideo SVD, and our ROI is at index 1.
* facemap derives the binned ROI as `arange(start, stop)` over the *inclusive*
  range endpoints, so a `(w, h)` ROI yields a `(h-1, w-1)` mask at `sbin=1`. The
  effective ROI is read back from `rois[0]['yrange_bin']` rather than assumed.

`motion` is the **sum** of |frame diff| over ROI pixels; it is divided by the ROI
pixel count here so the trace is comparable between ROIs of different sizes.
"""
from __future__ import annotations

import pathlib
import time
from typing import NamedTuple

import numpy as np

import build_minimal_proc as _bmp
import facemap_rois
import motion_trace

N_SVD_COMPONENTS = 500
"""Components facemap keeps (hard-coded as `ncomps` in facemap.process.run)."""


class FacemapResult(NamedTuple):
    camera: str
    video_path: pathlib.Path
    proc_path: pathlib.Path
    n_frames: int
    roi: facemap_rois.ROI
    effective_roi: facemap_rois.ROI
    """What facemap actually integrated over (one row/column smaller -- see module docstring)."""
    motion_sum: np.ndarray
    """Raw facemap trace, summed over ROI pixels, original (mixed) alignment."""
    motion: np.ndarray
    """Per-pixel motion energy, alignment-repaired: `|f(t) - f(t-1)| / n_pixels`."""
    motsvd: np.ndarray            # (n_frames, n_components)
    motmask: np.ndarray           # (roi_h, roi_w, n_components)
    singular_values: np.ndarray   # (n_components,)
    avgframe: np.ndarray          # full frame, binned by sbin
    avgmotion: np.ndarray         # full frame, binned by sbin
    sbin: int
    runtime_s: float


def build_proc(roi: facemap_rois.ROI, savepath: pathlib.Path, sbin: int = 1) -> dict:
    """The facemap `proc` dict for a single motion-SVD ROI on a single video."""
    return {
        "rois": [facemap_rois.to_facemap_roi(roi, sbin=sbin)],
        "fullSVD": False,   # keep False: the multivideo SVD would build an
                            # (n_pixels x 7500) matrix -- ~10 GB for a full 658x492
                            # frame -- and would be dominated by rig motion anyway.
        "save_mat": False,
        "sbin": int(sbin),
        "sy": np.array([0]),
        "sx": np.array([0]),
        "savepath": str(savepath),
    }


def run_facemap(camera: str, video_path: pathlib.Path, roi: facemap_rois.ROI | None,
                savepath: pathlib.Path, sbin: int = 1,
                auto_roi: bool = False) -> FacemapResult:
    """Run the motion SVD for one camera and load the results back.

    When ``auto_roi=True`` the ROI is detected automatically from the full video
    frame using ``build_minimal_proc``; ``roi`` is ignored in that case and the
    returned :attr:`FacemapResult.roi` reflects the full-frame dimensions.
    When ``auto_roi=False`` (default) ``roi`` must be a valid :class:`~facemap_rois.ROI`.
    """
    from facemap import process

    savepath = pathlib.Path(savepath)
    savepath.mkdir(parents=True, exist_ok=True)

    if auto_roi:
        # Detect the motion-energy centroid and place a fixed-size box centred
        # on it.  The box dimensions are taken from the per-camera fixed default
        # (scaled to the actual frame size if it differs from the reference).
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(str(video_path))
        frame_w = int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
        _cap.release()

        fixed = facemap_rois.default_roi(camera)
        ref_w, ref_h = facemap_rois.REFERENCE_FRAME_SIZE
        target_w = max(facemap_rois.MIN_ROI_SIDE,
                       int(round(fixed.w * frame_w / ref_w)))
        target_h = max(facemap_rois.MIN_ROI_SIDE,
                       int(round(fixed.h * frame_h / ref_h)))

        roi_yrange, roi_xrange = _bmp.detect_motion_roi(
            str(video_path), roi_w=target_w, roi_h=target_h
        )
        y0, y1 = roi_yrange
        x0, x1 = roi_xrange
        roi = facemap_rois.ROI(x0, y0, x1 - x0, y1 - y0)
        print(f"  [facemap] {camera}: AUTO ROI {roi.as_dict()} "
              f"(fixed size {target_w}×{target_h}, centred on motion) "
              f"sbin={sbin} → {savepath}", flush=True)
        proc_in = _bmp.build_minimal_proc(str(video_path), str(savepath), sbin=sbin,
                                           roi_yrange=roi_yrange, roi_xrange=roi_xrange)
    else:
        proc_in = build_proc(roi, savepath, sbin=sbin)
        print(f"  [facemap] {camera}: ROI {roi.as_dict()} sbin={sbin} -> {savepath}", flush=True)
    start = time.time()
    proc_path = process.run([[str(video_path)]], motSVD=True, movSVD=False,
                            proc=proc_in, savepath=str(savepath))
    runtime = time.time() - start
    # facemap names its output after the input file; with a trimmed copy that would be
    # `trimmed_<original>_proc.npy`. Rename to the camera so outputs are predictable.
    proc_path = pathlib.Path(proc_path)
    camera_named = proc_path.with_name(f"{camera}_proc.npy")
    proc_path.replace(camera_named)
    proc_path = camera_named
    proc = np.load(proc_path, allow_pickle=True).item()

    if len(proc["motion"]) < 2:
        raise RuntimeError(
            f"facemap returned no ROI motion for {camera} "
            f"(got {len(proc['motion'])} entries; expected a placeholder + 1 ROI)")

    motion_sum = np.asarray(proc["motion"][1], dtype=np.float64)
    fitted = proc["rois"][0]
    y0, x0 = int(fitted["yrange_bin"][0]), int(fitted["xrange_bin"][0])
    effective = facemap_rois.ROI(x0 * sbin, y0 * sbin,
                                 int(fitted["xrange_bin"].size) * sbin,
                                 int(fitted["yrange_bin"].size) * sbin)
    n_pixels = int(fitted["yrange_bin"].size * fitted["xrange_bin"].size)

    motion = motion_trace.repair_motion_alignment(motion_sum) / max(n_pixels, 1)
    print(f"  [facemap] {camera}: {motion_sum.size} frames in {runtime / 60:.1f} min "
          f"({motion_sum.size / max(runtime, 1e-9):.0f} fps), "
          f"effective ROI {effective.as_dict()} ({n_pixels} px)", flush=True)

    return FacemapResult(
        camera=camera,
        video_path=pathlib.Path(video_path),
        proc_path=pathlib.Path(proc_path),
        n_frames=int(motion_sum.size),
        roi=roi,
        effective_roi=effective,
        motion_sum=motion_sum,
        motion=motion,
        motsvd=np.asarray(proc["motSVD"][1], dtype=np.float32),
        motmask=np.asarray(proc["motMask_reshape"][1], dtype=np.float32),
        singular_values=np.asarray(proc["motSv"], dtype=np.float64),
        avgframe=np.asarray(proc["avgframe_reshape"], dtype=np.float32),
        avgmotion=np.asarray(proc["avgmotion_reshape"], dtype=np.float32),
        sbin=int(sbin),
        runtime_s=float(runtime),
    )
