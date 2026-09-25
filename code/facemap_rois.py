"""Fixed per-camera motion-SVD ROIs for the AIND multiplane-ophys behavior cameras.

Facemap's motion SVD is computed on a rectangular crop of the frame. The crop is
**fixed per camera** rather than found per session, because the pixel window is what
makes sessions comparable: the same rectangle on the same rig views the same body
part, so the ROI-mean motion energy is on the same scale from session to session.
(The SVD *basis* is still fit per session -- see README, "cross-session comparability".)

A fixed crop only works if the rig geometry is stable, so every run writes a
placement QC figure + metrics (`metrics.roi_placement_metrics`) that show where the
session's motion actually fell relative to the box. Check those before trusting a
session; the App Panel exposes a per-camera override for the cases that need one.

Coordinates are pixels in the raw frame, `(x, y, w, h)` with the origin at top-left,
matching the `--{camera}-roi` App Panel parameters. They are converted to facemap's
`yrange`/`xrange` index arrays by `to_facemap_roi`.

Provenance of the defaults
--------------------------
Derived from average motion-energy maps (|frame diff| averaged over 600-1500 frame
windows at three points in the session) of
`multiplane-ophys_786297_2025-05-12_09-46-40`, 658x492 MVR cameras. They are
deliberately tighter than the animal: each box excludes the head-plate arc, the
running-wheel edge, the stimulus panels and the specular/noise patches, all of which
carry more frame-to-frame signal than the animal does and would otherwise dominate
the SVD. THESE ARE PROVISIONAL -- validated on one session only. Re-derive them
across sessions/rigs before treating them as a standard.
"""
from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np

REFERENCE_FRAME_SIZE = (658, 492)
"""(width, height) the default ROIs were derived on. Other sizes are scaled + warned about."""

MIN_ROI_SIDE = 16
"""Below this a motion SVD is meaningless (and facemap's chunked SVD misbehaves)."""


class ROI(NamedTuple):
    x: int
    y: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    def as_dict(self) -> dict:
        return {"x": int(self.x), "y": int(self.y), "w": int(self.w), "h": int(self.h)}


CAMERA_ROIS: dict[str, ROI] = {
    # snout, mouth and both forepaws; excludes the head-plate arc above (y<215) and
    # the bright apparatus strip along the bottom of the frame (y>400).
    "Face": ROI(200, 220, 180, 150),
    # whisker pad + snout close-up; excludes the paw/grooming region that appears in
    # the lower-right corner and the dark upper-left corner.
    "Nose": ROI(110, 110, 240, 210),
    # head/shoulder of the animal in the side view; deliberately stops above the
    # running-wheel edge (a bright diagonal arc, the single strongest motion source
    # in this view) and to the left of the stimulus panel.
    "Behavior": ROI(120, 185, 180, 90),
}

CAMERAS = tuple(CAMERA_ROIS)
"""Cameras this capsule processes, in output order."""


def parse_roi(value: str | None) -> ROI | None:
    """Parse an `"x,y,w,h"` App Panel override. Empty/None means 'use the default'."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    parts = [p for p in value.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"ROI override must be 'x,y,w,h' (got {value!r})")
    x, y, w, h = (int(round(float(p))) for p in parts)
    return ROI(x, y, w, h)


def default_roi(camera: str) -> ROI:
    return CAMERA_ROIS[camera]


def resolve_roi(camera: str, frame_w: int, frame_h: int,
                override: ROI | None = None) -> tuple[ROI, dict]:
    """Return the ROI to use for `camera` on a `frame_w` x `frame_h` video, plus a
    provenance dict recording how it was obtained (logged to processing.json).

    An override is used verbatim (only clipped to the frame). A default is scaled if
    the video is not the reference size -- proportional scaling is a guess, so it is
    flagged in the provenance and should be checked against the placement QC figure.
    """
    info: dict = {"camera": camera, "frame_width": frame_w, "frame_height": frame_h}
    if override is not None:
        roi, info["source"] = override, "override"
    else:
        roi, info["source"] = default_roi(camera), "default"
        ref_w, ref_h = REFERENCE_FRAME_SIZE
        if (frame_w, frame_h) != (ref_w, ref_h):
            sx, sy = frame_w / ref_w, frame_h / ref_h
            roi = ROI(int(round(roi.x * sx)), int(round(roi.y * sy)),
                      int(round(roi.w * sx)), int(round(roi.h * sy)))
            info["source"] = "default_scaled"
            info["scale_xy"] = [sx, sy]
            print(f"  [roi] WARNING {camera}: video is {frame_w}x{frame_h}, but the default ROI "
                  f"was derived on {ref_w}x{ref_h}. Scaled to {roi.as_dict()} -- CHECK the "
                  f"placement QC figure.", flush=True)

    clipped = clip_to_frame(roi, frame_w, frame_h)
    if clipped != roi:
        print(f"  [roi] WARNING {camera}: ROI {roi.as_dict()} extends outside the frame; "
              f"clipped to {clipped.as_dict()}", flush=True)
        info["clipped_from"] = roi.as_dict()
    if clipped.w < MIN_ROI_SIDE or clipped.h < MIN_ROI_SIDE:
        raise ValueError(
            f"{camera} ROI {clipped.as_dict()} is smaller than {MIN_ROI_SIDE}px on a side "
            f"after clipping to the {frame_w}x{frame_h} frame")
    info["roi"] = clipped.as_dict()
    info["default_roi"] = default_roi(camera).as_dict()
    info["reference_frame_size"] = list(REFERENCE_FRAME_SIZE)
    return clipped, info


def clip_to_frame(roi: ROI, frame_w: int, frame_h: int) -> ROI:
    x = max(0, min(int(roi.x), frame_w - 1))
    y = max(0, min(int(roi.y), frame_h - 1))
    return ROI(x, y, max(0, min(int(roi.w), frame_w - x)), max(0, min(int(roi.h), frame_h - y)))


def to_facemap_roi(roi: ROI, sbin: int = 1) -> dict:
    """Build the facemap ROI dict for a motion-SVD ROI.

    `rind=1` selects "motion SVD"; facemap reads only `rind`, `ivid`, `yrange` and
    `xrange` for this ROI type (`saturation`/`pupil_sigma` are pupil/blink-only).
    Note facemap derives `yrange_bin`/`xrange_bin` as `arange(start//sbin, stop//sbin)`,
    which drops the last row/column -- harmless, but it is why the mask returned for a
    (w, h) ROI is (h-1, w-1) when sbin == 1.
    """
    return {
        "rind": 1,
        "rtype": "motion SVD",
        "iROI": 0,
        "ivid": 0,
        "yrange": np.arange(roi.y, roi.y1),
        "xrange": np.arange(roi.x, roi.x1),
    }


def roi_from_env(camera: str) -> ROI | None:
    """`FACEMAP_ROI_FACE=x,y,w,h` etc. -- an escape hatch for local runs."""
    return parse_roi(os.environ.get(f"FACEMAP_ROI_{camera.upper()}"))
