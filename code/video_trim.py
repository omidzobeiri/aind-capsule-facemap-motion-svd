"""Detect and remove the acquisition preamble at the start of an MVR video.

Every camera's recording opens with material that is not behaviour:

* **frame 0 is a near-white metadata frame** with the MovieID, pixel format and
  codec burned into the image (mean luminance 252-255 on all three cameras of
  `multiplane-ophys_786297_2025-05-12_09-46-40`), and
* on cameras whose `CustomInitialExposureTime` differs from their steady
  `ExposureTime`, the next ~10-13 frames are captured at the initial exposure and
  then step to the running one (Nose: mean 196 -> 71 at frame 12; Behavior: 13 ->
  36 at frame 13).

Both produce frame differences 27-519x the steady-state motion energy. That matters
more than its duration suggests: `facemap.process` always starts its first
subsample chunk at frame 0, so these few frames enter the matrix the motion masks
are fit to, and a whole-ROI brightness step is exactly the kind of structure an SVD
will spend a component on. Masking the samples afterwards fixes the trace but not
the basis.

So the preamble is removed *before* facemap sees the video, by **remuxing** (stream
copy -- no re-encode, no pixel change) from the first keyframe at or after the
settle point into a scratch file. Trimming has to land on a keyframe, so with this
rig's 250-frame GOP the cost is the first 250 frames (4.2 s at 60 fps) of an ~80
minute recording, before the experiment starts. The number of dropped frames is
recorded and applied to the sync alignment, so frame indices in the outputs still
refer to the original video.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np

PROBE_FRAMES = 90
"""Frames read when locating the settle point."""
REFERENCE_WINDOW = 30
"""Trailing frames of the probe used as the steady-state brightness reference."""
BRIGHTNESS_TOLERANCE = 0.15
"""Relative brightness deviation that still counts as 'not settled yet'."""
MAX_TRIM_FRAMES = 2000
"""Never drop more than this; beyond it something is wrong and we would rather not."""


def detect_settle_frame(video_path: str | pathlib.Path, mvr: dict | None = None) -> dict:
    """First frame whose exposure/brightness matches the steady state.

    Combines the measurement with what the MVR sidecar declares
    (`CustomInitialNumberOfFrames`), taking whichever is later.
    """
    capture = cv2.VideoCapture(str(video_path))
    brightness: list[float] = []
    try:
        for _ in range(PROBE_FRAMES):
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            brightness.append(float(gray.mean()))
    finally:
        capture.release()

    info: dict = {"probe_brightness_first10": [round(b, 1) for b in brightness[:10]]}
    if len(brightness) < REFERENCE_WINDOW + 2:
        info |= {"settle_frame": 0, "reason": "video too short to probe"}
        return info

    values = np.asarray(brightness)
    reference = float(np.median(values[-REFERENCE_WINDOW:]))
    deviation = np.abs(values - reference) / max(reference, 1.0)
    unsettled = np.flatnonzero(deviation > BRIGHTNESS_TOLERANCE)
    measured = int(unsettled[-1]) + 1 if unsettled.size else 0

    declared = int(mvr.get("custom_initial_number_of_frames") or 0) if mvr else 0
    info |= {
        "settle_frame": int(max(measured, declared)),
        "measured_settle_frame": measured,
        "declared_initial_frames": declared,
        "steady_brightness": round(reference, 1),
    }
    return info


def keyframe_at_or_after(frame_index: int, video_path: str | pathlib.Path,
                         keyframes: np.ndarray | None = None) -> int | None:
    """First keyframe index >= `frame_index` (trimming must start on a keyframe)."""
    if frame_index <= 0:
        return 0
    if keyframes is None:
        from motion_trace import keyframe_indices  # noqa: PLC0415

        keyframes, _ = keyframe_indices(video_path, stop_after=MAX_TRIM_FRAMES)
    if keyframes is None or not len(keyframes):
        return None
    later = np.asarray(keyframes)[np.asarray(keyframes) >= frame_index]
    return int(later[0]) if later.size else None


def remux_from(video_path: str | pathlib.Path, start_frame: int,
               out_path: str | pathlib.Path) -> pathlib.Path:
    """Copy the video from `start_frame` (a keyframe) onwards without re-encoding."""
    import av

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as inp, av.open(str(out_path), "w") as out:
        in_stream = inp.streams.video[0]
        in_stream.thread_type = "AUTO"
        # add_stream_from_template exists from PyAV 14; PyAV 13 (pinned) takes template=
        out_stream = (out.add_stream_from_template(in_stream)
                      if hasattr(out, "add_stream_from_template")
                      else out.add_stream(template=in_stream))
        index, written = 0, 0
        for packet in inp.demux(in_stream):
            if packet.dts is None:
                continue
            if index >= start_frame:
                packet.stream = out_stream
                out.mux(packet)
                written += 1
            index += 1
    print(f"  [trim] wrote {out_path.name}: {written} packets "
          f"(dropped the first {start_frame})", flush=True)
    return out_path


def trim_preamble(video_path: pathlib.Path, scratch_dir: pathlib.Path,
                  mvr: dict | None = None, mode: str = "auto") -> tuple[pathlib.Path, dict]:
    """Return `(video_to_process, info)`; `info['frames_dropped']` offsets the sync.

    `mode` is `"auto"` (detect), `"0"`/`"none"` (leave the video alone) or an explicit
    frame count. Any failure falls back to the original video with the reason
    recorded -- a preamble is a QC problem, not a reason to lose the session.
    """
    info: dict = {"mode": mode, "frames_dropped": 0, "trimmed": False}
    if mode in ("0", "none", ""):
        info["reason"] = "trimming disabled"
        return video_path, info

    if mode == "auto":
        detected = detect_settle_frame(video_path, mvr)
        info |= detected
        target = int(detected.get("settle_frame", 0))
    else:
        target = int(mode)
        info["settle_frame"] = target
    if target <= 0:
        info["reason"] = "no preamble detected"
        return video_path, info

    start = keyframe_at_or_after(target, video_path)
    if start is None:
        info["reason"] = "could not read keyframes; video left untrimmed"
        print(f"  [trim] WARNING {info['reason']} -- the motion SVD basis will include "
              f"the first {target} preamble frames", flush=True)
        return video_path, info
    if start > MAX_TRIM_FRAMES:
        info["reason"] = f"first keyframe after the preamble is at {start} (> {MAX_TRIM_FRAMES})"
        print(f"  [trim] WARNING {info['reason']}; video left untrimmed", flush=True)
        return video_path, info
    if start == 0:
        info["reason"] = "preamble already excluded"
        return video_path, info

    try:
        trimmed = remux_from(video_path, start, scratch_dir / f"trimmed_{video_path.name}")
    except Exception as e:  # noqa: BLE001
        info["reason"] = f"remux failed ({e}); video left untrimmed"
        print(f"  [trim] WARNING {info['reason']}", flush=True)
        return video_path, info
    info |= {"frames_dropped": int(start), "trimmed": True, "trimmed_path": str(trimmed)}
    return trimmed, info
