"""Camera frame timestamps from the AIBS/AIND sync file.

Generalised from the eye-tracking capsule's `generate_eye_tracking_table.py`: the
same h5py-only reader, but per-camera line labels and -- importantly -- candidate
lines are checked for *pulses*, not just for presence in `line_labels`.

That check is not defensive programming. On
`multiplane-ophys_786297_2025-05-12_09-46-40` the label `nose_cam_exposing` exists
(bit 30) and carries **zero** rising edges, while `nose_cam_frame_readout` carries
288,002 -- one per frame. Selecting a line by label alone silently yields an empty
timestamp array for the Nose camera.
"""
from __future__ import annotations

import ast
import pathlib
from typing import Sequence

import h5py
import numpy as np

CAMERA_SYNC_KEYS: dict[str, tuple[str, ...]] = {
    "Face": ("face_cam_exposing", "face_cam_frame_readout", "face_cam", "facetracking"),
    "Nose": ("nose_cam_exposing", "nose_cam_frame_readout", "nose_cam"),
    "Behavior": ("beh_cam_exposing", "beh_cam_frame_readout", "behavior_cam_exposing",
                 "behavior_cam_frame_readout", "beh_cam", "behavior_cam"),
    "Eye": ("eye_cam_exposing", "eye_cam_frame_readout", "eye_frame_received",
            "cam2_exposure", "eyetracking", "eye_tracking"),
}
"""Candidate sync-line labels per camera, most-preferred first."""

MIN_PULSE_FRACTION = 0.5
"""A candidate line must carry at least this fraction of the video's frame count."""


def _decode_labels(values) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _read_metadata(sync_hdf: h5py.File) -> dict:
    raw = sync_hdf["meta"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return ast.literal_eval(raw)


def _unwrap_counter(counter: np.ndarray) -> np.ndarray:
    """Correct rollover in the sync file's uint32 sample counter."""
    counter = counter.astype(np.int64, copy=True)
    for index in np.flatnonzero(np.diff(counter) < 0) + 1:
        counter[index:] += 2**32
    return counter


def find_sync_file(data_path: str | pathlib.Path = "/data",
                   video_path: str | pathlib.Path | None = None) -> pathlib.Path:
    """Locate the sync h5 in the `behavior/` folder of the video's raw data asset."""
    search_root = pathlib.Path(data_path)
    if video_path is not None:
        video_path = pathlib.Path(video_path).resolve()
        raw_asset = next((p for p in video_path.parents if (p / "behavior").is_dir()), None)
        if raw_asset is None:
            raise FileNotFoundError(
                f"no raw data asset with a behavior/ folder above {video_path}")
        search_root = raw_asset
        globs = ("behavior/*_sync.h5", "behavior/*.h5")
    else:
        globs = ("*/behavior/*_sync.h5", "*/behavior/*.h5")
    for pattern in globs:
        candidates = sorted(search_root.glob(pattern))
        if candidates:
            if len(candidates) > 1:
                raise RuntimeError(
                    f"expected one sync file under {search_root}/behavior/, "
                    f"found {len(candidates)}: {candidates}")
            return candidates[0]
    raise FileNotFoundError(f"no sync HDF5 file found under {search_root}/behavior/")


def rising_edges(sync_file: str | pathlib.Path, label: str) -> np.ndarray:
    """Rising-edge times (seconds) of one named sync line."""
    with h5py.File(sync_file, "r") as hdf:
        metadata = _read_metadata(hdf)
        labels = metadata["line_labels"]
        if label not in labels:
            raise KeyError(f"{label!r} not in sync line labels: {labels}")
        bit = np.uint64(labels.index(label))
        data = np.asarray(hdf["data"][:])
    counter = _unwrap_counter(data[:, 0])
    state = ((data[:, -1].astype(np.uint64) >> bit) & np.uint64(1)).astype(np.int8)
    edges = np.flatnonzero(np.diff(state, prepend=np.int8(0)) == 1)
    daq = metadata["ni_daq"]
    freq = daq.get("sample_freq", daq.get("counter_output_freq", daq.get("sample_rate")))
    if freq is None:
        raise KeyError("no sample frequency found in sync metadata")
    return counter[edges] / float(freq)


def camera_frame_times(sync_file: str | pathlib.Path, camera: str,
                       n_frames: int | None = None,
                       candidates: Sequence[str] | None = None) -> tuple[np.ndarray, str]:
    """Rising-edge times for `camera`, choosing the first candidate line that is both
    present *and* carries a plausible number of pulses.

    Returns `(times, label_used)`. Raises if no candidate line qualifies.
    """
    candidates = tuple(candidates or CAMERA_SYNC_KEYS.get(camera, ()))
    if not candidates:
        raise KeyError(f"no sync line candidates configured for camera {camera!r}")
    with h5py.File(sync_file, "r") as hdf:
        present = set(_read_metadata(hdf)["line_labels"])
    minimum = int(MIN_PULSE_FRACTION * n_frames) if n_frames else 1
    tried: list[str] = []
    for label in candidates:
        if label not in present:
            continue
        times = rising_edges(sync_file, label)
        tried.append(f"{label}({times.size})")
        if times.size >= minimum:
            if tried[:-1]:
                print(f"  [sync] {camera}: using {label!r} "
                      f"(skipped {', '.join(tried[:-1])} -- too few pulses)", flush=True)
            return times, label
    raise KeyError(
        f"no usable sync line for {camera}: candidates={candidates}, "
        f"tried={tried or 'none present'}, required>={minimum} pulses")


def align_frame_times(times: np.ndarray, n_frames: int,
                      offset: int = 0) -> tuple[np.ndarray, dict]:
    """Pair `n_frames` video frames with sync pulses, 1-to-1 from pulse `offset`.

    `offset` is the number of leading frames dropped from the video before
    processing (see `video_trim`); frame `i` of the processed video is frame
    `offset + i` of the original, so it pairs with pulse `offset + i`.

    The cameras on this rig emit one pulse per captured frame; the sync file
    typically holds 1-2 more pulses than the video has frames (acquisition is
    stopped between a pulse and its frame being written). Extra trailing pulses are
    dropped. If pulses are *missing* the trace is padded with NaN timestamps rather
    than silently shifting every frame, and the shortfall is reported so QC fails.
    """
    times = np.asarray(times, dtype=float)
    info = {"n_pulses": int(times.size), "n_frames": int(n_frames),
            "frame_offset": int(offset),
            "n_pulses_minus_frames": int(times.size - n_frames - offset)}
    times = times[offset:] if offset else times
    if times.size >= n_frames:
        out = times[:n_frames]
        info["n_frames_without_timestamp"] = 0
    else:
        out = np.full(n_frames, np.nan)
        out[:times.size] = times
        info["n_frames_without_timestamp"] = int(n_frames - times.size)
        print(f"  [sync] WARNING only {times.size} pulses for {n_frames} frames -- "
              f"{n_frames - times.size} frames have no timestamp", flush=True)
    finite = np.isfinite(out)
    if finite.sum() > 1:
        deltas = np.diff(out[finite])
        info["median_interval_s"] = float(np.median(deltas))
        info["estimated_fps"] = float(1.0 / np.median(deltas)) if np.median(deltas) else None
        info["max_interval_s"] = float(deltas.max())
        info["n_intervals_over_2x_median"] = int((deltas > 2 * np.median(deltas)).sum())
    return out, info
