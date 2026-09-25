"""Repair and clean facemap's ROI motion-energy trace.

Two separate problems are handled here; both were verified against frame-by-frame
`cv2` differences on this rig's videos (see README, "facemap quirks").

1. **Alignment.** `proc['motion'][i]` is not on a single time convention.
   `process_ROIs` diffs the movie in chunks of `CHUNK = 500` frames, carrying the
   last frame of each chunk into the next -- except for the first chunk, which has
   no carry-in. The result is that

       motion[0 : CHUNK-1]  ==  |f(t+1) - f(t)|   (forward difference)
       motion[CHUNK-1]      ==  0                 (never written)
       motion[CHUNK :]      ==  |f(t) - f(t-1)|   (backward difference)

   so the first 500 samples are shifted one frame relative to the rest and there is
   a spurious zero at index 499. `repair_motion_alignment` puts the whole trace on
   the backward convention -- which is also what `motSVD` uses, since facemap pads
   the first chunk's projection by duplicating its first row -- so the motion trace
   and the SVD components can be read on the same time base.

2. **Compression pops.** These videos are H.264 with a 250-frame GOP. Every
   keyframe is quantized differently from the inter-frames around it, so the frame
   difference into (and out of) a keyframe carries a step change that has nothing to
   do with the animal: on this rig it shows up as a >4 sigma spike every 250 frames
   in the raw motion energy. Facemap has no notion of this. The raw trace is never
   altered here -- contaminated samples are flagged in a mask and a separate cleaned
   trace is produced, following `aind-motion-energy`'s convention.
"""
from __future__ import annotations

import pathlib

import numpy as np

CHUNK = 500
"""facemap's `process_ROIs` projection chunk size (nt0). The alignment seam is here."""

INTRA_ONLY_CODECS = frozenset({"mjpeg", "jpeg2000", "jpegls", "rawvideo", "ffv1", "huffyuv"})
"""Codecs where every frame is a keyframe and nothing should be masked."""


def repair_motion_alignment(motion: np.ndarray) -> np.ndarray:
    """Return facemap's ROI motion trace on a consistent backward-difference basis.

    `out[t] = |f(t) - f(t-1)|` for every `t >= 1`; `out[0]` is NaN (there is no
    preceding frame). This also fills facemap's zero hole at index `CHUNK-1`.
    """
    motion = np.asarray(motion, dtype=np.float64)
    n = motion.size
    out = np.full(n, np.nan)
    if n == 0:
        return out
    seam = min(CHUNK, n)
    # forward-differenced head: motion[i] == backward[i+1] for i in 0 .. seam-2
    out[1:seam] = motion[: seam - 1]
    if n > CHUNK:
        out[CHUNK:] = motion[CHUNK:]
    return out


def keyframe_indices(video_path: str | pathlib.Path, stop_after: int | None = None
                     ) -> tuple[np.ndarray | None, str | None]:
    """Frame indices that are keyframes, read from the container with PyAV.

    Reads *packets* rather than decoding frames -- keyframe flags live in the packet
    header, so this costs no pixel decode (~30 s for a 288k-frame video instead of
    ~70 s, and no memory). Packet order is decode order, which only differs from
    presentation order when the stream has B-frames, so that case falls back to
    decoding. `stop_after` stops early when only the first keyframes are needed.

    Returns `(indices, codec_name)`, or `(None, codec)` if PyAV is unavailable or
    the file cannot be parsed -- callers fall back to `infer_keyframe_period`.
    """
    try:
        import av  # noqa: PLC0415  (optional dependency; fallback below)
    except ImportError:
        print("  [keyframes] PyAV not installed; falling back to period inference", flush=True)
        return None, None
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            codec = stream.codec_context.name
            indices: list[int] = []
            if stream.codec_context.has_b_frames:
                stream.thread_type = "AUTO"
                for index, frame in enumerate(container.decode(stream)):
                    if frame.key_frame:
                        indices.append(index)
                    if stop_after is not None and indices and index >= stop_after:
                        break
            else:
                index = 0
                for packet in container.demux(stream):
                    if packet.dts is None:
                        continue
                    if packet.is_keyframe:
                        indices.append(index)
                        if stop_after is not None and index >= stop_after:
                            break
                    index += 1
        return np.asarray(indices, dtype=np.int64), codec
    except Exception as e:  # noqa: BLE001 - never fail the run over a QC input
        print(f"  [keyframes] WARNING could not read keyframes from {video_path} ({e})", flush=True)
        return None, None


def infer_keyframe_period(motion: np.ndarray, min_period: int = 20, max_period: int = 600,
                          min_cycles: int = 8, min_score: float = 1.5,
                          max_samples: int = 20_000) -> tuple[int | None, int, float]:
    """Infer the GOP period and phase from periodic pops in the motion trace.

    Fallback for when the container cannot be parsed (no PyAV, unreadable file).
    Returns `(period, offset, score)`; `period` is None when no periodic
    contamination is detectable.

    For each candidate period the score is the **median** robust z-score of the
    samples at that phase. The median (not the mean) is what makes this work inside
    an ROI: one large bout of animal movement lands on a single cycle and would
    dominate a mean, whereas a compression pop is present on *every* cycle. The
    smallest period reaching 90% of the best score wins, so a true period of 250 is
    not reported as its multiple 500.
    """
    x = np.asarray(motion, dtype=float)[:max_samples]
    finite = np.isfinite(x)
    if finite.sum() < min_cycles * min_period:
        return None, 0, float("nan")
    med = float(np.median(x[finite]))
    mad = float(np.median(np.abs(x[finite] - med))) or float(np.std(x[finite]))
    if not mad:
        return None, 0, float("nan")
    z = (x - med) / (1.4826 * mad)
    scores: dict[int, tuple[float, int]] = {}
    hi = min(max_period, x.size // min_cycles)
    with np.errstate(invalid="ignore"):
        for period in range(min_period, hi + 1):
            best, best_offset = -np.inf, 0
            for offset in range(period):
                s = np.nanmedian(z[offset::period])
                if s > best:
                    best, best_offset = float(s), offset
            scores[period] = (best, best_offset)
    if not scores:
        return None, 0, float("nan")
    top = max(s for s, _ in scores.values())
    if top < min_score:
        return None, 0, top
    for period in sorted(scores):
        score, offset = scores[period]
        if score >= 0.9 * top:
            return period, offset, score
    return None, 0, top


def keyframe_mask(n_frames: int, indices: np.ndarray | None = None,
                  period: int | None = None, offset: int = 0) -> np.ndarray:
    """Bool mask over the backward-differenced trace: True where the sample spans a
    keyframe. Sample `t` is `|f(t) - f(t-1)|`, so it is contaminated when frame `t`
    or frame `t-1` is a keyframe.
    """
    mask = np.zeros(n_frames, dtype=bool)
    if indices is None:
        if not period:
            return mask
        indices = np.arange(offset, n_frames, period, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < n_frames)]
    mask[indices] = True
    nxt = indices + 1
    mask[nxt[nxt < n_frames]] = True
    return mask


def clean_trace(trace: np.ndarray, mask: np.ndarray, method: str = "interpolate") -> np.ndarray:
    """Copy of `trace` with masked samples handled. The input is never modified.

    `"interpolate"` fills masked samples by linear interpolation (continuous trace,
    for regression); `"nan"` leaves gaps; `"none"` returns the trace unchanged.
    """
    out = np.asarray(trace, dtype=float).copy()
    if method == "none" or not mask.any():
        return out
    if method == "nan":
        out[mask] = np.nan
        return out
    if method != "interpolate":
        raise ValueError(f"unknown clean method {method!r} (expected interpolate/nan/none)")
    good = ~mask & np.isfinite(out)
    if good.sum() < 2:
        out[mask] = np.nan
        return out
    idx = np.arange(out.size)
    out[mask] = np.interp(idx[mask], idx[good], out[good])
    return out
