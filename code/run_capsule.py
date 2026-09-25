"""Facemap capsule entry point.

For the one session attached under /data, runs facemap's motion SVD over a **fixed
per-camera ROI** on each available behaviour video (Face, Nose, Behavior), then:

    1. repairs facemap's motion-energy trace onto a single time convention
    2. flags H.264 keyframe-contaminated samples and writes a cleaned trace
    3. attaches per-frame timestamps from the session's sync file
    4. computes QC metrics + figures, and writes AIND metadata

Outputs (all under /results):

    facemap/{Camera}_proc.npy       facemap's native output, untouched
    facemap_table.h5                per-camera motion + SVD tables, synchronised
    {Camera}_motion_masks.npy       leading spatial components
    qc/{Camera}_*.png, *_metrics.json
    facemap_qc.png                  session dashboard (quality_control.json reference)
    processing.json, quality_control.json, data_description.json, session.json, ...

Metadata is written only after every requested camera has finished, so its presence
in /results marks a complete run.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import pathlib
import sys
import traceback

import numpy as np
import pandas as pd

import facemap_rois
import facemap_runner
import metrics as qc_metrics
import motion_trace
import qc_plots
import roi_alignment
import sync as sync_mod
import utils
import video_trim

RESULTS = utils.RESULTS_PATH
SCRATCH = pathlib.Path("/scratch")
DEFAULT_CAMERAS = ",".join(facemap_rois.CAMERAS)
# App Panel parameter order. Code Ocean passes App Panel values positionally in this
# order (e.g. `run_capsule.py Face,Nose,Behavior auto auto auto 2 100 ...`), so they
# are mapped back onto the flags below; must match .codeocean/app-panel.json.
APP_PANEL_ORDER = ("cameras", "face-roi", "nose-roi", "behavior-roi", "sbin",
                   "svd-components-saved", "keyframe-handling", "trim-leading-frames",
                   "save-proc-npy")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="facemap motion SVD on AIND behaviour videos")
    # nargs="?"/const on the string parameters: the Code Ocean App Panel passes an
    # empty parameter as a bare flag, which plain argparse would reject.
    parser.add_argument("--cameras", nargs="?", const=DEFAULT_CAMERAS, default=DEFAULT_CAMERAS,
                        help="comma-separated cameras to process (default %(default)s); "
                             "a camera with no video in the session is skipped")
    for camera in facemap_rois.CAMERAS:
        parser.add_argument(f"--{camera.lower()}-roi", nargs="?", const="auto", default="auto",
                            help=f"{camera} motion-SVD ROI: 'auto' (default) centres a "
                                 f"default-sized box on the session's motion; 'aligned' moves "
                                 f"the built-in box by the camera's rigid alignment to the "
                                 f"reference session; 'default' uses the built-in box "
                                 f"{tuple(facemap_rois.default_roi(camera))}; or 'x,y,w,h' in "
                                 f"pixels")
    parser.add_argument("--sbin", nargs="?", const=1, type=int, default=1,
                        help="spatial binning before the SVD (default %(default)s). "
                             "Use 1 unless the ROI is large: facemap bins x and y "
                             "inconsistently when the ROI is not divisible by sbin")
    parser.add_argument("--svd-components-saved", nargs="?", const=100, type=int, default=100,
                        help="components written to facemap_table.h5 and the mask npy "
                             "(facemap always computes 500; default %(default)s)")
    parser.add_argument("--keyframe-handling", nargs="?", const="interpolate",
                        default="interpolate", choices=("interpolate", "nan", "none"),
                        help="how the cleaned trace handles compression-contaminated "
                             "samples (default %(default)s); the raw trace is always kept")
    parser.add_argument("--trim-leading-frames", nargs="?", const="auto", default="auto",
                        help="drop the acquisition preamble (white metadata frame + "
                             "exposure settling) before facemap sees the video: 'auto' "
                             "detects it and trims losslessly to the next keyframe, '0' "
                             "disables trimming, or give an explicit frame count "
                             "(default %(default)s)")
    parser.add_argument("--save-proc-npy", nargs="?", const="true", default="true",
                        choices=("true", "false"),
                        help="keep facemap's native _proc.npy (~0.6 GB per camera, holds "
                             "all 500 components and masks; default %(default)s)")
    args, unknown = parser.parse_known_args(argv)
    positional = [u for u in unknown if not u.startswith("--")]
    if positional and len(positional) == len(unknown) and len(positional) <= len(APP_PANEL_ORDER):
        # App Panel run: re-parse the positional values as their flags. An empty value
        # keeps that parameter's default. Explicit flags given alongside still win.
        flags = [a for name, value in zip(APP_PANEL_ORDER, positional) if value.strip()
                 for a in (f"--{name}", value)]
        args = parser.parse_args(flags + [a for a in (argv if argv is not None else sys.argv[1:])
                                          if a not in unknown])
        print(f"[args] App Panel values mapped onto "
              f"{dict(zip(APP_PANEL_ORDER, positional))}", flush=True)
    elif unknown:
        print(f"[args] ignoring unrecognised arguments: {unknown}", flush=True)
    return args


def roi_mode(args: argparse.Namespace, camera: str) -> tuple[str, facemap_rois.ROI | None]:
    """Return ``('auto', None)``, ``('aligned', None)`` or ``('fixed', roi)``.

    'auto'    → box of the default size centred on the session's motion (build_minimal_proc).
    'aligned' → built-in box moved by the camera's rigid alignment to the reference session
                (roi_alignment); falls back to the built-in box if the alignment is unusable.
    'fixed'   → 'default' (built-in box) or an explicit 'x,y,w,h' crop.
    """
    raw = (getattr(args, f"{camera.lower()}_roi", "auto") or "auto").strip()
    if raw.lower() == "auto":
        return "auto", None
    if raw.lower() == "aligned":
        return "aligned", None
    if raw.lower() == "default":
        return "fixed", None   # resolve_roi's default path (scales to non-reference sizes)
    roi = facemap_rois.parse_roi(raw) or facemap_rois.roi_from_env(camera)
    return "fixed", roi


def keyframe_flags(video_path: pathlib.Path, motion: np.ndarray,
                   frame_offset: int = 0) -> tuple[np.ndarray, dict]:
    """Mask of motion samples that span an H.264 keyframe, plus how it was obtained.

    Keyframes are read from the *original* video, so `frame_offset` (the frames
    dropped by trimming) is subtracted to bring them into the processed video's
    frame numbering. The trim always lands on a keyframe, so the shift is exact.
    """
    indices, codec = motion_trace.keyframe_indices(video_path)
    info: dict = {"codec": codec, "frame_offset": int(frame_offset)}
    if codec in motion_trace.INTRA_ONLY_CODECS:
        info |= {"source": "intra_only_codec", "masked_fraction": 0.0, "n_masked": 0}
        return np.zeros(motion.size, dtype=bool), info
    if indices is not None and indices.size:
        gaps = np.diff(indices)
        if frame_offset:
            indices = indices[indices >= frame_offset] - frame_offset
        mask = motion_trace.keyframe_mask(motion.size, indices=indices)
        info |= {"source": "container", "n_keyframes": int(indices.size),
                 "median_gop": int(np.median(gaps)) if gaps.size else None}
    else:
        period, offset, score = motion_trace.infer_keyframe_period(motion)
        mask = motion_trace.keyframe_mask(motion.size, period=period, offset=offset)
        info |= {"source": "inferred" if period else "none",
                 "inferred_period": period, "inferred_offset": offset,
                 "inferred_score": float(score) if np.isfinite(score) else None}
    info["n_masked"] = int(mask.sum())
    info["masked_fraction"] = float(mask.mean()) if mask.size else 0.0
    return mask, info


def process_camera(camera: str, args: argparse.Namespace, results_dir: pathlib.Path) -> dict:
    """Run and post-process one camera. Returns the entry for the QC/metadata pass."""
    video_path = utils.find_camera_video(camera)
    if video_path is None:
        print(f"\n[{camera}] no video found in the session -- skipping", flush=True)
        return {}
    print(f"\n{'=' * 72}\n{camera}: {video_path.name}\n{'=' * 72}", flush=True)

    start = datetime.datetime.now()
    props = utils.video_properties(video_path)
    if min(props["width"], props["height"], props["n_frames"]) <= 0:
        raise RuntimeError(
            f"{camera}: could not read {video_path} -- the container reports "
            f"{props['width']}x{props['height']} and {props['n_frames']} frames. "
            f"The file is likely truncated or corrupt.")
    mvr = utils.mvr_metadata(video_path)
    fps = float(mvr.get("fps") or props.get("fps") or 60.0)

    mode, roi_val = roi_mode(args, camera)

    if mode == "auto":
        # Auto ROI: build_minimal_proc uses the full video frame.
        roi_info: dict = {
            "camera": camera, "source": "auto",
            "frame_width": props["width"], "frame_height": props["height"],
            "default_roi": facemap_rois.default_roi(camera).as_dict(),
            "reference_frame_size": list(facemap_rois.REFERENCE_FRAME_SIZE),
        }
    elif mode == "aligned":
        aligned, alignment = roi_alignment.align_default_roi(
            camera, video_path, facemap_rois.default_roi(camera), props["width"], props["height"],
            qc_path=results_dir / "qc" / f"{camera}_roi_alignment.png")
        print(f"  [align] {camera}: {alignment['status']}"
              + (f" -- rot={alignment['rot_deg']:.2f} deg dx={alignment['dx']:.1f} "
                 f"dy={alignment['dy']:.1f} ecc={alignment['ecc']:.2f}" if "ecc" in alignment else "")
              + (f" -> ROI {aligned.as_dict()}" if aligned else
                 f" ({alignment.get('reason')}); using the built-in ROI"), flush=True)
        roi, roi_info = facemap_rois.resolve_roi(camera, props["width"], props["height"],
                                                 override=aligned)
        roi_info["source"] = "aligned" if aligned else "aligned_fallback_default"
        roi_info["alignment"] = alignment
    else:
        roi, roi_info = facemap_rois.resolve_roi(camera, props["width"], props["height"],
                                                 override=roi_val)

    scratch = SCRATCH if SCRATCH.is_dir() else results_dir / "scratch"
    video_to_process, trim_info = video_trim.trim_preamble(
        video_path, scratch, mvr=mvr, mode=args.trim_leading_frames)
    dropped = int(trim_info.get("frames_dropped", 0))
    if dropped:
        print(f"  [trim] {camera}: processing from frame {dropped} "
              f"(settle at {trim_info.get('settle_frame')}, trimmed to the next keyframe)",
              flush=True)

    try:
        if mode == "auto":
            result = facemap_runner.run_facemap(camera, video_to_process, roi=None,
                                                savepath=results_dir / "facemap",
                                                sbin=args.sbin, auto_roi=True)
            # Back-fill the resolved full-frame ROI into roi_info after the run.
            roi_info["roi"] = result.roi.as_dict()
            roi = result.roi
        else:
            result = facemap_runner.run_facemap(camera, video_to_process, roi=roi,
                                                savepath=results_dir / "facemap",
                                                sbin=args.sbin)
    finally:
        if trim_info.get("trimmed") and video_to_process != video_path:
            pathlib.Path(video_to_process).unlink(missing_ok=True)
            if scratch != SCRATCH:
                with contextlib.suppress(OSError):
                    scratch.rmdir()   # only succeeds while it is empty

    mask, keyframe_info = keyframe_flags(video_path, result.motion, frame_offset=dropped)
    motion_clean = motion_trace.clean_trace(result.motion, mask, args.keyframe_handling)
    print(f"  [keyframes] {camera}: {keyframe_info['n_masked']} of {result.n_frames} samples "
          f"flagged ({keyframe_info['masked_fraction']:.2%}, source={keyframe_info['source']})",
          flush=True)

    # ── timestamps ──────────────────────────────────────────────────────────────
    timestamps, sync_info = np.full(result.n_frames, np.nan), {}
    try:
        sync_file = sync_mod.find_sync_file(utils.DATA_PATH, video_path)
        times, label = sync_mod.camera_frame_times(sync_file, camera,
                                                   result.n_frames + dropped)
        timestamps, sync_info = sync_mod.align_frame_times(times, result.n_frames,
                                                           offset=dropped)
        sync_info |= {"sync_file": str(sync_file), "line_label": label}
        print(f"  [sync] {camera}: {label} -- {sync_info['n_pulses']} pulses for "
              f"{result.n_frames} frames", flush=True)
    except Exception as e:  # noqa: BLE001 - a missing sync file must not lose the SVD
        print(f"  [sync] WARNING {camera}: no timestamps ({e})", flush=True)
        sync_info = {"error": str(e), "n_frames_without_timestamp": result.n_frames}

    # ── metrics ─────────────────────────────────────────────────────────────────
    placement = qc_metrics.roi_placement_metrics(result.avgmotion, result.avgframe,
                                                 result.roi, sbin=result.sbin)
    camera_metrics = {
        "camera": camera,
        "video": {"path": str(video_path), **props, **mvr,
                  "n_frames_processed": result.n_frames,
                  "n_frames_trimmed": dropped},
        "roi": roi_info | {"effective_roi": result.effective_roi.as_dict(),
                           "sbin": result.sbin},
        "roi_placement": placement,
        "trim": trim_info,
        "keyframes": keyframe_info,
        "motion": qc_metrics.motion_metrics(motion_clean, result.motion, mask, fps),
        "svd": qc_metrics.svd_metrics(result.singular_values, result.motsvd),
        "sync": sync_info,
        "runtime_s": result.runtime_s,
    }
    checks = qc_metrics.evaluate(camera_metrics)
    camera_metrics["checks"] = checks
    camera_metrics["status"] = qc_metrics.overall_status(checks)
    print(f"  [qc] {camera}: {camera_metrics['status'].upper()} -- "
          f"{ {k: v['status'] for k, v in checks.items()} }", flush=True)

    write_tables(result, motion_clean, mask, timestamps, results_dir,
                 n_components=args.svd_components_saved, frame_offset=dropped)
    if args.save_proc_npy == "false" and result.proc_path.is_file():
        result.proc_path.unlink()
        print(f"  [outputs] removed {result.proc_path.name} (--save-proc-npy false)", flush=True)

    return {"camera": camera, "result": result, "metrics": camera_metrics,
            "checks": checks, "motion_clean": motion_clean, "keyframe_mask": mask,
            "timestamps": timestamps, "fps": fps, "video_path": video_path,
            "start": start, "end": datetime.datetime.now(), "roi_info": roi_info}


def write_tables(result, motion_clean: np.ndarray, mask: np.ndarray,
                 timestamps: np.ndarray, results_dir: pathlib.Path,
                 n_components: int = 100, frame_offset: int = 0) -> None:
    """Write the synchronised motion table, the SVD table and the leading masks.

    The `frame` index is the frame number in the **original** video, so a trimmed
    preamble shifts the index rather than hiding the offset.
    """
    key = result.camera.lower()
    index = pd.RangeIndex(frame_offset, frame_offset + result.n_frames, name="frame")
    table = pd.DataFrame({
        "timestamps": timestamps,
        "motion_energy": result.motion,
        "motion_energy_clean": motion_clean,
        "motion_energy_sum_raw": result.motion_sum,
        "is_keyframe_contaminated": mask,
    })
    table.index = index
    table.to_hdf(results_dir / "facemap_table.h5", key=f"{key}/motion", mode="a")

    n_keep = int(min(max(n_components, 1), result.motsvd.shape[1]))
    components = pd.DataFrame(
        result.motsvd[:, :n_keep],
        columns=[f"motsvd_{i + 1}" for i in range(n_keep)])
    components.insert(0, "timestamps", timestamps)
    components.index = index
    components.to_hdf(results_dir / "facemap_table.h5", key=f"{key}/motsvd", mode="a")

    np.save(results_dir / f"{result.camera}_motion_masks.npy",
            result.motmask[:, :, :n_keep].astype(np.float32))
    print(f"  [outputs] {result.camera}: facemap_table.h5[{key}/motion, {key}/motsvd] "
          f"({len(table)} frames, {n_keep} components) + "
          f"{result.camera}_motion_masks.npy", flush=True)


def write_qc_json(entries: list[dict], results_dir: pathlib.Path) -> None:
    """quality_control.json: one evaluation per camera, one metric per check, plus a
    human review checkbox against the dashboard image."""
    try:
        from aind_data_schema.core.quality_control import (
            QCEvaluation, QCMetric, QCStatus, QualityControl, Stage, Status)
        from aind_data_schema_models.modalities import Modality
        from aind_qcportal_schema.metric_value import CheckboxMetric

        now = datetime.datetime.now()
        evaluations = []
        for entry in entries:
            camera = entry["camera"]
            qc_metric_list = []
            for name, check in entry["checks"].items():
                qc_metric_list.append(QCMetric(
                    name=f"{camera}: {name.replace('_', ' ')}",
                    description=check.get("note") or name,
                    value={"value": check["value"], "threshold": check["threshold"]},
                    reference=f"qc/{camera}_roi_placement.png",
                    status_history=[QCStatus(
                        evaluator="", timestamp=now,
                        status=Status.PASS if check["status"] == "pass" else Status.FAIL)],
                ))
            qc_metric_list.append(QCMetric(
                name=f"{camera}: ROI placement review",
                description="Does the fixed ROI sit on the animal in this session?",
                reference=f"qc/{camera}_roi_placement.png",
                value=CheckboxMetric(
                    value="Placeholder CheckboxMetric Value",
                    options=["ROI correctly placed", "ROI misplaced -- needs override",
                             "Video too dim / overexposed", "Other issues"],
                    status=[Status.PASS, Status.FAIL, Status.FAIL, Status.FAIL]),
                status_history=[QCStatus(evaluator="", timestamp=now, status=Status.PENDING)],
            ))
            evaluations.append(QCEvaluation(
                name=f"Facemap motion SVD -- {camera} camera",
                description=f"ROI placement, synchronisation and motion-energy quality "
                            f"for the {camera} camera",
                stage=Stage.PROCESSING,
                modality=Modality.from_abbreviation("behavior-videos"),
                notes=f"ROI {tuple(entry['result'].roi)}; "
                      f"source {entry['roi_info'].get('source')}",
                allow_failed_metrics=False,
                metrics=qc_metric_list,
            ))
        QualityControl(
            notes="Dataset-level quality control for facemap motion SVD on behaviour videos",
            evaluations=evaluations,
        ).write_standard_file(results_dir)
    except Exception as e:  # noqa: BLE001
        print(f"Could not write quality_control.json: {e}")
        traceback.print_exc()


def main(argv=None) -> None:
    args = parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "qc").mkdir(exist_ok=True)
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    unknown = [c for c in cameras if c not in facemap_rois.CAMERA_ROIS]
    if unknown:
        raise ValueError(f"no ROI configured for camera(s) {unknown}; "
                         f"known cameras: {list(facemap_rois.CAMERA_ROIS)}")
    print(f"[params] {vars(args)}", flush=True)

    # One bad camera must not cost the others their results: each is isolated, and the
    # run is failed at the very end (after metadata is written) if any of them errored.
    entries, failures = [], {}
    for camera in cameras:
        try:
            entry = process_camera(camera, args, RESULTS)
        except Exception as e:  # noqa: BLE001
            failures[camera] = f"{type(e).__name__}: {e}"
            print(f"\n[{camera}] FAILED -- continuing with the other cameras", flush=True)
            traceback.print_exc()
            continue
        if entry:
            entries.append(entry)
    if not entries:
        raise RuntimeError(
            f"no camera produced results. requested={cameras}, errors={failures or 'none'}, "
            f"videos searched under {utils.DATA_PATH}")

    # ── QC figures ──────────────────────────────────────────────────────────────
    qc_dir = RESULTS / "qc"
    for entry in entries:
        camera, result = entry["camera"], entry["result"]
        try:
            qc_plots.plot_roi_placement(result, entry["metrics"]["roi_placement"],
                                        entry["checks"],
                                        qc_dir / f"{camera}_roi_placement.png")
            qc_plots.plot_motion_energy(result, entry["motion_clean"], entry["keyframe_mask"],
                                        entry["timestamps"], entry["fps"],
                                        qc_dir / f"{camera}_motion_energy.png")
            qc_plots.plot_svd(result, entry["fps"], qc_dir / f"{camera}_svd.png")
        except Exception as e:  # noqa: BLE001 - figures must not lose the data
            print(f"  WARNING: QC figures failed for {camera} ({e})", flush=True)
            traceback.print_exc()
        (qc_dir / f"{camera}_metrics.json").write_text(
            json.dumps(entry["metrics"], indent=4, default=str))
    try:
        qc_plots.build_dashboard([e["result"] for e in entries],
                                 {e["camera"]: e for e in entries},
                                 RESULTS / "facemap_qc.png")
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: QC dashboard failed ({e})", flush=True)
        traceback.print_exc()

    # ── metadata (the success marker) ───────────────────────────────────────────
    print("\nAll cameras complete -- writing metadata.", flush=True)
    utils.copy_session_metadata_jsons(RESULTS)
    utils.write_data_description(RESULTS)
    parameters = {
        "sbin": args.sbin,
        "svd_components_computed": facemap_runner.N_SVD_COMPONENTS,
        "svd_components_saved": args.svd_components_saved,
        "keyframe_handling": args.keyframe_handling,
        "trim_leading_frames": args.trim_leading_frames,
        "fullSVD": False,
        "movSVD": False,
        "qc_thresholds": qc_metrics.thresholds(),
    }
    utils.write_processing(RESULTS, [{
        "camera": e["camera"], "start": e["start"], "end": e["end"],
        "video_path": e["video_path"], "roi": tuple(e["result"].roi),
        "parameters": {"roi": e["roi_info"],
                       "effective_roi": e["result"].effective_roi.as_dict(),
                       "sync_line": e["metrics"]["sync"].get("line_label"),
                       "trim": e["metrics"]["trim"]},
        "outputs": {"proc_npy": e["result"].proc_path.name,
                    "table_keys": [f"{e['camera'].lower()}/motion",
                                   f"{e['camera'].lower()}/motsvd"]},
    } for e in entries], parameters)
    write_qc_json(entries, RESULTS)

    statuses = {e["camera"]: e["metrics"]["status"] for e in entries}
    print(f"Done. QC status: {statuses}", flush=True)
    if failures:
        (RESULTS / "qc" / "camera_errors.json").write_text(json.dumps(failures, indent=4))
        raise RuntimeError(
            f"{len(failures)} camera(s) failed after the others completed: {failures}")


if __name__ == "__main__":
    main()
