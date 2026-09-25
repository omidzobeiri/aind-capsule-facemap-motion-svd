"""Shared helpers: paths, per-camera video discovery, and AIND metadata output."""
from __future__ import annotations

import datetime
import json
import pathlib
import re
from typing import Iterator

import cv2

DATA_PATH = pathlib.Path("/data/")
RESULTS_PATH = pathlib.Path("/results/")

VIDEO_SUFFIXES = (".mp4", ".avi", ".wmv", ".mov", ".mkv")

EXCLUDE_DIR_PATTERNS = (
    "lightningPose-eye-model",
    "lightningPose-training-dataset",
    "universal_eye_tracking",
    "facemap-model",
)
"""Directory-name fragments under /data that are model/training assets, not sessions."""

CODE_URL = "https://github.com/AllenNeuralDynamics/aind-capsule-facemap-motion-svd"
SOFTWARE_VERSION = "0.1.0"
PROCESS_NAME = "facemap"


def camera_pattern(camera: str) -> re.Pattern:
    """Match a camera name as a token in a file stem (`1435931266_Face_2025...`).

    Token-bounded so `Behavior` does not match `behavior-videos` and `Eye` does not
    match an arbitrary substring.
    """
    return re.compile(rf"(?:^|[_\-\s]){re.escape(camera)}(?:$|[_\-\s])", re.IGNORECASE)


def iter_session_videos() -> Iterator[pathlib.Path]:
    """All video files under /data that belong to a session asset."""
    for path in sorted(DATA_PATH.rglob("*")):
        if path.suffix.lower() not in VIDEO_SUFFIXES or not path.is_file():
            continue
        if any(any(pat in part for pat in EXCLUDE_DIR_PATTERNS) for part in path.parts):
            continue
        yield path


def find_camera_video(camera: str) -> pathlib.Path | None:
    """The video for `camera`, or None when the session does not have that camera.

    Supports both AIND layouts: a camera-tagged filename
    (`<id>_Face_<timestamp>.mp4`) and the nested `<CameraName>/video.mp4` form.
    """
    pattern = camera_pattern(camera)
    matches = [p for p in iter_session_videos()
               if pattern.search(p.stem) or (p.stem.lower() == "video"
                                             and pattern.search(p.parent.name))]
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  [video] WARNING {len(matches)} videos match {camera}: "
              f"{[m.name for m in matches]}; using {matches[0].name}", flush=True)
    return matches[0]


def video_properties(video_path: str | pathlib.Path) -> dict:
    """Frame count, size and fps straight from the container."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        props = {
            "n_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        capture.release()
    return props


def mvr_metadata(video_path: str | pathlib.Path) -> dict:
    """The MVR RecordingReport sidecar json next to the video, if present.

    Carries the acquisition-side truth (`FPS`, `FramesRecorded`, `FramesLostCount`,
    `LostFrames`), which is what dropped-frame QC compares the container against.
    """
    sidecar = pathlib.Path(video_path).with_suffix(".json")
    if not sidecar.is_file():
        return {}
    try:
        report = json.loads(sidecar.read_text()).get("RecordingReport", {})
    except Exception as e:  # noqa: BLE001
        print(f"  [video] could not read {sidecar.name}: {e}", flush=True)
        return {}
    return {
        "camera_label": report.get("CameraLabel"),
        "camera_id": report.get("CameraID"),
        "fps": report.get("FPS"),
        "frames_recorded": report.get("FramesRecorded"),
        "frames_lost_count": report.get("FramesLostCount"),
        "lost_frames": report.get("LostFrames"),
        "exposure_time": report.get("ExposureTime"),
        "custom_initial_exposure_time": report.get("CustomInitialExposureTime"),
        "custom_initial_number_of_frames": report.get("CustomInitialNumberOfFrames"),
        "camera_gain": report.get("CameraGain"),
        "image_dimensions": report.get("ImageDimensions"),
        "codec": report.get("Codec"),
        "mvr_version": report.get("MVR Version"),
    }


# ── AIND metadata ───────────────────────────────────────────────────────────────
def session_asset_dirs() -> list[pathlib.Path]:
    return [d for d in sorted(DATA_PATH.glob("*"))
            if d.is_dir() and not any(p in d.name for p in EXCLUDE_DIR_PATTERNS)]


def parse_session_id() -> str:
    """The AIND session id, from the attached raw data asset's directory name."""
    import npc_session

    session_id = None
    for path in session_asset_dirs():
        try:
            session_id = npc_session.parsing.extract_aind_session_id(path.stem)
        except ValueError:
            continue
    if session_id is None:
        raise FileNotFoundError("no data asset attached that follows the aind session format")
    return session_id


def attached_asset_id_for(mount_name: str) -> str | None:
    """Map a data-asset mount name -> Code Ocean asset id via .codeocean/datasets.json.

    Returns None on a local run or for assets attached ad hoc through the API (Code
    Ocean's own computation provenance is authoritative in that case).
    """
    for candidate in (pathlib.Path("/root/capsule/.codeocean/datasets.json"),
                      pathlib.Path(__file__).parent.parent / ".codeocean" / "datasets.json"):
        try:
            if candidate.is_file():
                for dataset in json.loads(candidate.read_text()).get("attached_datasets", []):
                    if dataset.get("mount") == mount_name:
                        return dataset.get("id") or None
        except Exception as e:  # noqa: BLE001
            print(f"[provenance] could not read asset id from {candidate} ({e})", flush=True)
    return None


def copy_session_metadata_jsons(results_dir: pathlib.Path) -> None:
    """Copy the session-level metadata jsons from the raw asset to /results."""
    import shutil

    for name in ("session.json", "subject.json", "procedures.json"):
        matches = [m for m in DATA_PATH.glob(f"*/{name}")
                   if not any(p in m.parent.name for p in EXCLUDE_DIR_PATTERNS)]
        if matches:
            shutil.copy(matches[0].as_posix(), (results_dir / name).as_posix())
        else:
            print(f"No {name} found")


def write_data_description(results_dir: pathlib.Path) -> None:
    try:
        from aind_data_schema.core.data_description import (
            DataDescription, DataLevel, DerivedDataDescription, Funding, Modality,
            Organization, Platform)
        from aind_data_schema_models.pid_names import PIDName
        import npc_session

        session_id = parse_session_id()
        modality = getattr(Modality, "BEHAVIOR_VIDEOS", None) or Modality.POPHYS
        description = DataDescription(
            creation_time=datetime.datetime.now(),
            name=session_id,
            institution=Organization.AIND,
            data_level=DataLevel.DERIVED,
            investigators=[PIDName(name="Unknown")],
            funding_source=[Funding(funder=Organization.AI)],
            modality=[modality],
            platform=Platform.MULTIPLANE_OPHYS,
            subject_id=str(npc_session.SessionRecord(session_id).subject),
        )
        derived = DerivedDataDescription.from_data_description(
            data_description=description, process_name=PROCESS_NAME)
        (results_dir / "data_description.json").write_text(derived.model_dump_json(indent=3))
    except Exception as e:  # noqa: BLE001
        print(f"Could not write data_description.json: {e}")


def write_processing(results_dir: pathlib.Path, camera_results: list[dict],
                     parameters: dict) -> None:
    """One processing.json with a DataProcess per camera processed."""
    try:
        from aind_data_schema.core.processing import (
            DataProcess, PipelineProcess, Processing)

        processes = []
        for result in camera_results:
            camera = result["camera"]
            processes.append(DataProcess(
                name="Other",
                software_version=SOFTWARE_VERSION,
                start_date_time=str(result["start"]),
                end_date_time=str(result["end"]),
                input_location=str(result["video_path"]),
                output_location=RESULTS_PATH.as_posix(),
                code_url=CODE_URL,
                parameters={**parameters, **result.get("parameters", {}), "camera": camera},
                notes=f"facemap motion SVD, {camera} camera, ROI {result['roi']}",
                outputs=result.get("outputs", {}),
            ))
        Processing(processing_pipeline=PipelineProcess(
            data_processes=processes,
            processor_full_name="AIND Scientific Computing")).write_standard_file(results_dir)
    except Exception as e:  # noqa: BLE001
        print(f"Could not write processing.json: {e}")
