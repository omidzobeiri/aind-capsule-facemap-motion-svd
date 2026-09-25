# aind-capsule-facemap-motion-svd

Runs [`MouseLand/facemap`](https://github.com/MouseLand/facemap)'s **motion SVD** over
a **fixed per-camera ROI** on a session's behaviour videos (Face, Nose, Behavior),
and produces synchronised traces, QC figures and AIND metadata.

Split out of the `lightningPose-eye-tracking` capsule's `facemap-motion-svd` branch
(Code Ocean capsule 6609641, commit `b3a0993`); the eye pipeline lives on in
[`lightningPose-eye-tracking`](https://github.com/AllenNeuralDynamics/lightningPose-eye-tracking).

---

## Why the ROI is fixed

Facemap can compute an SVD of the whole frame ("multivideo SVD"), but on this rig the
frame is mostly apparatus: the head-plate arc, the running-wheel edge, the stimulus
panels and specular patches all move more than the animal does, and a full-frame SVD
spends its leading components on them. Constraining the SVD to a crop that contains
only the animal is the established practice here — see
[`facemap-capsule-fixed-face-rois`](https://github.com/AllenNeuralDynamics/facemap-capsule-fixed-face-rois),
which this capsule takes the approach (and the `proc`-dict mechanics) from.

Fixing the crop *per camera*, rather than fitting it per session, is what makes
sessions comparable: the same rectangle on the same rig views the same body part, so
the ROI-mean motion energy is on the same scale from session to session.

That only holds while the rig geometry does. So every run writes
`qc/{Camera}_roi_placement.png` and a set of placement metrics that answer, from the
session's own average motion map: is there more motion inside the box than outside,
is it centred in the box, and is it pressed against the edge as if the animal were
being clipped. **Look at that figure before trusting a session.** When a session
needs a different box, the App Panel takes a per-camera override.

### Default ROIs

| camera | ROI `x, y, w, h` | covers |
|---|---|---|
| Face | `200, 220, 180, 150` | snout, mouth, both forepaws (excludes the head-plate arc and the bottom apparatus strip) |
| Nose | `110, 110, 240, 210` | whisker pad and snout (excludes the lower-right grooming/paw region) |
| Behavior | `120, 185, 180, 90` | head and shoulder (stops above the running-wheel edge, left of the stimulus panel) |

Derived from average motion-energy maps of
`multiplane-ophys_786297_2025-05-12_09-46-40` (658×492 MVR cameras) at three points
in the session. **They are provisional — validated on one session only.** Re-derive
them across sessions and rigs before treating them as a standard. On a video of a
different size the defaults are scaled proportionally and the run says so loudly.

### ROI modes: `auto`, `aligned`, `default`, `x,y,w,h`

The cameras are not perfectly fixed: across mice and sessions they shift by up to
~100 px and rotate by up to ~4° (Face and Nose move most). Each camera's ROI parameter
therefore takes one of four values:

| value | ROI used |
|---|---|
| `auto` *(default)* | a box of the default size centred on the session's own motion (`build_minimal_proc.detect_motion_roi`) |
| `aligned` | the default box moved to follow the camera, by rigid alignment to a reference session (below) |
| `default` | the built-in box above, as-is |
| `x,y,w,h` | an explicit crop |

`auto` follows wherever the motion is in *this* session; `aligned` follows the rig, so the
box covers the same body part it was drawn on regardless of how much the animal moves.

How `aligned` works:

1. **Session template:** the median of 9 frames spread across the video (5–95%, so the
   preamble is never sampled).
2. **Alignment to `code/alignment_reference/{Camera}.png`:** the median of 25 frames of
   `multiplane-ophys_786297_2025-05-20_09-30-57`. Matching is done on edge maps (log
   intensity → Sobel), with saturated screens and reflections masked out. For Face the
   top 30% of the frame (the objective ring) is also ignored.
   - A coarse search tries rotations from −15° to +15° in 1° steps, with phase
     correlation for the translation, at ¼ resolution.
   - OpenCV ECC (`MOTION_EUCLIDEAN`) then refines the top candidates plus the identity at
     ½ resolution.
   - This takes ~8 s per video, almost all of it decoding the 9 frames.
3. **Moving the ROI:** the default ROI's centre is carried through that rotation +
   translation, and its size is kept.
   - The box stays axis-aligned (facemap needs that), so the rotation itself is not
     followed. `max_corner_error_px` records the resulting error: 1–12 px on the
     validation sessions.
   - At the rotations seen on this rig, the motion-energy trace from the shifted box
     matched the exact rotated box to r = 0.9999.
4. **Guard rails:**
   - An alignment with ECC below `min_ecc` (0.2, in `alignment_reference/reference.json`)
     is rejected and the built-in box is used (`roi.source = aligned_fallback_default`). Validation ECC ranged 0.34–0.83.
   - A frame size different from the reference skips alignment.

Each camera's `qc/{Camera}_metrics.json` → `roi.alignment` records the warp, the shift,
the rotation, ECC, the status and the corner error. `roi.source` is `aligned` when the
alignment was applied. `qc/{Camera}_roi_alignment.png` shows the reference vs session
before and after alignment, with the used box drawn.

The default boxes were drawn on `786297_2025-05-12`, and the alignment reference is the
same mouse on `2025-05-20`. A different session of that mouse (2025-05-13) aligned to the
reference to within 3 px on Behavior and Face.

To change the reference session, rebuild `code/alignment_reference/` with
`make_reference.py` from `behavior_video_alignment` in the oPhys_Transcriptomics capsule.
The same folder holds the validation (one session per mouse, 10 mice).

### Cross-session comparability, precisely

A fixed ROI makes the *pixel space* comparable. It does **not** make
`motsvd_k` comparable: facemap refits the basis on every session, so component
order, sign and rotation are arbitrary. What is comparable across sessions:

* `motion_energy` — the ROI-mean motion energy (this is the scalar to use for
  session-to-session comparisons),
* summary statistics of the spectrum (`variance_fraction_first{k}`).

To get comparable *components* you have to project onto a fixed basis. The masks are
written out (`{Camera}_motion_masks.npy`, pixels × components) so a reference basis
from one session can be applied to others, but this capsule does not do that
projection — it is the natural next phase.

## What the capsule does

1. **Trim the acquisition preamble** (`--trim-leading-frames`, default `auto`).
   Every MVR recording opens with a near-white metadata frame (MovieID, pixel format
   and codec burned into the image) and, on cameras whose `CustomInitialExposureTime`
   differs from their running `ExposureTime`, ~10–13 further frames at the initial
   exposure. The steps out of these are 27–519× the steady-state motion energy, and
   facemap always starts its first subsample chunk at frame 0 — so without trimming,
   a motion mask gets spent on the preamble. The video is **remuxed** (stream copy,
   no re-encode) from the first keyframe after the settle point, which with this
   rig's 250-frame GOP costs the first 250 frames (4.2 s of an ~80 min recording).
2. **Motion SVD** — `facemap.process.run` with `fullSVD=False`, `movSVD=False`,
   `sbin=1`, 500 components, one motion-SVD ROI.
3. **Repair the motion trace** onto a single time convention (see below).
4. **Flag compression pops.** These videos are H.264 with a 250-frame GOP; the
   difference into and out of each keyframe carries a quantisation step that is not
   behaviour. Keyframes are read from the container with PyAV (packet flags, no
   pixel decode), or inferred from the trace's periodicity as a fallback. The raw
   trace is never altered — contaminated samples are flagged, and a separate cleaned
   trace is written (`--keyframe-handling`).
5. **Attach timestamps** from the session sync file, per camera.
6. **QC metrics, checks and figures**, then AIND metadata.

## Inputs

One AIND raw session asset under `/data`, containing

* `behavior-videos/*_{Face,Nose,Behavior}_*.mp4` (plus the MVR `.json` sidecars —
  `FramesRecorded`, `FramesLostCount` and the exposure fields are used by QC), and
* `behavior/*_sync.h5`.

Both AIND video layouts are recognised: a camera-tagged filename
(`<id>_Face_<timestamp>.mp4`) and the nested `<CameraName>/video.mp4` form. A camera
with no video is skipped with a logged warning, so sessions missing a camera still
produce results.

### Sync lines

Candidate lines are tried per camera **and checked for pulses**, not just for
presence in `line_labels`. This matters: in `786297_2025-05-12`, `nose_cam_exposing`
exists but carries **zero** rising edges, while `nose_cam_frame_readout` carries one
per frame. Selecting by label alone yields empty timestamps for the Nose camera.

| camera | candidates, in order |
|---|---|
| Face | `face_cam_exposing`, `face_cam_frame_readout`, … |
| Nose | `nose_cam_exposing`, `nose_cam_frame_readout`, … |
| Behavior | `beh_cam_exposing`, `beh_cam_frame_readout`, … |

Pulses typically outnumber frames by 1–2; the extra trailing pulses are dropped. If
pulses are *missing*, timestamps are padded with NaN (rather than silently shifting
every frame) and `sync_covers_all_frames` fails.

## Outputs (`/results`)

| path | contents |
|---|---|
| `facemap/{Camera}_proc.npy` | facemap's native output, untouched (all 500 components + masks, ~0.6 GB/camera; `--save-proc-npy false` to drop it) |
| `facemap_table.h5` → `{camera}/motion` | `frame` (index, **original** video frame number), `timestamps`, `motion_energy`, `motion_energy_clean`, `motion_energy_sum_raw`, `is_keyframe_contaminated` |
| `facemap_table.h5` → `{camera}/motsvd` | `timestamps` + `motsvd_1..N` (`--svd-components-saved`) |
| `{Camera}_motion_masks.npy` | leading spatial components, `(roi_h, roi_w, N)` |
| `qc/{Camera}_roi_alignment.png` | only for `aligned` ROIs: reference vs session, before/after alignment, used ROI |
| `qc/{Camera}_roi_placement.png` | **the figure to check first** — ROI over the average frame and average motion, with metrics and check results |
| `qc/{Camera}_motion_energy.png` | full trace, a zoom showing the flagged keyframe samples, distribution, per-block drift |
| `qc/{Camera}_svd.png` | spectrum, cumulative variance, leading masks and component traces |
| `qc/{Camera}_metrics.json` | every metric and check for that camera |
| `facemap_qc.png` | session dashboard; the image referenced by `quality_control.json` |
| `processing.json`, `quality_control.json`, `data_description.json`, `session.json`, `subject.json`, `procedures.json` | AIND metadata; written only after every requested camera finishes, so their presence marks a complete run |

`motion_energy` is **per pixel** (facemap's `motion` is a sum over ROI pixels; it is
divided by the pixel count here so ROIs of different sizes are on the same scale).

## QC checks

Each becomes a metric in `quality_control.json` with a pass/fail status. Thresholds
live in `metrics.py` and are recorded in `processing.json`.

| check | fails when |
|---|---|
| `sync_covers_all_frames` | a video frame has no sync pulse |
| `frame_count_matches_recording` | container frame count differs from MVR `FramesRecorded` by > 2 |
| `no_frames_lost_at_acquisition` | MVR reports `FramesLostCount > 0` |
| `acquisition_preamble_excluded` | the metadata/settling frames reached the SVD |
| `roi_contains_the_motion` | in-ROI mean motion < 1.5× out-of-ROI mean |
| `roi_motion_is_centred` | motion centroid > 0.6 of the way to the ROI corner |
| `roi_not_clipping_the_animal` | motion mass in the outer 10% border > 1.4× its area share |
| `roi_not_overexposed` | > 2% of ROI pixels at/above 250 in the average frame |
| `keyframe_contamination_low` | > 5% of samples flagged as compression-contaminated |
| `motion_trace_not_flat` | coefficient of variation < 0.05 (frozen or blank view) |

Plus a human-review checkbox per camera pointing at the placement figure.

## Parameters (App Panel)

| parameter | default | notes |
|---|---|---|
| `cameras` | `Face,Nose,Behavior` | missing cameras are skipped |
| `face-roi` / `nose-roi` / `behavior-roi` | `auto` | `auto` / `aligned` / `default` / `x,y,w,h` (see ROI modes) |
| `sbin` | `2` (App Panel), `1` (command line) | see the facemap quirk below |
| `svd-components-saved` | `100` | of the 500 facemap computes |
| `keyframe-handling` | `interpolate` | `interpolate` / `nan` / `none` |
| `trim-leading-frames` | `auto` | `auto` / `0` / an explicit frame count |
| `save-proc-npy` | `true` | `false` to save ~0.6 GB per camera |

`FACEMAP_ROI_FACE=x,y,w,h` (etc.) is an environment escape hatch for local runs.

Code Ocean passes App Panel values positionally, in the order of `.codeocean/app-panel.json`;
`run_capsule.py` maps them back onto the flags (`APP_PANEL_ORDER`), so keep the two in sync
when adding a parameter.

## facemap quirks this capsule works around

Verified against frame-by-frame `cv2` differences on this rig's videos.

* **The motion trace is not on one time convention.** `process_ROIs` diffs the movie
  in chunks of 500, carrying the last frame of each chunk into the next — except the
  first chunk, which has no carry-in. So `motion[0:499]` is the *forward* difference
  `|f(t+1) − f(t)|`, `motion[499]` is never written (it stays 0), and `motion[500:]`
  is the *backward* difference `|f(t) − f(t−1)|`. `motSVD` is backward-aligned
  throughout. `motion_trace.repair_motion_alignment` puts the whole trace on the
  backward convention, which also fills the zero hole, so the trace and the
  components share a time base. Exact, not approximate: `repaired[1:500] =
  motion[0:499]`, `repaired[500:] = motion[500:]`, `repaired[0] = NaN`.
* **With `fullSVD=False`, index 0 of `motion`/`motSVD`/`motMask` is an empty
  placeholder** for the multivideo SVD; the ROI is at index 1.
* **A `(w, h)` ROI yields an `(h−1, w−1)` mask** at `sbin=1` — facemap builds the
  binned range as `arange(start, stop)` over inclusive endpoints. The effective ROI
  is read back from `rois[0]['yrange_bin']` rather than assumed.
* **`sbin > 1` bins x and y inconsistently** (`floor(x_stop)/sbin` vs
  `floor(y_stop/sbin)` in `process.run`), so the x and y binned ranges can differ by
  one unless the ROI is divisible by `sbin`. Use `sbin=1`.
* **`motSv` is left as zeros** unless the video is long enough for more than one
  subsample chunk (~2000 frames). Variance fractions are therefore computed from the
  projections (always available, full-session) and the singular-value versions are
  reported separately when present.
* **`fullSVD=True` is a memory trap**: the subsample SVD builds an
  `n_pixels × (15 × 500)` matrix — ~10 GB for a full 658×492 frame, ~0.5 GB for a
  typical ROI.

## Runtime and resources

CPU only — nothing here runs on the GPU (torch is installed because facemap's import
chain requires it). Decoding dominates: ~100–160 frames/s per camera in testing, so
roughly 30–50 min per camera for a ~288k-frame (80 min) session, plus ~1 min to remux
the trim. Peak memory is a few GB per camera. `resource_class` is `large`.

## Development

```bash
python run_capsule.py --cameras Face --trim-leading-frames auto
```

Point `utils.DATA_PATH` / `utils.RESULTS_PATH` elsewhere to run outside Code Ocean.
For a fast end-to-end test, remux a couple of thousand frames of each camera into a
session-shaped tree (a stream copy keeps the real GOP, so the keyframe logic is
exercised); note that a short clip will fail `frame_count_matches_recording`, since
the MVR sidecar still describes the full recording.

## Prior art

* [`facemap-capsule-fixed-face-rois`](https://github.com/AllenNeuralDynamics/facemap-capsule-fixed-face-rois) — fixed per-camera ROIs driving `process.run` headless (Ben Hardcastle, Ethan McBride).
* [`facemap-capsule-template`](https://github.com/AllenNeuralDynamics/facemap-capsule-template) — the original facemap install recipe for Code Ocean.
* [`aind-motion-energy`](https://github.com/AllenNeuralDynamics/aind-motion-energy) — AIND's frame-difference library; the keyframe-masking convention here follows it.
* [`lightningPose-eye-tracking`](https://github.com/AllenNeuralDynamics/lightningPose-eye-tracking) — the eye pipeline this capsule replaced; the sync reader and metadata writers are adapted from it.
