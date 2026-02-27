# YOLO Segmentation Lab (Tkinter all-in-one)

This folder provides a practical pipeline for your object segmentation project:

1. Input object walkaround video
2. Auto-detect video length and sample N frames evenly + auto masks (pseudo labels)
3. Build YOLO-seg dataset structure
4. Train YOLO-seg model
5. Run inference on webcam or video with mask overlays (optional video save)

## Quick start (Windows student PC)

```powershell
cd yolo-segmentation-lab
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

## Notes
- Auto-labeling is best for controlled single-object videos.
- For best quality, manually review/fix a subset of labels.
- Human classes can be added from external datasets later.

## Main app
`app.py` has tabs for:
- Instructions (step-by-step guide for adding classes, adding more data, and safe retraining)
- Data prep (auto-label from video + split mode: all/real/synth/obs + mask quality selector + optional split-class selector + optional run filter)
- Class IDs are auto-suggested from `dataset.yaml` (reuse existing ID for known class, otherwise pick next free ID)
- When generating labels/data from the GUI, the class is auto-registered into `dataset.yaml` (no manual class-list edit required)
- Data Prep also has `Auto-sync dataset.yaml from data folders` to rebuild/repair class list from `data/images/*` at any time
- Synthetic BG (cut-paste augmentation from your background folder, with browsable preview samples via left/right)
- Multi-Instance Synth (generate multiple same-class objects per image with touching/overlap cases, with browsable preview samples)
- COMBO RUN (run Synthetic BG + Multi-Instance + Obstruction in one go, with per-mode settings and enable toggles)
- Obstruction Data (build on random backgrounds + configurable white-table ratio, overlay hands/arms as occluders, force top-to-center orientation, subtract overlap from object mask, with browsable preview samples)
- Add Masked Object (draw bbox on an image to extract object via GrabCut and save as a masked source sample for later synth generation)
- Manual Real Data (extract evenly sampled frames + edit masks interactively)
- Training (with configurable dataloader workers; on Windows use workers=0 for stability)
- DDP Multi-PC (guided multi-node launch helper for 2+ computers: host checks, rank settings, generated launch commands, local node start)
- Inference (including explicit webcam capture resolution controls, higher default 1920x1080 display, polygon-based high-quality mask overlay, per-instance colors, instance count logging, and optional human arm joint tracking overlay).

## Synthetic run folders
- Synthetic generation now keeps the selected background image proportions (no forced crop to source image aspect ratio).
- New synthetic and obstruction generations are stored in per-run folders:
  - `data/images/<class>/synth_runs/<run_name>/`
  - `data/images/<class>/obs_runs/<run_name>/`
  - matching label folders under `data/labels/...`
- Build split can filter to a specific run via run filter. For COMBO RUN, use the base name (e.g. `combo01`) to include `combo01_bg`, `combo01_multi`, `combo01_obs`.
- Synthetic BG tab includes separate brightness min/max controls for background and masked object.

## Manual reviewer usability
- Manual prep can use YOLO weights for better initial masks (`source=yolo`) or rembg fallback.
- You can choose init weights, confidence, image size, and device from the GUI manual tab (same style as inference settings).
- Drawing is smoother (line interpolation instead of sparse dots).
- Zoom controls added (`z` in, `x` out).

## DDP Multi-PC notes
- Use same commit + same dataset on all nodes.
- In DDP tab, set identical `master addr`, `master port`, and `nnodes` on all computers.
- Set `node rank` uniquely per PC (0..nnodes-1).
- Start rank 0 first, then rank 1/2/...
- Script used by DDP launcher: `scripts/train_yolo_seg_ddp.py`

## Incremental multiclass workflow
- You can add one object class at a time.
- Keep class IDs stable and update `dataset.yaml` class list in ID order.
- Rebuild split with `all` and continue training from latest `runs/.../best.pt` to avoid forgetting old classes.

## Output structure
- `data/images/<class>/` extracted frames
- `data/labels/<class>/` YOLO-seg labels
- `data/yolo_dataset/` train/val/test split
- `runs/segment/` training results
- `runs/predict/` inference outputs
