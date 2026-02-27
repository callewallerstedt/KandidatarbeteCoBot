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
- Data prep (auto-label from video + split mode: all/real/synth/obs + mask quality selector + optional split-class selector)
- Class IDs are auto-suggested from `dataset.yaml` (reuse existing ID for known class, otherwise pick next free ID)
- Synthetic BG (cut-paste augmentation from your background folder)
- Obstruction Data (build on random backgrounds + configurable white-table ratio, overlay hands/arms as occluders, force top-to-center orientation, subtract overlap from object mask, with live 1-sample preview)
- Manual Real Data (extract evenly sampled frames + edit masks interactively)
- Training
- Inference

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
