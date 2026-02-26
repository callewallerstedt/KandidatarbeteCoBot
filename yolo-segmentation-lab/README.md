# YOLO Segmentation Lab (Tkinter all-in-one)

This folder provides a practical pipeline for your object segmentation project:

1. Input object walkaround video
2. Auto-detect video length and sample N frames evenly + auto masks (pseudo labels)
3. Build YOLO-seg dataset structure
4. Train YOLO-seg model
5. Run inference on webcam or video with mask overlays

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
- Data prep (auto-label from video)
- Synthetic BG (cut-paste augmentation from your background folder)
- Manual Real Data (extract evenly sampled frames + edit masks interactively)
- Training
- Inference

## Output structure
- `data/images/<class>/` extracted frames
- `data/labels/<class>/` YOLO-seg labels
- `data/yolo_dataset/` train/val/test split
- `runs/segment/` training results
- `runs/predict/` inference outputs
