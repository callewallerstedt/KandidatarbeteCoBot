# Cobot Grip Pose Lab (YOLO Pose)

Separate pipeline for cobot head-camera grasp keypoints.

## Goal
Predict per-object grip direction with 3 keypoints:
1. center
2. grip_A
3. grip_B

## Workflow
1. Export Unity dataset (RGB + JSON annotations)
2. Convert JSON -> YOLO Pose labels
3. Build train/val split
4. Train YOLO pose model (`yolo11s-pose.pt` recommended)
5. Run inference overlay and export grip angle/points

## Quick start
```bash
cd cobot-grip-pose-lab
python3 -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Expected Unity export layout
```
unity_export/
  RGB/
    frame_000001.png
  annotations/
    frame_000001.json
```

JSON per frame (example):
```json
{
  "image": "frame_000001.png",
  "width": 1920,
  "height": 1080,
  "objects": [
    {
      "class_id": 0,
      "bbox_xyxy": [x1,y1,x2,y2],
      "center": [x,y,2],
      "grip_a": [x,y,2],
      "grip_b": [x,y,2]
    }
  ]
}
```

Visibility flag: `0/1/2` (YOLO pose style).
