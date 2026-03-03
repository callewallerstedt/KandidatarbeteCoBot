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

## Unity setup (exact)
Copy scripts from `unity-scripts/` into your Unity project:
- `GripAnnotatable.cs`
- `ObjectBoxRandomizer.cs`
- `GripPoseExporter.cs`
- `DatasetCaptureController.cs`

### Scene wiring
1. Add `ObjectBoxRandomizer` to an empty GameObject (e.g. `RandomizerRoot`).
   - Set `boxCenter` + `boxSize` to your bin region.
   - Add all target object transforms in `objects`.
2. On each target object, add `GripAnnotatable`.
   - Set `classId`
   - Assign `centerPoint`, `gripPointA`, `gripPointB` transforms.
   - (Optional) set explicit renderers.
3. Add `GripPoseExporter` to another GameObject.
   - Assign `renderCamera` (cobot head-camera simulation).
   - Add all `GripAnnotatable` objects to `objects` list.
   - Set `outputRoot` (e.g. `D:/unity_export`).
4. Add `DatasetCaptureController`.
   - Link `randomizer` + `exporter`.
   - Set `framesToCapture`.
   - Click `Start Capture` from Inspector context menu (or enable `autoStart`).

5. (Optional, recommended) Add `UnityCommandBridge` for remote control from Tkinter GUI.
   - Link `randomizer` + `exporter`.
   - Set `unityExportRoot` to same folder used by the GUI.
   - It polls `<unityExportRoot>/_commands/next_command.json`.

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

Then in this lab GUI:
1. Tab 1: choose Unity export folder and convert
2. Tab 2: train pose model
3. Tab 3: run inference (grip line + angle)
