# KandidatarbeteCoBot

This repository contains experimental tools for a cobot vision pipeline: generating or capturing training data, training YOLO segmentation and pose models, testing Unity camera streams, and converting Gaussian splat assets. The code is organized as separate labs rather than one single application.

## Repository structure

```text
.
|-- yolo-segmentation-lab/          # Main YOLO segmentation dataset, training, and inference GUI
|-- cobot-grip-pose-lab/            # YOLO pose pipeline for grip/keypoint prediction
|-- unity-python-camera-stream-test/# Unity camera TCP streaming and Python calibration/tracking
`-- splat-transform-lab/            # LCC/PLY conversion tools for Gaussian splat experiments
```

## Main workflow

The intended pipeline is:

1. Capture or generate object images.
   - Use real object videos, Unity RGB/mask exports, or synthetic cut-paste generation.
2. Convert the data into YOLO labels.
   - Segmentation labels are stored as YOLO polygon masks.
   - Grip pose labels use three keypoints: center, grip_A, and grip_B.
3. Build a train/validation/test split.
4. Train a YOLO model.
   - Segmentation uses YOLO segmentation models such as `yolo11n-seg.pt`.
   - Grip pose uses YOLO pose models such as `yolo11s-pose.pt`.
5. Run inference on webcam, video, saved images, or a Unity TCP stream.
6. Use the predicted masks or keypoints to estimate object location, orientation, and possible grip direction.

## Prerequisites

- Python 3.11 is recommended.
- A CUDA-capable GPU is useful for training but not required for all tools.
- Unity is needed for the Unity capture and camera streaming workflows.
- Node.js is needed only for `splat-transform-lab`.

Each lab has its own `requirements.txt` or setup instructions. Install dependencies inside the lab folder you are using.

## YOLO segmentation lab

Folder: `yolo-segmentation-lab/`

This is the largest part of the repo. It provides a Tkinter GUI around the scripts used to prepare segmentation data, generate synthetic samples, train YOLO segmentation models, and run inference.

Start it on Windows:

```powershell
cd yolo-segmentation-lab
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

The GUI tabs cover:

- data preparation from object walkaround videos
- importing Unity RGB and red-mask exports
- building train/val/test splits
- synthetic cut-paste background generation
- multi-object synthetic scenes
- obstruction data with hands/arms over objects
- manual mask review and repair
- YOLO segmentation training
- multi-PC distributed training helpers
- inference with mask overlays
- a combined Unity head-camera segmentation and pose workflow

Important generated paths:

```text
yolo-segmentation-lab/data/images/<class>/      # source and generated images
yolo-segmentation-lab/data/labels/<class>/      # matching YOLO labels
yolo-segmentation-lab/data/yolo_dataset/        # train/val/test split
yolo-segmentation-lab/runs/segment/             # training outputs and weights
yolo-segmentation-lab/runs/predict/             # inference outputs
```

The root segmentation config is `yolo-segmentation-lab/dataset.yaml`. The GUI can update this file when new classes are added.

Useful script entry points:

```powershell
python scripts/video_to_yoloseg_autolabel.py --video object.mp4 --class-name part --class-id 0
python scripts/build_dataset_split.py --mode all
python scripts/train_yolo_seg.py --model yolo11n-seg.pt --epochs 80 --imgsz 640
python scripts/run_inference_overlay.py --weights runs/segment/train/weights/best.pt --source 0
```

## Cobot grip pose lab

Folder: `cobot-grip-pose-lab/`

This lab trains a YOLO pose model to predict three grip-related keypoints for each object:

1. `center`
2. `grip_A`
3. `grip_B`

Start the GUI:

```powershell
cd cobot-grip-pose-lab
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

The workflow is:

1. Copy the scripts from `cobot-grip-pose-lab/unity-scripts/` into a Unity project.
2. Mark objects with `GripAnnotatable`.
3. Use `GripPoseExporter` and `DatasetCaptureController` to export RGB images and JSON annotations.
4. Convert the Unity JSON export to a YOLO pose dataset.
5. Train a YOLO pose model.
6. Run inference to display the predicted grip line and angle.

Command-line equivalents:

```powershell
python scripts/convert_unity_pose_json.py --unity-dir D:\unity_export --out-dir dataset
python scripts/train_yolo_pose.py --model yolo11s-pose.pt --data dataset/dataset.yaml
python scripts/run_inference_pose_overlay.py --weights runs/pose/train/weights/best.pt --source 0
```

## Unity camera stream test

Folder: `unity-python-camera-stream-test/`

This lab tests live camera streaming from Unity into Python. Unity sends encoded frames over TCP, and Python receives them with OpenCV.

Core files:

- `CameraTcpStreamer.cs` streams camera frames over TCP.
- `ObjectMaskRed.shader` renders segmentation masks as red objects on a black background.
- `receive_two_cams.py` receives two RGB streams.
- `receive_rgb_and_mask.py` receives two RGB streams and two mask streams.
- `calibrate_and_track_3d.py` provides a calibration and red-dot 3D tracking GUI.

Default ports:

```text
Camera 1 RGB:  5000
Camera 2 RGB:  5001
Camera 1 mask: 6000
Camera 2 mask: 6001
```

On Windows, the `.bat` files create the virtual environment and start the relevant receiver:

```powershell
start_receiver.bat
start_rgb_and_mask_receiver.bat
start_calibration_tracker.bat
```

Run the Python receiver first, then start Unity Play mode.

## Splat transform lab

Folder: `splat-transform-lab/`

This lab is for converting XGRIDS `.lcc` files into more usable formats through PlayCanvas `splat-transform`, and optionally converting PLY point clouds into meshes with Open3D.

Install `splat-transform`:

```powershell
npm install -g @playcanvas/splat-transform
splat-transform --version
```

Start the GUI:

```powershell
cd splat-transform-lab
python lcc_to_ply_gui.py
```

The GUI has two tabs:

- `LCC -> PLY`, backed by `splat-transform`
- `PLY -> Mesh`, backed by `ply_to_mesh_open3d.py`

For mesh conversion, also install:

```powershell
python -m pip install open3d numpy
```

## Tracked outputs and ignored local data

The repository tracks code, READMEs, selected training run metadata, and some model weights under `yolo-segmentation-lab/runs/`.

The `.gitignore` excludes local virtual environments, generated datasets, UI state, downloaded YOLO model files, and heavy visual prediction outputs such as images and videos under `runs/`.

## Notes

- The labs are independent. Create a separate virtual environment inside each lab folder.
- Most GUIs are wrappers around scripts in their local `scripts/` folders, so the same operations can be automated from the command line.
- On Windows, YOLO dataloader workers are often set to `0` for stability.
- Unity export folders are expected to contain matching RGB images, mask images, or annotation JSON files depending on the lab.
