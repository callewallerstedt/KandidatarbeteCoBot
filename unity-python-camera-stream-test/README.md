# Unity -> Python Dual Camera Stream Test

This folder contains a simple test setup where two Unity cameras stream JPEG frames over TCP to Python, and Python shows each camera in its own OpenCV window.

## Files

- `CameraTcpStreamer.cs` - Unity script to stream one camera over TCP (JPG/PNG modes).
- `ObjectMaskRed.shader` - replacement shader for pure red object masks.
- `receive_two_cams.py` - basic Python receiver that opens two windows (`Unity Cam 1`, `Unity Cam 2`).
- `receive_rgb_and_mask.py` - Python receiver for 4 streams (2 RGB + 2 masks).
- `calibrate_and_track_3d.py` - Tkinter + OpenCV calibration flow and 3D red-dot tracking.
- `requirements.txt` - Python dependencies.
- `start_receiver.bat` - Windows one-click start script for basic receiver.
- `start_calibration_tracker.bat` - Windows one-click start for calibration + 3D tracker.
- `start_rgb_and_mask_receiver.bat` - Windows one-click start for 2 RGB + 2 mask windows.

## Unity setup (one script for RGB + SEG)

1. In your Unity project, add:
   - `CameraTcpStreamer.cs`
   - `ObjectMaskRed.shader`
2. Create layer `SegmentationMask` and put target objects on that layer.
3. Attach `CameraTcpStreamer.cs` to your two main cameras.
4. Configure camera 1:
   - `host=127.0.0.1`
   - `port=5000` (RGB)
   - `enableMaskStream=true`
   - `maskPort=6000` (SEG)
5. Configure camera 2:
   - `host=127.0.0.1`
   - `port=5001` (RGB)
   - `enableMaskStream=true`
   - `maskPort=6001` (SEG)
6. Recommended start values:
   - `width=1280`, `height=720`
   - `fps=24`
   - `encodeMode=JPG` (RGB stream)
   - `jpegQuality=80`
   - `maxQueueSize=2`

Mask stream is auto-created by the same script and uses lossless PNG with:
- object pixels = pure red `(255,0,0)`
- background = black

## Python setup (Windows)

Double-click:

`start_receiver.bat`

It will:
1. Create `.venv` (first run only)
2. Install dependencies
3. Start `receive_two_cams.py`

Quit with `Q` or `ESC` in an OpenCV window.

## Calibration + 3D tracking mode

Double-click:

`start_calibration_tracker.bat`

This launches:
- two OpenCV stream windows (one per camera), and
- one Tkinter control window.

### Calibration flow (in app)

1. Click **Start Calibration**.
2. App prompts each position in this order: **TOP-LEFT, TOP-RIGHT, BOTTOM-LEFT, BOTTOM-RIGHT** on table.
3. Move the red Unity sphere to the shown position, make sure it's visible in both cameras, then click **Confirm Position**.
4. Repeat same order for in-air points at **z = +1m**.
5. Calibration saves automatically to `calibration_data.json` and tracking starts.

### Coordinate system

- X: width direction of table (0 to 2.5 m)
- Y: depth direction of table (0 to 1.5 m)
- Z: height above table (0 to 1.0 m)

The app tracks the red dot and shows estimated 3D coordinates in the Tkinter window.

### Dataset capture in Tkinter

Use the `N` input + **Take N Pics** button to capture dataset frames.

Output folders (auto-created):
- `images/RGB`  (raw RGB frames, cam1 + cam2)
- `images/SEG`  (segmentation masks, cam1 + cam2 — uses Unity mask streams when available)

## Notes

- Run Python app first, then start Unity Play mode.
- If Unity is on another machine, set `host` in Unity to the Python machine IP and open firewall ports 5000/5001/6000/6001.
- Use **Load Saved Calibration** to reuse previous calibration without clicking again.

## Performance tips (important)

`CameraTcpStreamer.cs` now uses:
- Async GPU readback (non-blocking capture)
- background sender thread
- queue with old-frame dropping for low latency

If Unity still lags at 24 FPS x2:
- Lower resolution first (e.g. 960x540)
- Then lower jpeg quality (e.g. 70)
- Keep `maxQueueSize` small (1-2) to avoid backlog
