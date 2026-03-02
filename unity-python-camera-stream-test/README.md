# Unity -> Python Dual Camera Stream Test

This folder contains a simple test setup where two Unity cameras stream JPEG frames over TCP to Python, and Python shows each camera in its own OpenCV window.

## Files

- `CameraTcpStreamer.cs` - Unity script to stream one camera over TCP.
- `receive_two_cams.py` - basic Python receiver that opens two windows (`Unity Cam 1`, `Unity Cam 2`).
- `calibrate_and_track_3d.py` - Tkinter + OpenCV calibration flow and 3D red-dot tracking.
- `requirements.txt` - Python dependencies.
- `start_receiver.bat` - Windows one-click start script for basic receiver.
- `start_calibration_tracker.bat` - Windows one-click start for calibration + 3D tracker.

## Unity setup

1. In your Unity project, add `CameraTcpStreamer.cs`.
2. Attach it to two different Camera objects.
3. Configure:
   - Camera 1: `host=127.0.0.1`, `port=5000`
   - Camera 2: `host=127.0.0.1`, `port=5001`
4. Keep both running in Play mode.

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
2. App asks for table corners in order: **TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT**.
   - For each corner, click in **CAM 1** first, then **same point** in **CAM 2**.
3. App then asks you to move the red Unity sphere to the same 4 corners but at **z = +1m** above table.
   - Again click sphere in CAM 1 then CAM 2 for each corner.
4. Calibration saves automatically to `calibration_data.json` and tracking starts.

### Coordinate system

- X: width direction of table (0 to 2.5 m)
- Y: depth direction of table (0 to 1.5 m)
- Z: height above table (0 to 1.0 m)

The app tracks the red dot and shows estimated 3D coordinates in the Tkinter window.

## Notes

- Run Python app first, then start Unity Play mode.
- If Unity is on another machine, set `host` in Unity to the Python machine IP and open firewall ports 5000/5001.
- Use **Load Saved Calibration** to reuse previous calibration without clicking again.
