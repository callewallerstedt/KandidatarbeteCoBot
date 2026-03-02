# Unity -> Python Dual Camera Stream Test

This folder contains a simple test setup where two Unity cameras stream JPEG frames over TCP to Python, and Python shows each camera in its own OpenCV window.

## Files

- `CameraTcpStreamer.cs` - Unity script to stream one camera over TCP.
- `receive_two_cams.py` - Python receiver that opens two windows (`Unity Cam 1`, `Unity Cam 2`).
- `requirements.txt` - Python dependencies.
- `start_receiver.bat` - Windows one-click start script for Python receiver.

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

## Notes

- Run Python receiver first, then start Unity Play mode.
- If Unity is on another machine, set `host` in Unity to the Python machine IP and open firewall ports 5000/5001.
