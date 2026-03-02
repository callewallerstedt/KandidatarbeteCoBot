import socket
import struct
import threading
import time
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox

# ---------------------------
# Stream config
# ---------------------------
HOST = "0.0.0.0"
PORTS = [5000, 5001, 6000, 6001]  # RGB1, RGB2, SEG1, SEG2
WINDOWS = ["Unity Cam 1", "Unity Cam 2"]

# ---------------------------
# World config (meters)
# ---------------------------
TABLE_WIDTH = 2.5   # X direction
TABLE_DEPTH = 1.5   # Y direction
CALIB_HEIGHT = 1.0  # Z for elevated calibration points

# Corner order used everywhere: TL, TR, BR, BL
WORLD_CORNERS_XY = np.array([
    [0.0, 0.0],
    [TABLE_WIDTH, 0.0],
    [TABLE_WIDTH, TABLE_DEPTH],
    [0.0, TABLE_DEPTH],
], dtype=np.float32)

CALIB_FILE = Path(__file__).parent / "calibration_data.json"

latest_frames = [None, None, None, None]
locks = [threading.Lock(), threading.Lock(), threading.Lock(), threading.Lock()]
running = True


class SharedState:
    def __init__(self):
        self.mode = "idle"  # idle | calibrating | tracking
        self.instructions = "Press 'Start Calibration'"

        # current red-dot detections from both cameras
        self.latest_uv = [None, None]

        # calibration sequence step index (0..7)
        self.calib_step = 0

        # per camera calibration points
        self.cam_points = {
            0: {
                "z0": [None] * 4,  # 4 image points for table corners
                "z1": [None] * 4,  # 4 image points for elevated corners
            },
            1: {
                "z0": [None] * 4,
                "z1": [None] * 4,
            },
        }

        # homographies image->worldXY for z=0 and z=1, per camera
        self.H = {
            0: {"z0": None, "z1": None},
            1: {"z0": None, "z1": None},
        }

        self.last_3d = None
        self.capture_in_progress = False
        self.lock = threading.Lock()


state = SharedState()


# ---------------------------
# Networking
# ---------------------------
def recv_exact(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def camera_server(index, port):
    global running
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, port))
    server.listen(1)
    print(f"[Cam {index+1}] Listening on {HOST}:{port}", flush=True)

    while running:
        try:
            conn, addr = server.accept()
            print(f"[Cam {index+1}] Connected: {addr}", flush=True)
            with conn:
                while running:
                    header = recv_exact(conn, 4)
                    if header is None:
                        break
                    (size,) = struct.unpack("<I", header)
                    payload = recv_exact(conn, size)
                    if payload is None:
                        break

                    arr = np.frombuffer(payload, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with locks[index]:
                            latest_frames[index] = frame
            print(f"[Cam {index+1}] Disconnected", flush=True)
        except Exception as e:
            print(f"[Cam {index+1}] Error: {e}", flush=True)
            time.sleep(0.2)

    server.close()


# ---------------------------
# Geometry
# ---------------------------
def build_homography_img_to_world(img_pts):
    img = np.array(img_pts, dtype=np.float32)
    H, _ = cv2.findHomography(img, WORLD_CORNERS_XY)
    return H


def map_pixel_to_world_xy(H_img_to_world, uv):
    p = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(p, H_img_to_world)[0, 0]
    return np.array([float(mapped[0]), float(mapped[1])], dtype=np.float64)


def line_from_camera_mapping(cam_idx, uv):
    H0 = state.H[cam_idx]["z0"]
    H1 = state.H[cam_idx]["z1"]
    if H0 is None or H1 is None:
        return None

    xy0 = map_pixel_to_world_xy(H0, uv)
    xy1 = map_pixel_to_world_xy(H1, uv)

    p0 = np.array([xy0[0], xy0[1], 0.0], dtype=np.float64)
    p1 = np.array([xy1[0], xy1[1], CALIB_HEIGHT], dtype=np.float64)
    d = p1 - p0
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    d /= n
    return p0, d


def closest_point_between_lines(line1, line2):
    p1, d1 = line1
    p2, d2 = line2
    w0 = p1 - p2

    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)

    denom = a * c - b * b
    if abs(denom) < 1e-9:
        return (p1 + p2) * 0.5

    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom

    cp1 = p1 + t1 * d1
    cp2 = p2 + t2 * d2
    return (cp1 + cp2) * 0.5


# ---------------------------
# Vision: red dot detection
# ---------------------------
def get_red_binary_mask(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # red wraps around hue range
    lower1 = np.array([0, 120, 80], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([165, 120, 80], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def make_red_on_black_mask_image(frame_bgr):
    mask = get_red_binary_mask(frame_bgr)
    out = np.zeros_like(frame_bgr)
    out[mask > 0] = (0, 0, 255)  # BGR pure red
    return out


def force_red_mask_from_any_mask_image(mask_like_bgr, threshold=10):
    # Converts any masked/shaded input (gray, lit, etc.) into pure red-on-black.
    gray = cv2.cvtColor(mask_like_bgr, cv2.COLOR_BGR2GRAY)
    out = np.zeros_like(mask_like_bgr)
    out[gray > threshold] = (0, 0, 255)  # BGR pure red
    return out


def detect_red_dot(frame_bgr):
    mask = get_red_binary_mask(frame_bgr)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 10:
        return None

    m = cv2.moments(c)
    if m["m00"] == 0:
        return None

    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return (cx, cy), area


# ---------------------------
# Calibration flow
# ---------------------------
def corner_name(i):
    return ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"][i]


def get_step_info(step_idx):
    # User-friendly order requested: TL, TR, BL, BR
    order = [0, 1, 3, 2]
    if step_idx < 4:
        corner_idx = order[step_idx]
        return "z0", corner_idx, f"Move RED dot to TABLE {corner_name(corner_idx)} and press 'Confirm Position'"

    corner_idx = order[step_idx - 4]
    return "z1", corner_idx, f"Move RED dot to IN-AIR {corner_name(corner_idx)} (+1m) and press 'Confirm Position'"


def start_calibration():
    with state.lock:
        state.mode = "calibrating"
        state.calib_step = 0
        state.cam_points = {0: {"z0": [None] * 4, "z1": [None] * 4}, 1: {"z0": [None] * 4, "z1": [None] * 4}}
        state.H = {0: {"z0": None, "z1": None}, 1: {"z0": None, "z1": None}}
        state.last_3d = None
        _, _, msg = get_step_info(0)
        state.instructions = msg


def finalize_calibration():
    try:
        for cam in [0, 1]:
            state.H[cam]["z0"] = build_homography_img_to_world(state.cam_points[cam]["z0"])
            state.H[cam]["z1"] = build_homography_img_to_world(state.cam_points[cam]["z1"])

        save_calibration()

        with state.lock:
            state.mode = "tracking"
            state.instructions = "Calibration done. Tracking RED dot in 3D."

        messagebox.showinfo("Calibration", "Calibration complete. Now tracking red dot in 3D.")
    except Exception as e:
        with state.lock:
            state.mode = "idle"
            state.instructions = f"Calibration failed: {e}"
        messagebox.showerror("Calibration failed", str(e))


def confirm_calibration_position():
    with state.lock:
        if state.mode != "calibrating":
            messagebox.showinfo("Calibration", "Start calibration first.")
            return

        uv0 = state.latest_uv[0]
        uv1 = state.latest_uv[1]
        if uv0 is None or uv1 is None:
            messagebox.showwarning("No red dot", "Red dot must be visible in BOTH camera windows before confirm.")
            return

        phase_key, corner_idx, _ = get_step_info(state.calib_step)
        state.cam_points[0][phase_key][corner_idx] = [int(uv0[0]), int(uv0[1])]
        state.cam_points[1][phase_key][corner_idx] = [int(uv1[0]), int(uv1[1])]
        state.calib_step += 1

        if state.calib_step >= 8:
            done = True
        else:
            done = False
            _, _, msg = get_step_info(state.calib_step)
            state.instructions = msg

    if done:
        finalize_calibration()


def save_calibration():
    data = {
        "table_width": TABLE_WIDTH,
        "table_depth": TABLE_DEPTH,
        "calib_height": CALIB_HEIGHT,
        "cam_points": {
            "0": {
                "z0": state.cam_points[0]["z0"],
                "z1": state.cam_points[0]["z1"],
            },
            "1": {
                "z0": state.cam_points[1]["z0"],
                "z1": state.cam_points[1]["z1"],
            },
        },
    }
    CALIB_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration_if_available():
    if not CALIB_FILE.exists():
        return False

    try:
        data = json.loads(CALIB_FILE.read_text(encoding="utf-8"))
        for cam in [0, 1]:
            z0 = data["cam_points"][str(cam)]["z0"]
            z1 = data["cam_points"][str(cam)]["z1"]
            state.cam_points[cam]["z0"] = z0
            state.cam_points[cam]["z1"] = z1
            state.H[cam]["z0"] = build_homography_img_to_world(z0)
            state.H[cam]["z1"] = build_homography_img_to_world(z1)

        with state.lock:
            state.mode = "tracking"
            state.instructions = "Loaded calibration from file. Tracking RED dot in 3D."
        return True
    except Exception as e:
        print(f"Failed to load calibration: {e}", flush=True)
        return False


# ---------------------------
# UI + display loop
# ---------------------------
def track_and_draw(frame, cam_idx):
    result = detect_red_dot(frame)
    if result is None:
        return frame, None

    (cx, cy), area = result
    cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2)
    cv2.putText(frame, f"red ({cx},{cy}) area={area:.0f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame, (cx, cy)


def clamp_world_point(p):
    x = max(0.0, min(TABLE_WIDTH, float(p[0])))
    y = max(0.0, min(TABLE_DEPTH, float(p[1])))
    z = max(0.0, min(CALIB_HEIGHT, float(p[2])))
    return np.array([x, y, z], dtype=np.float64)


def display_loop(root, label_var, coord_var):
    global running

    for i, p in enumerate(PORTS):
        t = threading.Thread(target=camera_server, args=(i, p), daemon=True)
        t.start()

    cv2.namedWindow(WINDOWS[0])
    cv2.namedWindow(WINDOWS[1])

    while running:
        frames = []
        for i in [0, 1]:
            with locks[i]:
                f = latest_frames[i].copy() if latest_frames[i] is not None else None
            frames.append(f)

        uv = [None, None]
        for i in [0, 1]:
            if frames[i] is None:
                blank = np.zeros((300, 500, 3), dtype=np.uint8)
                cv2.putText(blank, f"Waiting for CAM {i+1}...", (80, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                frames[i] = blank
            else:
                frames[i], uv[i] = track_and_draw(frames[i], i)

        with state.lock:
            state.latest_uv = [uv[0], uv[1]]
            instr = state.instructions
            mode = state.mode

        for i in [0, 1]:
            cv2.putText(frames[i], f"MODE: {mode}", (10, frames[i].shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frames[i], instr, (10, frames[i].shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(WINDOWS[i], frames[i])

        # 3D estimate if calibrated + both red detections available
        if mode == "tracking" and uv[0] is not None and uv[1] is not None:
            line1 = line_from_camera_mapping(0, uv[0])
            line2 = line_from_camera_mapping(1, uv[1])
            if line1 is not None and line2 is not None:
                p3d = closest_point_between_lines(line1, line2)
                p3d = clamp_world_point(p3d)
                with state.lock:
                    state.last_3d = p3d
                coord_var.set(f"X={p3d[0]:.3f} m, Y={p3d[1]:.3f} m, Z={p3d[2]:.3f} m")

        label_var.set(instr)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            running = False
            break

        try:
            root.update_idletasks()
            root.update()
        except tk.TclError:
            running = False
            break

    cv2.destroyAllWindows()
    try:
        root.destroy()
    except Exception:
        pass


def reset_calibration():
    with state.lock:
        state.mode = "idle"
        state.calib_step = 0
        state.instructions = "Calibration reset. Press 'Start Calibration'."
        state.H = {0: {"z0": None, "z1": None}, 1: {"z0": None, "z1": None}}
        state.cam_points = {0: {"z0": [None] * 4, "z1": [None] * 4}, 1: {"z0": [None] * 4, "z1": [None] * 4}}
        state.last_3d = None


def capture_dataset(n_pics):
    with state.lock:
        if state.capture_in_progress:
            return
        state.capture_in_progress = True

    try:
        base = Path(__file__).parent / "images"
        rgb_dir = base / "RGB"
        seg_dir = base / "SEG"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        seg_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i in range(n_pics):
            if not running:
                break

            with locks[0]:
                f0 = latest_frames[0].copy() if latest_frames[0] is not None else None
            with locks[1]:
                f1 = latest_frames[1].copy() if latest_frames[1] is not None else None
            with locks[2]:
                s0 = latest_frames[2].copy() if latest_frames[2] is not None else None
            with locks[3]:
                s1 = latest_frames[3].copy() if latest_frames[3] is not None else None

            if f0 is None or f1 is None:
                with state.lock:
                    state.instructions = "Take N Pics paused: waiting for both RGB streams..."
                time.sleep(0.1)
                continue

            # Prefer true mask streams from Unity (ports 6000/6001). Fallback to HSV-generated masks.
            # Always normalize to pure red-on-black output for training consistency.
            seg0 = force_red_mask_from_any_mask_image(s0) if s0 is not None else make_red_on_black_mask_image(f0)
            seg1 = force_red_mask_from_any_mask_image(s1) if s1 is not None else make_red_on_black_mask_image(f1)

            cv2.imwrite(str(rgb_dir / f"{stamp}_cam1_{i:04d}.png"), f0)
            cv2.imwrite(str(rgb_dir / f"{stamp}_cam2_{i:04d}.png"), f1)
            cv2.imwrite(str(seg_dir / f"{stamp}_cam1_{i:04d}.png"), seg0)
            cv2.imwrite(str(seg_dir / f"{stamp}_cam2_{i:04d}.png"), seg1)

            with state.lock:
                state.instructions = f"Capturing dataset: {i + 1}/{n_pics}"

            time.sleep(0.08)  # ~12.5 captures/sec max

        with state.lock:
            state.instructions = f"Done: saved {n_pics} pics to images/RGB and images/SEG"
    finally:
        with state.lock:
            state.capture_in_progress = False


def on_take_n_pics(count_var):
    try:
        n = int(count_var.get())
    except ValueError:
        messagebox.showwarning("Invalid number", "Enter an integer for N pics.")
        return

    if n <= 0:
        messagebox.showwarning("Invalid number", "N must be > 0.")
        return

    t = threading.Thread(target=capture_dataset, args=(n,), daemon=True)
    t.start()


def build_tk_window():
    root = tk.Tk()
    root.title("Dual Camera Calibration + 3D Tracker")
    root.geometry("760x320")

    title = tk.Label(root, text="Table calibration and 3D red-dot tracking", font=("Segoe UI", 14, "bold"))
    title.pack(pady=8)

    label_var = tk.StringVar(value="Press 'Start Calibration'")
    coord_var = tk.StringVar(value="X=--, Y=--, Z=--")

    info = tk.Label(root, textvariable=label_var, wraplength=650, justify="left", fg="blue", font=("Segoe UI", 10))
    info.pack(pady=6)

    coords = tk.Label(root, textvariable=coord_var, font=("Consolas", 16, "bold"), fg="darkred")
    coords.pack(pady=8)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    b1 = tk.Button(button_frame, text="Start Calibration", width=18, command=start_calibration)
    b1.grid(row=0, column=0, padx=6, pady=4)

    b_confirm = tk.Button(button_frame, text="Confirm Position", width=18, command=confirm_calibration_position)
    b_confirm.grid(row=0, column=1, padx=6, pady=4)

    b2 = tk.Button(button_frame, text="Load Saved Calibration", width=18,
                   command=lambda: messagebox.showinfo(
                       "Calibration", "Loaded." if load_calibration_if_available() else "No valid calibration file found."
                   ))
    b2.grid(row=0, column=2, padx=6, pady=4)

    b3 = tk.Button(button_frame, text="Reset Calibration", width=18, command=reset_calibration)
    b3.grid(row=0, column=3, padx=6, pady=4)

    count_var = tk.StringVar(value="50")
    tk.Label(button_frame, text="N:").grid(row=1, column=0, sticky="e")
    tk.Entry(button_frame, width=10, textvariable=count_var).grid(row=1, column=1, sticky="w")
    tk.Button(button_frame, text="Take N Pics", width=18, command=lambda: on_take_n_pics(count_var)).grid(row=1, column=2, padx=6, pady=4)

    root.protocol("WM_DELETE_WINDOW", lambda: close_app(root))
    return root, label_var, coord_var


def close_app(root):
    global running
    running = False
    try:
        root.destroy()
    except Exception:
        pass


def main():
    print("Starting dual-camera calibration + 3D tracking...", flush=True)
    root, label_var, coord_var = build_tk_window()
    display_loop(root, label_var, coord_var)


if __name__ == "__main__":
    main()
