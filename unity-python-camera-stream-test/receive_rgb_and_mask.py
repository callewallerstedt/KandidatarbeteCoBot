import socket
import struct
import threading

import cv2
import numpy as np

HOST = "0.0.0.0"
PORTS = [5000, 5001, 6000, 6001]  # RGB1, RGB2, MASK1, MASK2
WINDOWS = ["RGB Cam 1", "RGB Cam 2", "MASK Cam 1", "MASK Cam 2"]

latest_frames = [None, None, None, None]
locks = [threading.Lock() for _ in PORTS]
running = True


def recv_exact(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def stream_server(index, port):
    global running
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, port))
    server.listen(1)
    print(f"[{WINDOWS[index]}] Listening on {HOST}:{port}", flush=True)

    while running:
        try:
            conn, addr = server.accept()
            print(f"[{WINDOWS[index]}] Connected: {addr}", flush=True)
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
            print(f"[{WINDOWS[index]}] Disconnected", flush=True)
        except Exception as e:
            print(f"[{WINDOWS[index]}] Error: {e}", flush=True)

    server.close()


def main():
    for i, p in enumerate(PORTS):
        threading.Thread(target=stream_server, args=(i, p), daemon=True).start()

    while True:
        for i, win in enumerate(WINDOWS):
            frame = None
            with locks[i]:
                if latest_frames[i] is not None:
                    frame = latest_frames[i].copy()

            if frame is None:
                frame = np.zeros((240, 426, 3), dtype=np.uint8)
                cv2.putText(frame, f"Waiting: {win}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord("q")]:
            break

    global running
    running = False
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
