import socket
import struct
import threading

import cv2
import numpy as np

HOST = "0.0.0.0"
PORTS = [5000, 5001]
WINDOWS = ["Unity Cam 1", "Unity Cam 2"]

latest_frames = [None, None]
locks = [threading.Lock(), threading.Lock()]
running = True


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
    print(f"[Cam {index + 1}] Listening on {HOST}:{port}")

    while running:
        try:
            conn, addr = server.accept()
            print(f"[Cam {index + 1}] Connected: {addr}")
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
            print(f"[Cam {index + 1}] Disconnected")
        except Exception as e:
            print(f"[Cam {index + 1}] Error: {e}")

    server.close()


def main():
    threads = []
    for i, p in enumerate(PORTS):
        t = threading.Thread(target=camera_server, args=(i, p), daemon=True)
        t.start()
        threads.append(t)

    global running
    while True:
        for i, win in enumerate(WINDOWS):
            frame = None
            with locks[i]:
                if latest_frames[i] is not None:
                    frame = latest_frames[i].copy()

            if frame is not None:
                cv2.imshow(win, frame)
            else:
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    f"Waiting for cam {i + 1}...",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                )
                cv2.imshow(win, blank)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    running = False
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
