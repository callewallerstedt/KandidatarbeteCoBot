#!/usr/bin/env python3
import argparse
import socket
import struct

import cv2
import numpy as np
from ultralytics import YOLO


def recv_exact(conn, n):
    data = b''
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def smooth_polygon(poly, smooth_strength):
    if smooth_strength <= 0 or poly is None or len(poly) < 5:
        return poly
    p = np.round(poly).astype(np.int32)
    x1, y1 = p[:, 0].min(), p[:, 1].min()
    x2, y2 = p[:, 0].max(), p[:, 1].max()
    pad = max(3, int(smooth_strength) * 2)
    w = max(8, int(x2 - x1 + 1 + 2 * pad))
    h = max(8, int(y2 - y1 + 1 + 2 * pad))

    local = np.zeros((h, w), dtype=np.uint8)
    q = p.copy()
    q[:, 0] = q[:, 0] - x1 + pad
    q[:, 1] = q[:, 1] - y1 + pad
    cv2.fillPoly(local, [q.reshape(-1, 1, 2)], 255)

    k = max(3, 2 * int(smooth_strength) + 1)
    local = cv2.GaussianBlur(local, (k, k), 0)
    _, local = cv2.threshold(local, 127, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return poly
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    c[:, 0] = c[:, 0] + x1 - pad
    c[:, 1] = c[:, 1] + y1 - pad
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=5000)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--device', default='0')
    ap.add_argument('--mask-smooth', type=int, default=2)
    args = ap.parse_args()

    model = YOLO(args.weights)
    palette = [(255,90,90),(90,255,90),(90,170,255),(255,220,90),(255,90,220),(90,255,255)]

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f'[UnityTCP] Listening on {args.host}:{args.port}', flush=True)

    win = 'YOLO-Seg Overlay TCP (q quit)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        conn, addr = srv.accept()
        print(f'[UnityTCP] Connected: {addr}', flush=True)
        with conn:
            while True:
                header = recv_exact(conn, 4)
                if header is None:
                    break
                (size,) = struct.unpack('<I', header)
                payload = recv_exact(conn, size)
                if payload is None:
                    break

                arr = np.frombuffer(payload, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, retina_masks=True, verbose=False)[0]
                out = frame.copy()
                overlay = out.copy()
                count = 0
                if r.masks is not None and getattr(r.masks, 'xy', None) is not None:
                    polys = r.masks.xy
                    count = len(polys)
                    smoothed = []
                    for i, poly in enumerate(polys):
                        if poly is None or len(poly) < 3:
                            smoothed.append(None)
                            continue
                        sp = smooth_polygon(poly, args.mask_smooth)
                        smoothed.append(sp)
                        pts = sp.astype(np.int32).reshape(-1, 1, 2)
                        cv2.fillPoly(overlay, [pts], palette[i % len(palette)])
                    out = cv2.addWeighted(out, 0.62, overlay, 0.38, 0)
                    for i, poly in enumerate(smoothed):
                        if poly is None or len(poly) < 3:
                            continue
                        col = palette[i % len(palette)]
                        pts = poly.astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(out, [pts], True, col, 2, cv2.LINE_AA)
                cv2.putText(out, f'count={count}', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow(win, out)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    srv.close()
                    cv2.destroyAllWindows()
                    return
        print('[UnityTCP] Disconnected', flush=True)


if __name__ == '__main__':
    main()
