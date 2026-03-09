#!/usr/bin/env python3
import argparse
import socket
import struct

import cv2
import numpy as np
from ultralytics import YOLO
from pickable_instance import select_pickable_instance
from instance_tracker import update_tracks


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
    pts = np.round(poly).astype(np.float32).reshape(-1, 1, 2)
    peri = cv2.arcLength(pts, True)
    eps = max(0.8, float(smooth_strength) * 0.0035 * peri)
    approx = cv2.approxPolyDP(pts, eps, True)
    return approx.reshape(-1, 2).astype(np.float32) if approx is not None and len(approx) >= 3 else poly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=5000)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--device', default='0')
    ap.add_argument('--mask-smooth', type=int, default=2)
    ap.add_argument('--pick-overlay', action='store_true', help='Highlight the most free/pickable object')
    ap.add_argument('--grip-pose', action='store_true')
    ap.add_argument('--grip-model', default='')
    ap.add_argument('--grip-conf', type=float, default=0.20)
    args = ap.parse_args()

    model = YOLO(args.weights)
    grip_model = YOLO(args.grip_model) if (args.grip_pose and args.grip_model.strip()) else None
    palette = [(255,90,90),(90,255,90),(90,170,255),(255,220,90),(255,90,220),(90,255,255)]
    prev_mask_overlay = None
    prev_mask_alpha = None
    prev_pick_center = None
    tracks = {}
    next_track_id = 1

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
                overlay = np.zeros_like(out, dtype=np.float32)
                alpha_layer = np.zeros(out.shape[:2], dtype=np.float32)
                count = 0
                if r.masks is not None and getattr(r.masks, 'xy', None) is not None:
                    polys = r.masks.xy
                    count = len(polys)
                    smoothed = []
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and getattr(r.boxes, 'xyxy', None) is not None else None
                    detections = []
                    if boxes_xyxy is not None:
                        for i, poly in enumerate(polys):
                            if poly is None or len(poly) < 3 or i >= len(boxes_xyxy):
                                smoothed.append(None)
                                continue
                            sp = smooth_polygon(poly, args.mask_smooth if args.pick_overlay else min(args.mask_smooth, 1))
                            smoothed.append(sp)
                            detections.append({'poly': sp, 'box': tuple(float(v) for v in boxes_xyxy[i])})
                        tracks, next_track_id = update_tracks(tracks, next_track_id, detections, frame.shape, max_missed=2)
                    else:
                        tracks = {}
                    if args.pick_overlay:
                        active_boxes = [tracks[tid]['box'] for tid in tracks if tracks[tid].get('matched')]
                        best_local_idx, pick_infos = select_pickable_instance(active_boxes, frame.shape)
                        matched_track_ids = [tid for tid in tracks if tracks[tid].get('matched')]
                        best_track_id = matched_track_ids[best_local_idx] if best_local_idx is not None and best_local_idx < len(matched_track_ids) else None
                        for track_id, track in tracks.items():
                            poly = track.get('poly')
                            if poly is None or len(poly) < 3:
                                continue
                            pts = poly.astype(np.int32).reshape(-1, 1, 2)
                            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                            cv2.fillPoly(mask, [pts], 255)
                            sel = mask > 0
                            color = palette[(track_id - 1) % len(palette)]
                            overlay[sel] = color
                            base_alpha = 0.10 if not track.get('matched') else 0.18
                            hi_alpha = 0.16 if not track.get('matched') else 0.24
                            alpha_layer[sel] = np.maximum(alpha_layer[sel], hi_alpha if track_id == best_track_id else base_alpha)
                        if prev_mask_overlay is not None and prev_mask_overlay.shape == overlay.shape:
                            overlay = 0.45 * prev_mask_overlay + 0.55 * overlay
                            alpha_layer = 0.45 * prev_mask_alpha + 0.55 * alpha_layer
                        prev_mask_overlay = overlay.copy()
                        prev_mask_alpha = alpha_layer.copy()
                        out = np.clip(frame.astype(np.float32) * (1.0 - alpha_layer[:, :, None]) + overlay * alpha_layer[:, :, None], 0, 255).astype(np.uint8)
                    else:
                        best_track_id, best_local_idx, pick_infos = None, None, []
                        prev_mask_overlay = None
                        prev_mask_alpha = None
                        prev_pick_center = None
                        simple_overlay = frame.copy()
                        for track_id, track in tracks.items():
                            poly = track.get('poly')
                            if poly is None or len(poly) < 3:
                                continue
                            pts = poly.astype(np.int32).reshape(-1, 1, 2)
                            cv2.fillPoly(simple_overlay, [pts], palette[(track_id - 1) % len(palette)])
                        out = cv2.addWeighted(frame, 0.80, simple_overlay, 0.20, 0.0)
                    for track_id, track in tracks.items():
                        poly = track.get('poly')
                        if poly is None or len(poly) < 3:
                            continue
                        col = palette[(track_id - 1) % len(palette)]
                        cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())
                        if args.pick_overlay and track_id == best_track_id:
                            if prev_pick_center is not None:
                                cx = int(round(0.55 * prev_pick_center[0] + 0.45 * cx))
                                cy = int(round(0.55 * prev_pick_center[1] + 0.45 * cy))
                            prev_pick_center = (cx, cy)
                            x1, y1, x2, y2 = [int(round(v)) for v in track['box']]
                            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.circle(out, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
                            cv2.putText(out, f'PICK {track_id}', (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
                        else:
                            txt_col = col if track.get('matched') else (180, 180, 180)
                            cv2.putText(out, str(track_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, txt_col, 2, cv2.LINE_AA)
                    if args.pick_overlay and best_track_id is not None and best_local_idx is not None and 0 <= best_local_idx < len(pick_infos):
                        info = pick_infos[best_local_idx]
                        cv2.putText(out, f"pick={best_track_id} free={info['score']:.2f} contacts={info['contact_count']}", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255,255,255), 2, cv2.LINE_AA)
                else:
                    prev_mask_overlay = None
                    prev_mask_alpha = None
                    prev_pick_center = None
                    tracks = {}
                if grip_model is not None:
                    gr = grip_model.predict(frame, imgsz=args.imgsz, conf=args.grip_conf, device=args.device, verbose=False)[0]
                    if gr is not None and gr.keypoints is not None and gr.keypoints.xy is not None:
                        for p in gr.keypoints.xy.cpu().numpy():
                            if len(p) < 3:
                                continue
                            c = tuple(np.round(p[0]).astype(int))
                            a = tuple(np.round(p[1]).astype(int))
                            b = tuple(np.round(p[2]).astype(int))
                            cv2.line(out, a, b, (0, 255, 255), 2, cv2.LINE_AA)
                            cv2.circle(out, c, 3, (255, 255, 255), -1, cv2.LINE_AA)
                            cv2.circle(out, a, 3, (0, 255, 0), -1, cv2.LINE_AA)
                            cv2.circle(out, b, 3, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(out, f'count={count}', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow(win, out)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    srv.close()
                    cv2.destroyAllWindows()
                    return
        print('[UnityTCP] Disconnected', flush=True)


if __name__ == '__main__':
    main()
