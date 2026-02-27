#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--source', default='0', help='0 for webcam, or video path')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--device', default='0')
    ap.add_argument('--view-width', type=int, default=1920, help='Max overlay window width')
    ap.add_argument('--view-height', type=int, default=1080, help='Max overlay window height')
    ap.add_argument('--save-video', action='store_true', help='Save displayed overlay video')
    ap.add_argument('--save-path', default='', help='Output video path (.mp4). If empty, auto path in runs/predict_overlay')
    ap.add_argument('--save-fps', type=float, default=20.0, help='Fallback FPS for saved output')
    ap.add_argument('--cam-width', type=int, default=1920, help='Requested webcam capture width')
    ap.add_argument('--cam-height', type=int, default=1080, help='Requested webcam capture height')
    ap.add_argument('--count-log', action='store_true', help='Print per-frame instance count to terminal')
    ap.add_argument('--count-log-every', type=int, default=10, help='Log every N frames when --count-log is enabled')
    ap.add_argument('--human-joints', action='store_true', help='Enable human arm joint tracking overlay')
    ap.add_argument('--human-model', default='yolo11n-pose.pt', help='Pose model for human joint tracking')
    ap.add_argument('--human-conf', type=float, default=0.20)
    ap.add_argument('--human-alpha', type=float, default=0.30, help='Opacity for thick arm corridor')
    ap.add_argument('--mask-smooth', type=int, default=2, help='Mask smoothing strength (0=off, higher=smoother)')
    args = ap.parse_args()

    source = 0 if args.source == '0' else args.source
    model = YOLO(args.weights)
    pose_model = YOLO(args.human_model) if args.human_joints else None

    win = 'YOLO-Seg Overlay (q to quit)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    writer = None
    save_path = None
    frame_idx = 0

    palette = [
        (255, 90, 90),
        (90, 255, 90),
        (90, 170, 255),
        (255, 220, 90),
        (255, 90, 220),
        (90, 255, 255),
    ]

    def smooth_polygon(poly, smooth_strength):
        if smooth_strength <= 0 or poly is None or len(poly) < 5:
            return poly
        pts = poly.astype('float32').reshape(-1, 1, 2)
        # upscale then smooth contour for less blocky edges
        factor = max(1, int(smooth_strength))
        up = cv2.resize(pts, (1, len(pts) * factor), interpolation=cv2.INTER_CUBIC)
        up = up.reshape(-1, 2)
        # moving average smoothing
        k = min(11, max(3, 2 * factor + 1))
        pad = k // 2
        ext = np.vstack([up[-pad:], up, up[:pad]])
        sm = np.array([ext[i:i + k].mean(axis=0) for i in range(len(up))], dtype=np.float32)
        return sm

    def render_result(r):
        if getattr(r, 'orig_img', None) is None:
            frame = r.plot()
            return frame, 0

        frame = r.orig_img.copy()
        overlay = frame.copy()
        count = 0

        if r.masks is not None and getattr(r.masks, 'xy', None) is not None:
            polys = r.masks.xy
            count = len(polys)
            smoothed_polys = []
            for i, poly in enumerate(polys):
                color = palette[i % len(palette)]
                if poly is None or len(poly) < 3:
                    smoothed_polys.append(None)
                    continue
                sp = smooth_polygon(poly, args.mask_smooth)
                smoothed_polys.append(sp)
                pts = sp.astype('int32').reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], color)

            # Blend once to preserve image quality (avoid repeated resampling)
            frame = cv2.addWeighted(frame, 0.62, overlay, 0.38, 0.0)

            for i, poly in enumerate(smoothed_polys):
                color = palette[i % len(palette)]
                if poly is None or len(poly) < 3:
                    continue
                pts = poly.astype('int32').reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
                cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())
                cv2.putText(frame, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.putText(frame, f'count={count}', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        if count == 0 and r.boxes is not None and len(r.boxes) > 0:
            cv2.putText(frame, 'Warning: boxes detected but no seg masks', (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)
        return frame, count

    def draw_human_arm_overlay(frame, pose_r):
        if pose_r is None or pose_r.keypoints is None:
            return frame

        out = frame.copy()
        thick = frame.copy()
        kpts = pose_r.keypoints.xy
        if kpts is None:
            return frame
        kpts = kpts.cpu().numpy()

        # COCO keypoint indices used by YOLO pose:
        # 5 l-shoulder, 6 r-shoulder, 7 l-elbow, 8 r-elbow, 9 l-wrist, 10 r-wrist
        arms = [(5, 7, 9), (6, 8, 10)]

        for person in kpts:
            for s_idx, e_idx, w_idx in arms:
                p_s = person[s_idx] if s_idx < len(person) else None
                p_e = person[e_idx] if e_idx < len(person) else None
                p_w = person[w_idx] if w_idx < len(person) else None

                pts = []
                for p in [p_s, p_e, p_w]:
                    if p is None:
                        pts.append(None)
                    else:
                        x, y = int(p[0]), int(p[1])
                        if x <= 1 and y <= 1:
                            pts.append(None)
                        else:
                            pts.append((x, y))

                ps, pe, pw = pts
                color = (80, 255, 255)
                corridor_thick = 180  # intentionally very large (10x-style corridor)
                center_thick = 3
                hand_end = None

                # shoulder->elbow if available
                if ps is not None and pe is not None:
                    cv2.line(thick, ps, pe, color, corridor_thick, cv2.LINE_AA)
                    cv2.line(out, ps, pe, color, center_thick, cv2.LINE_AA)

                # elbow->wrist if available
                if pe is not None and pw is not None:
                    cv2.line(thick, pe, pw, color, corridor_thick, cv2.LINE_AA)
                    cv2.line(out, pe, pw, color, center_thick, cv2.LINE_AA)

                    # approximate wrist->middle-of-hand extension
                    dx, dy = (pw[0] - pe[0], pw[1] - pe[1])
                    hx, hy = int(pw[0] + 0.35 * dx), int(pw[1] + 0.35 * dy)
                    hand_mid = (hx, hy)
                    hand_end = hand_mid
                    cv2.line(thick, pw, hand_mid, color, int(corridor_thick * 0.8), cv2.LINE_AA)
                    cv2.line(out, pw, hand_mid, color, 2, cv2.LINE_AA)

                # if shoulder is missing but forearm visible, still draw forearm (requested)
                if ps is None and pe is not None and pw is not None:
                    cv2.line(thick, pe, pw, color, corridor_thick, cv2.LINE_AA)
                    cv2.line(out, pe, pw, color, center_thick, cv2.LINE_AA)
                    hand_end = pw

                # big circle at hand end: diameter = 1.5x line diameter
                if hand_end is not None:
                    r = max(8, int(0.75 * corridor_thick))
                    cv2.circle(thick, hand_end, r, color, -1, cv2.LINE_AA)
                    cv2.circle(out, hand_end, max(2, int(0.12 * r)), color, 2, cv2.LINE_AA)

                for p in [ps, pe, pw]:
                    if p is not None:
                        cv2.circle(out, p, 4, (255, 255, 255), -1, cv2.LINE_AA)

        cv2.addWeighted(thick, max(0.0, min(1.0, args.human_alpha)), out, 1.0 - max(0.0, min(1.0, args.human_alpha)), 0.0, out)
        return out

    def show_and_maybe_save(frame):
        nonlocal writer, save_path
        h, w = frame.shape[:2]
        scale = min(args.view_width / max(w, 1), args.view_height / max(h, 1), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        if args.save_video:
            if writer is None:
                if args.save_path.strip():
                    save_path = Path(args.save_path)
                else:
                    out_dir = Path(__file__).resolve().parents[1] / 'runs' / 'predict_overlay'
                    out_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_path = out_dir / f'overlay_{ts}.mp4'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(save_path), fourcc, max(args.save_fps, 1.0), (frame.shape[1], frame.shape[0]))
            writer.write(frame)

        cv2.imshow(win, frame)
        return (cv2.waitKey(1) & 0xFF) == ord('q')

    if source == 0:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam source 0')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        print(f'Webcam capture resolution requested: {args.cam_width}x{args.cam_height}')
        print(f'Webcam capture resolution actual: {actual_w}x{actual_h}')

        while True:
            ok, raw = cap.read()
            if not ok:
                break
            frame_idx += 1
            r = model.predict(raw, imgsz=args.imgsz, conf=args.conf, device=args.device, retina_masks=True, verbose=False)[0]
            frame, count = render_result(r)
            if pose_model is not None:
                pr = pose_model.predict(raw, imgsz=args.imgsz, conf=args.human_conf, device=args.device, verbose=False)[0]
                frame = draw_human_arm_overlay(frame, pr)
            if args.count_log and (frame_idx % max(1, args.count_log_every) == 0):
                print(f'frame {frame_idx}: count={count}')
            if show_and_maybe_save(frame):
                break
        cap.release()
    else:
        for r in model.predict(source=source, stream=True, imgsz=args.imgsz, conf=args.conf, device=args.device, retina_masks=True):
            frame_idx += 1
            frame, count = render_result(r)
            if pose_model is not None and getattr(r, 'orig_img', None) is not None:
                pr = pose_model.predict(r.orig_img, imgsz=args.imgsz, conf=args.human_conf, device=args.device, verbose=False)[0]
                frame = draw_human_arm_overlay(frame, pr)
            if args.count_log and (frame_idx % max(1, args.count_log_every) == 0):
                print(f'frame {frame_idx}: count={count}')
            if show_and_maybe_save(frame):
                break

    if writer is not None:
        writer.release()
        print(f'Saved overlay video: {save_path}')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
