#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import cv2

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--source', default='0', help='0 for webcam, or video/image path')
    ap.add_argument('--imgsz', type=int, default=1280)
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
    ap.add_argument('--grip-pose', action='store_true', help='Enable grip keypoint overlay from a pose model')
    ap.add_argument('--grip-model', default='', help='Pose model weights for grip keypoints (best.pt)')
    ap.add_argument('--grip-conf', type=float, default=0.20)
    ap.add_argument('--mask-smooth', type=int, default=2, help='Mask smoothing strength (0=off, higher=smoother)')
    args = ap.parse_args()

    source = 0 if args.source == '0' else args.source
    model = YOLO(args.weights)
    pose_model = YOLO(args.human_model) if args.human_joints else None
    grip_model = YOLO(args.grip_model) if (args.grip_pose and args.grip_model.strip()) else None

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

    def is_image_source(src):
        if not isinstance(src, str):
            return False
        return Path(src).suffix.lower() in IMG_EXTS

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

    def add_controls_overlay(frame, mode_text, paused=False):
        out = frame.copy()
        controls = 'q/esc quit | space pause/play | n next frame | j/k -/+30 frames'
        cv2.putText(out, mode_text, (12, out.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, controls, (12, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
        if paused:
            cv2.putText(out, 'PAUSED', (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)
        return out

    def draw_grip_overlay(frame, grip_r):
        if grip_r is None or grip_r.keypoints is None:
            return frame
        out = frame.copy()
        kpts = grip_r.keypoints.xy
        if kpts is None:
            return frame
        kpts = kpts.cpu().numpy()
        for person in kpts:
            if len(person) < 3:
                continue
            c = tuple(np.round(person[0]).astype(int))
            a = tuple(np.round(person[1]).astype(int))
            b = tuple(np.round(person[2]).astype(int))
            cv2.line(out, a, b, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(out, c, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, a, 3, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(out, b, 3, (0, 0, 255), -1, cv2.LINE_AA)
        return out

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

    def save_single_image(frame):
        save_path = Path(args.save_path) if args.save_path.strip() else None
        if save_path is None:
            out_dir = Path(__file__).resolve().parents[1] / 'runs' / 'predict_overlay'
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = out_dir / f'overlay_{ts}.jpg'
        elif save_path.suffix.lower() not in IMG_EXTS:
            save_path = save_path.with_suffix('.jpg')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), frame)
        print(f'Saved overlay image: {save_path}')

    def show_and_maybe_save(frame, wait_ms=1):
        nonlocal writer, save_path
        h, w = frame.shape[:2]
        scale = min(args.view_width / max(w, 1), args.view_height / max(h, 1), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        if args.save_video:
            if is_image_source(args.source):
                save_single_image(frame)
            else:
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
        return cv2.waitKey(wait_ms) & 0xFF

    def process_raw_frame(raw):
        nonlocal frame_idx
        frame_idx += 1
        r = model.predict(raw, imgsz=args.imgsz, conf=args.conf, device=args.device, retina_masks=True, verbose=False)[0]
        frame, count = render_result(r)
        if pose_model is not None:
            pr = pose_model.predict(raw, imgsz=args.imgsz, conf=args.human_conf, device=args.device, verbose=False)[0]
            frame = draw_human_arm_overlay(frame, pr)
        if grip_model is not None:
            gr = grip_model.predict(raw, imgsz=args.imgsz, conf=args.grip_conf, device=args.device, verbose=False)[0]
            frame = draw_grip_overlay(frame, gr)
        if args.count_log and (frame_idx % max(1, args.count_log_every) == 0):
            print(f'frame {frame_idx}: count={count}')
        return frame, count

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
            frame, _ = process_raw_frame(raw)
            key = show_and_maybe_save(add_controls_overlay(frame, 'webcam mode'))
            if key in (ord('q'), 27):
                break
        cap.release()
    else:
        if is_image_source(source):
            r = model.predict(source=source, imgsz=args.imgsz, conf=args.conf, device=args.device, retina_masks=True, verbose=False)[0]
            frame_idx += 1
            frame, count = render_result(r)
            if pose_model is not None and getattr(r, 'orig_img', None) is not None:
                pr = pose_model.predict(r.orig_img, imgsz=args.imgsz, conf=args.human_conf, device=args.device, verbose=False)[0]
                frame = draw_human_arm_overlay(frame, pr)
            if grip_model is not None and getattr(r, 'orig_img', None) is not None:
                gr = grip_model.predict(r.orig_img, imgsz=args.imgsz, conf=args.grip_conf, device=args.device, verbose=False)[0]
                frame = draw_grip_overlay(frame, gr)
            if args.count_log:
                print(f'image {Path(source).name}: count={count}')
            show_and_maybe_save(add_controls_overlay(frame, f'image mode: {Path(source).name}'), wait_ms=0)
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open video source: {source}')

            paused = False
            last_frame = None
            video_name = Path(source).name

            def seek_relative(delta_frames):
                cur_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                target = max(0, cur_idx + delta_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)

            while True:
                if not paused or last_frame is None:
                    ok, raw = cap.read()
                    if not ok:
                        break
                    frame, _ = process_raw_frame(raw)
                    last_frame = frame

                shown = add_controls_overlay(last_frame, f'video mode: {video_name}', paused=paused)
                key = show_and_maybe_save(shown, wait_ms=0 if paused else 1)
                if key in (ord('q'), 27):
                    break
                if key == ord(' '):
                    paused = not paused
                    continue
                if key == ord('n'):
                    paused = True
                    ok, raw = cap.read()
                    if not ok:
                        break
                    frame, _ = process_raw_frame(raw)
                    last_frame = frame
                    continue
                if key == ord('j'):
                    seek_relative(-30)
                    paused = True
                    last_frame = None
                    continue
                if key == ord('k'):
                    seek_relative(30)
                    paused = True
                    last_frame = None
                    continue

            cap.release()

    if writer is not None:
        writer.release()
        print(f'Saved overlay video: {save_path}')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
