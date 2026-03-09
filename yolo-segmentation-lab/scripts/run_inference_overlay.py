#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import cv2
from pickable_instance import select_pickable_instance
from instance_tracker import update_tracks

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
    ap.add_argument('--pick-overlay', action='store_true', help='Highlight the most free/pickable object')
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
    prev_mask_overlay = None
    prev_mask_alpha = None
    prev_pick_center = None
    tracks = {}
    next_track_id = 1
    manual_selected_track_id = None
    toolbox_proc = None
    toolbox_cmd_path = Path(tempfile.gettempdir()) / f'infer_toolbox_cmd_{Path(__file__).stem}_{id(object())}.json'
    toolbox_status_path = Path(tempfile.gettempdir()) / f'infer_toolbox_status_{Path(__file__).stem}_{id(object())}.json'
    toolbox_paused = False
    toolbox_choose_mode = False
    toolbox_last_seq = -1
    display_scale = 1.0
    last_render_shape = None

    palette = [
        (255, 90, 90),
        (90, 255, 90),
        (90, 170, 255),
        (255, 220, 90),
        (255, 90, 220),
        (90, 255, 255),
    ]
    try:
        toolbox_proc = subprocess.Popen([
            sys.executable,
            str(Path(__file__).with_name('inference_toolbox.py')),
            '--command-file', str(toolbox_cmd_path),
            '--status-file', str(toolbox_status_path),
        ])
    except Exception:
        toolbox_proc = None

    def choose_track_at(x, y):
        nonlocal manual_selected_track_id, toolbox_choose_mode
        if last_render_shape is None:
            return
        scale = max(1e-6, float(display_scale))
        fx = int(round(x / scale))
        fy = int(round(y / scale))
        candidates = []
        for track_id, track in tracks.items():
            x1, y1, x2, y2 = [int(round(v)) for v in track['box']]
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                dist2 = (fx - cx) ** 2 + (fy - cy) ** 2
                candidates.append((dist2, track_id))
        if candidates:
            manual_selected_track_id = min(candidates)[1]
            toolbox_choose_mode = False
            toolbox_status_path.write_text(json.dumps({'status': f'Selected object {manual_selected_track_id}'}), encoding='utf-8')
        else:
            toolbox_status_path.write_text(json.dumps({'status': 'No object under click'}), encoding='utf-8')

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and toolbox_choose_mode:
            choose_track_at(x, y)

    cv2.setMouseCallback(win, on_mouse)

    def is_image_source(src):
        if not isinstance(src, str):
            return False
        return Path(src).suffix.lower() in IMG_EXTS

    def smooth_polygon(poly, smooth_strength):
        if smooth_strength <= 0 or poly is None or len(poly) < 5:
            return poly
        pts = np.round(poly).astype(np.float32).reshape(-1, 1, 2)
        peri = cv2.arcLength(pts, True)
        eps = max(0.8, float(smooth_strength) * 0.0035 * peri)
        approx = cv2.approxPolyDP(pts, eps, True)
        return approx.reshape(-1, 2).astype(np.float32) if approx is not None and len(approx) >= 3 else poly

    def render_result(r):
        nonlocal prev_mask_overlay, prev_mask_alpha, prev_pick_center, tracks, next_track_id, manual_selected_track_id
        if getattr(r, 'orig_img', None) is None:
            frame = r.plot()
            return frame, 0

        frame = r.orig_img.copy()
        overlay = np.zeros_like(frame, dtype=np.float32)
        alpha_layer = np.zeros(frame.shape[:2], dtype=np.float32)
        count = 0

        if r.masks is not None and getattr(r.masks, 'xy', None) is not None:
            polys = r.masks.xy
            count = len(polys)
            smoothed_polys = []
            best_track_id = None
            best_local_idx = None
            pick_infos = []
            boxes_xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and getattr(r.boxes, 'xyxy', None) is not None else None
            detections = []
            if boxes_xyxy is not None:
                for i, poly in enumerate(polys):
                    if poly is None or len(poly) < 3 or i >= len(boxes_xyxy):
                        smoothed_polys.append(None)
                        continue
                    sp = smooth_polygon(poly, args.mask_smooth if args.pick_overlay else min(args.mask_smooth, 1))
                    smoothed_polys.append(sp)
                    detections.append({'poly': sp, 'box': tuple(float(v) for v in boxes_xyxy[i])})
                tracks, next_track_id = update_tracks(tracks, next_track_id, detections, frame.shape, max_missed=2)
            else:
                tracks = {}
            if args.pick_overlay:
                active_boxes = [tracks[tid]['box'] for tid in tracks if tracks[tid].get('matched')]
                best_local_idx, pick_infos = select_pickable_instance(active_boxes, frame.shape)
                matched_track_ids = [tid for tid in tracks if tracks[tid].get('matched')]
                best_track_id = matched_track_ids[best_local_idx] if best_local_idx is not None and best_local_idx < len(matched_track_ids) else None
                if manual_selected_track_id is not None:
                    best_track_id = manual_selected_track_id if manual_selected_track_id in tracks else None
                for track_id, track in tracks.items():
                    poly = track.get('poly')
                    if poly is None or len(poly) < 3:
                        continue
                    color = palette[(track_id - 1) % len(palette)]
                    pts = poly.astype('int32').reshape(-1, 1, 2)
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255)
                    sel = mask > 0
                    overlay[sel] = color
                    base_alpha = 0.10 if not track.get('matched') else 0.18
                    hi_alpha = 0.16 if not track.get('matched') else 0.24
                    alpha_layer[sel] = np.maximum(alpha_layer[sel], hi_alpha if track_id == best_track_id else base_alpha)

                if prev_mask_overlay is not None and prev_mask_overlay.shape == overlay.shape:
                    overlay = 0.45 * prev_mask_overlay + 0.55 * overlay
                    alpha_layer = 0.45 * prev_mask_alpha + 0.55 * alpha_layer
                prev_mask_overlay = overlay.copy()
                prev_mask_alpha = alpha_layer.copy()

                frame_f = frame.astype(np.float32)
                frame = np.clip(frame_f * (1.0 - alpha_layer[:, :, None]) + overlay * alpha_layer[:, :, None], 0, 255).astype(np.uint8)
            else:
                prev_mask_overlay = None
                prev_mask_alpha = None
                prev_pick_center = None
                simple_overlay = frame.copy()
                for track_id, track in tracks.items():
                    poly = track.get('poly')
                    if poly is None or len(poly) < 3:
                        continue
                    pts = poly.astype('int32').reshape(-1, 1, 2)
                    color = palette[(track_id - 1) % len(palette)]
                    cv2.fillPoly(simple_overlay, [pts], color)
                frame = cv2.addWeighted(frame, 0.80, simple_overlay, 0.20, 0.0)

            for track_id, track in tracks.items():
                poly = track.get('poly')
                if poly is None or len(poly) < 3:
                    continue
                cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())
                color = palette[(track_id - 1) % len(palette)]
                if args.pick_overlay and track_id == best_track_id:
                    if prev_pick_center is not None:
                        cx = int(round(0.55 * prev_pick_center[0] + 0.45 * cx))
                        cy = int(round(0.55 * prev_pick_center[1] + 0.45 * cy))
                    prev_pick_center = (cx, cy)
                    x1, y1, x2, y2 = [int(round(v)) for v in track['box']]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.putText(frame, f'PICK {track_id}', (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    text_col = color if track.get('matched') else (180, 180, 180)
                    cv2.putText(frame, str(track_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_col, 2, cv2.LINE_AA)

            if args.pick_overlay and manual_selected_track_id is None and best_track_id is not None and best_local_idx is not None and 0 <= best_local_idx < len(pick_infos):
                info = pick_infos[best_local_idx]
                cv2.putText(
                    frame,
                    f'pick={best_track_id} free={info["score"]:.2f} contacts={info["contact_count"]}',
                    (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif best_track_id is not None:
                cv2.putText(
                    frame,
                    f'pick={best_track_id}',
                    (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        else:
            prev_mask_overlay = None
            prev_mask_alpha = None
            prev_pick_center = None
            tracks = {}
            manual_selected_track_id = None

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
        nonlocal writer, save_path, display_scale, last_render_shape
        h, w = frame.shape[:2]
        last_render_shape = frame.shape[:2]
        scale = min(args.view_width / max(w, 1), args.view_height / max(h, 1), 1.0)
        display_scale = scale
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

    def step_toolbox():
        nonlocal manual_selected_track_id, toolbox_paused, toolbox_choose_mode, toolbox_last_seq
        if toolbox_proc is None:
            return True
        if toolbox_proc.poll() is not None:
            return False
        try:
            if toolbox_cmd_path.exists():
                data = json.loads(toolbox_cmd_path.read_text(encoding='utf-8'))
                seq = int(data.get('seq', -1))
                if seq > toolbox_last_seq:
                    toolbox_last_seq = seq
                    cmd = str(data.get('command', '')).strip()
                    if cmd == 'toggle_pause':
                        toolbox_paused = not toolbox_paused
                    elif cmd == 'choose_object':
                        toolbox_paused = True
                        toolbox_choose_mode = True
                    elif cmd == 'clear_selection':
                        manual_selected_track_id = None
                    toolbox_status_path.write_text(json.dumps({'status': 'Paused - click object in video' if toolbox_choose_mode else ('Paused' if toolbox_paused else 'Running')}), encoding='utf-8')
        except Exception:
            pass
        return True

    if source == 0:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam source 0')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        last_frame = None
        print(f'Webcam capture resolution requested: {args.cam_width}x{args.cam_height}')
        print(f'Webcam capture resolution actual: {actual_w}x{actual_h}')

        while True:
            if not step_toolbox():
                break
            if not toolbox_paused or last_frame is None:
                ok, raw = cap.read()
                if not ok:
                    break
                frame, _ = process_raw_frame(raw)
                last_frame = frame
            shown = add_controls_overlay(last_frame, 'webcam mode', paused=toolbox_paused)
            if toolbox_choose_mode:
                cv2.putText(shown, 'CLICK OBJECT TO SELECT', (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)
            key = show_and_maybe_save(shown)
            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                toolbox_paused = not toolbox_paused
                toolbox_status_path.write_text(json.dumps({'status': 'Paused' if toolbox_paused else 'Running'}), encoding='utf-8')
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

            last_frame = None
            video_name = Path(source).name

            def seek_relative(delta_frames):
                cur_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                target = max(0, cur_idx + delta_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)

            while True:
                if not step_toolbox():
                    break
                paused = toolbox_paused
                if not paused or last_frame is None:
                    ok, raw = cap.read()
                    if not ok:
                        break
                    frame, _ = process_raw_frame(raw)
                    last_frame = frame

                shown = add_controls_overlay(last_frame, f'video mode: {video_name}', paused=paused)
                if toolbox_choose_mode:
                    cv2.putText(shown, 'CLICK OBJECT TO SELECT', (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)
                key = show_and_maybe_save(shown, wait_ms=0 if paused else 1)
                if key in (ord('q'), 27):
                    break
                if key == ord(' '):
                    toolbox_paused = not toolbox_paused
                    toolbox_status_path.write_text(json.dumps({'status': 'Paused' if toolbox_paused else 'Running'}), encoding='utf-8')
                    continue
                if key == ord('n'):
                    toolbox_paused = True
                    toolbox_status_path.write_text(json.dumps({'status': 'Paused'}), encoding='utf-8')
                    ok, raw = cap.read()
                    if not ok:
                        break
                    frame, _ = process_raw_frame(raw)
                    last_frame = frame
                    continue
                if key == ord('j'):
                    seek_relative(-30)
                    toolbox_paused = True
                    toolbox_status_path.write_text(json.dumps({'status': 'Paused'}), encoding='utf-8')
                    last_frame = None
                    continue
                if key == ord('k'):
                    seek_relative(30)
                    toolbox_paused = True
                    toolbox_status_path.write_text(json.dumps({'status': 'Paused'}), encoding='utf-8')
                    last_frame = None
                    continue

            cap.release()

    if writer is not None:
        writer.release()
        print(f'Saved overlay video: {save_path}')

    if toolbox_proc is not None and toolbox_proc.poll() is None:
        toolbox_proc.terminate()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
