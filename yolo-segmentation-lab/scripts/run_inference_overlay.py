#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime
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
    args = ap.parse_args()

    source = 0 if args.source == '0' else args.source
    model = YOLO(args.weights)

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
            for i, poly in enumerate(polys):
                color = palette[i % len(palette)]
                if poly is None or len(poly) < 3:
                    continue
                pts = poly.astype('int32').reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], color)

            # Blend once to preserve image quality (avoid repeated resampling)
            frame = cv2.addWeighted(frame, 0.62, overlay, 0.38, 0.0)

            for i, poly in enumerate(polys):
                color = palette[i % len(palette)]
                if poly is None or len(poly) < 3:
                    continue
                pts = poly.astype('int32').reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
                cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())
                cv2.putText(frame, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.putText(frame, f'count={count}', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return frame, count

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
            r = model.predict(raw, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
            frame, count = render_result(r)
            if args.count_log and (frame_idx % max(1, args.count_log_every) == 0):
                print(f'frame {frame_idx}: count={count}')
            if show_and_maybe_save(frame):
                break
        cap.release()
    else:
        for r in model.predict(source=source, stream=True, imgsz=args.imgsz, conf=args.conf, device=args.device):
            frame_idx += 1
            frame, count = render_result(r)
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
