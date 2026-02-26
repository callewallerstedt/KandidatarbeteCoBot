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
    ap.add_argument('--view-width', type=int, default=1280, help='Max overlay window width')
    ap.add_argument('--view-height', type=int, default=720, help='Max overlay window height')
    ap.add_argument('--save-video', action='store_true', help='Save displayed overlay video')
    ap.add_argument('--save-path', default='', help='Output video path (.mp4). If empty, auto path in runs/predict_overlay')
    ap.add_argument('--save-fps', type=float, default=20.0, help='Fallback FPS for saved output')
    args = ap.parse_args()

    source = 0 if args.source == '0' else args.source
    model = YOLO(args.weights)

    win = 'YOLO-Seg Overlay (q to quit)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    writer = None
    save_path = None

    for r in model.predict(source=source, stream=True, imgsz=args.imgsz, conf=args.conf, device=args.device):
        frame = r.plot()
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if writer is not None:
        writer.release()
        print(f'Saved overlay video: {save_path}')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
