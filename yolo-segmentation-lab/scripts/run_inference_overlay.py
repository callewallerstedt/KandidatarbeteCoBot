#!/usr/bin/env python3
import argparse
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
    args = ap.parse_args()

    source = 0 if args.source == '0' else args.source
    model = YOLO(args.weights)

    win = 'YOLO-Seg Overlay (q to quit)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    for r in model.predict(source=source, stream=True, imgsz=args.imgsz, conf=args.conf, device=args.device):
        frame = r.plot()
        h, w = frame.shape[:2]
        scale = min(args.view_width / max(w, 1), args.view_height / max(h, 1), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
