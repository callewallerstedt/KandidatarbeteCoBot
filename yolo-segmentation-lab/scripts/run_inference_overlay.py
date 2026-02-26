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
    args = ap.parse_args()

    source = 0 if args.source == '0' else args.source
    model = YOLO(args.weights)

    for r in model.predict(source=source, stream=True, imgsz=args.imgsz, conf=args.conf, device=args.device):
        frame = r.plot()
        cv2.imshow('YOLO-Seg Overlay (q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
