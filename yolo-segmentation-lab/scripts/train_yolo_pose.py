#!/usr/bin/env python3
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='yolo11s-pose.pt')
    ap.add_argument('--data', default=str(Path(__file__).resolve().parents[1] / 'data_pose' / 'dataset' / 'dataset.yaml'))
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--project', default=str(Path(__file__).resolve().parents[1] / 'runs' / 'pose'))
    ap.add_argument('--name', default='train')
    args = ap.parse_args()

    YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        task='pose',
    )


if __name__ == '__main__':
    main()
