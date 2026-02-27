#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='yolo11n-seg.pt')
    ap.add_argument('--data', default=str(Path(__file__).resolve().parents[1] / 'dataset.yaml'))
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='0')
    ap.add_argument('--project', default=str(Path(__file__).resolve().parents[1] / 'runs' / 'segment'))
    ap.add_argument('--name', default='train')
    ap.add_argument('--workers', type=int, default=(0 if os.name == 'nt' else 4))
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        task='segment',
        workers=args.workers,
    )


if __name__ == '__main__':
    main()
