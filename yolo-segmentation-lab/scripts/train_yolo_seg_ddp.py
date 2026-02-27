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
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--batch', type=int, default=8, help='Per-process batch size')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--project', default=str(Path(__file__).resolve().parents[1] / 'runs' / 'segment'))
    ap.add_argument('--name', default='train_ddp')
    args = ap.parse_args()

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=local_rank,
        project=args.project,
        name=args.name,
        task='segment',
    )


if __name__ == '__main__':
    main()
