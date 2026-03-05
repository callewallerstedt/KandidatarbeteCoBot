#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description='Extract evenly spaced frames from video into a background image folder.')
    ap.add_argument('--video', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--num-frames', type=int, default=120)
    ap.add_argument('--prefix', default='bg')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        raise RuntimeError('Could not read total frame count from video.')

    n = max(1, min(args.num_frames, total))
    idxs = np.linspace(0, total - 1, num=n, dtype=int)

    made = 0
    for i, idx in enumerate(idxs, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        fp = out / f'{args.prefix}_{i:04d}.jpg'
        cv2.imwrite(str(fp), frame)
        made += 1

    cap.release()
    print(f'Extracted {made} frames to: {out}')


if __name__ == '__main__':
    main()
