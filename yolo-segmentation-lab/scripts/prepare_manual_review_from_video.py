#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
from rembg import remove


def mask_to_polygon(mask, eps=0.003):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps * peri, True)
    if len(approx) < 3:
        return None
    return approx.reshape(-1, 2)


def write_yolo(label_path: Path, class_id: int, poly, w, h):
    vals = [str(class_id)]
    for x, y in poly:
        vals.append(f'{x / w:.6f}')
        vals.append(f'{y / h:.6f}')
    label_path.write_text(' '.join(vals) + '\n', encoding='utf-8')


def foreground_mask(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out = remove(rgb)
    if out.shape[2] == 4:
        alpha = out[:, :, 3]
    else:
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--class-id', type=int, required=True)
    ap.add_argument('--num-samples', type=int, default=80)
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--prefix', default='manual')
    args = ap.parse_args()

    root = Path(args.data_root)
    img_dir = root / 'images' / args.class_name
    lbl_dir = root / 'labels' / args.class_name
    viz_dir = root / 'staging' / f'{args.class_name}_manual_review'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        raise RuntimeError('Could not detect total frame count for even sampling.')

    n = min(max(args.num_samples, 1), total)
    picks = set(np.linspace(0, total - 1, num=n, dtype=int).tolist())

    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx not in picks:
            idx += 1
            continue

        mask = foreground_mask(frame)
        poly = mask_to_polygon(mask)
        if poly is None:
            idx += 1
            continue

        h, w = frame.shape[:2]
        stem = f'{args.class_name}_{args.prefix}_{saved + 1:06d}'
        imp = img_dir / f'{stem}.jpg'
        lbp = lbl_dir / f'{stem}.txt'
        vsp = viz_dir / f'{stem}_overlay.jpg'

        cv2.imwrite(str(imp), frame)
        write_yolo(lbp, args.class_id, poly, w, h)
        ov = frame.copy()
        cv2.drawContours(ov, [poly.reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
        cv2.imwrite(str(vsp), ov)

        saved += 1
        idx += 1

    cap.release()
    print(f'Prepared {saved} frames for manual review.')
    print(f'Images: {img_dir}')
    print(f'Labels: {lbl_dir}')
    print(f'Overlays: {viz_dir}')


if __name__ == '__main__':
    main()
