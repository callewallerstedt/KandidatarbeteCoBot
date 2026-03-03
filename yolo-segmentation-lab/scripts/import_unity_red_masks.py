#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def mask_to_polygon(mask, eps=0.0008):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps * peri, True)
    if len(approx) < 3:
        return None
    return approx.reshape(-1, 2)


def write_yolo(label_path: Path, class_id: int, poly: np.ndarray, w: int, h: int):
    vals = [str(class_id)]
    for x, y in poly:
        vals.append(f'{x / w:.6f}')
        vals.append(f'{y / h:.6f}')
    label_path.write_text(' '.join(vals) + '\n', encoding='utf-8')


def load_red_mask(mask_path: Path, threshold: int):
    m = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if m is None:
        return None
    # Unity mask format: red object on black background (BGR in OpenCV)
    red = m[:, :, 2]
    green = m[:, :, 1]
    blue = m[:, :, 0]
    bin_mask = ((red >= threshold) & (green <= threshold // 3) & (blue <= threshold // 3)).astype(np.uint8) * 255
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return bin_mask


def main():
    ap = argparse.ArgumentParser(description='Import Unity RGB + red mask pairs into YOLO-seg dataset format.')
    ap.add_argument('--rgb-dir', required=True)
    ap.add_argument('--mask-dir', required=True)
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--class-id', type=int, required=True)
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--run-name', default='')
    ap.add_argument('--red-threshold', type=int, default=120)
    args = ap.parse_args()

    rgb_dir = Path(args.rgb_dir)
    mask_dir = Path(args.mask_dir)
    if not rgb_dir.exists() or not mask_dir.exists():
        raise RuntimeError('rgb-dir and mask-dir must both exist')

    run_name = args.run_name.strip() or datetime.now().strftime('unity_%Y%m%d_%H%M%S')
    root = Path(args.data_root)
    out_img_dir = root / 'images' / args.class_name / 'unity_runs' / run_name
    out_lbl_dir = root / 'labels' / args.class_name / 'unity_runs' / run_name
    out_viz_dir = root / 'staging' / f'{args.class_name}_unity' / run_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    out_viz_dir.mkdir(parents=True, exist_ok=True)

    rgb_files = [p for p in sorted(rgb_dir.rglob('*')) if p.suffix.lower() in IMG_EXTS]
    if not rgb_files:
        raise RuntimeError('No RGB images found')

    made = 0
    for i, rgb_path in enumerate(rgb_files, start=1):
        rel = rgb_path.relative_to(rgb_dir)
        mask_path = (mask_dir / rel).with_suffix('.png')
        if not mask_path.exists():
            mask_path = (mask_dir / rel).with_suffix(rgb_path.suffix)
        if not mask_path.exists():
            continue

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        h, w = rgb.shape[:2]

        bin_mask = load_red_mask(mask_path, args.red_threshold)
        if bin_mask is None or (bin_mask > 0).sum() < 20:
            continue

        poly = mask_to_polygon(bin_mask)
        if poly is None:
            continue

        stem = f'{args.class_name}_unity_{i:06d}'
        out_img = out_img_dir / f'{stem}.jpg'
        out_lbl = out_lbl_dir / f'{stem}.txt'
        out_viz = out_viz_dir / f'{stem}_overlay.jpg'

        cv2.imwrite(str(out_img), rgb)
        write_yolo(out_lbl, args.class_id, poly, w, h)

        ov = rgb.copy()
        cv2.drawContours(ov, [poly.astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
        cv2.imwrite(str(out_viz), ov)
        made += 1

    print(f'Imported Unity samples: {made}')
    print(f'Run name: {run_name}')
    print(f'Images dir: {out_img_dir}')
    print(f'Labels dir: {out_lbl_dir}')
    print(f'Overlays dir: {out_viz_dir}')


if __name__ == '__main__':
    main()
