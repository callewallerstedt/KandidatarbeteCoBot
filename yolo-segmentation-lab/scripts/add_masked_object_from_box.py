#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def mask_to_polygon(mask, eps=0.0015):
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


def main():
    ap = argparse.ArgumentParser(description='Create masked object sample by drawing a bbox (GrabCut).')
    ap.add_argument('--image', required=True)
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--class-id', type=int, required=True)
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--prefix', default='cutout')
    args = ap.parse_args()

    imp = Path(args.image)
    src = cv2.imread(str(imp))
    if src is None:
        raise RuntimeError(f'Cannot read image: {imp}')

    h, w = src.shape[:2]
    win = 'Draw bbox around object and press ENTER/SPACE (c to cancel)'
    rect = cv2.selectROI(win, src, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win)

    x, y, bw, bh = [int(v) for v in rect]
    if bw <= 2 or bh <= 2:
        raise RuntimeError('No valid bbox selected.')

    mask = np.zeros(src.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(src, mask, (x, y, bw, bh), bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    poly = mask_to_polygon(fg)
    if poly is None:
        raise RuntimeError('Could not extract object polygon from selection.')

    out = np.zeros_like(src)
    out[fg > 0] = src[fg > 0]

    root = Path(args.data_root)
    img_dir = root / 'images' / args.class_name / 'manual_cutouts'
    lbl_dir = root / 'labels' / args.class_name / 'manual_cutouts'
    viz_dir = root / 'staging' / f'{args.class_name}_cutouts'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(img_dir.glob(f'{args.class_name}_{args.prefix}_*.png'))
    idx = len(existing) + 1
    stem = f'{args.class_name}_{args.prefix}_{idx:06d}'

    out_img = img_dir / f'{stem}.png'
    out_lbl = lbl_dir / f'{stem}.txt'
    out_viz = viz_dir / f'{stem}_overlay.jpg'

    cv2.imwrite(str(out_img), out)
    write_yolo(out_lbl, args.class_id, poly, w, h)

    ov = src.copy()
    cv2.drawContours(ov, [poly.reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
    cv2.imwrite(str(out_viz), ov)

    print(f'Created cutout sample: {out_img}')
    print(f'Label: {out_lbl}')
    print(f'Overlay: {out_viz}')


if __name__ == '__main__':
    main()
