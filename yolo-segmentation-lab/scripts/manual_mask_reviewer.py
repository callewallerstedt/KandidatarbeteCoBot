#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def load_label(label_path: Path, w: int, h: int):
    if not label_path.exists():
        return None, None
    txt = label_path.read_text(encoding='utf-8').strip()
    if not txt:
        return None, None
    vals = txt.split()
    cid = int(float(vals[0]))
    pts = np.array([float(v) for v in vals[1:]], dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return cid, pts


def mask_from_poly(poly, w, h):
    m = np.zeros((h, w), dtype=np.uint8)
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    return m


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


def save_label(path: Path, cid: int, poly: np.ndarray, w: int, h: int):
    vals = [str(cid)]
    for x, y in poly:
        vals.append(f'{x / w:.6f}')
        vals.append(f'{y / h:.6f}')
    path.write_text(' '.join(vals) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='Simple manual mask reviewer/editor for YOLO-seg labels')
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--class-id', type=int, required=True)
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--contains', default='manual_', help='Only review filenames containing this token')
    ap.add_argument('--brush', type=int, default=14)
    args = ap.parse_args()

    root = Path(args.data_root)
    img_dir = root / 'images' / args.class_name
    lbl_dir = root / 'labels' / args.class_name
    viz_dir = root / 'staging' / f'{args.class_name}_manual_review'
    viz_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(img_dir.glob('*')) if p.suffix.lower() in IMG_EXTS and args.contains in p.stem]
    if not files:
        raise RuntimeError(f'No files found in {img_dir} with token {args.contains}')

    idx = 0
    brush = max(1, args.brush)
    mode = 'add'  # add or erase

    win = 'Manual Mask Reviewer'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    drawing = {'down': False, 'mask': None}

    def on_mouse(event, x, y, flags, param):
        if drawing['mask'] is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing['down'] = True
        elif event == cv2.EVENT_LBUTTONUP:
            drawing['down'] = False
        if drawing['down'] and (flags & cv2.EVENT_FLAG_LBUTTON):
            val = 255 if mode == 'add' else 0
            cv2.circle(drawing['mask'], (x, y), brush, val, -1)

    cv2.setMouseCallback(win, on_mouse)

    while 0 <= idx < len(files):
        img_path = files[idx]
        lbl_path = lbl_dir / f'{img_path.stem}.txt'
        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue
        h, w = img.shape[:2]
        cid, poly = load_label(lbl_path, w, h)
        if cid is None:
            cid = args.class_id
        mask = mask_from_poly(poly, w, h)
        drawing['mask'] = mask.copy()

        while True:
            overlay = img.copy()
            m = drawing['mask']
            if m is not None:
                green = np.zeros_like(overlay)
                green[:, :, 1] = 180
                alpha = (m > 0)[:, :, None]
                overlay = np.where(alpha, (0.65 * overlay + 0.35 * green).astype(np.uint8), overlay)

            info = f'{idx+1}/{len(files)} | mode={mode} | brush={brush} | n/p next-prev, s save, r reset, a add, e erase, +/- brush, q quit'
            cv2.putText(overlay, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, overlay)

            k = cv2.waitKey(20) & 0xFF
            if k == ord('q'):
                cv2.destroyAllWindows()
                return
            if k == ord('a'):
                mode = 'add'
            elif k == ord('e'):
                mode = 'erase'
            elif k in (ord('+'), ord('=')):
                brush = min(200, brush + 2)
            elif k == ord('-'):
                brush = max(1, brush - 2)
            elif k == ord('r'):
                drawing['mask'] = mask.copy()
            elif k == ord('s'):
                poly_new = mask_to_polygon(drawing['mask'])
                if poly_new is not None:
                    save_label(lbl_path, cid, poly_new, w, h)
                    viz = img.copy()
                    cv2.drawContours(viz, [poly_new.astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
                    cv2.imwrite(str(viz_dir / f'{img_path.stem}_edited_overlay.jpg'), viz)
                    print(f'Saved: {lbl_path.name}')
                else:
                    print(f'Skipped save (mask empty): {lbl_path.name}')
            elif k == ord('n'):
                idx += 1
                break
            elif k == ord('p'):
                idx = max(0, idx - 1)
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
