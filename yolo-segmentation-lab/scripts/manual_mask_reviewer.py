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


def mask_to_polygon(mask, eps=0.002):
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
    ap = argparse.ArgumentParser(description='Manual mask reviewer/editor (smooth draw + zoom)')
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--class-id', type=int, required=True)
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--contains', default='manual_', help='Only review filenames containing this token')
    ap.add_argument('--brush', type=int, default=16)
    args = ap.parse_args()

    root = Path(args.data_root)
    img_dir = root / 'images' / args.class_name
    lbl_dir = root / 'labels' / args.class_name
    viz_dir = root / 'staging' / f'{args.class_name}_manual_review'
    viz_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(img_dir.rglob('*')) if p.suffix.lower() in IMG_EXTS and args.contains in p.stem]
    if not files:
        raise RuntimeError(f'No files found in {img_dir} with token {args.contains}')

    idx = 0
    brush = max(1, args.brush)
    mode = 'add'
    zoom = 1.0

    win = 'Manual Mask Reviewer'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    state = {'down': False, 'mask': None, 'last_pt': None, 'img_shape': (1, 1)}

    def to_img_coords(xd, yd):
        h, w = state['img_shape']
        xi = int(np.clip(round(xd / zoom), 0, w - 1))
        yi = int(np.clip(round(yd / zoom), 0, h - 1))
        return xi, yi

    def on_mouse(event, x, y, flags, param):
        if state['mask'] is None:
            return
        xi, yi = to_img_coords(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            state['down'] = True
            state['last_pt'] = (xi, yi)
            val = 255 if mode == 'add' else 0
            cv2.circle(state['mask'], (xi, yi), brush, val, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            state['down'] = False
            state['last_pt'] = None
        elif event == cv2.EVENT_MOUSEMOVE and state['down'] and (flags & cv2.EVENT_FLAG_LBUTTON):
            val = 255 if mode == 'add' else 0
            if state['last_pt'] is not None:
                cv2.line(state['mask'], state['last_pt'], (xi, yi), val, thickness=max(1, brush * 2), lineType=cv2.LINE_AA)
            cv2.circle(state['mask'], (xi, yi), brush, val, -1)
            state['last_pt'] = (xi, yi)

    cv2.setMouseCallback(win, on_mouse)

    while 0 <= idx < len(files):
        img_path = files[idx]
        rel = img_path.relative_to(img_dir)
        lbl_path = (lbl_dir / rel).with_suffix('.txt')
        lbl_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue

        h, w = img.shape[:2]
        state['img_shape'] = (h, w)
        cid, poly = load_label(lbl_path, w, h)
        if cid is None:
            cid = args.class_id
        mask0 = mask_from_poly(poly, w, h)
        state['mask'] = mask0.copy()

        while True:
            overlay = img.copy()
            m = state['mask']
            green = np.zeros_like(overlay)
            green[:, :, 1] = 180
            alpha = (m > 0)[:, :, None]
            overlay = np.where(alpha, (0.65 * overlay + 0.35 * green).astype(np.uint8), overlay)

            info = (
                f'{idx+1}/{len(files)} mode={mode} brush={brush} zoom={zoom:.1f} | '
                'draw LMB, a/e add-erase, +/- brush, z/x zoom, s save, n/p next-prev, r reset, q quit'
            )
            cv2.putText(overlay, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            if abs(zoom - 1.0) > 1e-6:
                disp = cv2.resize(overlay, (int(w * zoom), int(h * zoom)), interpolation=cv2.INTER_LINEAR)
            else:
                disp = overlay
            cv2.imshow(win, disp)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                cv2.destroyAllWindows()
                return
            if k == ord('a'):
                mode = 'add'
            elif k == ord('e'):
                mode = 'erase'
            elif k in (ord('+'), ord('=')):
                brush = min(220, brush + 2)
            elif k == ord('-'):
                brush = max(1, brush - 2)
            elif k == ord('z'):
                zoom = min(4.0, zoom + 0.2)
            elif k == ord('x'):
                zoom = max(0.5, zoom - 0.2)
            elif k == ord('r'):
                state['mask'] = mask0.copy()
            elif k == ord('s'):
                poly_new = mask_to_polygon(state['mask'])
                if poly_new is not None:
                    save_label(lbl_path, cid, poly_new, w, h)
                    viz = img.copy()
                    cv2.drawContours(viz, [poly_new.astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
                    out_viz = viz_dir / (rel.as_posix().replace('/', '__') + '_edited_overlay.jpg')
                    cv2.imwrite(str(out_viz), viz)
                    print(f'Saved: {lbl_path}')
                else:
                    print(f'Skipped save (mask empty): {lbl_path}')
            elif k == ord('n'):
                idx += 1
                break
            elif k == ord('p'):
                idx = max(0, idx - 1)
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
