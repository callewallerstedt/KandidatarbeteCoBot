#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import cv2

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def fit(img, max_w=1600, max_h=900):
    h, w = img.shape[:2]
    s = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if s < 1:
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), s
    return img.copy(), 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bg-dir', required=True)
    ap.add_argument('--profile', required=True)
    args = ap.parse_args()

    bg_dir = Path(args.bg_dir)
    files = sorted([p for p in bg_dir.rglob('*') if p.suffix.lower() in IMG_EXTS])
    if not files:
        raise RuntimeError('No background images found')

    p = Path(args.profile)
    data = {'items': {}}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            if 'items' not in data:
                data = {'items': {}}
        except Exception:
            data = {'items': {}}

    idx = 0
    while True:
        fp = files[idx]
        key = fp.name
        item = data['items'].get(key, {'rect': [0.0, 0.0, 1.0, 1.0], 'min_scale': 0.55, 'max_scale': 1.25})

        img = cv2.imread(str(fp))
        if img is None:
            idx = (idx + 1) % len(files)
            continue

        disp, s = fit(img)
        h, w = disp.shape[:2]
        rx1, ry1, rx2, ry2 = item['rect']
        x1, y1, x2, y2 = int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h)
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
        txt = f'{idx+1}/{len(files)} {fp.name}  min={item.get("min_scale",0.55):.2f} max={item.get("max_scale",1.25):.2f}'
        cv2.putText(disp, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(disp, 'n/p next-prev | r draw rect | [/ ] min -/+ | -/= max -/+ | s save | q quit', (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow('BG Placement Profile Editor', disp)
        k = cv2.waitKey(0) & 0xFF

        if k in (ord('q'), 27):
            break
        if k == ord('n'):
            idx = min(len(files)-1, idx + 1)
            continue
        if k == ord('p'):
            idx = max(0, idx - 1)
            continue
        if k == ord('r'):
            sel_img, s2 = fit(img)
            x, y, ww, hh = cv2.selectROI('BG Placement Profile Editor', sel_img, fromCenter=False, showCrosshair=True)
            if ww > 1 and hh > 1:
                ih, iw = img.shape[:2]
                # ROI was on fitted image
                fx1 = (x / max(1, sel_img.shape[1]))
                fy1 = (y / max(1, sel_img.shape[0]))
                fx2 = ((x + ww) / max(1, sel_img.shape[1]))
                fy2 = ((y + hh) / max(1, sel_img.shape[0]))
                item['rect'] = [round(float(fx1), 4), round(float(fy1), 4), round(float(fx2), 4), round(float(fy2), 4)]
                data['items'][key] = item
            continue
        if k == ord('['):
            item['min_scale'] = round(max(0.05, float(item.get('min_scale', 0.55)) - 0.05), 3)
            data['items'][key] = item
            continue
        if k == ord(']'):
            item['min_scale'] = round(min(3.0, float(item.get('min_scale', 0.55)) + 0.05), 3)
            data['items'][key] = item
            continue
        if k == ord('-'):
            item['max_scale'] = round(max(0.05, float(item.get('max_scale', 1.25)) - 0.05), 3)
            data['items'][key] = item
            continue
        if k == ord('='):
            item['max_scale'] = round(min(3.0, float(item.get('max_scale', 1.25)) + 0.05), 3)
            data['items'][key] = item
            continue
        if k == ord('s'):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=2), encoding='utf-8')
            print(f'Saved profile: {p}')
            continue

    cv2.destroyAllWindows()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f'Saved profile: {p}')


if __name__ == '__main__':
    main()
