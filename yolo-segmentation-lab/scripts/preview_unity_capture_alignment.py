#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import cv2
import numpy as np


def pick_latest(dir_path: Path):
    files = sorted([p for p in dir_path.glob('frame_*.png')])
    return files[-1] if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unity-dir', required=True)
    ap.add_argument('--camera-tag', default='cobot')
    args = ap.parse_args()

    u = Path(args.unity_dir)
    rgb = u / 'RGB' / args.camera_tag
    msk = u / 'MASK' / args.camera_tag
    ann = u / 'annotations' / args.camera_tag

    if not rgb.exists() or not ann.exists():
        raise RuntimeError('Expected RGB/<tag> and annotations/<tag> folders')

    rgb_file = pick_latest(rgb)
    if rgb_file is None:
        raise RuntimeError('No RGB frame found')

    stem = rgb_file.stem
    msk_file = msk / f'{stem}.png'
    ann_file = ann / f'{stem}.json'
    if not ann_file.exists():
        raise RuntimeError(f'Annotation not found for {stem}')

    img = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError('Failed to read RGB image')

    overlay = img.copy()

    if msk_file.exists():
        mm = cv2.imread(str(msk_file), cv2.IMREAD_COLOR)
        if mm is not None and mm.shape[:2] == img.shape[:2]:
            # red mask to alpha overlay
            red = mm[:, :, 2].astype(np.int16)
            green = mm[:, :, 1].astype(np.int16)
            blue = mm[:, :, 0].astype(np.int16)
            mask = (red > 80) & (green < 60) & (blue < 60)
            tint = overlay.copy()
            tint[mask] = (0, 0, 255)
            overlay = cv2.addWeighted(overlay, 0.7, tint, 0.3, 0)

    data = json.loads(ann_file.read_text(encoding='utf-8'))
    objs = data.get('objects', [])
    for i, o in enumerate(objs, start=1):
        bb = o.get('bbox_xyxy', [0, 0, 0, 0])
        if len(bb) == 4:
            x1, y1, x2, y2 = [int(round(v)) for v in bb]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 2, cv2.LINE_AA)

        def draw_kp(name, color, r=5):
            kp = o.get(name, None)
            if kp and len(kp) >= 2:
                p = (int(round(kp[0])), int(round(kp[1])))
                cv2.circle(overlay, p, r, color, -1, cv2.LINE_AA)
                return p
            return None

        c = draw_kp('center', (255, 255, 255), 5)
        a = draw_kp('grip_a', (0, 255, 0), 6)
        b = draw_kp('grip_b', (0, 0, 255), 6)
        if a and b:
            cv2.line(overlay, a, b, (0, 255, 255), 2, cv2.LINE_AA)
        if c:
            cv2.putText(overlay, f'#{i}', (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    txt = f'{stem} | objs={len(objs)} | tag={args.camera_tag}'
    cv2.putText(overlay, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Unity Capture Alignment Preview (q/esc)', overlay)
    while True:
        k = cv2.waitKey(10) & 0xFF
        if k in (27, ord('q')):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
