#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def norm(v, m):
    return max(0.0, min(1.0, float(v) / max(1.0, float(m))))


def main():
    ap = argparse.ArgumentParser(description='Convert Unity RGB+annotations JSON to YOLO pose dataset.')
    ap.add_argument('--unity-dir', required=True, help='Folder containing RGB/ and annotations/')
    ap.add_argument('--out-dir', required=True, help='Output dataset folder (contains images/ labels/ dataset.yaml)')
    ap.add_argument('--train-ratio', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--class-name', default='object')
    args = ap.parse_args()

    u = Path(args.unity_dir)
    rgb_dir = u / 'RGB'
    ann_dir = u / 'annotations'
    if not rgb_dir.exists() or not ann_dir.exists():
        raise RuntimeError('unity-dir must contain RGB/ and annotations/')

    frames = []
    for jp in sorted(ann_dir.glob('*.json')):
        d = json.loads(jp.read_text(encoding='utf-8'))
        img_name = d.get('image', '')
        img = rgb_dir / img_name
        if not img.exists() or img.suffix.lower() not in IMG_EXTS:
            continue
        if not isinstance(d.get('objects', []), list):
            continue
        frames.append((img, d))

    if not frames:
        raise RuntimeError('No valid frame annotations found.')

    random.seed(args.seed)
    random.shuffle(frames)
    n_train = int(len(frames) * args.train_ratio)
    splits = {
        'train': frames[:n_train],
        'val': frames[n_train:],
    }

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)

    for split, items in splits.items():
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

        for i, (img, d) in enumerate(items, start=1):
            stem = f'frame_{i:06d}'
            shutil.copy2(img, out / 'images' / split / f'{stem}{img.suffix.lower()}')

            w = float(d.get('width', 0))
            h = float(d.get('height', 0))
            if w <= 0 or h <= 0:
                continue

            lines = []
            for o in d.get('objects', []):
                c = int(o.get('class_id', 0))
                if 'bbox_xyxy' not in o or len(o['bbox_xyxy']) != 4:
                    continue
                x1, y1, x2, y2 = o['bbox_xyxy']
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                bw, bh = (x2 - x1), (y2 - y1)
                if bw <= 1 or bh <= 1:
                    continue

                k1 = o.get('center', [cx, cy, 2])
                k2 = o.get('grip_a', [x1, cy, 2])
                k3 = o.get('grip_b', [x2, cy, 2])
                if len(k1) < 3 or len(k2) < 3 or len(k3) < 3:
                    continue

                vals = [
                    c,
                    norm(cx, w), norm(cy, h), norm(bw, w), norm(bh, h),
                    norm(k1[0], w), norm(k1[1], h), int(k1[2]),
                    norm(k2[0], w), norm(k2[1], h), int(k2[2]),
                    norm(k3[0], w), norm(k3[1], h), int(k3[2]),
                ]
                lines.append(' '.join(str(v) for v in vals))

            (out / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')

    (out / 'dataset.yaml').write_text(
        f"""path: {out.as_posix()}\ntrain: images/train\nval: images/val\nkpt_shape: [3, 3]\nflip_idx: [0, 2, 1]\nnames:\n  0: {args.class_name}\n""",
        encoding='utf-8',
    )

    print(f'Converted Unity pose frames: {len(frames)}')
    print(f'Pose dataset: {out}')


if __name__ == '__main__':
    main()
