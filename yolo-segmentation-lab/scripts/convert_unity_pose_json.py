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
    rgb_dir = None
    ann_dir = None
    for d in u.iterdir():
        if not d.is_dir():
            continue
        n = d.name.lower()
        if rgb_dir is None and n in {'rgb', 'images', 'image', 'color', 'render', 'renders', 'camera', 'frames'}:
            rgb_dir = d
        if ann_dir is None and n in {'annotations', 'annotation', 'ann', 'json'}:
            ann_dir = d
    if rgb_dir is None:
        c = u / 'RGB'
        if c.exists(): rgb_dir = c
    if ann_dir is None:
        c = u / 'annotations'
        if c.exists(): ann_dir = c
    if not rgb_dir or not ann_dir or not rgb_dir.exists() or not ann_dir.exists():
        raise RuntimeError('unity-dir must contain an RGB-like image folder and an annotations JSON folder')
    frames = []
    for jp in sorted(ann_dir.rglob('*.json')):
        d = json.loads(jp.read_text(encoding='utf-8'))
        img_name = d.get('image', '')
        img = rgb_dir / img_name
        if not img.exists():
            # try same relative subdir as annotation file
            relp = jp.relative_to(ann_dir).parent
            img = rgb_dir / relp / img_name
        if not img.exists() or img.suffix.lower() not in IMG_EXTS:
            continue
        if not isinstance(d.get('objects', []), list):
            continue
        frames.append((img, d))

    if not frames:
        raise RuntimeError('No valid frame annotations found.')

    # Build contiguous class-id map for YOLO (0..N-1)
    raw_class_ids = set()
    for _img, d in frames:
        for o in d.get('objects', []):
            raw_class_ids.add(int(o.get('class_id', o.get('class', 0))))
    if not raw_class_ids:
        raw_class_ids = {0}
    class_map = {cid: i for i, cid in enumerate(sorted(raw_class_ids))}

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
                c = int(o.get('class_id', o.get('class', 0)))
                raw_class_ids.add(c)
                bbox = o.get('bbox_xyxy') or o.get('bbox') or o.get('xyxy')
                if bbox is None and all(k in o for k in ['x1', 'y1', 'x2', 'y2']):
                    bbox = [o['x1'], o['y1'], o['x2'], o['y2']]
                if bbox is None or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                bw, bh = (x2 - x1), (y2 - y1)
                if bw <= 1 or bh <= 1:
                    continue

                k1 = o.get('center', o.get('kp_center', [cx, cy, 2]))
                k2 = o.get('grip_a', o.get('gripA', o.get('kp_a', [x1, cy, 2])))
                k3 = o.get('grip_b', o.get('gripB', o.get('kp_b', [x2, cy, 2])))
                if len(k1) < 3 or len(k2) < 3 or len(k3) < 3:
                    continue

                vals = [
                    class_map[c],
                    norm(cx, w), norm(cy, h), norm(bw, w), norm(bh, h),
                    norm(k1[0], w), norm(k1[1], h), int(k1[2]),
                    norm(k2[0], w), norm(k2[1], h), int(k2[2]),
                    norm(k3[0], w), norm(k3[1], h), int(k3[2]),
                ]
                lines.append(' '.join(str(v) for v in vals))

            (out / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')

    names_lines = []
    for raw_cid in sorted(raw_class_ids):
        mapped = class_map[raw_cid]
        cname = args.class_name if mapped == 0 else f'class_{raw_cid}'
        names_lines.append(f'  {mapped}: {cname}')

    (out / 'dataset.yaml').write_text(
        f"""path: {out.as_posix()}\ntrain: images/train\nval: images/val\nkpt_shape: [3, 3]\nflip_idx: [0, 2, 1]\nnames:\n""" + "\n".join(names_lines) + "\n",
        encoding='utf-8',
    )

    print(f'Converted Unity pose frames: {len(frames)}')
    print(f'Pose dataset: {out}')


if __name__ == '__main__':
    main()
