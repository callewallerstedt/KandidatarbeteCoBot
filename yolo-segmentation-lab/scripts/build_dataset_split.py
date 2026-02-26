#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--train', type=float, default=0.8)
    ap.add_argument('--val', type=float, default=0.1)
    ap.add_argument('--test', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mode', choices=['all', 'real', 'synth'], default='all',
                    help='all: use all pairs, real: exclude *_synth_* files, synth: include only *_synth_* files')
    args = ap.parse_args()

    if round(args.train + args.val + args.test, 6) != 1.0:
        raise ValueError('train+val+test must sum to 1.0')

    root = Path(args.data_root)
    images_root = root / 'images'
    labels_root = root / 'labels'
    dst = root / 'yolo_dataset'

    all_pairs = []
    classes = sorted([d.name for d in images_root.iterdir() if d.is_dir()]) if images_root.exists() else []
    if not classes:
        raise RuntimeError('No class folders found in data/images')

    for cls in classes:
        for im in sorted((images_root / cls).iterdir()):
            if im.suffix.lower() not in IMG_EXTS:
                continue

            is_synth = '_synth_' in im.stem
            if args.mode == 'real' and is_synth:
                continue
            if args.mode == 'synth' and not is_synth:
                continue

            lb = labels_root / cls / f'{im.stem}.txt'
            if lb.exists():
                all_pairs.append((im, lb))

    if not all_pairs:
        raise RuntimeError('No image-label pairs found.')

    random.seed(args.seed)
    random.shuffle(all_pairs)

    n = len(all_pairs)
    n_train = int(n * args.train)
    n_val = int(n * args.val)

    splits = {
        'train': all_pairs[:n_train],
        'val': all_pairs[n_train:n_train+n_val],
        'test': all_pairs[n_train+n_val:],
    }

    for split, items in splits.items():
        (dst / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dst / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for im, lb in items:
            shutil.copy2(im, dst / 'images' / split / im.name)
            shutil.copy2(lb, dst / 'labels' / split / lb.name)

    print(f'Dataset built at: {dst}')
    print(f'mode: {args.mode}')
    for k, v in splits.items():
        print(f'{k}: {len(v)}')


if __name__ == '__main__':
    main()
