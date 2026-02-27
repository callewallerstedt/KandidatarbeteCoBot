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
    ap.add_argument('--mode', choices=['all', 'real', 'synth', 'obs'], default='all',
                    help='all: all pairs, real: exclude *_synth_* and *_obs_*, synth: only *_synth_*, obs: only *_obs_*')
    ap.add_argument('--class-name', default='', help='Optional single class folder name to include (e.g. bottle). Empty = all classes')
    ap.add_argument('--run-name', default='', help='Optional run folder filter (e.g. run_20260227_120000)')
    args = ap.parse_args()

    if round(args.train + args.val + args.test, 6) != 1.0:
        raise ValueError('train+val+test must sum to 1.0')

    root = Path(args.data_root)
    images_root = root / 'images'
    labels_root = root / 'labels'
    dst = root / 'yolo_dataset'

    all_pairs = []
    skipped_overlay_like = 0
    classes = sorted([d.name for d in images_root.iterdir() if d.is_dir()]) if images_root.exists() else []
    if not classes:
        raise RuntimeError('No class folders found in data/images')

    if args.class_name.strip():
        wanted = args.class_name.strip()
        if wanted not in classes:
            raise RuntimeError(f'class-name not found in data/images: {wanted}')
        classes = [wanted]

    run_filter = args.run_name.strip()

    for cls in classes:
        cls_img_root = images_root / cls
        cls_lbl_root = labels_root / cls
        for im in sorted(cls_img_root.rglob('*')):
            if im.suffix.lower() not in IMG_EXTS:
                continue

            rel = im.relative_to(cls_img_root)
            rel_str = rel.as_posix()
            stem = im.stem
            stem_l = stem.lower()

            # Safety guard: never train on visualization/debug/overlay renders.
            if any(tag in stem_l for tag in ['overlay', 'preview', 'debug', '_viz']):
                skipped_overlay_like += 1
                continue

            is_synth = ('_synth_' in stem) or ('synth_runs' in rel.parts)
            is_obs = ('_obs_' in stem) or ('obs_runs' in rel.parts)
            if args.mode == 'real' and (is_synth or is_obs):
                continue
            if args.mode == 'synth' and not is_synth:
                continue
            if args.mode == 'obs' and not is_obs:
                continue
            if run_filter and run_filter not in rel_str:
                continue

            lb = (cls_lbl_root / rel).with_suffix('.txt')
            if lb.exists():
                all_pairs.append((cls, rel, im, lb))

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

    if dst.exists():
        shutil.rmtree(dst)

    for split, items in splits.items():
        (dst / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dst / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for cls, rel, im, lb in items:
            rel_base = rel.with_suffix('').as_posix().replace('/', '__')
            out_name = f'{cls}__{rel_base}{im.suffix.lower()}'
            out_label = f'{cls}__{rel_base}.txt'
            shutil.copy2(im, dst / 'images' / split / out_name)
            shutil.copy2(lb, dst / 'labels' / split / out_label)

    print(f'Dataset built at: {dst}')
    print(f'mode: {args.mode}')
    print(f'classes: {", ".join(classes)}')
    for k, v in splits.items():
        print(f'{k}: {len(v)}')
    print(f'skipped overlay/debug-like images: {skipped_overlay_like}')


if __name__ == '__main__':
    main()
