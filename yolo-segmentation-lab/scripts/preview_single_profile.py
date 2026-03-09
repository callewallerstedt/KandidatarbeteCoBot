#!/usr/bin/env python3
import argparse
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from profile_scene_builders import build_single_scene
from profile_scene_builders import collect_cutouts
from profile_scene_builders import load_profile
from profile_scene_builders import IMG_EXTS


def show_preview_window(frames, max_width, max_height):
    idx = 0
    win = 'Single Synth Profile Preview (left/right, q to quit)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        frame = frames[idx].copy()
        cv2.putText(frame, f'{idx + 1}/{len(frames)}', (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        h, w = frame.shape[:2]
        scale = min(max_width / max(1, w), max_height / max(1, h), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imshow(win, frame)
        key = cv2.waitKey(0)
        if key in (ord('q'), 27):
            break
        if key in (81, 2424832, ord('a')):
            idx = max(0, idx - 1)
        elif key in (83, 2555904, ord('d')):
            idx = min(len(frames) - 1, idx + 1)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--class-id', type=int, required=True)
    ap.add_argument('--background-dir', required=True)
    ap.add_argument('--placement-profile', required=True)
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--preview-count', type=int, default=12)
    ap.add_argument('--preview-window', action='store_true')
    ap.add_argument('--preview-max-width', type=int, default=1600)
    ap.add_argument('--preview-max-height', type=int, default=900)
    ap.add_argument('--max-rotation', type=float, default=360.0)
    ap.add_argument('--object-temp-bias', type=float, default=0.35)
    ap.add_argument('--object-temp-variance', type=float, default=0.18)
    ap.add_argument('--object-shade-prob', type=float, default=0.45)
    ap.add_argument('--object-shade-strength', type=float, default=0.22)
    ap.add_argument('--source-run-filter', default='', help='Optional comma-separated filter for real source run folder names used to build cutouts')
    ap.add_argument('--run-name', default='')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    args.preview_mode = 'random'
    random.seed(args.seed)
    np.random.seed(args.seed)

    bg_dir = Path(args.background_dir)
    bg_files = [p for p in bg_dir.rglob('*') if p.suffix.lower() in IMG_EXTS]
    if not bg_files:
        raise RuntimeError(f'No background images found in {bg_dir}')

    profile_data = load_profile(Path(args.placement_profile))
    cutouts = collect_cutouts(Path(args.data_root), args.class_name, args.source_run_filter)
    reference_long_side = float(cutouts[0]['long_side'])

    run_name = args.run_name.strip() or datetime.now().strftime('preview_%Y%m%d_%H%M%S')
    out_dir = Path(args.data_root) / 'staging' / f'{args.class_name}_synth_preview2' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    made = 0
    while made < args.preview_count:
        scene = build_single_scene(args, bg_files, profile_data, cutouts, reference_long_side)
        if scene is None:
            break
        out_path = out_dir / f'{args.class_name}_previewsingle_{made + 1:06d}_overlay.jpg'
        cv2.imwrite(str(out_path), scene['overlay'])
        frames.append(scene['overlay'])
        made += 1

    print(f'Profile-based single preview created: {made}')
    print(f'Overlays dir: {out_dir}')
    if made == 0:
        print('No preview frames produced. Check placement profile coverage and object size settings.')
        return

    if args.preview_window:
        show_preview_window(frames, args.preview_max_width, args.preview_max_height)


if __name__ == '__main__':
    main()
