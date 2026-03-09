#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
_WORKER_STATE = {}


def load_yolo_polys(label_path: Path, w: int, h: int):
    txt = label_path.read_text(encoding='utf-8').strip()
    if not txt:
        return []
    polys = []
    for line in txt.splitlines():
        vals = line.strip().split()
        if len(vals) < 7:
            continue
        cls_id = int(float(vals[0]))
        pts = np.array([float(x) for x in vals[1:]], dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        polys.append((cls_id, pts))
    return polys


def poly_to_mask(poly: np.ndarray, w: int, h: int):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    return m


def mask_to_polygon(mask, eps=0.0008):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps * peri, True)
    if len(approx) < 3:
        return None
    return approx.reshape(-1, 2)


def point_in_poly(x, y, poly_pts):
    if poly_pts is None or len(poly_pts) < 3:
        return True
    contour = np.array(poly_pts, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def bbox_fully_inside_poly(px, py, ow, oh, poly_pts):
    if poly_pts is None or len(poly_pts) < 3:
        return True
    # strict containment: all bbox corners must be inside polygon
    corners = [(px, py), (px + ow, py), (px, py + oh), (px + ow, py + oh)]
    return all(point_in_poly(cx, cy, poly_pts) for cx, cy in corners)


def profile_for_bg(args, bg_path):
    if bg_path is None:
        return None
    prof = getattr(args, 'placement_profile_data', None)
    if not prof:
        return None
    items = prof.get('items', {})
    k_abs = str(bg_path).replace('\\', '/')
    k_rel = None
    try:
        k_rel = str(bg_path.relative_to(Path(args.background_dir))).replace('\\', '/')
        if k_rel in items:
            return items[k_rel]
    except Exception:
        pass

    if k_abs in items:
        return items[k_abs]

    if bg_path.name in items:
        return items[bg_path.name]

    k_abs_low = k_abs.lower()
    k_rel_low = k_rel.lower() if k_rel else None
    for kk, vv in items.items():
        k_norm = str(kk).replace('\\', '/').lower()
        if k_rel_low and (k_norm == k_rel_low or k_norm.endswith('/' + k_rel_low)):
            return vv
        if k_norm == k_abs_low or k_norm.endswith('/' + bg_path.name.lower()):
            return vv
    return None


def pick_random_background(bg_files, max_dim=1920):
    bg_path = random.choice(bg_files)
    bg = cv2.imread(str(bg_path))
    if bg is None:
        return np.full((1080, 1920, 3), 127, dtype=np.uint8), None
    bh, bw = bg.shape[:2]
    md = max(bw, bh)
    if md > max_dim:
        s = max_dim / float(md)
        bg = cv2.resize(bg, (max(1, int(bw * s)), max(1, int(bh * s))), interpolation=cv2.INTER_AREA)
    return bg.copy(), bg_path


def rotate_bound_pair(img, mask, angle):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2.0) - cx
    M[1, 2] += (nh / 2.0) - cy
    rot_img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rot_mask = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rot_img, rot_mask


def write_yolo_multi(label_path: Path, class_id: int, polys, w, h):
    lines = []
    for poly in polys:
        vals = [str(class_id)]
        for x, y in poly:
            vals.append(f'{x / w:.6f}')
            vals.append(f'{y / h:.6f}')
        lines.append(' '.join(vals))
    label_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def _worker_init(config):
    from profile_scene_builders import build_multi_scene as core_build_multi_scene
    from profile_scene_builders import collect_cutouts as core_collect_cutouts
    from profile_scene_builders import load_profile as core_load_profile

    args = SimpleNamespace(**config['args'])
    data_root = Path(config['data_root'])
    profile_data = core_load_profile(Path(config['placement_profile']))
    cutouts = core_collect_cutouts(data_root, args.class_name, getattr(args, 'source_run_filter', ''))
    _WORKER_STATE['build_multi_scene'] = core_build_multi_scene
    _WORKER_STATE['args'] = args
    _WORKER_STATE['bg_files'] = [Path(p) for p in config['bg_files']]
    _WORKER_STATE['profile_data'] = profile_data
    _WORKER_STATE['cutouts'] = cutouts
    _WORKER_STATE['reference_long_side'] = float(cutouts[0]['long_side'])


def _worker_generate_scene(task_idx):
    build_multi_scene = _WORKER_STATE['build_multi_scene']
    args = _WORKER_STATE['args']
    seed = int(args.seed) + int(task_idx) * 9973 + 17
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    scene = build_multi_scene(
        args,
        _WORKER_STATE['bg_files'],
        _WORKER_STATE['profile_data'],
        _WORKER_STATE['cutouts'],
        _WORKER_STATE['reference_long_side'],
    )
    return scene


def main():
    from profile_scene_builders import build_multi_scene as core_build_multi_scene
    from profile_scene_builders import collect_cutouts as core_collect_cutouts
    from profile_scene_builders import load_profile as core_load_profile
    from profile_scene_builders import next_output_index as core_next_output_index

    ap = argparse.ArgumentParser()
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--class-id', type=int, required=True)
    ap.add_argument('--background-dir', required=True)
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--num-synthetic', type=int, default=300)
    ap.add_argument('--min-objects', type=int, default=2)
    ap.add_argument('--max-objects', type=int, default=5)
    ap.add_argument('--overlap-prob', type=float, default=0.8)
    ap.add_argument('--max-overlap-ratio', type=float, default=0.5, help='Maximum allowed overlap ratio between any two object masks')
    ap.add_argument('--cluster-distance-factor', type=float, default=1.0, help='Max center distance in object-size units to keep cluster tight')
    ap.add_argument('--min-scale', type=float, default=0.45)
    ap.add_argument('--max-scale', type=float, default=1.10)
    ap.add_argument('--max-rotation', type=float, default=360.0)
    ap.add_argument('--overlap-spread', type=float, default=0.35, help='Jitter factor for overlap placement (higher spreads objects more)')
    ap.add_argument('--brightness-min', type=float, default=-20.0)
    ap.add_argument('--brightness-max', type=float, default=20.0)
    ap.add_argument('--object-brightness-min', type=float, default=-10.0)
    ap.add_argument('--object-brightness-max', type=float, default=10.0)
    ap.add_argument('--object-temp-bias', type=float, default=0.35, help='Object temperature bias: -1 cooler, +1 warmer')
    ap.add_argument('--object-temp-variance', type=float, default=0.18, help='Random object temperature variance around the bias')
    ap.add_argument('--object-shade-prob', type=float, default=0.45, help='Probability of partial one-sided shading per object')
    ap.add_argument('--object-shade-strength', type=float, default=0.22, help='Strength of partial object shading')
    ap.add_argument('--workers', type=int, default=1, help='Parallel workers for generation only; preview stays single-process')
    ap.add_argument('--source-run-filter', default='', help='Optional comma-separated filter for real source run folder names used to build cutouts')
    ap.add_argument('--run-name', default='')
    ap.add_argument('--preview-only', action='store_true')
    ap.add_argument('--preview-window', action='store_true')
    ap.add_argument('--preview-count', type=int, default=12)
    ap.add_argument('--preview-max-width', type=int, default=1600)
    ap.add_argument('--preview-max-height', type=int, default=900)
    ap.add_argument('--preview-mode', default='random', choices=['random', 'min_scale', 'max_scale', 'bg_bri_min', 'bg_bri_max'])
    ap.add_argument('--placement-profile', default='', help='JSON profile with per-background polygon/min_scale/max_scale')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.data_root)

    run_name = args.run_name.strip() or datetime.now().strftime('run_%Y%m%d_%H%M%S')
    out_img_dir = root / 'images' / args.class_name / 'synth_multi_runs' / run_name
    out_lbl_dir = root / 'labels' / args.class_name / 'synth_multi_runs' / run_name
    out_viz_dir = root / 'staging' / f'{args.class_name}_synth_multi' / run_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    out_viz_dir.mkdir(parents=True, exist_ok=True)

    bg_files = [p for p in Path(args.background_dir).rglob('*') if p.suffix.lower() in IMG_EXTS]
    if not bg_files:
        raise RuntimeError('No background images found')

    if not args.placement_profile.strip():
        raise RuntimeError('placement_profile is required for multi-instance generation')
    profile_data = core_load_profile(Path(args.placement_profile))
    cutouts = core_collect_cutouts(root, args.class_name, args.source_run_filter)
    reference_long_side = float(cutouts[0]['long_side'])

    preview_frames = []
    target_n = args.preview_count if args.preview_only else args.num_synthetic
    stem_prefix = f'{args.class_name}_synthmulti'
    next_idx = core_next_output_index(out_img_dir, stem_prefix)

    def _consume_scene(scene, made_count):
        if scene is None:
            return False
        h, w = scene['image'].shape[:2]
        polys_new = []
        for vm in scene['visible_masks']:
            p = mask_to_polygon(vm)
            if p is not None:
                polys_new.append(p)
        if len(polys_new) < args.min_objects:
            return False

        stem = f'{stem_prefix}_{next_idx + made_count:06d}'
        out_img = out_img_dir / f'{stem}.jpg'
        out_lbl = out_lbl_dir / f'{stem}.txt'
        out_viz = out_viz_dir / f'{stem}_overlay.jpg'

        if args.preview_only:
            preview_frames.append(scene['overlay'])
        else:
            cv2.imwrite(str(out_img), scene['image'])
            write_yolo_multi(out_lbl, args.class_id, polys_new, w, h)
        cv2.imwrite(str(out_viz), scene['overlay'])
        return True

    made = 0
    workers = max(1, int(args.workers or 1))
    if args.preview_only or workers == 1:
        while made < target_n:
            scene = core_build_multi_scene(args, bg_files, profile_data, cutouts, reference_long_side)
            if scene is None:
                break
            if _consume_scene(scene, made):
                made += 1
    else:
        worker_config = {
            'args': vars(args).copy(),
            'data_root': str(root),
            'placement_profile': args.placement_profile,
            'bg_files': [str(p) for p in bg_files],
        }
        task_idx = 0
        attempts = 0
        max_attempts = max(target_n * 6, workers * 4)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(worker_config,),
        ) as ex:
            futures = {}
            while len(futures) < workers * 2 and attempts < max_attempts:
                fut = ex.submit(_worker_generate_scene, task_idx)
                futures[fut] = task_idx
                task_idx += 1
                attempts += 1

            while futures and made < target_n:
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    futures.pop(fut, None)
                    scene = fut.result()
                    if _consume_scene(scene, made):
                        made += 1
                        if made >= target_n:
                            break
                    if attempts < max_attempts and made < target_n:
                        nf = ex.submit(_worker_generate_scene, task_idx)
                        futures[nf] = task_idx
                        task_idx += 1
                        attempts += 1

    print(f'Multi-instance synthetic created: {made}')
    if args.preview_only and made == 0:
        print('Preview note: no valid samples with current strict settings. Try lower min objects or wider placement area/profile.')
    print(f'Run name: {run_name}')
    if args.preview_only:
        print('Preview-only mode: no train images/labels written.')
    else:
        print(f'Images dir: {out_img_dir}')
        print(f'Labels dir: {out_lbl_dir}')
    print(f'Overlays dir: {out_viz_dir}')

    if args.preview_only and args.preview_window and preview_frames:
        idx = 0
        win = 'Multi-instance Preview (left/right, q to quit)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        while True:
            show = preview_frames[idx].copy()
            cv2.putText(show, f'{idx+1}/{len(preview_frames)}', (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            h0, w0 = show.shape[:2]
            s = min(args.preview_max_width / max(1, w0), args.preview_max_height / max(1, h0), 1.0)
            if s < 1.0:
                show = cv2.resize(show, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
            cv2.imshow(win, show)
            k = cv2.waitKey(0)
            if k in (ord('q'), 27):
                break
            if k in (81, 2424832, ord('a')):
                idx = max(0, idx - 1)
            elif k in (83, 2555904, ord('d')):
                idx = min(len(preview_frames)-1, idx + 1)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
