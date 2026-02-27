#!/usr/bin/env python3
import argparse
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def load_yolo_polygon(label_path: Path, w: int, h: int):
    txt = label_path.read_text(encoding='utf-8').strip()
    if not txt:
        return None, None
    vals = txt.split()
    cls_id = int(float(vals[0]))
    pts = np.array([float(x) for x in vals[1:]], dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return cls_id, pts


def poly_to_mask(poly: np.ndarray, w: int, h: int):
    m = np.zeros((h, w), dtype=np.uint8)
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


def write_yolo(label_path: Path, class_id: int, poly: np.ndarray, w: int, h: int):
    vals = [str(class_id)]
    for x, y in poly:
        vals.append(f'{x / w:.6f}')
        vals.append(f'{y / h:.6f}')
    label_path.write_text(' '.join(vals) + '\n', encoding='utf-8')


def pick_random_background(bg_files, max_dim=1920):
    bg = cv2.imread(str(random.choice(bg_files)))
    if bg is None:
        return np.full((1080, 1920, 3), 127, dtype=np.uint8)
    bh, bw = bg.shape[:2]
    # Keep original background proportions; only downscale very large images.
    md = max(bw, bh)
    if md > max_dim:
        s = max_dim / float(md)
        bg = cv2.resize(bg, (max(1, int(bw * s)), max(1, int(bh * s))), interpolation=cv2.INTER_AREA)
    return bg.copy()


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


def compose_once(pairs, bg_files, args, fixed_scale=None, fixed_bg_beta=None, fixed_obj_beta=None):
    im_path, lb_path = random.choice(pairs)
    src = cv2.imread(str(im_path))
    if src is None:
        return None
    src_h, src_w = src.shape[:2]
    cls_id, poly = load_yolo_polygon(lb_path, src_w, src_h)
    if poly is None:
        return None

    mask = poly_to_mask(poly, src_w, src_h)
    x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
    crop = src[y:y + bh, x:x + bw]
    crop_mask = mask[y:y + bh, x:x + bw]

    bg = pick_random_background(bg_files)
    h, w = bg.shape[:2]

    scale = fixed_scale if fixed_scale is not None else random.uniform(args.min_scale, args.max_scale)
    angle = random.uniform(-args.max_rotation, args.max_rotation)
    tw = max(8, int(bw * scale))
    th = max(8, int(bh * scale))

    crop_r = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)
    m_r = cv2.resize(crop_mask, (tw, th), interpolation=cv2.INTER_NEAREST)

    crop_rr, m_rr = rotate_bound_pair(crop_r, m_r, angle)

    ys, xs = np.where(m_rr > 0)
    if len(xs) < 20:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    crop_rr = crop_rr[y1:y2 + 1, x1:x2 + 1]
    m_rr = m_rr[y1:y2 + 1, x1:x2 + 1]
    oh, ow = m_rr.shape[:2]

    if ow >= w or oh >= h:
        return None

    px = random.randint(0, w - ow)
    py = random.randint(0, h - oh)

    obj_beta = fixed_obj_beta if fixed_obj_beta is not None else random.uniform(args.object_brightness_min, args.object_brightness_max)
    crop_rr = cv2.convertScaleAbs(crop_rr, alpha=1.0, beta=obj_beta)

    roi = bg[py:py + oh, px:px + ow]
    alpha = (m_rr > 0)
    roi[alpha] = crop_rr[alpha]
    bg[py:py + oh, px:px + ow] = roi

    bg_beta = fixed_bg_beta if fixed_bg_beta is not None else random.uniform(args.brightness_min, args.brightness_max)
    bg = cv2.convertScaleAbs(bg, alpha=1.0, beta=bg_beta)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[py:py + oh, px:px + ow] = m_rr
    poly_new = mask_to_polygon(full_mask)
    if poly_new is None:
        return None

    return {
        'image': bg,
        'poly': poly_new,
        'w': w,
        'h': h,
        'cls_id': args.class_id if cls_id is None else cls_id,
        'scale': scale,
        'bg_beta': bg_beta,
        'obj_beta': obj_beta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--class-id', type=int, required=True)
    ap.add_argument('--background-dir', required=True)
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--num-synthetic', type=int, default=300)
    ap.add_argument('--min-scale', type=float, default=0.55)
    ap.add_argument('--max-scale', type=float, default=1.25)
    ap.add_argument('--max-rotation', type=float, default=25.0)
    ap.add_argument('--brightness-min', type=float, default=-20.0, help='Background brightness min shift (beta)')
    ap.add_argument('--brightness-max', type=float, default=20.0, help='Background brightness max shift (beta)')
    ap.add_argument('--object-brightness-min', type=float, default=-10.0, help='Object brightness min shift (beta)')
    ap.add_argument('--object-brightness-max', type=float, default=10.0, help='Object brightness max shift (beta)')
    ap.add_argument('--run-name', default='', help='Synthetic run folder name (default timestamp)')
    ap.add_argument('--preview-only', action='store_true', help='Show/save preview only, do not write training data')
    ap.add_argument('--preview-window', action='store_true', help='Show preview window')
    ap.add_argument('--preview-count', type=int, default=12, help='How many random previews to generate in preview-only mode')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.data_root)
    src_img_dir = root / 'images' / args.class_name
    src_lbl_dir = root / 'labels' / args.class_name

    run_name = args.run_name.strip() or datetime.now().strftime('run_%Y%m%d_%H%M%S')
    out_img_dir = root / 'images' / args.class_name / 'synth_runs' / run_name
    out_lbl_dir = root / 'labels' / args.class_name / 'synth_runs' / run_name
    out_viz_dir = root / 'staging' / f'{args.class_name}_synth' / run_name

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    out_viz_dir.mkdir(parents=True, exist_ok=True)

    bg_dir = Path(args.background_dir)
    bg_files = [p for p in bg_dir.rglob('*') if p.suffix.lower() in IMG_EXTS]
    if not bg_files:
        raise RuntimeError(f'No background images found in {bg_dir}')

    pairs = []
    for im in sorted(src_img_dir.rglob('*')):
        if im.suffix.lower() not in IMG_EXTS:
            continue
        rel_parts = set(im.relative_to(src_img_dir).parts)
        if 'synth_runs' in rel_parts or 'obs_runs' in rel_parts:
            continue
        if any(tag in im.stem for tag in ['_synth_', '_obs_']):
            continue
        lb = (src_lbl_dir / im.relative_to(src_img_dir)).with_suffix('.txt')
        if lb.exists():
            pairs.append((im, lb))
    if not pairs:
        raise RuntimeError(f'No source image-label pairs found for class={args.class_name}')

    if args.preview_only:
        previews = []
        for i in range(max(1, args.preview_count)):
            sample = None
            for _ in range(20):
                sample = compose_once(pairs, bg_files, args)
                if sample is not None:
                    break
            if sample is None:
                continue
            vis = sample['image'].copy()
            cv2.drawContours(vis, [sample['poly'].astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
            cv2.putText(vis, f'preview {i+1}/{max(1,args.preview_count)} scale={sample["scale"]:.2f}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f'bg_beta={sample["bg_beta"]:.1f} obj_beta={sample["obj_beta"]:.1f}', (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            previews.append(vis)
            cv2.imwrite(str(out_viz_dir / f'preview_{i+1:03d}.jpg'), vis)

        if not previews:
            raise RuntimeError('Could not create preview samples with current settings.')

        print(f'Preview images saved in: {out_viz_dir}')
        if args.preview_window:
            idx = 0
            win = 'Synth Preview (left/right, q to quit)'
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            while True:
                show = previews[idx].copy()
                cv2.putText(show, f'{idx+1}/{len(previews)}', (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(win, show)
                k = cv2.waitKey(0)
                if k in (ord('q'), 27):
                    break
                if k in (81, 2424832, ord('a')):
                    idx = max(0, idx - 1)
                elif k in (83, 2555904, ord('d')):
                    idx = min(len(previews) - 1, idx + 1)
            cv2.destroyAllWindows()
        print('Preview-only mode: no training images/labels written.')
        return

    made = 0
    for i in range(args.num_synthetic):
        sample = None
        for _ in range(20):
            sample = compose_once(pairs, bg_files, args)
            if sample is not None:
                break
        if sample is None:
            continue

        stem = f'{args.class_name}_synth_{i + 1:06d}'
        out_img = out_img_dir / f'{stem}.jpg'
        out_lbl = out_lbl_dir / f'{stem}.txt'
        out_viz = out_viz_dir / f'{stem}_overlay.jpg'

        cv2.imwrite(str(out_img), sample['image'])
        write_yolo(out_lbl, sample['cls_id'], sample['poly'], sample['w'], sample['h'])

        ov = sample['image'].copy()
        cv2.drawContours(ov, [sample['poly'].astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
        cv2.imwrite(str(out_viz), ov)
        made += 1

    print(f'Synthetic created: {made}')
    print(f'Run name: {run_name}')
    print(f'Images dir: {out_img_dir}')
    print(f'Labels dir: {out_lbl_dir}')
    print(f'Overlays dir: {out_viz_dir}')


if __name__ == '__main__':
    main()
