#!/usr/bin/env python3
import argparse
import json
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


def write_yolo(label_path: Path, class_id: int, poly: np.ndarray, w: int, h: int):
    vals = [str(class_id)]
    for x, y in poly:
        vals.append(f'{x / w:.6f}')
        vals.append(f'{y / h:.6f}')
    label_path.write_text(' '.join(vals) + '\n', encoding='utf-8')


def pick_random_background(bg_files, max_dim=1920):
    bg_path = random.choice(bg_files)
    bg = cv2.imread(str(bg_path))
    if bg is None:
        return np.full((1080, 1920, 3), 127, dtype=np.uint8), None
    bh, bw = bg.shape[:2]
    # Keep original background proportions; only downscale very large images.
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


def parse_rect_norm(rect_text):
    if not rect_text:
        return None
    try:
        vals = [float(x.strip()) for x in rect_text.split(',')]
        if len(vals) != 4:
            return None
        x1, y1, x2, y2 = vals
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1 = max(0.0, min(1.0, x1)); x2 = max(0.0, min(1.0, x2))
        y1 = max(0.0, min(1.0, y1)); y2 = max(0.0, min(1.0, y2))
        if x2 - x1 < 0.01 or y2 - y1 < 0.01:
            return None
        return (x1, y1, x2, y2)
    except Exception:
        return None


def point_in_poly(x, y, poly_pts):
    if poly_pts is None or len(poly_pts) < 3:
        return True
    contour = np.array(poly_pts, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def bbox_fully_inside_poly(px, py, ow, oh, poly_pts):
    if poly_pts is None or len(poly_pts) < 3:
        return True
    corners = [
        (px, py),
        (px + ow, py),
        (px, py + oh),
        (px + ow, py + oh),
    ]
    for cx, cy in corners:
        if not point_in_poly(cx, cy, poly_pts):
            return False
    return True


def profile_for_bg(args, bg_path):
    if bg_path is None:
        return None
    prof = getattr(args, 'placement_profile_data', None)
    if not prof:
        return None
    items = prof.get('items', {})
    k_abs = str(bg_path).replace('\\', '/')
    k1 = bg_path.name
    k_rel = None
    try:
        k_rel = str(bg_path.relative_to(Path(args.background_dir))).replace('\\', '/')
        if k_rel in items:
            return items[k_rel]
    except Exception:
        pass

    if k_abs in items:
        return items[k_abs]

    if k1 in items:
        return items[k1]

    # Windows/path normalization fallback
    k_abs_low = k_abs.lower()
    k_rel_low = k_rel.lower() if k_rel else None
    for kk, vv in items.items():
        k_norm = str(kk).replace('\\', '/').lower()
        if k_rel_low and (k_norm == k_rel_low or k_norm.endswith('/' + k_rel_low)):
            return vv
        if k_norm == k_abs_low or k_norm.endswith('/' + k1.lower()):
            return vv
    return None


def extract_crop_from_pair(im_path: Path, lb_path: Path):
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
    if crop.size == 0 or crop_mask.size == 0:
        return None
    return cls_id, crop, crop_mask


def compose_once(pairs, bg_files, args, fixed_scale=None, fixed_bg_beta=None, fixed_obj_beta=None):
    if getattr(args, 'preview_only', False) and getattr(args, 'preview_pair', None) is not None:
        im_path, lb_path = args.preview_pair
    else:
        im_path, lb_path = random.choice(pairs)
    crop_info = extract_crop_from_pair(im_path, lb_path)
    if crop_info is None:
        return None
    cls_id, crop, crop_mask = crop_info
    bw = crop.shape[1]
    bh = crop.shape[0]

    bg, bg_path = pick_random_background(bg_files)
    h, w = bg.shape[:2]

    prof = profile_for_bg(args, bg_path)
    p_min_scale = prof.get('min_scale') if prof else None
    p_max_scale = prof.get('max_scale') if prof else None
    p_bg_min = prof.get('bg_brightness_min') if prof else None
    p_bg_max = prof.get('bg_brightness_max') if prof else None
    p_obj_min = prof.get('obj_brightness_min') if prof else None
    p_obj_max = prof.get('obj_brightness_max') if prof else None
    if prof and args.class_name and isinstance(prof.get('class_settings'), dict):
        cset = prof.get('class_settings', {})
        cs = cset.get(args.class_name)
        if cs is None:
            want = str(args.class_name).strip().lower()
            for ck, cv in cset.items():
                if str(ck).strip().lower() == want:
                    cs = cv
                    break
        if isinstance(cs, dict):
            p_min_scale = cs.get('min_scale', p_min_scale)
            p_max_scale = cs.get('max_scale', p_max_scale)
            p_bg_min = cs.get('bg_brightness_min', p_bg_min)
            p_bg_max = cs.get('bg_brightness_max', p_bg_max)
            p_obj_min = cs.get('obj_brightness_min', p_obj_min)
            p_obj_max = cs.get('obj_brightness_max', p_obj_max)

    smin = p_min_scale if p_min_scale is not None else args.min_scale
    smax = p_max_scale if p_max_scale is not None else args.max_scale

    requested_scale = fixed_scale if fixed_scale is not None else random.uniform(min(smin, smax), max(smin, smax))
    source_long = float(max(1, max(bw, bh)))
    reference_long = float(getattr(args, 'reference_long_side', source_long) or source_long)
    scale = requested_scale * (reference_long / source_long)
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

    rect_norm = args.placement_rect_norm
    poly_norm = None
    if prof:
        if isinstance(prof.get('rect'), (list, tuple)) and len(prof.get('rect')) == 4:
            try:
                rect_norm = tuple(float(x) for x in prof.get('rect'))
            except Exception:
                pass
        if isinstance(prof.get('poly'), list) and len(prof.get('poly')) >= 3:
            try:
                poly_norm = [(float(p[0]), float(p[1])) for p in prof.get('poly') if isinstance(p, (list, tuple)) and len(p) >= 2]
            except Exception:
                poly_norm = None

    if poly_norm and len(poly_norm) >= 3:
        poly_px = [(int(x * w), int(y * h)) for x, y in poly_norm]
        xs = [p[0] for p in poly_px]
        ys = [p[1] for p in poly_px]
        min_px = max(0, min(xs))
        min_py = max(0, min(ys))
        max_px = min(w - ow, max(xs))
        max_py = min(h - oh, max(ys))
        px = random.randint(0, w - ow)
        py = random.randint(0, h - oh)
        for _ in range(60):
            tx = random.randint(min_px, max_px) if max_px >= min_px else px
            ty = random.randint(min_py, max_py) if max_py >= min_py else py
            if bbox_fully_inside_poly(tx, ty, ow, oh, poly_px):
                px, py = tx, ty
                break
    elif rect_norm is not None:
        rx1, ry1, rx2, ry2 = rect_norm
        bx1 = int(rx1 * w)
        by1 = int(ry1 * h)
        bx2 = int(rx2 * w)
        by2 = int(ry2 * h)
        max_px = max(bx1, bx2 - ow)
        max_py = max(by1, by2 - oh)
        min_px = min(bx1, max_px)
        min_py = min(by1, max_py)
        px = random.randint(min_px, max_px) if max_px >= min_px else random.randint(0, w - ow)
        py = random.randint(min_py, max_py) if max_py >= min_py else random.randint(0, h - oh)
    else:
        px = random.randint(0, w - ow)
        py = random.randint(0, h - oh)

    obj_min = p_obj_min if p_obj_min is not None else args.object_brightness_min
    obj_max = p_obj_max if p_obj_max is not None else args.object_brightness_max
    obj_beta = fixed_obj_beta if fixed_obj_beta is not None else random.uniform(obj_min, obj_max)
    crop_rr = cv2.convertScaleAbs(crop_rr, alpha=1.0, beta=obj_beta)

    roi = bg[py:py + oh, px:px + ow]
    alpha = (m_rr > 0)
    roi[alpha] = crop_rr[alpha]
    bg[py:py + oh, px:px + ow] = roi

    bg_min = p_bg_min if p_bg_min is not None else args.brightness_min
    bg_max = p_bg_max if p_bg_max is not None else args.brightness_max
    bg_beta = fixed_bg_beta if fixed_bg_beta is not None else random.uniform(bg_min, bg_max)
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
        'scale': requested_scale,
        'applied_scale': scale,
        'bg_beta': bg_beta,
        'obj_beta': obj_beta,
    }


def main():
    from profile_scene_builders import build_single_scene as core_build_single_scene
    from profile_scene_builders import collect_cutouts as core_collect_cutouts
    from profile_scene_builders import load_profile as core_load_profile
    from profile_scene_builders import next_output_index as core_next_output_index

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
    ap.add_argument('--object-temp-bias', type=float, default=0.35, help='Object temperature bias: -1 cooler, +1 warmer')
    ap.add_argument('--object-temp-variance', type=float, default=0.18, help='Random object temperature variance around the bias')
    ap.add_argument('--object-shade-prob', type=float, default=0.45, help='Probability of partial one-sided shading per object')
    ap.add_argument('--object-shade-strength', type=float, default=0.22, help='Strength of partial object shading')
    ap.add_argument('--run-name', default='', help='Synthetic run folder name (default timestamp)')
    ap.add_argument('--preview-only', action='store_true', help='Show/save preview only, do not write training data')
    ap.add_argument('--preview-window', action='store_true', help='Show preview window')
    ap.add_argument('--preview-count', type=int, default=12, help='How many random previews to generate in preview-only mode')
    ap.add_argument('--preview-max-width', type=int, default=1600)
    ap.add_argument('--preview-max-height', type=int, default=900)
    ap.add_argument('--placement-rect', default='', help='Normalized placement rectangle x1,y1,x2,y2 (0..1), objects stay within this box')
    ap.add_argument('--placement-profile', default='', help='JSON profile with per-background rect/min_scale/max_scale')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    args.preview_mode = 'random'
    args.placement_rect_norm = parse_rect_norm(args.placement_rect)
    args.placement_profile_data = None
    if args.placement_profile.strip():
        p = Path(args.placement_profile)
        if p.exists():
            try:
                args.placement_profile_data = json.loads(p.read_text(encoding='utf-8-sig'))
            except Exception as e:
                print(f'Warning: failed reading placement profile: {e}')

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

    args.reference_long_side = None
    args.preview_pair = None
    for im, lb in pairs:
        crop_info = extract_crop_from_pair(im, lb)
        if crop_info is None:
            continue
        _, crop, _ = crop_info
        args.reference_long_side = float(max(1, max(crop.shape[1], crop.shape[0])))
        args.preview_pair = (im, lb)
        break
    if args.reference_long_side is None:
        raise RuntimeError(f'No valid source cutouts found for class={args.class_name}')

    if args.preview_only:
        previews = []
        specs = [
            ('min_scale', {'fixed_scale': args.min_scale}),
            ('max_scale', {'fixed_scale': args.max_scale}),
            ('min_bg_brightness', {'fixed_bg_beta': args.brightness_min}),
            ('max_bg_brightness', {'fixed_bg_beta': args.brightness_max}),
        ]

        for label, kw in specs:
            sample = None
            for _ in range(30):
                sample = compose_once(pairs, bg_files, args, **kw)
                if sample is not None:
                    break
            if sample is None:
                continue
            vis = sample['image'].copy()
            cv2.drawContours(vis, [sample['poly'].astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
            cv2.putText(vis, f'{label} requested={sample["scale"]:.2f} applied={sample["applied_scale"]:.2f}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f'scale range={args.min_scale:.2f}..{args.max_scale:.2f}', (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f'bg beta={args.brightness_min:.0f}..{args.brightness_max:.0f} obj beta={args.object_brightness_min:.0f}..{args.object_brightness_max:.0f}', (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            previews.append(vis)

        while len(previews) < max(1, args.preview_count):
            sample = None
            for _ in range(20):
                sample = compose_once(pairs, bg_files, args)
                if sample is not None:
                    break
            if sample is None:
                break
            vis = sample['image'].copy()
            cv2.drawContours(vis, [sample['poly'].astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
            cv2.putText(vis, f'random scale={sample["scale"]:.2f}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f'scale range={args.min_scale:.2f}..{args.max_scale:.2f}', (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f'bg beta={args.brightness_min:.0f}..{args.brightness_max:.0f} obj beta={args.object_brightness_min:.0f}..{args.object_brightness_max:.0f}', (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            previews.append(vis)

        if not previews:
            raise RuntimeError('Could not create preview samples with current settings.')

        print(f'Preview images saved in: {out_viz_dir}')
        if args.preview_window:
            idx = 0
            win = 'Synth Preview (left/right, q to quit)'
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            while True:
                show = previews[idx].copy()
                cv2.putText(show, f'{idx+1}/{len(previews)}', (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
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
                    idx = min(len(previews) - 1, idx + 1)
            cv2.destroyAllWindows()
        print('Preview-only mode: no training images/labels written.')
        return

    if not args.placement_profile.strip():
        raise RuntimeError('placement_profile is required for synthetic generation')
    profile_data = core_load_profile(Path(args.placement_profile))
    cutouts = core_collect_cutouts(root, args.class_name)
    reference_long_side = float(cutouts[0]['long_side'])

    made = 0
    stem_prefix = f'{args.class_name}_synth'
    next_idx = core_next_output_index(out_img_dir, stem_prefix)
    while made < args.num_synthetic:
        scene = core_build_single_scene(args, bg_files, profile_data, cutouts, reference_long_side)
        if scene is None:
            continue

        poly = mask_to_polygon(scene['mask'])
        if poly is None:
            continue

        stem = f'{stem_prefix}_{next_idx + made:06d}'
        out_img = out_img_dir / f'{stem}.jpg'
        out_lbl = out_lbl_dir / f'{stem}.txt'
        out_viz = out_viz_dir / f'{stem}_overlay.jpg'

        h, w = scene['image'].shape[:2]
        cv2.imwrite(str(out_img), scene['image'])
        write_yolo(out_lbl, args.class_id, poly, w, h)
        cv2.imwrite(str(out_viz), scene['overlay'])
        made += 1

    print(f'Synthetic created: {made}')
    print(f'Run name: {run_name}')
    print(f'Images dir: {out_img_dir}')
    print(f'Labels dir: {out_lbl_dir}')
    print(f'Overlays dir: {out_viz_dir}')


if __name__ == '__main__':
    main()
