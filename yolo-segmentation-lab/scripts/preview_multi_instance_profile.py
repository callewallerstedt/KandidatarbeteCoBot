#!/usr/bin/env python3
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def load_profile(profile_path: Path):
    if not profile_path.exists():
        raise RuntimeError(f'Profile not found: {profile_path}')
    data = json.loads(profile_path.read_text(encoding='utf-8-sig'))
    items = data.get('items')
    if not isinstance(items, dict):
        raise RuntimeError('Invalid placement profile: missing items object')
    return data


def resolve_profile_item(profile_data, background_dir: Path, bg_path: Path):
    items = profile_data.get('items', {})
    rel_key = None
    abs_key = str(bg_path).replace('\\', '/')
    try:
        rel_key = str(bg_path.relative_to(background_dir)).replace('\\', '/')
    except Exception:
        pass

    if rel_key and rel_key in items:
        return items[rel_key]
    if abs_key in items:
        return items[abs_key]
    if bg_path.name in items:
        return items[bg_path.name]

    rel_low = rel_key.lower() if rel_key else None
    abs_low = abs_key.lower()
    name_low = bg_path.name.lower()
    for key, value in items.items():
        norm = str(key).replace('\\', '/').lower()
        if rel_low and (norm == rel_low or norm.endswith('/' + rel_low)):
            return value
        if norm == abs_low or norm.endswith('/' + name_low):
            return value
    return None


def parse_first_polygon(label_path: Path, width: int, height: int):
    text = label_path.read_text(encoding='utf-8', errors='ignore').strip()
    if not text:
        return None
    for line in text.splitlines():
        vals = line.strip().split()
        if len(vals) < 7 or (len(vals) - 1) % 2 != 0:
            continue
        pts = np.array([float(x) for x in vals[1:]], dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= width
        pts[:, 1] *= height
        return pts
    return None


def polygon_to_mask(poly: np.ndarray, width: int, height: int):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask


def rotate_image_mask(image, mask, angle_deg):
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    mat[0, 2] += (nw / 2.0) - cx
    mat[1, 2] += (nh / 2.0) - cy
    rot_img = cv2.warpAffine(image, mat, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rot_mask = cv2.warpAffine(mask, mat, (nw, nh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rot_img, rot_mask


def contour_from_mask(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    contour = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.0008 * peri, True)
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
    corners = [(px, py), (px + ow, py), (px, py + oh), (px + ow, py + oh)]
    return all(point_in_poly(cx, cy, poly_pts) for cx, cy in corners)


def collect_cutouts(data_root: Path, class_name: str):
    image_root = data_root / 'images' / class_name
    label_root = data_root / 'labels' / class_name
    if not image_root.exists() or not label_root.exists():
        raise RuntimeError(f'Missing source data for class={class_name}')

    cutouts = []
    for image_path in sorted(image_root.rglob('*')):
        if image_path.suffix.lower() not in IMG_EXTS:
            continue
        rel = image_path.relative_to(image_root)
        if any(tag in rel.parts for tag in ['synth_runs', 'obs_runs', 'synth_multi_runs']):
            continue
        label_path = (label_root / rel).with_suffix('.txt')
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        poly = parse_first_polygon(label_path, w, h)
        if poly is None:
            continue
        mask = polygon_to_mask(poly, w, h)
        x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
        crop = image[y:y + bh, x:x + bw].copy()
        crop_mask = mask[y:y + bh, x:x + bw].copy()
        if crop.size == 0 or int((crop_mask > 0).sum()) < 20:
            continue
        cutouts.append({
            'image': crop,
            'mask': crop_mask,
            'long_side': float(max(crop.shape[0], crop.shape[1])),
            'source': image_path.name,
        })

    if not cutouts:
        raise RuntimeError(f'No valid cutouts found for class={class_name}')
    return cutouts


def choose_effective_settings(item, class_name: str):
    min_scale = float(item.get('min_scale', 0.55))
    max_scale = float(item.get('max_scale', 1.25))
    bg_min = float(item.get('bg_brightness_min', 0.0))
    bg_max = float(item.get('bg_brightness_max', 0.0))
    obj_min = float(item.get('obj_brightness_min', 0.0))
    obj_max = float(item.get('obj_brightness_max', 0.0))

    cset = item.get('class_settings', {})
    if class_name and isinstance(cset, dict):
        cls = cset.get(class_name)
        if cls is None:
            want = class_name.strip().lower()
            for key, value in cset.items():
                if str(key).strip().lower() == want:
                    cls = value
                    break
        if isinstance(cls, dict):
            min_scale = float(cls.get('min_scale', min_scale))
            max_scale = float(cls.get('max_scale', max_scale))
            bg_min = float(cls.get('bg_brightness_min', bg_min))
            bg_max = float(cls.get('bg_brightness_max', bg_max))
            obj_min = float(cls.get('obj_brightness_min', obj_min))
            obj_max = float(cls.get('obj_brightness_max', obj_max))

    poly_px = None
    poly = item.get('poly')
    if isinstance(poly, list) and len(poly) >= 3:
        poly_px = poly

    return {
        'min_scale': min_scale,
        'max_scale': max_scale,
        'bg_brightness_min': bg_min,
        'bg_brightness_max': bg_max,
        'obj_brightness_min': obj_min,
        'obj_brightness_max': obj_max,
        'poly': poly_px,
    }


def render_preview_frame(args, bg_files, profile_data, cutouts, reference_long_side):
    for _ in range(200):
        bg_path = random.choice(bg_files)
        entry = resolve_profile_item(profile_data, Path(args.background_dir), bg_path)
        if not isinstance(entry, dict):
            continue

        background = cv2.imread(str(bg_path))
        if background is None:
            continue
        h, w = background.shape[:2]

        effective = choose_effective_settings(entry, args.class_name)
        poly_norm = effective['poly']
        poly_px = None
        if poly_norm is not None:
            poly_px = [(int(pt[0] * w), int(pt[1] * h)) for pt in poly_norm]

        if args.preview_mode == 'min_scale':
            requested_scale = min(effective['min_scale'], effective['max_scale'])
        elif args.preview_mode == 'max_scale':
            requested_scale = max(effective['min_scale'], effective['max_scale'])
        else:
            requested_scale = random.uniform(min(effective['min_scale'], effective['max_scale']), max(effective['min_scale'], effective['max_scale']))

        if args.preview_mode == 'bg_bri_min':
            bg_beta = effective['bg_brightness_min']
        elif args.preview_mode == 'bg_bri_max':
            bg_beta = effective['bg_brightness_max']
        else:
            bg_beta = random.uniform(effective['bg_brightness_min'], effective['bg_brightness_max'])

        canvas = cv2.convertScaleAbs(background.copy(), alpha=1.0, beta=bg_beta)
        target_objects = random.randint(args.min_objects, args.max_objects)
        visible_masks = []
        centers = []
        placed_instances = []

        attempts_budget = max(120, target_objects * 120)
        for _ in range(attempts_budget):
            if len(placed_instances) >= target_objects:
                break

            cutout = random.choice(cutouts)
            normalized_scale = requested_scale * (reference_long_side / max(1.0, cutout['long_side']))
            tw = max(8, int(cutout['image'].shape[1] * normalized_scale))
            th = max(8, int(cutout['image'].shape[0] * normalized_scale))
            obj_img = cv2.resize(cutout['image'], (tw, th), interpolation=cv2.INTER_LINEAR)
            obj_mask = cv2.resize(cutout['mask'], (tw, th), interpolation=cv2.INTER_NEAREST)

            angle = random.uniform(-args.max_rotation, args.max_rotation)
            obj_img, obj_mask = rotate_image_mask(obj_img, obj_mask, angle)
            ys, xs = np.where(obj_mask > 0)
            if len(xs) < 20:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            obj_img = obj_img[y1:y2 + 1, x1:x2 + 1]
            obj_mask = obj_mask[y1:y2 + 1, x1:x2 + 1]
            oh, ow = obj_mask.shape[:2]
            if ow >= w or oh >= h:
                continue

            obj_beta = random.uniform(effective['obj_brightness_min'], effective['obj_brightness_max'])
            obj_img = cv2.convertScaleAbs(obj_img, alpha=1.0, beta=obj_beta)

            placed = False
            new_mask = None
            for _ in range(60):
                if centers:
                    anchor_x, anchor_y = centers[random.randrange(len(centers))]
                    if random.random() < args.overlap_prob:
                        jitter_x = int(random.uniform(-1.0, 1.0) * args.overlap_spread * ow)
                        jitter_y = int(random.uniform(-1.0, 1.0) * args.overlap_spread * oh)
                    else:
                        max_dist = int(args.cluster_distance_factor * max(ow, oh))
                        jitter_x = random.randint(-max_dist, max_dist)
                        jitter_y = random.randint(-max_dist, max_dist)
                    px = int(np.clip(anchor_x - ow // 2 + jitter_x, 0, w - ow))
                    py = int(np.clip(anchor_y - oh // 2 + jitter_y, 0, h - oh))
                else:
                    px = random.randint(0, w - ow)
                    py = random.randint(0, h - oh)

                if poly_px is not None and not bbox_fully_inside_poly(px, py, ow, oh, poly_px):
                    continue

                candidate = np.zeros((h, w), dtype=np.uint8)
                candidate[py:py + oh, px:px + ow][obj_mask > 0] = 255
                cand_area = max(1, int((candidate > 0).sum()))

                overlap_ok = True
                for prev in visible_masks:
                    inter = int(((prev > 0) & (candidate > 0)).sum())
                    prev_area = max(1, int((prev > 0).sum()))
                    if (inter / cand_area) > args.max_overlap_ratio or (inter / prev_area) > args.max_overlap_ratio:
                        overlap_ok = False
                        break

                if overlap_ok and centers:
                    cx_new = px + ow // 2
                    cy_new = py + oh // 2
                    nearest = min((((cx_new - cx) ** 2 + (cy_new - cy) ** 2) ** 0.5) for cx, cy in centers)
                    if nearest > args.cluster_distance_factor * max(ow, oh):
                        overlap_ok = False

                if not overlap_ok:
                    continue

                roi = canvas[py:py + oh, px:px + ow]
                alpha = obj_mask > 0
                roi[alpha] = obj_img[alpha]
                canvas[py:py + oh, px:px + ow] = roi
                new_mask = candidate
                placed = True
                break

            if not placed or new_mask is None:
                continue

            updated_masks = []
            for prev in visible_masks:
                vis = ((prev > 0) & (new_mask == 0)).astype(np.uint8) * 255
                if int((vis > 0).sum()) > 20:
                    updated_masks.append(vis)
            visible_masks = updated_masks
            visible_masks.append(new_mask)

            ys2, xs2 = np.where(new_mask > 0)
            centers.append((int(xs2.mean()), int(ys2.mean())))
            placed_instances.append({
                'mask': new_mask,
                'source': cutout['source'],
            })

        if len(visible_masks) < args.min_objects:
            continue

        overlay = canvas.copy()
        colors = [
            (80, 220, 120),
            (120, 140, 255),
            (255, 190, 90),
            (220, 120, 255),
            (120, 255, 255),
        ]
        for idx, mask in enumerate(visible_masks):
            color = colors[idx % len(colors)]
            tint = np.zeros_like(overlay)
            tint[:] = color
            alpha = mask > 0
            overlay[alpha] = cv2.addWeighted(overlay[alpha], 0.55, tint[alpha], 0.45, 0)
            contour = contour_from_mask(mask)
            if contour is not None:
                cv2.drawContours(overlay, [contour.astype(np.int32).reshape(-1, 1, 2)], -1, color, 2)

        if poly_px is not None:
            cv2.polylines(overlay, [np.array(poly_px, dtype=np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(overlay, f'instances={len(visible_masks)} target={target_objects} mode={args.preview_mode}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f'profile scale={effective["min_scale"]:.2f}..{effective["max_scale"]:.2f}', (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f'bg={bg_path.name} source={placed_instances[-1]["source"]}', (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        return {
            'overlay': overlay,
            'background': bg_path.name,
            'effective': effective,
            'count': len(visible_masks),
        }

    return None


def show_preview_window(frames, max_width, max_height):
    idx = 0
    win = 'Multi-instance Profile Preview (left/right, q to quit)'
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
    from profile_scene_builders import build_multi_scene as core_build_multi_scene
    from profile_scene_builders import collect_cutouts as core_collect_cutouts
    from profile_scene_builders import load_profile as core_load_profile

    ap = argparse.ArgumentParser()
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--class-id', type=int, required=True)
    ap.add_argument('--background-dir', required=True)
    ap.add_argument('--placement-profile', required=True)
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--preview-count', type=int, default=12)
    ap.add_argument('--preview-mode', default='random', choices=['random', 'min_scale', 'max_scale', 'bg_bri_min', 'bg_bri_max'])
    ap.add_argument('--min-objects', type=int, default=2)
    ap.add_argument('--max-objects', type=int, default=5)
    ap.add_argument('--overlap-prob', type=float, default=0.8)
    ap.add_argument('--max-overlap-ratio', type=float, default=0.5)
    ap.add_argument('--cluster-distance-factor', type=float, default=1.0)
    ap.add_argument('--overlap-spread', type=float, default=0.25)
    ap.add_argument('--object-temp-bias', type=float, default=0.35)
    ap.add_argument('--object-temp-variance', type=float, default=0.18)
    ap.add_argument('--object-shade-prob', type=float, default=0.45)
    ap.add_argument('--object-shade-strength', type=float, default=0.22)
    ap.add_argument('--max-rotation', type=float, default=360.0)
    ap.add_argument('--source-run-filter', default='', help='Optional comma-separated filter for real source run folder names used to build cutouts')
    ap.add_argument('--preview-window', action='store_true')
    ap.add_argument('--preview-max-width', type=int, default=1600)
    ap.add_argument('--preview-max-height', type=int, default=900)
    ap.add_argument('--run-name', default='')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    bg_dir = Path(args.background_dir)
    bg_files = [p for p in bg_dir.rglob('*') if p.suffix.lower() in IMG_EXTS]
    if not bg_files:
        raise RuntimeError(f'No background images found in {bg_dir}')

    profile_data = core_load_profile(Path(args.placement_profile))
    cutouts = core_collect_cutouts(Path(args.data_root), args.class_name, args.source_run_filter)
    reference_long_side = cutouts[0]['long_side']

    run_name = args.run_name.strip() or datetime.now().strftime('preview_%Y%m%d_%H%M%S')
    out_dir = Path(args.data_root) / 'staging' / f'{args.class_name}_synth_multi_preview2' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    made = 0
    while made < args.preview_count:
        scene = core_build_multi_scene(args, bg_files, profile_data, cutouts, reference_long_side)
        if scene is None:
            break
        out_path = out_dir / f'{args.class_name}_previewmulti_{made + 1:06d}_overlay.jpg'
        cv2.imwrite(str(out_path), scene['overlay'])
        frames.append(scene['overlay'])
        made += 1

    print(f'Profile-based multi preview created: {made}')
    print(f'Overlays dir: {out_dir}')
    if made == 0:
        print('No preview frames produced. Check placement profile coverage and object size settings.')
        return

    if args.preview_window:
        show_preview_window(frames, args.preview_max_width, args.preview_max_height)


if __name__ == '__main__':
    main()
