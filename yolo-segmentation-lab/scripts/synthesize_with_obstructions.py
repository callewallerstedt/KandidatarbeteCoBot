#!/usr/bin/env python3
import argparse
import random
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


def write_yolo(label_path: Path, class_id: int, poly: np.ndarray, w: int, h: int):
    vals = [str(class_id)]
    for x, y in poly:
        vals.append(f'{x / w:.6f}')
        vals.append(f'{y / h:.6f}')
    label_path.write_text(' '.join(vals) + '\n', encoding='utf-8')


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


def poly_to_mask(poly: np.ndarray, w: int, h: int):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    return m


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


def load_obstruction(path: Path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None, None
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        m = np.where(im[:, :, 0] > 0, 255, 0).astype(np.uint8)
    elif im.shape[2] == 4:
        m = im[:, :, 3]
        im = im[:, :, :3]
    else:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    ys, xs = np.where(m > 0)
    if len(xs) < 20:
        return None, None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return im[y1:y2 + 1, x1:x2 + 1], m[y1:y2 + 1, x1:x2 + 1]


def top_point(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    iy = ys.argmin()
    return np.array([float(xs[iy]), float(ys[iy])], dtype=np.float32)


def top_bottom_mid_from_image_shape(img):
    h, w = img.shape[:2]
    top_mid = np.array([w / 2.0, 0.0], dtype=np.float32)
    bottom_mid = np.array([w / 2.0, h - 1.0], dtype=np.float32)
    return top_mid, bottom_mid


def centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


def angle_deg(v):
    return np.degrees(np.arctan2(v[1], v[0]))


def pick_boundary_point(obj_mask, center, d_unit):
    h, w = obj_mask.shape
    for t in np.linspace(0, max(w, h), num=max(w, h) * 2):
        x = int(round(center[0] - d_unit[0] * t))
        y = int(round(center[1] - d_unit[1] * t))
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        if obj_mask[y, x] == 0:
            # previous point is boundary-ish
            px = int(round(center[0] - d_unit[0] * max(t - 1, 0)))
            py = int(round(center[1] - d_unit[1] * max(t - 1, 0)))
            return np.array([px, py], dtype=np.float32), max(t - 1, 1.0)
    return center.copy(), 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--class-id', type=int, required=True)
    ap.add_argument('--obstruction-dir', required=True)
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--num-synthetic', type=int, default=300)
    ap.add_argument('--entry-angle-min', type=float, default=0)
    ap.add_argument('--entry-angle-max', type=float, default=360)
    ap.add_argument('--rotation-deviation', type=float, default=20)
    ap.add_argument('--overlap-level', type=float, default=0.8, help='0 boundary, 1 center, 2 past-center')
    ap.add_argument('--obstruction-scale', type=float, default=0.8, help='Target obstruction size relative to object bbox height')
    ap.add_argument('--preview-only', action='store_true', help='Create preview image(s) in staging only, do not write train labels/images')
    ap.add_argument('--preview-window', action='store_true', help='Show preview in an OpenCV window instead of saving overlay image')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.data_root)
    src_img_dir = root / 'images' / args.class_name
    src_lbl_dir = root / 'labels' / args.class_name
    out_img_dir = root / 'images' / args.class_name
    out_lbl_dir = root / 'labels' / args.class_name
    out_viz_dir = root / 'staging' / f'{args.class_name}_obstruction'

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    out_viz_dir.mkdir(parents=True, exist_ok=True)

    obs_files = [p for p in Path(args.obstruction_dir).rglob('*') if p.suffix.lower() in IMG_EXTS]
    if not obs_files:
        raise RuntimeError('No obstruction images found')

    pairs = []
    for im in sorted(src_img_dir.glob('*')):
        if im.suffix.lower() not in IMG_EXTS:
            continue

        stem = im.stem
        # Use only original base samples by default (exclude manual/synth/obstruction derivatives).
        if any(tag in stem for tag in ['_manual_', '_synth_', '_obs_']):
            continue

        lb = src_lbl_dir / f'{stem}.txt'
        if lb.exists():
            pairs.append((im, lb))
    if not pairs:
        raise RuntimeError('No source image-label pairs found')

    made = 0
    for i in range(args.num_synthetic):
        im_path, lb_path = random.choice(pairs)
        base = cv2.imread(str(im_path))
        if base is None:
            continue
        h, w = base.shape[:2]
        cls_id, poly = load_yolo_polygon(lb_path, w, h)
        if poly is None:
            continue

        obj_mask = poly_to_mask(poly, w, h)
        obj_poly_i = poly.astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(obj_poly_i)
        center = np.array([x + bw / 2.0, y + bh / 2.0], dtype=np.float32)

        obs_img, obs_mask = load_obstruction(random.choice(obs_files))
        if obs_img is None:
            continue

        # size normalize by object bbox height
        oh0, ow0 = obs_mask.shape[:2]
        target_h = max(8, int(bh * args.obstruction_scale * random.uniform(0.85, 1.15)))
        s = target_h / max(oh0, 1)
        tw = max(8, int(ow0 * s))
        th = max(8, int(oh0 * s))
        obs_img = cv2.resize(obs_img, (tw, th), interpolation=cv2.INTER_LINEAR)
        obs_mask = cv2.resize(obs_mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        entry_ang = random.uniform(args.entry_angle_min, args.entry_angle_max)
        d = np.array([np.cos(np.radians(entry_ang)), np.sin(np.radians(entry_ang))], dtype=np.float32)

        # Strict orientation rule from user:
        # vector (bottom-middle -> top-middle) of obstruction image must point toward object center.
        # Unrotated bottom->top vector is (0, -1), i.e. -90 degrees.
        base_ang = -90.0
        target_ang = angle_deg(d)
        rot = (target_ang - base_ang) + random.uniform(-args.rotation_deviation, args.rotation_deviation)
        obs_img, obs_mask = rotate_bound_pair(obs_img, obs_mask, rot)

        tp2, _ = top_bottom_mid_from_image_shape(obs_img)

        boundary_pt, r = pick_boundary_point(obj_mask, center, d)
        target_top = boundary_pt + d * (args.overlap_level * r)

        px = int(round(target_top[0] - tp2[0]))
        py = int(round(target_top[1] - tp2[1]))

        oh, ow = obs_mask.shape[:2]
        x1 = max(0, px)
        y1 = max(0, py)
        x2 = min(w, px + ow)
        y2 = min(h, py + oh)
        if x2 <= x1 or y2 <= y1:
            continue

        sx1 = x1 - px
        sy1 = y1 - py
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        roi = base[y1:y2, x1:x2]
        om = obs_mask[sy1:sy2, sx1:sx2] > 0
        oi = obs_img[sy1:sy2, sx1:sx2]
        roi[om] = oi[om]
        base[y1:y2, x1:x2] = roi

        full_obs = np.zeros((h, w), dtype=np.uint8)
        full_obs[y1:y2, x1:x2][om] = 255

        new_obj_mask = (obj_mask > 0) & (full_obs == 0)
        new_obj_mask = (new_obj_mask.astype(np.uint8) * 255)
        poly_new = mask_to_polygon(new_obj_mask)
        if poly_new is None:
            continue

        stem = f'{args.class_name}_{"obs_preview" if args.preview_only else "obs"}_{i + 1:06d}'

        ov = base.copy()
        cv2.drawContours(ov, [poly_new.astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 2)

        if args.preview_only and args.preview_window:
            cv2.namedWindow('Obstruction Preview (press q/esc to close)', cv2.WINDOW_NORMAL)
            cv2.imshow('Obstruction Preview (press q/esc to close)', ov)
            while True:
                k = cv2.waitKey(20) & 0xFF
                if k in (ord('q'), 27):
                    break
            cv2.destroyAllWindows()
        else:
            out_viz = out_viz_dir / f'{stem}_overlay.jpg'
            cv2.imwrite(str(out_viz), ov)

        if not args.preview_only:
            out_img = out_img_dir / f'{stem}.jpg'
            out_lbl = out_lbl_dir / f'{stem}.txt'
            cv2.imwrite(str(out_img), base)
            write_yolo(out_lbl, args.class_id if cls_id is None else cls_id, poly_new, w, h)

        made += 1

    print(f'Obstruction synthetic created: {made}')
    if args.preview_only:
        print('Preview-only mode: no train images/labels were written.')
        if args.preview_window:
            print('Preview shown in window (no overlay file saved).')
        else:
            print(f'Overlays dir: {out_viz_dir}')
    else:
        print(f'Images dir: {out_img_dir}')
        print(f'Labels dir: {out_lbl_dir}')
        print(f'Overlays dir: {out_viz_dir}')


if __name__ == '__main__':
    main()
