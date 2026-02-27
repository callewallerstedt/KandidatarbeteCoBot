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


def pick_random_background(bg_files, out_w, out_h):
    bg = cv2.imread(str(random.choice(bg_files)))
    if bg is None:
        return np.full((out_h, out_w, 3), 127, dtype=np.uint8)
    bh, bw = bg.shape[:2]
    scale = max(out_w / max(bw, 1), out_h / max(bh, 1))
    nw, nh = int(bw * scale), int(bh * scale)
    bg = cv2.resize(bg, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = random.randint(0, max(nw - out_w, 0))
    y0 = random.randint(0, max(nh - out_h, 0))
    return bg[y0:y0 + out_h, x0:x0 + out_w].copy()


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


def rotate_bound_with_points(img, mask, angle_deg, points_xy):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))

    M[0, 2] += (nw / 2.0) - cx
    M[1, 2] += (nh / 2.0) - cy

    rot_img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rot_mask = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    pts = np.array(points_xy, dtype=np.float32)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    t = (M @ pts_h.T).T
    return rot_img, rot_mask, t


def angle_deg(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def pick_boundary_point(obj_mask, center, d_unit):
    h, w = obj_mask.shape
    for t in np.linspace(0, max(w, h), num=max(w, h) * 2):
        x = int(round(center[0] - d_unit[0] * t))
        y = int(round(center[1] - d_unit[1] * t))
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        if obj_mask[y, x] == 0:
            px = int(round(center[0] - d_unit[0] * max(t - 1, 0)))
            py = int(round(center[1] - d_unit[1] * max(t - 1, 0)))
            return np.array([px, py], dtype=np.float32), max(t - 1, 1.0)
    return center.copy(), 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--class-id', type=int, required=True)
    ap.add_argument('--obstruction-dir', required=True)
    ap.add_argument('--background-dir', required=True)
    ap.add_argument('--white-bg-prob', type=float, default=0.10, help='Probability to keep original white-table background')
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--num-synthetic', type=int, default=300)
    ap.add_argument('--entry-angle-min', type=float, default=0)
    ap.add_argument('--entry-angle-max', type=float, default=360)
    ap.add_argument('--rotation-deviation', type=float, default=20)
    ap.add_argument('--overlap-level', type=float, default=0.8, help='0=edge, 1=center, 2=past center')
    ap.add_argument('--obstruction-scale', type=float, default=0.8, help='Relative to object bbox height')
    ap.add_argument('--preview-only', action='store_true')
    ap.add_argument('--preview-window', action='store_true')
    ap.add_argument('--run-name', default='', help='Obstruction run folder name (default timestamp)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.data_root)
    src_img_dir = root / 'images' / args.class_name
    src_lbl_dir = root / 'labels' / args.class_name
    run_name = args.run_name.strip() or datetime.now().strftime('run_%Y%m%d_%H%M%S')
    out_img_dir = root / 'images' / args.class_name / 'obs_runs' / run_name
    out_lbl_dir = root / 'labels' / args.class_name / 'obs_runs' / run_name
    out_viz_dir = root / 'staging' / f'{args.class_name}_obstruction' / run_name

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    out_viz_dir.mkdir(parents=True, exist_ok=True)

    obs_files = [p for p in Path(args.obstruction_dir).rglob('*') if p.suffix.lower() in IMG_EXTS]
    if not obs_files:
        raise RuntimeError('No obstruction images found')

    bg_files = [p for p in Path(args.background_dir).rglob('*') if p.suffix.lower() in IMG_EXTS]
    if not bg_files:
        raise RuntimeError('No background images found')

    pairs = []
    for im in sorted(src_img_dir.glob('*')):
        if im.suffix.lower() not in IMG_EXTS:
            continue
        stem = im.stem
        # only original white-table seeds
        if any(tag in stem for tag in ['_manual_', '_synth_', '_obs_']):
            continue
        lb = src_lbl_dir / f'{stem}.txt'
        if lb.exists():
            pairs.append((im, lb))
    if not pairs:
        raise RuntimeError('No source base image-label pairs found')

    made = 0
    for i in range(args.num_synthetic):
        im_path, lb_path = random.choice(pairs)
        src = cv2.imread(str(im_path))
        if src is None:
            continue
        h, w = src.shape[:2]

        cls_id, poly = load_yolo_polygon(lb_path, w, h)
        if poly is None:
            continue

        obj_mask_full = poly_to_mask(poly, w, h)
        x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
        obj_crop = src[y:y + bh, x:x + bw]
        obj_crop_mask = obj_mask_full[y:y + bh, x:x + bw]

        # Generate base image: 10% keep original white background, else random background
        if random.random() < args.white_bg_prob:
            base = src.copy()
            placed_obj_mask = obj_mask_full.copy()
        else:
            base = pick_random_background(bg_files, w, h)
            scale = random.uniform(0.85, 1.15)
            tw = max(8, int(bw * scale))
            th = max(8, int(bh * scale))
            obj_r = cv2.resize(obj_crop, (tw, th), interpolation=cv2.INTER_LINEAR)
            m_r = cv2.resize(obj_crop_mask, (tw, th), interpolation=cv2.INTER_NEAREST)

            px = random.randint(0, max(w - tw, 0))
            py = random.randint(0, max(h - th, 0))
            roi = base[py:py + th, px:px + tw]
            aa = m_r > 0
            roi[aa] = obj_r[aa]
            base[py:py + th, px:px + tw] = roi

            placed_obj_mask = np.zeros((h, w), dtype=np.uint8)
            placed_obj_mask[py:py + th, px:px + tw][aa] = 255

        # obstruction element
        obs_img, obs_mask = load_obstruction(random.choice(obs_files))
        if obs_img is None:
            continue

        obj_poly_new = mask_to_polygon(placed_obj_mask)
        if obj_poly_new is None:
            continue
        ox, oy, obw, obh = cv2.boundingRect(obj_poly_new.astype(np.int32))
        center = np.array([ox + obw / 2.0, oy + obh / 2.0], dtype=np.float32)

        oh0, ow0 = obs_mask.shape[:2]
        target_h = max(8, int(obh * args.obstruction_scale * random.uniform(0.9, 1.1)))
        s = target_h / max(oh0, 1)
        tw = max(8, int(ow0 * s))
        th = max(8, int(oh0 * s))
        obs_img = cv2.resize(obs_img, (tw, th), interpolation=cv2.INTER_LINEAR)
        obs_mask = cv2.resize(obs_mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        # strict orientation: bottom-middle -> top-middle points toward center
        top_mid = np.array([tw / 2.0, 0.0], dtype=np.float32)
        bot_mid = np.array([tw / 2.0, th - 1.0], dtype=np.float32)
        base_vec = top_mid - bot_mid  # unrotated direction
        base_ang = angle_deg(base_vec)

        entry_ang = random.uniform(args.entry_angle_min, args.entry_angle_max)
        d = np.array([np.cos(np.radians(entry_ang)), np.sin(np.radians(entry_ang))], dtype=np.float32)
        target_ang = angle_deg(d)
        rot = (target_ang - base_ang) + random.uniform(-args.rotation_deviation, args.rotation_deviation)

        obs_img, obs_mask, tpts = rotate_bound_with_points(obs_img, obs_mask, rot, [top_mid, bot_mid])
        top_r = tpts[0]
        bot_r = tpts[1]

        boundary_pt, r = pick_boundary_point(placed_obj_mask, center, d)
        target_top = boundary_pt + d * (args.overlap_level * r)

        px = int(round(target_top[0] - top_r[0]))
        py = int(round(target_top[1] - top_r[1]))

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

        new_obj_mask = ((placed_obj_mask > 0) & (full_obs == 0)).astype(np.uint8) * 255
        poly_new = mask_to_polygon(new_obj_mask)
        if poly_new is None:
            continue

        stem = f'{args.class_name}_{"obs_preview" if args.preview_only else "obs"}_{i + 1:06d}'
        ov = base.copy()
        cv2.drawContours(ov, [poly_new.astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 2)

        # Debug visualization: object center + obstruction direction vector
        center_i = (int(round(center[0])), int(round(center[1])))
        top_g = (int(round(px + top_r[0])), int(round(py + top_r[1])))
        bot_g = (int(round(px + bot_r[0])), int(round(py + bot_r[1])))

        cv2.circle(ov, center_i, 7, (0, 255, 255), -1)  # yellow center
        cv2.putText(ov, 'center', (center_i[0] + 8, center_i[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # hand axis bottom->top (should point to center)
        cv2.arrowedLine(ov, bot_g, top_g, (255, 0, 255), 2, tipLength=0.18)  # magenta
        cv2.putText(ov, 'hand vec', (bot_g[0] + 8, bot_g[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

        # vector from top to center (target direction)
        cv2.arrowedLine(ov, top_g, center_i, (255, 255, 0), 2, tipLength=0.12)  # cyan

        if args.preview_only and args.preview_window:
            cv2.namedWindow('Obstruction Preview (q/esc close)', cv2.WINDOW_NORMAL)
            cv2.imshow('Obstruction Preview (q/esc close)', ov)
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
            print('Preview shown in window (no preview image saved).')
        else:
            print(f'Overlays dir: {out_viz_dir}')
    else:
        print(f'Images dir: {out_img_dir}')
        print(f'Labels dir: {out_lbl_dir}')
        print(f'Overlays dir: {out_viz_dir}')


if __name__ == '__main__':
    main()
