#!/usr/bin/env python3
import argparse
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


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


def pick_random_background(bg_files, max_dim=1920):
    bg = cv2.imread(str(random.choice(bg_files)))
    if bg is None:
        return np.full((1080, 1920, 3), 127, dtype=np.uint8)
    bh, bw = bg.shape[:2]
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


def write_yolo_multi(label_path: Path, class_id: int, polys, w, h):
    lines = []
    for poly in polys:
        vals = [str(class_id)]
        for x, y in poly:
            vals.append(f'{x / w:.6f}')
            vals.append(f'{y / h:.6f}')
        lines.append(' '.join(vals))
    label_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--class-id', type=int, required=True)
    ap.add_argument('--background-dir', required=True)
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--num-synthetic', type=int, default=300)
    ap.add_argument('--min-objects', type=int, default=2)
    ap.add_argument('--max-objects', type=int, default=5)
    ap.add_argument('--overlap-prob', type=float, default=0.5)
    ap.add_argument('--min-scale', type=float, default=0.45)
    ap.add_argument('--max-scale', type=float, default=1.10)
    ap.add_argument('--max-rotation', type=float, default=30.0)
    ap.add_argument('--brightness-min', type=float, default=-20.0)
    ap.add_argument('--brightness-max', type=float, default=20.0)
    ap.add_argument('--object-brightness-min', type=float, default=-10.0)
    ap.add_argument('--object-brightness-max', type=float, default=10.0)
    ap.add_argument('--run-name', default='')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.data_root)
    src_img_dir = root / 'images' / args.class_name
    src_lbl_dir = root / 'labels' / args.class_name

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

    cutouts = []
    for im in sorted(src_img_dir.rglob('*')):
        if im.suffix.lower() not in IMG_EXTS:
            continue
        rel = im.relative_to(src_img_dir)
        if any(tag in rel.parts for tag in ['synth_runs', 'obs_runs', 'synth_multi_runs']):
            continue
        if any(tag in im.stem for tag in ['_synth_', '_obs_']):
            continue
        src = cv2.imread(str(im))
        if src is None:
            continue
        h, w = src.shape[:2]
        lb = (src_lbl_dir / rel).with_suffix('.txt')
        if not lb.exists():
            continue
        polys = load_yolo_polys(lb, w, h)
        if not polys:
            continue
        cls_id, poly = polys[0]
        m = poly_to_mask(poly, w, h)
        x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
        crop = src[y:y + bh, x:x + bw]
        crop_m = m[y:y + bh, x:x + bw]
        if crop.size > 0 and (crop_m > 0).sum() > 20:
            cutouts.append((cls_id, crop, crop_m))

    if not cutouts:
        raise RuntimeError('No valid source cutouts found for multi-instance synth')

    made = 0
    for i in range(args.num_synthetic):
        bg = pick_random_background(bg_files)
        h, w = bg.shape[:2]

        nobj = random.randint(max(1, args.min_objects), max(args.min_objects, args.max_objects))

        visible_masks = []
        for _ in range(nobj):
            cls_id, crop, crop_m = random.choice(cutouts)
            ch, cw = crop_m.shape[:2]
            scale = random.uniform(args.min_scale, args.max_scale)
            tw = max(8, int(cw * scale))
            th = max(8, int(ch * scale))
            c_r = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)
            m_r = cv2.resize(crop_m, (tw, th), interpolation=cv2.INTER_NEAREST)

            angle = random.uniform(-args.max_rotation, args.max_rotation)
            c_rr, m_rr = rotate_bound_pair(c_r, m_r, angle)
            ys, xs = np.where(m_rr > 0)
            if len(xs) < 20:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            c_rr = c_rr[y1:y2 + 1, x1:x2 + 1]
            m_rr = m_rr[y1:y2 + 1, x1:x2 + 1]
            oh, ow = m_rr.shape[:2]
            if ow >= w or oh >= h:
                continue

            if visible_masks and random.random() < args.overlap_prob:
                prev = random.choice(visible_masks)
                pyx = np.column_stack(np.where(prev > 0))
                if len(pyx) > 0:
                    yy, xx = pyx[random.randrange(len(pyx))]
                    px = int(np.clip(xx - ow // 2, 0, w - ow))
                    py = int(np.clip(yy - oh // 2, 0, h - oh))
                else:
                    px = random.randint(0, w - ow)
                    py = random.randint(0, h - oh)
            else:
                px = random.randint(0, w - ow)
                py = random.randint(0, h - oh)

            obj_beta = random.uniform(args.object_brightness_min, args.object_brightness_max)
            c_rr = cv2.convertScaleAbs(c_rr, alpha=1.0, beta=obj_beta)

            new_full = np.zeros((h, w), dtype=np.uint8)
            new_full[py:py + oh, px:px + ow][m_rr > 0] = 255

            # Occlusion handling: new object is on top, subtract from previous visible masks
            updated_prev = []
            for pm in visible_masks:
                vis = ((pm > 0) & (new_full == 0)).astype(np.uint8) * 255
                if vis.sum() > 20:
                    updated_prev.append(vis)
            visible_masks = updated_prev

            # Blend new object
            roi = bg[py:py + oh, px:px + ow]
            aa = m_rr > 0
            roi[aa] = c_rr[aa]
            bg[py:py + oh, px:px + ow] = roi
            visible_masks.append(new_full)

        bg_beta = random.uniform(args.brightness_min, args.brightness_max)
        bg = cv2.convertScaleAbs(bg, alpha=1.0, beta=bg_beta)

        polys_new = []
        for vm in visible_masks:
            p = mask_to_polygon(vm)
            if p is not None:
                polys_new.append(p)

        if not polys_new:
            continue

        stem = f'{args.class_name}_synthmulti_{i + 1:06d}'
        out_img = out_img_dir / f'{stem}.jpg'
        out_lbl = out_lbl_dir / f'{stem}.txt'
        out_viz = out_viz_dir / f'{stem}_overlay.jpg'

        cv2.imwrite(str(out_img), bg)
        write_yolo_multi(out_lbl, args.class_id, polys_new, w, h)

        viz = bg.copy()
        colors = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 220, 80), (255, 80, 220)]
        for j, p in enumerate(polys_new):
            col = colors[j % len(colors)]
            cv2.drawContours(viz, [p.astype(np.int32).reshape(-1, 1, 2)], -1, col, 2)
        cv2.putText(viz, f'instances={len(polys_new)}', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_viz), viz)
        made += 1

    print(f'Multi-instance synthetic created: {made}')
    print(f'Run name: {run_name}')
    print(f'Images dir: {out_img_dir}')
    print(f'Labels dir: {out_lbl_dir}')
    print(f'Overlays dir: {out_viz_dir}')


if __name__ == '__main__':
    main()
