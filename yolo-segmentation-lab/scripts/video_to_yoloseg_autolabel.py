#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
from rembg import remove


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


def write_yolo_seg(label_path: Path, class_id: int, poly, w, h):
    vals = [str(class_id)]
    for x, y in poly:
        vals.append(f"{x / w:.6f}")
        vals.append(f"{y / h:.6f}")
    label_path.write_text(" ".join(vals) + "\n", encoding="utf-8")


def augment(img):
    out = img.copy()
    alpha = random.uniform(0.85, 1.20)
    beta = random.uniform(-20, 20)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if random.random() < 0.2:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def foreground_mask_bgr(frame_bgr, quality='high'):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out = remove(frame_rgb)
    # rembg can return RGBA
    if out.shape[2] == 4:
        alpha = out[:, :, 3]
    else:
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    if quality == 'high':
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        _, mask = cv2.threshold(alpha, 100, 255, cv2.THRESH_BINARY)
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((7, 7), np.uint8)
    else:
        _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        kernel_open = np.ones((5, 5), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--class-id', type=int, required=True)
    ap.add_argument('--out-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--every', type=int, default=8, help='Use every N frames when --num-samples is not set')
    ap.add_argument('--num-samples', type=int, default=0, help='If >0, sample this many frames evenly across the full video')
    ap.add_argument('--max-frames', type=int, default=400)
    ap.add_argument('--aug-per-frame', type=int, default=1)
    ap.add_argument('--min-area-ratio', type=float, default=0.01)
    ap.add_argument('--max-area-ratio', type=float, default=0.80)
    ap.add_argument('--mask-quality', choices=['fast', 'high'], default='high')
    ap.add_argument('--poly-eps', type=float, default=0.0008, help='Polygon simplification ratio; smaller = more detail')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out_root)
    images_dir = out_root / 'images' / args.class_name
    labels_dir = out_root / 'labels' / args.class_name
    viz_dir = out_root / 'staging' / args.class_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        print('Warning: could not detect total frame count, fallback to --every sampling.')

    sample_indices = None
    if args.num_samples > 0 and total_frames > 0:
        n = min(args.num_samples, total_frames)
        sample_indices = set(np.linspace(0, total_frames - 1, num=n, dtype=int).tolist())
        print(f'Video frames detected: {total_frames}. Evenly sampling {len(sample_indices)} frames.')
    else:
        print('Using --every sampling mode.')

    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if sample_indices is not None:
            if idx not in sample_indices:
                idx += 1
                continue
        else:
            if idx % args.every != 0:
                idx += 1
                continue

        for a in range(args.aug_per_frame + 1):
            img = frame if a == 0 else augment(frame)
            mask = foreground_mask_bgr(img, quality=args.mask_quality)
            h, w = mask.shape
            area = float((mask > 0).sum()) / float(h * w)
            if area < args.min_area_ratio or area > args.max_area_ratio:
                continue

            poly = mask_to_polygon(mask, eps=args.poly_eps)
            if poly is None:
                continue

            kept += 1
            stem = f"{args.class_name}_{kept:06d}"
            imp = images_dir / f"{stem}.jpg"
            lbp = labels_dir / f"{stem}.txt"
            vsp = viz_dir / f"{stem}_overlay.jpg"

            cv2.imwrite(str(imp), img)
            write_yolo_seg(lbp, args.class_id, poly, w, h)

            overlay = img.copy()
            cv2.drawContours(overlay, [poly.reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
            cv2.imwrite(str(vsp), overlay)

            if kept >= args.max_frames:
                break

        if kept >= args.max_frames:
            break
        idx += 1

    cap.release()
    print(f'Created {kept} labeled images for class={args.class_name} class_id={args.class_id}')
    print(f'Images: {images_dir}')
    print(f'Labels: {labels_dir}')
    print(f'Overlays: {viz_dir}')


if __name__ == '__main__':
    main()
