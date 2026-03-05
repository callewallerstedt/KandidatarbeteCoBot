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


def foreground_mask_bgr(frame_bgr, quality='high', alpha_threshold=-1):
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
        thr = alpha_threshold if alpha_threshold >= 0 else 100
        _, mask = cv2.threshold(alpha, int(thr), 255, cv2.THRESH_BINARY)
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((7, 7), np.uint8)
    else:
        thr = alpha_threshold if alpha_threshold >= 0 else 127
        _, mask = cv2.threshold(alpha, int(thr), 255, cv2.THRESH_BINARY)
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
    ap.add_argument('--component-name', default='main', help='Component/subpart grouping, e.g. head/handle/body')
    ap.add_argument('--run-name', default='', help='Optional run name. Auto-generated when empty')
    ap.add_argument('--every', type=int, default=8, help='Use every N frames when --num-samples is not set')
    ap.add_argument('--num-samples', type=int, default=0, help='If >0, sample this many frames evenly across the full video')
    ap.add_argument('--max-frames', type=int, default=400)
    ap.add_argument('--aug-per-frame', type=int, default=1)
    ap.add_argument('--min-area-ratio', type=float, default=0.01)
    ap.add_argument('--max-area-ratio', type=float, default=0.80)
    ap.add_argument('--mask-quality', choices=['fast', 'high'], default='high')
    ap.add_argument('--poly-eps', type=float, default=0.0008, help='Polygon simplification ratio; smaller = more detail')
    ap.add_argument('--alpha-threshold', type=int, default=-1, help='Alpha threshold for foreground mask (0-255). Lower=more sensitive, higher=stricter.')
    ap.add_argument('--preview-only', action='store_true', help='Generate previews only (no train image/label writes)')
    ap.add_argument('--preview-count', type=int, default=16, help='Limit previews when --preview-only is set')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out_root)
    run_name = args.run_name.strip() or __import__('datetime').datetime.now().strftime('autolabel_%Y%m%d_%H%M%S')
    comp = args.component_name.strip() or 'main'

    images_dir = out_root / 'images' / args.class_name / 'components' / comp / 'runs' / run_name / 'frames'
    labels_dir = out_root / 'labels' / args.class_name / 'components' / comp / 'runs' / run_name / 'frames'
    viz_dir = out_root / 'staging' / args.class_name / 'autolabel' / comp / run_name
    overlay_dir = viz_dir / 'overlays'
    mask_dir = viz_dir / 'masks'
    reject_dir = viz_dir / 'rejected'

    if not args.preview_only:
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    reject_dir.mkdir(parents=True, exist_ok=True)

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
    rejected = 0
    target_kept = args.preview_count if args.preview_only else args.max_frames
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
            mask = foreground_mask_bgr(img, quality=args.mask_quality, alpha_threshold=args.alpha_threshold)
            h, w = mask.shape
            area = float((mask > 0).sum()) / float(h * w)
            if area < args.min_area_ratio or area > args.max_area_ratio:
                rejected += 1
                if rejected <= 50:
                    cv2.imwrite(str(reject_dir / f'reject_area_{idx:06d}_{a}.jpg'), img)
                continue

            poly = mask_to_polygon(mask, eps=args.poly_eps)
            if poly is None:
                rejected += 1
                if rejected <= 50:
                    cv2.imwrite(str(reject_dir / f'reject_poly_{idx:06d}_{a}.jpg'), img)
                continue

            kept += 1
            stem = f"{args.class_name}_{kept:06d}"
            imp = images_dir / f"{stem}.jpg"
            lbp = labels_dir / f"{stem}.txt"
            vsp = overlay_dir / f"{stem}_overlay.jpg"
            msp = mask_dir / f"{stem}_mask.png"

            overlay = img.copy()
            cv2.drawContours(overlay, [poly.reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
            cv2.imwrite(str(vsp), overlay)
            cv2.imwrite(str(msp), mask)

            if not args.preview_only:
                cv2.imwrite(str(imp), img)
                write_yolo_seg(lbp, args.class_id, poly, w, h)

            if kept >= target_kept:
                break

        if kept >= target_kept:
            break
        idx += 1

    cap.release()
    mode = 'PREVIEW_ONLY' if args.preview_only else 'WRITE_DATASET'
    print(f'Autolabel mode: {mode}')
    print(f'Created {kept} samples for class={args.class_name} class_id={args.class_id} component={comp} run={run_name}')
    print(f'Rejected samples: {rejected}')
    if not args.preview_only:
        print(f'Images: {images_dir}')
        print(f'Labels: {labels_dir}')
    print(f'Preview overlays: {overlay_dir}')
    print(f'Preview masks: {mask_dir}')
    print(f'Rejected previews: {reject_dir}')


if __name__ == '__main__':
    main()
