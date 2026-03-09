#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
from rembg import remove


def magenta_suppression_mask(work_bgr, hsv, med_hsv=None, hue_tol=28):
    b = work_bgr[:, :, 0].astype(np.int16)
    g = work_bgr[:, :, 1].astype(np.int16)
    r = work_bgr[:, :, 2].astype(np.int16)

    med_h = 150 if med_hsv is None else int(med_hsv[0])
    strength = max(1, int(hue_tol))
    hsv_sat_min = min(140, max(35, 110 - 3 * strength))
    hsv_val_min = min(120, max(0, 50 - 2 * strength))
    mag_low = np.array([max(0, med_h - strength), hsv_sat_min, hsv_val_min], dtype=np.uint8)
    mag_high = np.array([min(179, med_h + strength), 255, 255], dtype=np.uint8)
    hsv_magenta = cv2.inRange(hsv, mag_low, mag_high) > 0

    # Scale RGB gating with key strength so the UI slider actually changes behavior.
    rg_delta = max(18, 44 - strength)
    sum_min = max(70, 150 - 3 * strength)
    chroma_min = max(22, 62 - 2 * strength)
    green_max = min(150, 70 + 2 * strength)
    rb_balance = min(90, 25 + 2 * strength)

    magenta_rgb = (
        (r >= g + rg_delta) &
        (b >= g + rg_delta) &
        (r + b >= sum_min) &
        ((r + b) - 2 * g >= chroma_min)
    )
    magenta_rgb_strict = (
        (r >= max(35, sum_min // 2)) &
        (b >= max(35, sum_min // 2)) &
        (g <= green_max) &
        (np.abs(r - b) <= rb_balance) &
        (np.maximum(r, b) >= g + max(10, rg_delta - 4))
    )

    magenta_mask = np.where(hsv_magenta | magenta_rgb | magenta_rgb_strict, 255, 0).astype(np.uint8)
    return magenta_mask


def border_median_hsv(hsv_img):
    border = np.concatenate([hsv_img[0, :, :], hsv_img[-1, :, :], hsv_img[:, 0, :], hsv_img[:, -1, :]], axis=0)
    return np.median(border, axis=0).astype(np.uint8)


def largest_component_mask(bin_mask):
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if nlab <= 1:
        return bin_mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return np.where(labels == best, 255, 0).astype(np.uint8)


def centered_component_mask(bin_mask):
    nlab, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if nlab <= 1:
        return bin_mask
    h, w = bin_mask.shape[:2]
    cx = w / 2.0
    cy = h / 2.0
    best_label = 0
    best_score = None
    img_area = float(max(1, h * w))
    for lab in range(1, nlab):
        area = float(stats[lab, cv2.CC_STAT_AREA])
        if area < 0.0005 * img_area:
            continue
        px, py = centroids[lab]
        dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 / max(1.0, (w * w + h * h) ** 0.5)
        score = dist - 0.35 * min(area / img_area, 0.5)
        if best_score is None or score < best_score:
            best_score = score
            best_label = lab
    if best_label == 0:
        return largest_component_mask(bin_mask)
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def rembg_mask_from_bgr(frame_bgr, quality='high', alpha_threshold=-1, mask_upscale=1, keep_largest=True):
    h0, w0 = frame_bgr.shape[:2]
    work = frame_bgr
    up = max(1, int(mask_upscale))
    if up > 1:
        work = cv2.resize(frame_bgr, (w0 * up, h0 * up), interpolation=cv2.INTER_CUBIC)

    frame_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    out = remove(frame_rgb)
    if out.shape[2] == 4:
        alpha = out[:, :, 3]
    else:
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    if quality == 'ultra':
        alpha = cv2.bilateralFilter(alpha, 7, 35, 35)
        thr = alpha_threshold if alpha_threshold >= 0 else 82
        _, mask = cv2.threshold(alpha, int(thr), 255, cv2.THRESH_BINARY)
        kernel_open = np.ones((2, 2), np.uint8)
        kernel_close = np.ones((3, 3), np.uint8)
    elif quality == 'high':
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

    if keep_largest:
        mask = centered_component_mask(mask)

    if up > 1:
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_AREA)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def chroma_key_mask_bgr(frame_bgr, bg_color_threshold=0, mask_upscale=1, keep_largest=True, key_threshold=24):
    h0, w0 = frame_bgr.shape[:2]
    up = max(1, int(mask_upscale))
    work = frame_bgr if up == 1 else cv2.resize(frame_bgr, (w0 * up, h0 * up), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    med = border_median_hsv(hsv)

    hue_tol = max(1, int(key_threshold))
    bg_mask = magenta_suppression_mask(work, hsv, med_hsv=med, hue_tol=hue_tol)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    bg_mask = largest_component_mask(bg_mask)

    cnts, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((h0, w0), dtype=np.uint8)

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    pad = max(2, int(round(0.02 * max(w, h))))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(work.shape[1], x + w + pad)
    y2 = min(work.shape[0], y + h + pad)

    screen_mask = np.zeros_like(bg_mask)
    screen_mask[y1:y2, x1:x2] = 255

    roi = work[y1:y2, x1:x2]
    roi_mask = rembg_mask_from_bgr(
        roi,
        quality='high',
        alpha_threshold=-1,
        mask_upscale=1,
        keep_largest=True,
    )

    # Remove obvious magenta remnants inside the screen crop, but keep the foreground chosen by rembg.
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_magenta = magenta_suppression_mask(roi, roi_hsv, med_hsv=med, hue_tol=hue_tol)
    roi_mask[roi_magenta > 0] = 0
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    roi_mask = centered_component_mask(roi_mask)

    fg_mask = np.zeros_like(bg_mask)
    fg_mask[y1:y2, x1:x2] = roi_mask

    if keep_largest:
        fg_mask = centered_component_mask(fg_mask)

    if up > 1:
        fg_mask = cv2.resize(fg_mask, (w0, h0), interpolation=cv2.INTER_AREA)
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    return fg_mask


def mask_to_polygon(mask, eps=0.0008):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if eps <= 0:
        poly = c.reshape(-1, 2)
        if len(poly) < 3:
            return None
        return poly
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


def compose_preview_image(img, mask, poly, show_bg=True, show_overlay=True):
    if show_bg:
        preview = img.copy()
    else:
        preview = np.zeros_like(img)
        preview[mask > 0] = img[mask > 0]

    if show_overlay:
        tint = np.zeros_like(preview)
        tint[:, :] = (40, 220, 40)
        alpha = (mask > 0)
        preview[alpha] = cv2.addWeighted(preview[alpha], 0.72, tint[alpha], 0.28, 0)
        cv2.drawContours(preview, [poly.reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
    return preview


def foreground_mask_bgr(frame_bgr, quality='high', alpha_threshold=-1, bg_color_threshold=0, mask_upscale=1, keep_largest=True, magenta_bg_boost=False, magenta_key_threshold=24):
    if magenta_bg_boost:
        return chroma_key_mask_bgr(
            frame_bgr,
            bg_color_threshold=bg_color_threshold,
            mask_upscale=mask_upscale,
            keep_largest=keep_largest,
            key_threshold=magenta_key_threshold,
        )

    mask = rembg_mask_from_bgr(
        frame_bgr,
        quality=quality,
        alpha_threshold=alpha_threshold,
        mask_upscale=mask_upscale,
        keep_largest=keep_largest,
    )

    hsv = None
    med = None
    if bg_color_threshold and bg_color_threshold > 0:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        med = border_median_hsv(hsv)

    # Optional background-color suppression: sample border color and remove similar pixels.
    if bg_color_threshold and bg_color_threshold > 0:
        tol = int(bg_color_threshold)
        hue_tol = tol
        sat_low = max(0, int(med[1]) - tol)
        val_low = max(0, int(med[2]) - tol)
        sat_high = min(255, int(med[1]) + tol)
        val_high = min(255, int(med[2]) + tol)
        is_magenta_border = 135 <= int(med[0]) <= 175 and int(med[1]) >= 40
        if is_magenta_border:
            hue_tol = min(35, tol + 8)
            sat_low = max(0, int(med[1]) - tol - 35)
            val_low = max(0, int(med[2]) - max(40, 2 * tol))
            val_high = min(255, int(med[2]) + tol + 10)
        low = np.array([max(0, int(med[0]) - hue_tol), sat_low, val_low], dtype=np.uint8)
        high = np.array([min(179, int(med[0]) + hue_tol), sat_high, val_high], dtype=np.uint8)
        bg_like = cv2.inRange(hsv, low, high)
        mask[bg_like > 0] = 0
        if keep_largest:
            mask = centered_component_mask(mask)

    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='')
    ap.add_argument('--video-dir', default='')
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
    ap.add_argument('--mask-quality', choices=['fast', 'high', 'ultra'], default='high')
    ap.add_argument('--poly-eps', type=float, default=0.00035, help='Polygon simplification ratio; smaller = more detail (0 = full contour)')
    ap.add_argument('--mask-upscale', type=int, default=1, help='Upscale factor before rembg (ultra detail). 1=off, 2=better edges')
    ap.add_argument('--keep-largest-component', action='store_true', help='Keep only largest connected foreground component')
    ap.add_argument('--alpha-threshold', type=int, default=-1, help='Alpha threshold for foreground mask (0-255). Lower=more sensitive, higher=stricter.')
    ap.add_argument('--bg-color-threshold', type=int, default=0, help='HSV tolerance to remove border-like background colors (0=off)')
    ap.add_argument('--magenta-bg-boost', action='store_true', help='If border color is magenta, remove a wider and darker magenta range.')
    ap.add_argument('--magenta-key-threshold', type=int, default=24, help='Magenta chroma-key strength; higher catches a wider magenta range')
    ap.add_argument('--preview-only', action='store_true', help='Generate previews only (no train image/label writes)')
    ap.add_argument('--preview-count', type=int, default=16, help='Limit previews when --preview-only is set')
    ap.add_argument('--preview-max-width', type=int, default=1600)
    ap.add_argument('--preview-max-height', type=int, default=900)
    ap.add_argument('--preview-hide-bg', action='store_true', help='Preview only the masked object without original background')
    ap.add_argument('--preview-overlay-mask', action='store_true', help='Overlay the accepted mask/contour on the preview image')
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
    if not args.preview_only:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        reject_dir.mkdir(parents=True, exist_ok=True)

    video_files = []
    if args.video_dir.strip():
        vdir = Path(args.video_dir)
        exts = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv'}
        video_files = sorted([p for p in vdir.rglob('*') if p.suffix.lower() in exts])
        if not video_files:
            raise RuntimeError(f'No video files found in folder: {vdir}')
    elif args.video.strip():
        video_files = [Path(args.video)]
    else:
        raise RuntimeError('Provide --video or --video-dir')

    print(f'Autolabel video sources: {len(video_files)}')

    idx = 0
    kept = 0
    rejected = 0
    previews = []
    target_kept = args.preview_count if args.preview_only else args.max_frames
    for vf in video_files:
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            print(f'Warning: cannot open video: {vf}')
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_indices = None
        if args.num_samples > 0 and total_frames > 0:
            per_video = max(1, int(np.ceil(args.num_samples / max(1, len(video_files)))))
            n = min(per_video, total_frames)
            sample_indices = set(np.linspace(0, total_frames - 1, num=n, dtype=int).tolist())
        local_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if sample_indices is not None:
                if local_idx not in sample_indices:
                    local_idx += 1
                    continue
            else:
                if local_idx % args.every != 0:
                    local_idx += 1
                    continue

            for a in range(args.aug_per_frame + 1):
                mask_src = frame
                if mask_src is None or mask_src.size == 0:
                    continue
                upscale_eff = args.mask_upscale
                if args.mask_quality == 'ultra' and upscale_eff < 2:
                    upscale_eff = 2
                keep_largest_eff = args.keep_largest_component or (args.mask_quality == 'ultra')
                mask = foreground_mask_bgr(
                    mask_src,
                    quality=args.mask_quality,
                    alpha_threshold=args.alpha_threshold,
                    bg_color_threshold=args.bg_color_threshold,
                    mask_upscale=upscale_eff,
                    keep_largest=keep_largest_eff,
                    magenta_bg_boost=args.magenta_bg_boost,
                    magenta_key_threshold=args.magenta_key_threshold,
                )
                img = frame if a == 0 else augment(frame)
                if img is None or img.size == 0:
                    continue
                h, w = mask.shape
                area = float((mask > 0).sum()) / float(h * w)
                if area < args.min_area_ratio or area > args.max_area_ratio:
                    rejected += 1
                    if (not args.preview_only) and rejected <= 50:
                        cv2.imwrite(str(reject_dir / f'reject_area_{idx:06d}_{a}.jpg'), img)
                    continue

                poly = mask_to_polygon(mask, eps=args.poly_eps)
                if poly is None:
                    rejected += 1
                    if (not args.preview_only) and rejected <= 50:
                        cv2.imwrite(str(reject_dir / f'reject_poly_{idx:06d}_{a}.jpg'), img)
                    continue

                kept += 1
                stem = f"{args.class_name}_{kept:06d}"
                imp = images_dir / f"{stem}.jpg"
                lbp = labels_dir / f"{stem}.txt"
                vsp = overlay_dir / f"{stem}_overlay.jpg"
                msp = mask_dir / f"{stem}_mask.png"

                overlay = compose_preview_image(
                    img,
                    mask,
                    poly,
                    show_bg=not args.preview_hide_bg,
                    show_overlay=args.preview_overlay_mask,
                )

                if args.preview_only:
                    previews.append(overlay)
                else:
                    cv2.imwrite(str(vsp), overlay)
                    cv2.imwrite(str(msp), mask)
                    cv2.imwrite(str(imp), img)
                    write_yolo_seg(lbp, args.class_id, poly, w, h)

                if kept >= target_kept:
                    break

            local_idx += 1
            if kept >= target_kept:
                break

        cap.release()
        if kept >= target_kept:
            break
        idx += 1
    mode = 'PREVIEW_ONLY' if args.preview_only else 'WRITE_DATASET'
    print(f'Autolabel mode: {mode}')
    print(f'Created {kept} samples for class={args.class_name} class_id={args.class_id} component={comp} run={run_name}')
    print(f'Rejected samples: {rejected}')
    if args.preview_only:
        if args.preview_count > 0 and previews:
            win = 'Auto-label Preview (left/right, q to quit)'
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            i = 0
            while True:
                show = previews[i].copy()
                cv2.putText(show, f'{i+1}/{len(previews)}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                h0, w0 = show.shape[:2]
                s = min(args.preview_max_width / max(1, w0), args.preview_max_height / max(1, h0), 1.0)
                if s < 1.0:
                    show = cv2.resize(show, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
                cv2.imshow(win, show)
                k = cv2.waitKey(0)
                if k in (ord('q'), 27):
                    break
                if k in (81, 2424832, ord('a')):
                    i = max(0, i - 1)
                elif k in (83, 2555904, ord('d')):
                    i = min(len(previews) - 1, i + 1)
            cv2.destroyAllWindows()
        print('Preview-only mode: nothing written to dataset folders.')
    else:
        print(f'Images: {images_dir}')
        print(f'Labels: {labels_dir}')
        print(f'Preview overlays: {overlay_dir}')
        print(f'Preview masks: {mask_dir}')
        print(f'Rejected previews: {reject_dir}')


if __name__ == '__main__':
    main()
