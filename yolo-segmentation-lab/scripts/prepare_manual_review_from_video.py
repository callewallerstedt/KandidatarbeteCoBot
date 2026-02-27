#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
from rembg import remove


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


def write_yolo(label_path: Path, class_id: int, poly, w, h):
    vals = [str(class_id)]
    for x, y in poly:
        vals.append(f'{x / w:.6f}')
        vals.append(f'{y / h:.6f}')
    label_path.write_text(' '.join(vals) + '\n', encoding='utf-8')


def foreground_mask_rembg(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out = remove(rgb)
    if out.shape[2] == 4:
        alpha = out[:, :, 3]
    else:
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def foreground_mask_yolo(frame_bgr, model, class_id, conf=0.20, imgsz=640, device='0'):
    r = model.predict(frame_bgr, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        return None

    cls = r.boxes.cls.cpu().numpy().astype(int)
    scores = r.boxes.conf.cpu().numpy()
    masks = r.masks.data.cpu().numpy()  # N x H x W

    idxs = np.where(cls == int(class_id))[0]
    if len(idxs) == 0:
        idxs = np.arange(len(cls))

    best = idxs[int(np.argmax(scores[idxs]))]
    m = (masks[best] > 0.5).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--class-id', type=int, required=True)
    ap.add_argument('--num-samples', type=int, default=80)
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--prefix', default='manual')
    ap.add_argument('--init-source', choices=['rembg', 'yolo'], default='yolo')
    ap.add_argument('--weights', default=str(Path(__file__).resolve().parents[1] / 'runs' / 'segment' / 'train' / 'weights' / 'best.pt'))
    ap.add_argument('--init-conf', type=float, default=0.20)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0')
    args = ap.parse_args()

    root = Path(args.data_root)
    img_dir = root / 'images' / args.class_name
    lbl_dir = root / 'labels' / args.class_name
    viz_dir = root / 'staging' / f'{args.class_name}_manual_review'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    yolo_model = None
    if args.init_source == 'yolo':
        from ultralytics import YOLO
        wp = Path(args.weights)
        if not wp.exists():
            raise RuntimeError(f'YOLO init weights not found: {wp}')
        yolo_model = YOLO(str(wp))
        print(f'Using YOLO initial masks from: {wp}')
    else:
        print('Using rembg initial masks.')

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        raise RuntimeError('Could not detect total frame count for even sampling.')

    n = min(max(args.num_samples, 1), total)
    picks = set(np.linspace(0, total - 1, num=n, dtype=int).tolist())

    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx not in picks:
            idx += 1
            continue

        if args.init_source == 'yolo':
            mask = foreground_mask_yolo(
                frame,
                yolo_model,
                class_id=args.class_id,
                conf=args.init_conf,
                imgsz=args.imgsz,
                device=args.device,
            )
            if mask is None:
                idx += 1
                continue
        else:
            mask = foreground_mask_rembg(frame)

        poly = mask_to_polygon(mask)
        if poly is None:
            idx += 1
            continue

        h, w = frame.shape[:2]
        stem = f'{args.class_name}_{args.prefix}_{saved + 1:06d}'
        imp = img_dir / f'{stem}.jpg'
        lbp = lbl_dir / f'{stem}.txt'
        vsp = viz_dir / f'{stem}_overlay.jpg'

        cv2.imwrite(str(imp), frame)
        write_yolo(lbp, args.class_id, poly, w, h)
        ov = frame.copy()
        cv2.drawContours(ov, [poly.reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
        cv2.imwrite(str(vsp), ov)

        saved += 1
        idx += 1

    cap.release()
    print(f'Prepared {saved} frames for manual review.')
    print(f'Images: {img_dir}')
    print(f'Labels: {lbl_dir}')
    print(f'Overlays: {viz_dir}')


if __name__ == '__main__':
    main()
