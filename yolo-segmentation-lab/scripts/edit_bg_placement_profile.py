#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from queue import Queue, Empty
import threading
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def fit(img, max_w=1600, max_h=900):
    h, w = img.shape[:2]
    s = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if s < 1:
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), s
    return img.copy(), 1.0


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


def render_live_preview(bg, item, sample_obj, mode, args):
    if sample_obj is None:
        return None
    obj_crop, obj_mask = sample_obj
    out = bg.copy()
    h, w = out.shape[:2]

    min_s = float(item.get('min_scale', 0.55))
    max_s = float(item.get('max_scale', 1.25))
    if mode == 'pv_obj_scale_min':
        scale = min_s
    elif mode == 'pv_obj_scale_max':
        scale = max_s
    else:
        scale = float(np.random.uniform(min(min_s, max_s), max(min_s, max_s)))

    if mode == 'pv_bg_bri_min':
        bgb = float(args.bg_brightness_min)
    elif mode == 'pv_bg_bri_max':
        bgb = float(args.bg_brightness_max)
    else:
        bgb = float(np.random.uniform(args.bg_brightness_min, args.bg_brightness_max))

    if mode == 'pv_obj_bri_min':
        objb = float(args.obj_brightness_min)
    elif mode == 'pv_obj_bri_max':
        objb = float(args.obj_brightness_max)
    else:
        objb = float(np.random.uniform(args.obj_brightness_min, args.obj_brightness_max))

    out = cv2.convertScaleAbs(out, alpha=1.0, beta=bgb)

    oh0, ow0 = obj_crop.shape[:2]
    ow = max(8, int(ow0 * scale))
    oh = max(8, int(oh0 * scale))
    obj = cv2.resize(obj_crop, (ow, oh), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(obj_mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
    obj = cv2.convertScaleAbs(obj, alpha=1.0, beta=objb)

    poly_px = None
    if isinstance(item.get('poly'), list) and len(item['poly']) >= 3:
        poly_px = [(int(p[0] * w), int(p[1] * h)) for p in item['poly']]
        xs = [p[0] for p in poly_px]; ys = [p[1] for p in poly_px]
        minx, maxx = max(0, min(xs)), min(w - ow, max(xs))
        miny, maxy = max(0, min(ys)), min(h - oh, max(ys))
        px = max(0, min(w - ow, minx))
        py = max(0, min(h - oh, miny))
        for _ in range(60):
            tx = np.random.randint(minx, maxx + 1) if maxx >= minx else px
            ty = np.random.randint(miny, maxy + 1) if maxy >= miny else py
            if bbox_fully_inside_poly(tx, ty, ow, oh, poly_px):
                px, py = int(tx), int(ty)
                break
    else:
        px = np.random.randint(0, max(1, w - ow + 1))
        py = np.random.randint(0, max(1, h - oh + 1))

    roi = out[py:py+oh, px:px+ow]
    aa = m > 0
    roi[aa] = obj[aa]
    out[py:py+oh, px:px+ow] = roi

    if poly_px:
        cv2.polylines(out, [np.array(poly_px, dtype=np.int32).reshape(-1, 1, 2)], True, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, f'{mode} scale={scale:.2f} bgb={bgb:.0f} objb={objb:.0f}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    return out


def parse_yolo_polygon(label_path, w, h):
    toks = label_path.read_text(encoding='utf-8').strip().split()
    if len(toks) < 7:
        return None
    vals = list(map(float, toks[1:]))
    if len(vals) % 2 != 0:
        return None
    pts = np.array(vals, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return np.round(pts).astype(np.int32)


def extract_object_sample(data_root: Path, class_name: str):
    img_root = data_root / 'images' / class_name
    lbl_root = data_root / 'labels' / class_name
    if not img_root.exists() or not lbl_root.exists():
        return None
    imgs = [p for p in img_root.rglob('*') if p.suffix.lower() in IMG_EXTS]
    for im in sorted(imgs):
        rel = im.relative_to(img_root)
        if 'synth_runs' in rel.parts or 'obs_runs' in rel.parts:
            continue
        lb = (lbl_root / rel).with_suffix('.txt')
        if not lb.exists():
            continue
        src = cv2.imread(str(im))
        if src is None:
            continue
        h, w = src.shape[:2]
        poly = parse_yolo_polygon(lb, w, h)
        if poly is None or len(poly) < 3:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)
        x, y, ww, hh = cv2.boundingRect(poly.reshape(-1, 1, 2))
        crop = src[y:y+hh, x:x+ww].copy()
        m = mask[y:y+hh, x:x+ww].copy()
        if crop.size > 0 and (m > 0).sum() > 30:
            return crop, m
    return None


def launch_control_panel(cmd_q):
    root = tk.Tk()
    root.title('BG Profile Controls')
    root.geometry('320x280')

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill='both', expand=True)

    def push(c):
        cmd_q.put(c)

    ttk.Button(frm, text='Prev (p)', command=lambda: push('prev')).grid(row=0, column=0, sticky='we', padx=4, pady=4)
    ttk.Button(frm, text='Next (n)', command=lambda: push('next')).grid(row=0, column=1, sticky='we', padx=4, pady=4)
    ttk.Button(frm, text='Draw Polygon (r)', command=lambda: push('draw')).grid(row=1, column=0, columnspan=2, sticky='we', padx=4, pady=4)

    ttk.Button(frm, text='Min - ([)', command=lambda: push('min-')).grid(row=2, column=0, sticky='we', padx=4, pady=4)
    ttk.Button(frm, text='Min + (])', command=lambda: push('min+')).grid(row=2, column=1, sticky='we', padx=4, pady=4)
    ttk.Button(frm, text='Max - (-)', command=lambda: push('max-')).grid(row=3, column=0, sticky='we', padx=4, pady=4)
    ttk.Button(frm, text='Max + (=)', command=lambda: push('max+')).grid(row=3, column=1, sticky='we', padx=4, pady=4)

    ttk.Button(frm, text='Save (s)', command=lambda: push('save')).grid(row=4, column=0, sticky='we', padx=4, pady=4)
    ttk.Button(frm, text='Quit (q)', command=lambda: push('quit')).grid(row=4, column=1, sticky='we', padx=4, pady=4)

    ttk.Separator(frm, orient='horizontal').grid(row=5, column=0, columnspan=2, sticky='we', pady=6)
    ttk.Label(frm, text='Live Preview Mode').grid(row=6, column=0, columnspan=2, sticky='w', padx=4)
    ttk.Button(frm, text='ObjScale MIN', command=lambda: push('pv_obj_scale_min')).grid(row=7, column=0, sticky='we', padx=4, pady=3)
    ttk.Button(frm, text='ObjScale MAX', command=lambda: push('pv_obj_scale_max')).grid(row=7, column=1, sticky='we', padx=4, pady=3)
    ttk.Button(frm, text='ObjBright MIN', command=lambda: push('pv_obj_bri_min')).grid(row=8, column=0, sticky='we', padx=4, pady=3)
    ttk.Button(frm, text='ObjBright MAX', command=lambda: push('pv_obj_bri_max')).grid(row=8, column=1, sticky='we', padx=4, pady=3)
    ttk.Button(frm, text='BgBright MIN', command=lambda: push('pv_bg_bri_min')).grid(row=9, column=0, sticky='we', padx=4, pady=3)
    ttk.Button(frm, text='BgBright MAX', command=lambda: push('pv_bg_bri_max')).grid(row=9, column=1, sticky='we', padx=4, pady=3)
    ttk.Button(frm, text='Random Preview', command=lambda: push('pv_random')).grid(row=10, column=0, columnspan=2, sticky='we', padx=4, pady=3)

    ttk.Label(frm, text='Use mouse in image window:\nLeft click add corners\nRight click/Enter finish polygon').grid(row=11, column=0, columnspan=2, sticky='w', padx=4, pady=8)

    frm.columnconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bg-dir', required=True)
    ap.add_argument('--profile', required=True)
    ap.add_argument('--control-window', action='store_true', help='Open a small Tk control window with clickable buttons')
    ap.add_argument('--class-name', default='')
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--bg-brightness-min', type=float, default=-20)
    ap.add_argument('--bg-brightness-max', type=float, default=20)
    ap.add_argument('--obj-brightness-min', type=float, default=-15)
    ap.add_argument('--obj-brightness-max', type=float, default=15)
    args = ap.parse_args()

    bg_dir = Path(args.bg_dir)
    files = sorted([p for p in bg_dir.rglob('*') if p.suffix.lower() in IMG_EXTS])
    if not files:
        raise RuntimeError('No background images found')

    profile_path = Path(args.profile)
    data = {'items': {}}
    if profile_path.exists():
        try:
            data = json.loads(profile_path.read_text(encoding='utf-8'))
            if 'items' not in data:
                data = {'items': {}}
        except Exception:
            data = {'items': {}}

    def save_now():
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        print(f'Auto-saved profile: {profile_path}')

    sample_obj = None
    if args.class_name.strip():
        sample_obj = extract_object_sample(Path(args.data_root), args.class_name.strip())

    preview_mode = 'pv_random'

    cmd_q = Queue()
    if args.control_window:
        th = threading.Thread(target=launch_control_panel, args=(cmd_q,), daemon=True)
        th.start()

    idx = 0
    while True:
        fp = files[idx]
        key = fp.name
        item = data['items'].get(key, {'rect': [0.0, 0.0, 1.0, 1.0], 'min_scale': 0.55, 'max_scale': 1.25})

        img = cv2.imread(str(fp))
        if img is None:
            idx = (idx + 1) % len(files)
            continue

        disp, s = fit(img)
        h, w = disp.shape[:2]
        if isinstance(item.get('poly'), list) and len(item.get('poly')) >= 3:
            poly = np.array([[int(pt[0] * w), int(pt[1] * h)] for pt in item['poly']], dtype=np.int32)
            cv2.polylines(disp, [poly.reshape(-1, 1, 2)], True, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            rx1, ry1, rx2, ry2 = item['rect']
            x1, y1, x2, y2 = int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h)
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
        txt = f'{idx+1}/{len(files)} {fp.name}  min={item.get("min_scale",0.55):.2f} max={item.get("max_scale",1.25):.2f}'
        cv2.putText(disp, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(disp, 'n/p next-prev | r draw polygon | [/ ] min -/+ | -/= max -/+ | s save | q quit', (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow('BG Placement Profile Editor', disp)
        live = render_live_preview(img, item, sample_obj, preview_mode, args)
        if live is not None:
            live_show, _ = fit(live, 1600, 900)
            cv2.imshow('BG Live Component Preview', live_show)

        # Non-blocking key read to support optional control window queue.
        k = cv2.waitKey(50) & 0xFF
        cmd = None
        try:
            cmd = cmd_q.get_nowait()
        except Empty:
            pass

        if cmd in {'pv_obj_scale_min','pv_obj_scale_max','pv_obj_bri_min','pv_obj_bri_max','pv_bg_bri_min','pv_bg_bri_max','pv_random'}:
            preview_mode = cmd
            continue

        if cmd == 'quit' or k in (ord('q'), 27):
            break
        if cmd == 'next' or k == ord('n'):
            idx = min(len(files)-1, idx + 1)
            continue
        if cmd == 'prev' or k == ord('p'):
            idx = max(0, idx - 1)
            continue
        if cmd == 'draw' or k == ord('r'):
            sel_img, s2 = fit(img)
            points = []
            done = {'v': False}

            def on_mouse(evt, x, y, flags, param):
                if evt == cv2.EVENT_LBUTTONDOWN:
                    points.append((x, y))
                elif evt == cv2.EVENT_RBUTTONDOWN:
                    done['v'] = True

            cv2.setMouseCallback('BG Placement Profile Editor', on_mouse)
            while True:
                disp2 = sel_img.copy()
                if len(points) > 0:
                    for pt in points:
                        cv2.circle(disp2, pt, 3, (0, 255, 255), -1)
                    for i2 in range(1, len(points)):
                        cv2.line(disp2, points[i2 - 1], points[i2], (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(disp2, 'Left click: add corner, Right click/Enter: finish polygon, c: cancel', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow('BG Placement Profile Editor', disp2)
                kk = cv2.waitKey(20) & 0xFF
                if kk in (13, 10):
                    done['v'] = True
                if kk == ord('c'):
                    points = []
                    break
                if done['v']:
                    break

            if len(points) >= 3:
                ph, pw = sel_img.shape[:2]
                poly = []
                for x, y in points:
                    poly.append([round(float(x / max(1, pw)), 4), round(float(y / max(1, ph)), 4)])
                item['poly'] = poly
                # keep a compatible rect bbox too
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                item['rect'] = [min(xs), min(ys), max(xs), max(ys)]
                data['items'][key] = item
                save_now()
            cv2.setMouseCallback('BG Placement Profile Editor', lambda *a: None)
            continue
        if cmd == 'min-' or k == ord('['):
            item['min_scale'] = round(max(0.05, float(item.get('min_scale', 0.55)) - 0.05), 3)
            data['items'][key] = item
            save_now()
            continue
        if k == ord(']'):
            item['min_scale'] = round(min(3.0, float(item.get('min_scale', 0.55)) + 0.05), 3)
            data['items'][key] = item
            save_now()
            continue
        if cmd == 'max-' or k == ord('-'):
            item['max_scale'] = round(max(0.05, float(item.get('max_scale', 1.25)) - 0.05), 3)
            data['items'][key] = item
            save_now()
            continue
        if cmd == 'max+' or k == ord('='):
            item['max_scale'] = round(min(3.0, float(item.get('max_scale', 1.25)) + 0.05), 3)
            data['items'][key] = item
            save_now()
            continue
        if cmd == 'save' or k == ord('s'):
            save_now()
            continue

    cv2.destroyAllWindows()
    save_now()


if __name__ == '__main__':
    main()
