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

    ttk.Label(frm, text='Use mouse in image window:\nLeft click add corners\nRight click/Enter finish polygon').grid(row=5, column=0, columnspan=2, sticky='w', padx=4, pady=8)

    frm.columnconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bg-dir', required=True)
    ap.add_argument('--profile', required=True)
    ap.add_argument('--control-window', action='store_true', help='Open a small Tk control window with clickable buttons')
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

        # Non-blocking key read to support optional control window queue.
        k = cv2.waitKey(50) & 0xFF
        cmd = None
        try:
            cmd = cmd_q.get_nowait()
        except Empty:
            pass

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
