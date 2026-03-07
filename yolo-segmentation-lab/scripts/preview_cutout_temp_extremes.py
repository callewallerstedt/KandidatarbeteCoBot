#!/usr/bin/env python3
import argparse
import random
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from profile_scene_builders import collect_cutouts, weighted_choice, _apply_temperature_variation


def composite_on_neutral(obj_img, obj_mask, bg_value=238):
    h, w = obj_mask.shape[:2]
    canvas = np.full((h, w, 3), bg_value, dtype=np.uint8)
    alpha = np.clip(obj_mask.astype(np.float32) / 255.0, 0.0, 1.0)[:, :, None]
    out = canvas.astype(np.float32) * (1.0 - alpha) + obj_img.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_tile(cutout, temp_bias, title):
    img = _apply_temperature_variation(cutout['image'], cutout['mask'], temp_bias, 0.0)
    comp = composite_on_neutral(img, cutout['mask'])
    comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(comp)
    pil.thumbnail((280, 220))
    return pil, title


class TempPreviewApp(tk.Tk):
    def __init__(self, class_name, cutouts, cool_bias, warm_bias, sample_count):
        super().__init__()
        self.title(f'Temperature Extremes Preview - {class_name}')
        self.geometry('980x900')

        header = ttk.Frame(self, padding=10)
        header.pack(fill='x')
        ttk.Label(
            header,
            text=f'Quick cutout preview only. cool={cool_bias:.3f} warm={warm_bias:.3f}',
        ).pack(anchor='w')

        outer = ttk.Frame(self, padding=10)
        outer.pack(fill='both', expand=True)

        samples = [weighted_choice(cutouts) for _ in range(max(1, sample_count))]
        self.photos = []

        for row_idx, cutout in enumerate(samples):
            rel = cutout.get('source_rel', cutout.get('source', ''))
            ttk.Label(outer, text=rel, wraplength=900).grid(row=row_idx * 2, column=0, columnspan=3, sticky='w', pady=(0, 4))
            for col_idx, (bias, title) in enumerate(((0.0, 'Base'), (cool_bias, 'Cool extreme'), (warm_bias, 'Warm extreme'))):
                pil, caption = make_tile(cutout, bias, title)
                photo = ImageTk.PhotoImage(pil)
                self.photos.append(photo)
                lbl = tk.Label(outer, image=photo, bd=2, relief='groove')
                lbl.grid(row=row_idx * 2 + 1, column=col_idx, padx=8, pady=8, sticky='n')
                ttk.Label(outer, text=caption).grid(row=row_idx * 2 + 2, column=col_idx, pady=(0, 8))

        for i in range(3):
            outer.columnconfigure(i, weight=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--cool-bias', type=float, required=True)
    ap.add_argument('--warm-bias', type=float, required=True)
    ap.add_argument('--sample-count', type=int, default=4)
    args = ap.parse_args()

    random.seed()
    cutouts = collect_cutouts(Path(args.data_root), args.class_name)
    app = TempPreviewApp(args.class_name, cutouts, args.cool_bias, args.warm_bias, args.sample_count)
    app.mainloop()


if __name__ == '__main__':
    main()
