#!/usr/bin/env python3
import argparse
import json
import random
import re
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
EXCLUDED_TAGS = {'synth_runs', 'obs_runs', 'synth_multi_runs', 'unity_runs'}


def parse_first_polygon(label_path: Path, width: int, height: int):
    text = label_path.read_text(encoding='utf-8', errors='ignore').strip()
    if not text:
        return None
    for line in text.splitlines():
        vals = line.strip().split()
        if len(vals) < 7 or (len(vals) - 1) % 2 != 0:
            continue
        pts = np.array([float(x) for x in vals[1:]], dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= width
        pts[:, 1] *= height
        return pts
    return None


def numeric_sort_key(path: Path):
    nums = re.findall(r'(\d+)', path.stem)
    if nums:
        return (0, int(nums[-1]), path.stem.lower())
    return (1, path.stem.lower(), path.stem.lower())


def score_to_weight(score):
    return float(np.clip(1.0 + 0.35 * float(score), 0.15, 6.0))


def gather_items(data_root: Path, class_name: str, component_name: str):
    image_root = data_root / 'images' / class_name
    label_root = data_root / 'labels' / class_name
    if not image_root.exists() or not label_root.exists():
        raise RuntimeError(f'Missing image/label folders for class={class_name}')

    items = []
    for image_path in sorted(image_root.rglob('*')):
        if image_path.suffix.lower() not in IMG_EXTS:
            continue
        rel = image_path.relative_to(image_root)
        if any(tag in rel.parts for tag in EXCLUDED_TAGS):
            continue
        if 'components' in rel.parts:
            idx = rel.parts.index('components')
            comp = rel.parts[idx + 1] if idx + 1 < len(rel.parts) else ''
            if component_name and component_name != comp:
                continue
        elif component_name not in ('', 'main'):
            continue

        label_path = (label_root / rel).with_suffix('.txt')
        if not label_path.exists():
            continue
        items.append({
            'image_path': image_path,
            'label_path': label_path,
            'rel_path': str(rel).replace('\\', '/'),
            'group_key': str(rel.parent).replace('\\', '/'),
            'sort_key': numeric_sort_key(rel),
        })
    if not items:
        raise RuntimeError(f'No rankable labeled images found for class={class_name}, component={component_name}')
    return items


def build_neighbor_map(items):
    groups = {}
    for item in items:
        groups.setdefault(item['group_key'], []).append(item)

    neighbors = {}
    for group_items in groups.values():
        ordered = sorted(group_items, key=lambda x: x['sort_key'])
        for idx, item in enumerate(ordered):
            rel = item['rel_path']
            neighbors[rel] = []
            for dist, decay in ((1, 0.65), (2, 0.35), (3, 0.15)):
                for off in (-dist, dist):
                    j = idx + off
                    if 0 <= j < len(ordered):
                        neighbors[rel].append((ordered[j]['rel_path'], decay))
    return neighbors


def load_preferences(pref_path: Path):
    if not pref_path.exists():
        return {'scores': {}, 'votes': {}, 'selected_batches': 0}
    try:
        loaded = json.loads(pref_path.read_text(encoding='utf-8'))
        if not isinstance(loaded, dict):
            return {'scores': {}, 'votes': {}, 'selected_batches': 0}
        loaded.setdefault('scores', {})
        loaded.setdefault('votes', {})
        loaded.setdefault('selected_batches', 0)
        return loaded
    except Exception:
        return {'scores': {}, 'votes': {}, 'selected_batches': 0}


def save_preferences(pref_path: Path, prefs, class_name: str, component_name: str):
    pref_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'class_name': class_name,
        'component_name': component_name,
        'selected_batches': int(prefs.get('selected_batches', 0)),
        'scores': prefs.get('scores', {}),
        'votes': prefs.get('votes', {}),
    }
    pref_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


class RankerApp(tk.Tk):
    def __init__(self, args, items, neighbors, prefs, pref_path):
        super().__init__()
        self.args = args
        self.items = items
        self.neighbors = neighbors
        self.prefs = prefs
        self.pref_path = pref_path
        self.thumb_cache = {}
        self.batch_items = []
        self.rank_order = []

        self.title(f'Realism Ranker - {args.class_name}')
        self.geometry('1380x980')

        header = ttk.Frame(self)
        header.pack(fill='x', padx=12, pady=10)
        self.status_var = tk.StringVar()
        ttk.Label(
            header,
            text='Click the images from most realistic to least realistic. The top choices and nearby frames will be favored during synthesis.',
            wraplength=1180,
        ).pack(anchor='w')
        ttk.Label(header, textvariable=self.status_var).pack(anchor='w', pady=(6, 0))

        self.grid_frame = ttk.Frame(self)
        self.grid_frame.pack(fill='both', expand=True, padx=12, pady=8)

        footer = ttk.Frame(self)
        footer.pack(fill='x', padx=12, pady=(0, 12))
        ttk.Button(footer, text='Undo last pick', command=self.undo_pick).pack(side='left')
        ttk.Button(footer, text='Skip this batch', command=self.skip_batch).pack(side='left', padx=(8, 0))
        ttk.Button(footer, text='Save progress', command=self.save_progress).pack(side='left', padx=(8, 0))
        ttk.Button(footer, text='Save and close', command=self.close_and_save).pack(side='right')

        self.protocol('WM_DELETE_WINDOW', self.close_and_save)
        self.load_next_batch()

    def _votes(self, rel_path):
        return int(self.prefs.get('votes', {}).get(rel_path, 0))

    def _score(self, rel_path):
        return float(self.prefs.get('scores', {}).get(rel_path, 0.0))

    def choose_batch_candidates(self):
        batch_size = max(2, self.args.batch_size)
        if len(self.items) <= batch_size:
            return list(self.items)
        ranked = sorted(
            self.items,
            key=lambda x: (self._votes(x['rel_path']), -score_to_weight(self._score(x['rel_path'])), random.random()),
        )
        pool_size = min(len(ranked), max(batch_size * 4, len(ranked) // 2))
        pool = ranked[:pool_size]
        weights = [1.0 / (1.0 + self._votes(x['rel_path'])) for x in pool]
        picked = []
        while pool and len(picked) < batch_size:
            idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
            picked.append(pool.pop(idx))
            weights.pop(idx)
        return picked

    def render_thumbnail(self, item, rank_text=''):
        rel = item['rel_path']
        cache_key = (rel, rank_text)
        if cache_key in self.thumb_cache:
            return self.thumb_cache[cache_key]

        image = cv2.imread(str(item['image_path']))
        if image is None:
            image = np.full((480, 640, 3), 48, dtype=np.uint8)
        h, w = image.shape[:2]
        poly = parse_first_polygon(item['label_path'], w, h)
        overlay = image.copy()
        if poly is not None:
            cv2.polylines(overlay, [poly.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2, cv2.LINE_AA)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
            tint = np.zeros_like(overlay)
            tint[:] = (30, 180, 255)
            alpha = mask > 0
            overlay[alpha] = cv2.addWeighted(overlay[alpha], 0.80, tint[alpha], 0.20, 0)
        if rank_text:
            cv2.putText(overlay, rank_text, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 3, cv2.LINE_AA)
        thumb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        thumb = Image.fromarray(thumb)
        thumb.thumbnail((600, 360))
        photo = ImageTk.PhotoImage(thumb)
        self.thumb_cache[cache_key] = photo
        return photo

    def render_batch(self):
        for child in self.grid_frame.winfo_children():
            child.destroy()

        for idx, item in enumerate(self.batch_items):
            row = idx // 2
            col = idx % 2
            card = ttk.Frame(self.grid_frame, padding=8)
            card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
            rel = item['rel_path']
            rank_idx = self.rank_order.index(rel) + 1 if rel in self.rank_order else None
            rank_text = f'Rank {rank_idx}' if rank_idx is not None else ''
            photo = self.render_thumbnail(item, rank_text=rank_text)

            lbl = tk.Label(card, image=photo, cursor='hand2', bd=2, relief='groove')
            lbl.image = photo
            lbl.pack(fill='both', expand=True)
            lbl.bind('<Button-1>', lambda _e, rel_path=rel: self.pick_image(rel_path))

            info = f'{rel}\ncurrent weight={score_to_weight(self._score(rel)):.2f} votes={self._votes(rel)}'
            ttk.Label(card, text=info, wraplength=620, justify='left').pack(anchor='w', pady=(6, 0))
            btn_text = 'Selected' if rank_idx is not None else 'Pick next'
            btn = ttk.Button(card, text=btn_text, command=lambda rel_path=rel: self.pick_image(rel_path))
            if rank_idx is not None:
                btn.state(['disabled'])
            btn.pack(anchor='w', pady=(6, 0))

        for i in range(2):
            self.grid_frame.columnconfigure(i, weight=1)
            self.grid_frame.rowconfigure(i, weight=1)

        self.status_var.set(
            f'Batch picks: {len(self.rank_order)}/{len(self.batch_items)} | '
            f'batches saved: {int(self.prefs.get("selected_batches", 0))} | '
            f'preferences file: {self.pref_path}'
        )

    def load_next_batch(self):
        self.batch_items = self.choose_batch_candidates()
        self.rank_order = []
        self.render_batch()

    def pick_image(self, rel_path):
        if rel_path in self.rank_order:
            return
        self.rank_order.append(rel_path)
        self.render_batch()
        if len(self.rank_order) == len(self.batch_items):
            self.apply_current_ranking()
            self.load_next_batch()

    def undo_pick(self):
        if not self.rank_order:
            return
        self.rank_order.pop()
        self.render_batch()

    def skip_batch(self):
        self.rank_order = []
        self.load_next_batch()

    def apply_current_ranking(self):
        n = len(self.rank_order)
        if n < 2:
            return
        deltas = np.linspace(3.0, -1.5, n, dtype=np.float32)
        scores = self.prefs.setdefault('scores', {})
        votes = self.prefs.setdefault('votes', {})
        for rel_path, delta in zip(self.rank_order, deltas):
            scores[rel_path] = float(scores.get(rel_path, 0.0) + float(delta))
            votes[rel_path] = int(votes.get(rel_path, 0) + 1)
            for neighbor_rel, decay in self.neighbors.get(rel_path, []):
                scores[neighbor_rel] = float(scores.get(neighbor_rel, 0.0) + float(delta) * decay)
        self.prefs['selected_batches'] = int(self.prefs.get('selected_batches', 0)) + 1
        save_preferences(self.pref_path, self.prefs, self.args.class_name, self.args.component_name)

    def save_progress(self):
        save_preferences(self.pref_path, self.prefs, self.args.class_name, self.args.component_name)
        self.status_var.set(
            f'Progress saved. batches={int(self.prefs.get("selected_batches", 0))} file={self.pref_path}'
        )

    def close_and_save(self):
        save_preferences(self.pref_path, self.prefs, self.args.class_name, self.args.component_name)
        print(f'Realism ranking saved: {self.pref_path}')
        self.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--class-name', required=True)
    ap.add_argument('--component-name', default='main')
    ap.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument('--batch-size', type=int, default=4)
    args = ap.parse_args()

    random.seed()
    data_root = Path(args.data_root)
    items = gather_items(data_root, args.class_name, args.component_name)
    pref_path = data_root / 'preferences' / f'{args.class_name}_realism.json'
    prefs = load_preferences(pref_path)
    neighbors = build_neighbor_map(items)
    app = RankerApp(args, items, neighbors, prefs, pref_path)
    app.mainloop()


if __name__ == '__main__':
    main()
