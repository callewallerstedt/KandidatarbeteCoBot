#!/usr/bin/env python3
import os
import sys
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText
import yaml

ROOT = Path(__file__).resolve().parent
PY = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
if not PY.exists():
    PY = Path(sys.executable)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('YOLO Segmentation Lab')
        self.geometry('980x720')

        paned = ttk.Panedwindow(self, orient='vertical')
        paned.pack(fill='both', expand=True)

        top_container = ttk.Frame(paned)
        log_container = ttk.Frame(paned)
        paned.add(top_container, weight=4)
        paned.add(log_container, weight=1)

        self.top_canvas = tk.Canvas(top_container, highlightthickness=0)
        self.top_scroll = ttk.Scrollbar(top_container, orient='vertical', command=self.top_canvas.yview)
        self.top_canvas.configure(yscrollcommand=self.top_scroll.set)
        self.top_scroll.pack(side='right', fill='y')
        self.top_canvas.pack(side='left', fill='both', expand=True)

        self.top_inner = ttk.Frame(self.top_canvas)
        self._top_window = self.top_canvas.create_window((0, 0), window=self.top_inner, anchor='nw')

        def _on_top_inner_config(_e=None):
            self.top_canvas.configure(scrollregion=self.top_canvas.bbox('all'))

        def _on_top_canvas_config(e):
            self.top_canvas.itemconfig(self._top_window, width=e.width)

        self.top_inner.bind('<Configure>', _on_top_inner_config)
        self.top_canvas.bind('<Configure>', _on_top_canvas_config)

        def _on_mousewheel(event):
            delta = event.delta
            if delta == 0:
                return
            self.top_canvas.yview_scroll(int(-delta / 120), 'units')

        self.bind_all('<MouseWheel>', _on_mousewheel)

        self.log = tk.Text(log_container, wrap='word', height=10)
        self.log.pack(fill='both', expand=True)

        notebook = ttk.Notebook(self.top_inner)
        notebook.pack(fill='both', expand=True)

        self.tab_instructions = ttk.Frame(notebook)
        self.tab_data = ttk.Frame(notebook)
        self.tab_synth = ttk.Frame(notebook)
        self.tab_synth_multi = ttk.Frame(notebook)
        self.tab_synth_all = ttk.Frame(notebook)
        self.tab_cutout = ttk.Frame(notebook)
        self.tab_manual = ttk.Frame(notebook)
        self.tab_obstruction = ttk.Frame(notebook)
        self.tab_train = ttk.Frame(notebook)
        self.tab_ddp = ttk.Frame(notebook)
        self.tab_infer = ttk.Frame(notebook)
        notebook.add(self.tab_instructions, text='0) Instructions')
        notebook.add(self.tab_data, text='1) Data Prep')
        notebook.add(self.tab_synth, text='2) Synthetic BG')
        notebook.add(self.tab_synth_multi, text='3) Multi-Instance Synth')
        notebook.add(self.tab_synth_all, text='4) COMBO RUN')
        notebook.add(self.tab_obstruction, text='5) Obstruction Data')
        notebook.add(self.tab_cutout, text='6) Add Masked Object')
        notebook.add(self.tab_manual, text='7) Manual Real Data')
        notebook.add(self.tab_train, text='8) Train')
        notebook.add(self.tab_ddp, text='9) DDP Multi-PC')
        notebook.add(self.tab_infer, text='10) Inference')

        self.class_id_choices = [str(i) for i in range(0, 25)]

        self.build_instructions_tab()
        self.build_data_tab()
        self.build_synth_tab()
        self.build_synth_multi_tab()
        self.build_synth_all_tab()
        self.build_obstruction_tab()
        self.build_cutout_tab()
        self.build_manual_tab()
        self.build_train_tab()
        self.build_ddp_tab()
        self.build_infer_tab()
        self.refresh_class_options()
        self.sync_yaml_from_folders()
        self.auto_assign_class_id(self.class_var, self.class_id_var)
        self.auto_assign_class_id(self.synth_class_var, self.synth_class_id_var)
        self.auto_assign_class_id(self.synth_multi_class_var, self.synth_multi_class_id_var)
        self.auto_assign_class_id(self.synth_all_class_var, self.synth_all_class_id_var)
        self.auto_assign_class_id(self.cutout_class_var, self.cutout_class_id_var)
        self.auto_assign_class_id(self.obs_class_var, self.obs_class_id_var)
        self.auto_assign_class_id(self.manual_class_var, self.manual_class_id_var)

    def log_line(self, text):
        self.log.insert('end', text + '\n')
        self.log.see('end')
        self.update_idletasks()

    def run_cmd(self, cmd, cwd=ROOT):
        self.log_line('> ' + ' '.join(map(str, cmd)))

        def _target():
            p = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
            )
            for line in p.stdout:
                self.log_line(line.rstrip())
            rc = p.wait()
            self.log_line(f'[exit {rc}]')

        threading.Thread(target=_target, daemon=True).start()

    def build_instructions_tab(self):
        frm = self.tab_instructions
        txt = ScrolledText(frm, wrap='word', height=20)
        txt.pack(fill='both', expand=True, padx=8, pady=8)
        txt.insert('1.0',
            'YOLO Segmentation Lab – Recommended workflow\n\n'
            'A) Add a NEW class (incremental multiclass)\n'
            '1. Go to Data Prep. Enter/select Class name + Class id.\n'
            '2. Click Auto-label from video to create initial labels.\n'
            '3. Go to Manual Real Data and run Prepare frames + initial masks, then Open manual mask reviewer and fix bad masks.\n'
            '4. (Optional) Go to Synthetic BG / Obstruction Data to generate more varied training data for this class.\n'
            '5. Back in Data Prep: update class list (space-separated, class-id order) and click Update dataset.yaml.\n'
            '6. Click Build train/val/test split (mode=all recommended). Use Split class to limit to one class if needed.\n'
            '7. Train tab: use latest runs/.../best.pt to continue training and preserve old classes.\n\n'
            'B) Add MORE data to an existing class\n'
            '1. Select existing class name/id.\n'
            '2. Add new real video labels and/or synthetic data.\n'
            '3. Build split again (all).\n'
            '4. Continue training from latest best.pt.\n\n'
            'C) Retrain safely without forgetting old classes\n'
            '- Always train with mixed dataset containing old + new classes.\n'
            '- Do NOT train only on newest class data.\n\n'
            'D) Mask quality tips\n'
            '- If auto labels look jagged, increase Image size in Train tab (e.g. 960/1280).\n'
            '- Use Manual Real Data reviewer to clean edges on hard samples.\n'
            '- Auto-label now supports higher-quality contour mode (less polygon simplification).\n\n'
            'E) DDP Multi-PC (2+ computers)\n'
            '1. Pull same repo commit on all PCs and ensure same dataset files.\n'
            '2. In DDP tab, set hosts list and click Check connected.\n'
            '3. Set master addr to rank-0 computer IP, set same port/nnodes on all PCs.\n'
            '4. On each PC set its node rank (0,1,2...).\n'
            '5. Click Show launch commands and run matching rank command on each PC (rank 0 first).\n'
            '6. Use workers 0-1 on Windows for stability.\n'
        )
        txt.configure(state='disabled')

    def load_dataset_class_map(self):
        dataset_yaml = ROOT / 'dataset.yaml'
        if not dataset_yaml.exists():
            return {}, {}
        try:
            data = yaml.safe_load(dataset_yaml.read_text(encoding='utf-8')) or {}
            names = data.get('names', {})
            id_to_name = {}
            if isinstance(names, list):
                id_to_name = {int(i): str(n) for i, n in enumerate(names)}
            elif isinstance(names, dict):
                for k, v in names.items():
                    id_to_name[int(k)] = str(v)
            name_to_id = {v: k for k, v in id_to_name.items()}
            return name_to_id, id_to_name
        except Exception as e:
            self.log_line(f'Warning: could not parse dataset.yaml for class map: {e}')
            return {}, {}

    def suggest_class_id(self, class_name: str):
        name = class_name.strip()
        if not name:
            return None
        name_to_id, id_to_name = self.load_dataset_class_map()
        if name in name_to_id:
            return name_to_id[name]

        used = set(id_to_name.keys())
        i = 0
        while i in used:
            i += 1
        return i

    def auto_assign_class_id(self, class_var: tk.StringVar, id_var: tk.StringVar):
        cid = self.suggest_class_id(class_var.get())
        if cid is not None:
            id_var.set(str(cid))

    def ensure_class_registered(self, class_name: str, class_id_text: str):
        cname = (class_name or '').strip()
        if not cname:
            self.log_line('Class name is empty; cannot auto-register class.')
            return False
        try:
            cid = int(str(class_id_text).strip())
        except Exception:
            self.log_line(f'Invalid class id: {class_id_text}')
            return False

        dataset_yaml = ROOT / 'dataset.yaml'
        base = {
            'path': 'data/yolo_dataset',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {},
        }
        try:
            if dataset_yaml.exists():
                data = yaml.safe_load(dataset_yaml.read_text(encoding='utf-8')) or {}
            else:
                data = base

            names = data.get('names', {})
            if isinstance(names, list):
                id_to_name = {int(i): str(n) for i, n in enumerate(names)}
            elif isinstance(names, dict):
                id_to_name = {int(k): str(v) for k, v in names.items()}
            else:
                id_to_name = {}

            # already correct
            if id_to_name.get(cid) == cname:
                return True

            # same class name exists on another id -> keep existing stable mapping
            for k, v in id_to_name.items():
                if v == cname and k != cid:
                    self.log_line(f'Class "{cname}" already mapped to id={k}; keeping existing mapping.')
                    return True

            # id occupied by another class -> choose next free id to keep things safe
            if cid in id_to_name and id_to_name[cid] != cname:
                used = set(id_to_name.keys())
                nid = 0
                while nid in used:
                    nid += 1
                self.log_line(f'Class id {cid} already used by "{id_to_name[cid]}"; auto-assigning "{cname}" -> id {nid}.')
                cid = nid

            id_to_name[cid] = cname
            data.setdefault('path', 'data/yolo_dataset')
            data.setdefault('train', 'images/train')
            data.setdefault('val', 'images/val')
            data.setdefault('test', 'images/test')
            data['names'] = {int(k): id_to_name[k] for k in sorted(id_to_name.keys())}
            dataset_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
            self.log_line(f'Auto-registered class in dataset.yaml: id={cid}, name={cname}')
            self.refresh_class_options()
            return True
        except Exception as e:
            self.log_line(f'Failed to auto-register class in dataset.yaml: {e}')
            return False

    def refresh_class_options(self):
        data_images = ROOT / 'data' / 'images'
        names = []
        if data_images.exists():
            names = sorted([p.name for p in data_images.iterdir() if p.is_dir()])

        name_to_id, id_to_name = self.load_dataset_class_map()
        for n in sorted(name_to_id.keys()):
            if n not in names:
                names.append(n)

        if not names:
            names = ['object_name']
        self.class_choices = names

        # Split selector display: ALL + "id:name"
        split_vals = ['ALL classes']
        if id_to_name:
            for cid in sorted(id_to_name.keys()):
                split_vals.append(f'{cid}:{id_to_name[cid]}')
        else:
            for i, n in enumerate(self.class_choices):
                split_vals.append(f'{i}:{n}')
        self.split_class_choices = split_vals

        for cb_name in ['data_class_cb', 'synth_class_cb', 'synth_multi_class_cb', 'synth_all_class_cb', 'cutout_class_cb', 'obs_class_cb', 'manual_class_cb']:
            cb = getattr(self, cb_name, None)
            if cb is not None:
                cb['values'] = self.class_choices

        for cb_name in ['data_class_id_cb', 'synth_class_id_cb', 'synth_multi_class_id_cb', 'synth_all_class_id_cb', 'cutout_class_id_cb', 'obs_class_id_cb', 'manual_class_id_cb']:
            cb = getattr(self, cb_name, None)
            if cb is not None:
                cb['values'] = self.class_id_choices

        split_cb = getattr(self, 'split_class_cb', None)
        if split_cb is not None:
            split_cb['values'] = self.split_class_choices
            if not self.split_class_var.get() or self.split_class_var.get() not in self.split_class_choices:
                self.split_class_var.set('ALL classes')

    def build_data_tab(self):
        frm = self.tab_data

        self.video_var = tk.StringVar()
        self.class_var = tk.StringVar(value='object_name')
        self.class_id_var = tk.StringVar(value='0')
        self.every_var = tk.StringVar(value='8')
        self.num_samples_var = tk.StringVar(value='120')
        self.max_var = tk.StringVar(value='300')
        self.aug_var = tk.StringVar(value='1')
        self.mask_quality_var = tk.StringVar(value='high')
        self.classes_var = tk.StringVar(value='object_name')
        self.split_mode_var = tk.StringVar(value='all')
        self.split_class_var = tk.StringVar(value='ALL classes')
        self.split_run_var = tk.StringVar(value='')
        self.class_var.trace_add('write', lambda *_: self.auto_assign_class_id(self.class_var, self.class_id_var))

        ttk.Label(frm, text='Video file').grid(row=0, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.video_var, width=70).grid(row=0, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_video).grid(row=0, column=2)

        ttk.Label(frm, text='Class name').grid(row=1, column=0, sticky='w')
        self.data_class_cb = ttk.Combobox(frm, textvariable=self.class_var)
        self.data_class_cb.grid(row=1, column=1, sticky='we')

        ttk.Label(frm, text='Class id').grid(row=2, column=0, sticky='w')
        self.data_class_id_cb = ttk.Combobox(frm, textvariable=self.class_id_var, values=self.class_id_choices, width=8)
        self.data_class_id_cb.grid(row=2, column=1, sticky='w')

        ttk.Label(frm, text='Target samples (evenly across video)').grid(row=3, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.num_samples_var).grid(row=3, column=1, sticky='we')

        ttk.Label(frm, text='Fallback: every N frames (if target=0)').grid(row=4, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.every_var).grid(row=4, column=1, sticky='we')

        ttk.Label(frm, text='Max frames').grid(row=5, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.max_var).grid(row=5, column=1, sticky='we')

        ttk.Label(frm, text='Aug per frame').grid(row=6, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.aug_var).grid(row=6, column=1, sticky='we')

        ttk.Label(frm, text='Mask quality').grid(row=7, column=0, sticky='w')
        ttk.Combobox(frm, textvariable=self.mask_quality_var, values=['fast', 'high'], state='readonly', width=10).grid(row=7, column=1, sticky='w')

        ttk.Button(frm, text='Auto-label from video', command=self.autolabel).grid(row=8, column=0, pady=8)

        ttk.Label(frm, text='Split mode').grid(row=9, column=1, sticky='e')
        ttk.Combobox(frm, textvariable=self.split_mode_var, values=['all', 'real', 'synth', 'obs'], state='readonly', width=10).grid(row=9, column=2, sticky='w')

        ttk.Label(frm, text='Split class').grid(row=10, column=0, sticky='w')
        self.split_class_cb = ttk.Combobox(frm, textvariable=self.split_class_var, state='readonly', width=28)
        self.split_class_cb.grid(row=10, column=1, sticky='w')

        ttk.Label(frm, text='Run filter (optional). Example: combo01 to include combo01_bg + combo01_multi + combo01_obs').grid(row=11, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.split_run_var, width=40).grid(row=11, column=1, sticky='w')

        ttk.Label(frm, text='all = all data | real = no synth/obs | synth = only synth_runs | obs = only obs_runs').grid(row=12, column=0, columnspan=3, sticky='w')
        ttk.Button(frm, text='Build train/val/test split', command=self.build_split).grid(row=13, column=0, pady=8, sticky='w')

        ttk.Separator(frm, orient='horizontal').grid(row=14, column=0, columnspan=3, sticky='we', pady=8)

        ttk.Label(frm, text='All classes (space-separated, in class-id order)').grid(row=15, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.classes_var, width=70).grid(row=15, column=1, sticky='we')
        ttk.Button(frm, text='Update dataset.yaml', command=self.update_yaml).grid(row=15, column=2)
        ttk.Button(frm, text='Auto-sync dataset.yaml from data folders', command=self.sync_yaml_from_folders).grid(row=16, column=0, pady=6, sticky='w')

        frm.columnconfigure(1, weight=1)

    def build_synth_tab(self):
        frm = self.tab_synth
        self.synth_class_var = tk.StringVar(value='object_name')
        self.synth_class_id_var = tk.StringVar(value='0')
        self.bg_dir_var = tk.StringVar()
        self.synth_n_var = tk.StringVar(value='300')
        self.synth_min_scale_var = tk.StringVar(value='0.55')
        self.synth_max_scale_var = tk.StringVar(value='1.25')
        self.synth_rot_var = tk.StringVar(value='25')
        self.synth_bri_min_var = tk.StringVar(value='-20')
        self.synth_bri_max_var = tk.StringVar(value='20')
        self.synth_obj_bri_min_var = tk.StringVar(value='-10')
        self.synth_obj_bri_max_var = tk.StringVar(value='10')
        self.synth_preview_count_var = tk.StringVar(value='12')
        self.synth_run_var = tk.StringVar(value='')
        self.synth_class_var.trace_add('write', lambda *_: self.auto_assign_class_id(self.synth_class_var, self.synth_class_id_var))

        ttk.Label(frm, text='Class name').grid(row=0, column=0, sticky='w')
        self.synth_class_cb = ttk.Combobox(frm, textvariable=self.synth_class_var)
        self.synth_class_cb.grid(row=0, column=1, sticky='we')

        ttk.Label(frm, text='Class id').grid(row=1, column=0, sticky='w')
        self.synth_class_id_cb = ttk.Combobox(frm, textvariable=self.synth_class_id_var, values=self.class_id_choices, width=8)
        self.synth_class_id_cb.grid(row=1, column=1, sticky='w')

        ttk.Label(frm, text='Background images folder').grid(row=2, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.bg_dir_var, width=70).grid(row=2, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_bg_dir).grid(row=2, column=2)

        ttk.Label(frm, text='Num synthetic images').grid(row=3, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_n_var).grid(row=3, column=1, sticky='we')

        ttk.Label(frm, text='Min scale').grid(row=4, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_min_scale_var).grid(row=4, column=1, sticky='we')

        ttk.Label(frm, text='Max scale').grid(row=5, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_max_scale_var).grid(row=5, column=1, sticky='we')

        ttk.Label(frm, text='Max rotation deg').grid(row=6, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_rot_var).grid(row=6, column=1, sticky='we')

        ttk.Label(frm, text='Background brightness min (beta)').grid(row=7, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_bri_min_var).grid(row=7, column=1, sticky='we')

        ttk.Label(frm, text='Background brightness max (beta)').grid(row=8, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_bri_max_var).grid(row=8, column=1, sticky='we')

        ttk.Label(frm, text='Object brightness min (beta)').grid(row=9, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_obj_bri_min_var).grid(row=9, column=1, sticky='we')

        ttk.Label(frm, text='Object brightness max (beta)').grid(row=10, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_obj_bri_max_var).grid(row=10, column=1, sticky='we')

        ttk.Label(frm, text='Synth run name (optional)').grid(row=11, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_run_var).grid(row=11, column=1, sticky='we')

        ttk.Label(frm, text='Preview count').grid(row=12, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_preview_count_var, width=10).grid(row=12, column=1, sticky='w')

        ttk.Button(frm, text='Preview synth settings (left/right browse)', command=self.preview_synth).grid(row=13, column=0, pady=8)
        ttk.Button(frm, text='Generate synthetic cut-paste set', command=self.generate_synth).grid(row=13, column=1, pady=8, sticky='w')
        frm.columnconfigure(1, weight=1)

    def build_synth_multi_tab(self):
        frm = self.tab_synth_multi
        self.synth_multi_class_var = tk.StringVar(value='object_name')
        self.synth_multi_class_id_var = tk.StringVar(value='0')
        self.synth_multi_bg_dir_var = tk.StringVar()
        self.synth_multi_n_var = tk.StringVar(value='300')
        self.synth_multi_min_obj_var = tk.StringVar(value='2')
        self.synth_multi_max_obj_var = tk.StringVar(value='5')
        self.synth_multi_overlap_var = tk.StringVar(value='0.8')
        self.synth_multi_max_overlap_ratio_var = tk.StringVar(value='0.5')
        self.synth_multi_cluster_dist_var = tk.StringVar(value='1.0')
        self.synth_multi_min_scale_var = tk.StringVar(value='0.45')
        self.synth_multi_max_scale_var = tk.StringVar(value='1.10')
        self.synth_multi_overlap_spread_var = tk.StringVar(value='0.25')
        self.synth_multi_bg_bri_min_var = tk.StringVar(value='-20')
        self.synth_multi_bg_bri_max_var = tk.StringVar(value='20')
        self.synth_multi_obj_bri_min_var = tk.StringVar(value='-10')
        self.synth_multi_obj_bri_max_var = tk.StringVar(value='10')
        self.synth_multi_preview_count_var = tk.StringVar(value='12')
        self.synth_multi_run_var = tk.StringVar(value='')
        self.synth_multi_class_var.trace_add('write', lambda *_: self.auto_assign_class_id(self.synth_multi_class_var, self.synth_multi_class_id_var))

        ttk.Label(frm, text='Class name').grid(row=0, column=0, sticky='w')
        self.synth_multi_class_cb = ttk.Combobox(frm, textvariable=self.synth_multi_class_var)
        self.synth_multi_class_cb.grid(row=0, column=1, sticky='we')

        ttk.Label(frm, text='Class id').grid(row=1, column=0, sticky='w')
        self.synth_multi_class_id_cb = ttk.Combobox(frm, textvariable=self.synth_multi_class_id_var, values=self.class_id_choices, width=8)
        self.synth_multi_class_id_cb.grid(row=1, column=1, sticky='w')

        ttk.Label(frm, text='Background images folder').grid(row=2, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_bg_dir_var, width=70).grid(row=2, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_synth_multi_bg_dir).grid(row=2, column=2)

        ttk.Label(frm, text='Num synthetic images').grid(row=3, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_n_var).grid(row=3, column=1, sticky='we')

        ttk.Label(frm, text='Min objects per image').grid(row=4, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_min_obj_var).grid(row=4, column=1, sticky='we')

        ttk.Label(frm, text='Max objects per image').grid(row=5, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_max_obj_var).grid(row=5, column=1, sticky='we')

        ttk.Label(frm, text='Overlap probability').grid(row=6, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_overlap_var).grid(row=6, column=1, sticky='we')

        ttk.Label(frm, text='Max overlap ratio (<=0.5)').grid(row=7, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_max_overlap_ratio_var).grid(row=7, column=1, sticky='we')

        ttk.Label(frm, text='Cluster distance factor').grid(row=8, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_cluster_dist_var).grid(row=8, column=1, sticky='we')

        ttk.Label(frm, text='Min scale').grid(row=9, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_min_scale_var).grid(row=9, column=1, sticky='we')

        ttk.Label(frm, text='Max scale').grid(row=10, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_max_scale_var).grid(row=10, column=1, sticky='we')

        ttk.Label(frm, text='Overlap spread (0 tight overlap, 1 wider)').grid(row=11, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_overlap_spread_var).grid(row=11, column=1, sticky='we')

        ttk.Label(frm, text='BG brightness min/max').grid(row=12, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_bg_bri_min_var, width=8).grid(row=12, column=1, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_bg_bri_max_var, width=8).grid(row=12, column=1, padx=(70,0), sticky='w')

        ttk.Label(frm, text='Object brightness min/max').grid(row=13, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_obj_bri_min_var, width=8).grid(row=13, column=1, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_obj_bri_max_var, width=8).grid(row=13, column=1, padx=(70,0), sticky='w')

        ttk.Label(frm, text='Run name (optional)').grid(row=14, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_run_var).grid(row=14, column=1, sticky='we')

        ttk.Label(frm, text='Preview count').grid(row=15, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_multi_preview_count_var, width=10).grid(row=15, column=1, sticky='w')

        ttk.Button(frm, text='Preview multi-instance samples (left/right browse)', command=self.preview_synth_multi).grid(row=16, column=0, pady=8)
        ttk.Button(frm, text='Generate multi-instance synthetic set', command=self.generate_synth_multi).grid(row=16, column=1, pady=8, sticky='w')
        ttk.Label(frm, text='Cluster mode: close/touching by default, max overlap enforced.').grid(row=17, column=0, columnspan=3, sticky='w')
        frm.columnconfigure(1, weight=1)

    def build_synth_all_tab(self):
        frm = self.tab_synth_all
        self.synth_all_class_var = tk.StringVar(value='object_name')
        self.synth_all_class_id_var = tk.StringVar(value='0')
        self.synth_all_bg_dir_var = tk.StringVar()
        self.synth_all_obs_dir_var = tk.StringVar()
        self.synth_all_run_var = tk.StringVar(value='combo01')

        self.synth_all_use_bg_var = tk.BooleanVar(value=True)
        self.synth_all_use_multi_var = tk.BooleanVar(value=True)
        self.synth_all_use_obs_var = tk.BooleanVar(value=True)

        self.synth_all_n_bg_var = tk.StringVar(value='300')
        self.synth_all_n_multi_var = tk.StringVar(value='200')
        self.synth_all_n_obs_var = tk.StringVar(value='200')
        self.synth_all_preview_count_var = tk.StringVar(value='12')

        self.synth_all_min_scale_var = tk.StringVar(value='0.55')
        self.synth_all_max_scale_var = tk.StringVar(value='1.25')
        self.synth_all_bg_bri_min_var = tk.StringVar(value='-20')
        self.synth_all_bg_bri_max_var = tk.StringVar(value='20')
        self.synth_all_obj_bri_min_var = tk.StringVar(value='-10')
        self.synth_all_obj_bri_max_var = tk.StringVar(value='10')

        self.synth_all_multi_min_obj_var = tk.StringVar(value='2')
        self.synth_all_multi_max_obj_var = tk.StringVar(value='5')
        self.synth_all_multi_overlap_var = tk.StringVar(value='0.8')
        self.synth_all_multi_max_overlap_ratio_var = tk.StringVar(value='0.5')
        self.synth_all_multi_cluster_dist_var = tk.StringVar(value='1.0')
        self.synth_all_multi_min_scale_var = tk.StringVar(value='0.45')
        self.synth_all_multi_max_scale_var = tk.StringVar(value='1.10')
        self.synth_all_multi_spread_var = tk.StringVar(value='0.25')

        self.synth_all_obs_angle_min_var = tk.StringVar(value='0')
        self.synth_all_obs_angle_max_var = tk.StringVar(value='360')
        self.synth_all_obs_rot_dev_var = tk.StringVar(value='20')
        self.synth_all_obs_overlap_min_var = tk.StringVar(value='0.6')
        self.synth_all_obs_overlap_max_var = tk.StringVar(value='1.1')
        self.synth_all_obs_scale_min_var = tk.StringVar(value='0.7')
        self.synth_all_obs_scale_max_var = tk.StringVar(value='1.0')
        self.synth_all_obs_white_prob_var = tk.StringVar(value='0.10')

        self.synth_all_class_var.trace_add('write', lambda *_: self.auto_assign_class_id(self.synth_all_class_var, self.synth_all_class_id_var))

        ttk.Label(frm, text='Class name').grid(row=0, column=0, sticky='w')
        self.synth_all_class_cb = ttk.Combobox(frm, textvariable=self.synth_all_class_var)
        self.synth_all_class_cb.grid(row=0, column=1, sticky='we')

        ttk.Label(frm, text='Class id').grid(row=1, column=0, sticky='w')
        self.synth_all_class_id_cb = ttk.Combobox(frm, textvariable=self.synth_all_class_id_var, values=self.class_id_choices, width=8)
        self.synth_all_class_id_cb.grid(row=1, column=1, sticky='w')

        ttk.Label(frm, text='Background folder').grid(row=2, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_bg_dir_var, width=70).grid(row=2, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_synth_all_bg_dir).grid(row=2, column=2)

        ttk.Label(frm, text='Obstruction folder').grid(row=3, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_obs_dir_var, width=70).grid(row=3, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_synth_all_obs_dir).grid(row=3, column=2)

        ttk.Label(frm, text='Run base name (for run filter)').grid(row=4, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_run_var).grid(row=4, column=1, sticky='we')
        ttk.Button(frm, text='Copy base -> Run filter', command=self.copy_combo_base_to_run_filter).grid(row=4, column=2)

        ttk.Separator(frm, orient='horizontal').grid(row=5, column=0, columnspan=3, sticky='we', pady=6)
        ttk.Label(frm, text='SYNTHETIC BG').grid(row=6, column=0, sticky='w')
        ttk.Checkbutton(frm, text='Enable', variable=self.synth_all_use_bg_var).grid(row=6, column=1, sticky='w')
        ttk.Label(frm, text='Count').grid(row=7, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_n_bg_var, width=10).grid(row=7, column=1, sticky='w')
        ttk.Label(frm, text='Scale min/max').grid(row=8, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_min_scale_var, width=6).grid(row=8, column=1, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_max_scale_var, width=6).grid(row=8, column=1, padx=(58,0), sticky='w')
        ttk.Label(frm, text='Rotation fixed: 360').grid(row=9, column=0, sticky='w')
        ttk.Label(frm, text='BG brightness min/max').grid(row=10, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_bg_bri_min_var, width=6).grid(row=10, column=1, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_bg_bri_max_var, width=6).grid(row=10, column=1, padx=(58,0), sticky='w')
        ttk.Label(frm, text='Object brightness min/max').grid(row=11, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_obj_bri_min_var, width=6).grid(row=11, column=1, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_obj_bri_max_var, width=6).grid(row=11, column=1, padx=(58,0), sticky='w')
        ttk.Button(frm, text='Preview SYNTHETIC BG', command=self.preview_combo_bg).grid(row=11, column=2, sticky='w')

        ttk.Separator(frm, orient='horizontal').grid(row=12, column=0, columnspan=3, sticky='we', pady=6)
        ttk.Label(frm, text='SYNTHETIC MULTI-INSTANCE').grid(row=13, column=0, sticky='w')
        ttk.Checkbutton(frm, text='Enable', variable=self.synth_all_use_multi_var).grid(row=13, column=1, sticky='w')
        ttk.Label(frm, text='Count').grid(row=14, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_n_multi_var, width=10).grid(row=14, column=1, sticky='w')
        ttk.Label(frm, text='Min/Max objects').grid(row=15, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_multi_min_obj_var, width=8).grid(row=15, column=1, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_multi_max_obj_var, width=8).grid(row=15, column=2, sticky='w')
        ttk.Label(frm, text='Overlap prob').grid(row=16, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_multi_overlap_var, width=8).grid(row=16, column=1, sticky='w')
        ttk.Label(frm, text='Max overlap ratio').grid(row=16, column=2, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_multi_max_overlap_ratio_var, width=8).grid(row=16, column=2, padx=(120,0), sticky='w')
        ttk.Label(frm, text='Scale min/max').grid(row=17, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_multi_min_scale_var, width=6).grid(row=17, column=1, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_multi_max_scale_var, width=6).grid(row=17, column=1, padx=(58,0), sticky='w')
        ttk.Label(frm, text='Overlap spread (0 tight, 1 spread)').grid(row=18, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_multi_spread_var, width=8).grid(row=18, column=1, sticky='w')
        ttk.Label(frm, text='Cluster dist factor').grid(row=18, column=2, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_multi_cluster_dist_var, width=8).grid(row=18, column=2, padx=(120,0), sticky='w')
        ttk.Label(frm, text='Rotation fixed: 360').grid(row=19, column=0, sticky='w')

        ttk.Button(frm, text='Preview SYNTHETIC MULTI-INSTANCE', command=self.preview_combo_multi).grid(row=19, column=0, sticky='w')

        ttk.Separator(frm, orient='horizontal').grid(row=20, column=0, columnspan=3, sticky='we', pady=6)
        ttk.Label(frm, text='OBSTRUCTION').grid(row=21, column=0, sticky='w')
        ttk.Checkbutton(frm, text='Enable', variable=self.synth_all_use_obs_var).grid(row=21, column=1, sticky='w')
        ttk.Label(frm, text='Count').grid(row=22, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_n_obs_var, width=10).grid(row=22, column=1, sticky='w')
        ttk.Label(frm, text='Entry angle min/max').grid(row=23, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_obs_angle_min_var, width=6).grid(row=23, column=1, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_obs_angle_max_var, width=6).grid(row=23, column=1, padx=(58,0), sticky='w')
        ttk.Label(frm, text='Rotation deviation').grid(row=24, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_obs_rot_dev_var, width=8).grid(row=24, column=1, sticky='w')
        ttk.Label(frm, text='Overlap min/max').grid(row=25, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_obs_overlap_min_var, width=6).grid(row=25, column=1, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_obs_overlap_max_var, width=6).grid(row=25, column=1, padx=(58,0), sticky='w')
        ttk.Label(frm, text='Obstruction scale min/max').grid(row=26, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_obs_scale_min_var, width=6).grid(row=26, column=1, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_obs_scale_max_var, width=6).grid(row=26, column=1, padx=(58,0), sticky='w')
        ttk.Label(frm, text='White-bg prob').grid(row=27, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_obs_white_prob_var, width=8).grid(row=27, column=1, sticky='w')
        ttk.Button(frm, text='Preview OBSTRUCTION', command=self.preview_combo_obs).grid(row=27, column=2, sticky='w')

        ttk.Label(frm, text='Preview count (all previews)').grid(row=28, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.synth_all_preview_count_var, width=8).grid(row=28, column=1, sticky='w')

        ttk.Button(frm, text='Run COMBO RUN', command=self.run_all_synth).grid(row=29, column=0, pady=10)
        ttk.Label(frm, text='After generation: Data Prep -> Run filter = base name (e.g. combo01), then Build split.').grid(row=30, column=0, columnspan=3, sticky='w')
        frm.columnconfigure(1, weight=1)

    def build_obstruction_tab(self):
        frm = self.tab_obstruction
        self.obs_class_var = tk.StringVar(value='object_name')
        self.obs_class_id_var = tk.StringVar(value='0')
        self.obs_dir_var = tk.StringVar()
        self.obs_bg_dir_var = tk.StringVar()
        self.obs_num_var = tk.StringVar(value='300')
        self.obs_angle_min_var = tk.StringVar(value='0')
        self.obs_angle_max_var = tk.StringVar(value='360')
        self.obs_rot_dev_var = tk.StringVar(value='20')
        self.obs_overlap_var = tk.StringVar(value='0.8')
        self.obs_scale_var = tk.StringVar(value='0.8')
        self.obs_white_prob_var = tk.StringVar(value='0.10')
        self.obs_preview_count_var = tk.StringVar(value='12')
        self.obs_run_var = tk.StringVar(value='')
        self.obs_class_var.trace_add('write', lambda *_: self.auto_assign_class_id(self.obs_class_var, self.obs_class_id_var))

        ttk.Label(frm, text='Class name').grid(row=0, column=0, sticky='w')
        self.obs_class_cb = ttk.Combobox(frm, textvariable=self.obs_class_var)
        self.obs_class_cb.grid(row=0, column=1, sticky='we')

        ttk.Label(frm, text='Class id').grid(row=1, column=0, sticky='w')
        self.obs_class_id_cb = ttk.Combobox(frm, textvariable=self.obs_class_id_var, values=self.class_id_choices, width=8)
        self.obs_class_id_cb.grid(row=1, column=1, sticky='w')

        ttk.Label(frm, text='Obstruction folder (contains hands/arms subfolders)').grid(row=2, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.obs_dir_var, width=70).grid(row=2, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_obs_dir).grid(row=2, column=2)

        ttk.Label(frm, text='Background folder (for synthetic base images)').grid(row=3, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.obs_bg_dir_var, width=70).grid(row=3, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_obs_bg_dir).grid(row=3, column=2)

        ttk.Label(frm, text='Num obstruction images').grid(row=4, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.obs_num_var).grid(row=4, column=1, sticky='we')

        ttk.Label(frm, text='Entry angle min (deg)').grid(row=5, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.obs_angle_min_var).grid(row=5, column=1, sticky='we')

        ttk.Label(frm, text='Entry angle max (deg)').grid(row=6, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.obs_angle_max_var).grid(row=6, column=1, sticky='we')

        ttk.Label(frm, text='Rotation deviation (deg)').grid(row=7, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.obs_rot_dev_var).grid(row=7, column=1, sticky='we')

        ttk.Label(frm, text='Overlap level (0 edge, 1 center, 2 past center)').grid(row=8, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.obs_overlap_var).grid(row=8, column=1, sticky='we')

        ttk.Label(frm, text='Obstruction scale vs object height').grid(row=9, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.obs_scale_var).grid(row=9, column=1, sticky='we')

        ttk.Label(frm, text='Keep original white-table background probability (e.g. 0.10)').grid(row=10, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.obs_white_prob_var).grid(row=10, column=1, sticky='we')

        ttk.Label(frm, text='Obstruction run name (optional)').grid(row=11, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.obs_run_var).grid(row=11, column=1, sticky='we')

        ttk.Label(frm, text='Preview count').grid(row=12, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.obs_preview_count_var, width=10).grid(row=12, column=1, sticky='w')

        ttk.Button(frm, text='Preview obstruction samples (left/right browse)', command=self.preview_obstruction).grid(row=13, column=0, pady=8)
        ttk.Button(frm, text='Generate obstruction synthetic set', command=self.generate_obstruction).grid(row=13, column=1, pady=8, sticky='w')
        ttk.Label(frm, text='Preview shows debug: yellow=center, magenta=hand vector (bottom→top), cyan=top→center target.').grid(row=14, column=0, columnspan=3, sticky='w')
        frm.columnconfigure(1, weight=1)

    def build_cutout_tab(self):
        frm = self.tab_cutout
        self.cutout_image_var = tk.StringVar()
        self.cutout_class_var = tk.StringVar(value='object_name')
        self.cutout_class_id_var = tk.StringVar(value='0')
        self.cutout_prefix_var = tk.StringVar(value='cutout')
        self.cutout_class_var.trace_add('write', lambda *_: self.auto_assign_class_id(self.cutout_class_var, self.cutout_class_id_var))

        ttk.Label(frm, text='Source image').grid(row=0, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.cutout_image_var, width=70).grid(row=0, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_cutout_image).grid(row=0, column=2)

        ttk.Label(frm, text='Class name').grid(row=1, column=0, sticky='w')
        self.cutout_class_cb = ttk.Combobox(frm, textvariable=self.cutout_class_var)
        self.cutout_class_cb.grid(row=1, column=1, sticky='we')

        ttk.Label(frm, text='Class id').grid(row=2, column=0, sticky='w')
        self.cutout_class_id_cb = ttk.Combobox(frm, textvariable=self.cutout_class_id_var, values=self.class_id_choices, width=8)
        self.cutout_class_id_cb.grid(row=2, column=1, sticky='w')

        ttk.Label(frm, text='Filename prefix').grid(row=3, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.cutout_prefix_var).grid(row=3, column=1, sticky='we')

        ttk.Button(frm, text='Draw box and create masked object sample', command=self.create_cutout_sample).grid(row=4, column=0, pady=8)
        ttk.Label(frm, text='You will draw bbox, then GrabCut creates mask and saves image+label for synth source use.').grid(row=5, column=0, columnspan=3, sticky='w')
        frm.columnconfigure(1, weight=1)

    def build_manual_tab(self):
        frm = self.tab_manual
        self.manual_video_var = tk.StringVar()
        self.manual_class_var = tk.StringVar(value='object_name')
        self.manual_class_id_var = tk.StringVar(value='0')
        self.manual_samples_var = tk.StringVar(value='80')
        self.manual_prefix_var = tk.StringVar(value='manual')
        self.manual_init_source_var = tk.StringVar(value='yolo')
        self.manual_init_weights_var = tk.StringVar(value=str(ROOT / 'runs' / 'segment' / 'train' / 'weights' / 'best.pt'))
        self.manual_init_conf_var = tk.StringVar(value='0.20')
        self.manual_init_imgsz_var = tk.StringVar(value='960')
        self.manual_init_device_var = tk.StringVar(value='0')
        self.manual_class_var.trace_add('write', lambda *_: self.auto_assign_class_id(self.manual_class_var, self.manual_class_id_var))

        ttk.Label(frm, text='Video file').grid(row=0, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.manual_video_var, width=70).grid(row=0, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_manual_video).grid(row=0, column=2)

        ttk.Label(frm, text='Class name').grid(row=1, column=0, sticky='w')
        self.manual_class_cb = ttk.Combobox(frm, textvariable=self.manual_class_var)
        self.manual_class_cb.grid(row=1, column=1, sticky='we')

        ttk.Label(frm, text='Class id').grid(row=2, column=0, sticky='w')
        self.manual_class_id_cb = ttk.Combobox(frm, textvariable=self.manual_class_id_var, values=self.class_id_choices, width=8)
        self.manual_class_id_cb.grid(row=2, column=1, sticky='w')

        ttk.Label(frm, text='Evenly sampled frames').grid(row=3, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.manual_samples_var).grid(row=3, column=1, sticky='we')

        ttk.Label(frm, text='Filename prefix').grid(row=4, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.manual_prefix_var).grid(row=4, column=1, sticky='we')

        ttk.Label(frm, text='Initial mask source').grid(row=5, column=0, sticky='w')
        ttk.Combobox(frm, textvariable=self.manual_init_source_var, values=['yolo', 'rembg'], state='readonly', width=10).grid(row=5, column=1, sticky='w')

        ttk.Label(frm, text='Init weights (used when source=yolo)').grid(row=6, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.manual_init_weights_var, width=70).grid(row=6, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_manual_weights).grid(row=6, column=2)

        ttk.Label(frm, text='Init conf').grid(row=7, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.manual_init_conf_var, width=10).grid(row=7, column=1, sticky='w')

        ttk.Label(frm, text='Init image size').grid(row=8, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.manual_init_imgsz_var, width=10).grid(row=8, column=1, sticky='w')

        ttk.Label(frm, text='Init device').grid(row=9, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.manual_init_device_var, width=10).grid(row=9, column=1, sticky='w')

        ttk.Button(frm, text='Prepare frames + initial masks', command=self.prepare_manual).grid(row=10, column=0, pady=8)
        ttk.Button(frm, text='Open manual mask reviewer', command=self.open_manual_reviewer).grid(row=10, column=1, pady=8, sticky='w')

        ttk.Label(frm, text='Reviewer hotkeys: draw LMB | a/e add-erase | s save | n/p next-prev | +/- brush | z/x zoom | q quit').grid(row=11, column=0, columnspan=3, sticky='w')
        frm.columnconfigure(1, weight=1)

    def build_train_tab(self):
        frm = self.tab_train
        self.model_var = tk.StringVar(value='yolo11n-seg.pt')
        self.epochs_var = tk.StringVar(value='80')
        self.imgsz_var = tk.StringVar(value='960')
        self.batch_var = tk.StringVar(value='8')
        self.device_var = tk.StringVar(value='0')
        self.workers_var = tk.StringVar(value='0')

        ttk.Label(frm, text='Model/weights').grid(row=0, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.model_var).grid(row=0, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_train_model).grid(row=0, column=2)
        ttk.Label(frm, text='Tip: yolo11n-seg.pt = fresh train, runs/.../best.pt = continue/fine-tune').grid(row=1, column=0, columnspan=3, sticky='w')

        ttk.Label(frm, text='Epochs').grid(row=2, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.epochs_var).grid(row=2, column=1, sticky='we')

        ttk.Label(frm, text='Image size').grid(row=3, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.imgsz_var).grid(row=3, column=1, sticky='we')

        ttk.Label(frm, text='Batch').grid(row=4, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.batch_var).grid(row=4, column=1, sticky='we')

        ttk.Label(frm, text='Device').grid(row=5, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.device_var).grid(row=5, column=1, sticky='we')

        ttk.Label(frm, text='Workers (Windows: use 0)').grid(row=6, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.workers_var).grid(row=6, column=1, sticky='we')

        ttk.Button(frm, text='Start training', command=self.train).grid(row=7, column=0, pady=8)
        frm.columnconfigure(1, weight=1)

    def build_ddp_tab(self):
        frm = self.tab_ddp
        self.ddp_hosts_var = tk.StringVar(value='127.0.0.1')
        self.ddp_master_addr_var = tk.StringVar(value='127.0.0.1')
        self.ddp_master_port_var = tk.StringVar(value='29500')
        self.ddp_nnodes_var = tk.StringVar(value='2')
        self.ddp_node_rank_var = tk.StringVar(value='0')
        self.ddp_nproc_var = tk.StringVar(value='1')
        self.ddp_model_var = tk.StringVar(value='yolo11n-seg.pt')
        self.ddp_epochs_var = tk.StringVar(value='50')
        self.ddp_imgsz_var = tk.StringVar(value='960')
        self.ddp_batch_var = tk.StringVar(value='8')
        self.ddp_workers_var = tk.StringVar(value='0')
        self.ddp_status_var = tk.StringVar(value='Connection status: not checked')

        ttk.Label(frm, text='Computers (comma-separated host/IP)').grid(row=0, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.ddp_hosts_var, width=70).grid(row=0, column=1, sticky='we')
        ttk.Button(frm, text='Check connected', command=self.check_ddp_connections).grid(row=0, column=2)

        ttk.Label(frm, textvariable=self.ddp_status_var).grid(row=1, column=0, columnspan=3, sticky='w')

        ttk.Label(frm, text='Master addr').grid(row=2, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.ddp_master_addr_var).grid(row=2, column=1, sticky='w')

        ttk.Label(frm, text='Master port').grid(row=3, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.ddp_master_port_var).grid(row=3, column=1, sticky='w')

        ttk.Label(frm, text='Total nodes (nnodes)').grid(row=4, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.ddp_nnodes_var).grid(row=4, column=1, sticky='w')

        ttk.Label(frm, text='This node rank (0..nnodes-1)').grid(row=5, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.ddp_node_rank_var).grid(row=5, column=1, sticky='w')

        ttk.Label(frm, text='GPUs on this node (nproc_per_node)').grid(row=6, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.ddp_nproc_var).grid(row=6, column=1, sticky='w')

        ttk.Separator(frm, orient='horizontal').grid(row=7, column=0, columnspan=3, sticky='we', pady=6)

        ttk.Label(frm, text='Model/weights').grid(row=8, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.ddp_model_var).grid(row=8, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_ddp_model).grid(row=8, column=2)

        ttk.Label(frm, text='Epochs').grid(row=9, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.ddp_epochs_var).grid(row=9, column=1, sticky='w')

        ttk.Label(frm, text='Image size').grid(row=10, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.ddp_imgsz_var).grid(row=10, column=1, sticky='w')

        ttk.Label(frm, text='Batch (per process)').grid(row=11, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.ddp_batch_var).grid(row=11, column=1, sticky='w')

        ttk.Label(frm, text='Workers (recommend 0-1 on Windows)').grid(row=12, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.ddp_workers_var).grid(row=12, column=1, sticky='w')

        ttk.Button(frm, text='Show launch commands (all nodes)', command=self.show_ddp_commands).grid(row=13, column=0, pady=8, sticky='w')
        ttk.Button(frm, text='Start DDP on THIS node', command=self.start_ddp_local).grid(row=13, column=1, pady=8, sticky='w')

        ttk.Label(frm, text='Use same repo/data on all PCs. Run rank 0 on master first, then other ranks.').grid(row=14, column=0, columnspan=3, sticky='w')
        frm.columnconfigure(1, weight=1)

    def build_infer_tab(self):
        frm = self.tab_infer
        self.weights_var = tk.StringVar(value=str(ROOT / 'runs' / 'segment' / 'train' / 'weights' / 'best.pt'))
        self.source_var = tk.StringVar(value='0')
        self.infer_imgsz_var = tk.StringVar(value='960')
        self.infer_conf_var = tk.StringVar(value='0.25')
        self.infer_device_var = tk.StringVar(value='0')
        self.view_w_var = tk.StringVar(value='1920')
        self.view_h_var = tk.StringVar(value='1080')
        self.cam_w_var = tk.StringVar(value='1920')
        self.cam_h_var = tk.StringVar(value='1080')
        self.save_video_var = tk.BooleanVar(value=False)
        self.save_path_var = tk.StringVar(value='')
        self.count_log_var = tk.BooleanVar(value=True)
        self.count_log_every_var = tk.StringVar(value='10')
        self.human_joints_var = tk.BooleanVar(value=False)
        self.human_model_var = tk.StringVar(value='yolo11n-pose.pt')
        self.human_conf_var = tk.StringVar(value='0.20')
        self.human_alpha_var = tk.StringVar(value='0.30')

        ttk.Label(frm, text='Weights').grid(row=0, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.weights_var, width=70).grid(row=0, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_weights).grid(row=0, column=2)

        ttk.Label(frm, text='Source (0 webcam or video path)').grid(row=1, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.source_var).grid(row=1, column=1, sticky='we')
        ttk.Button(frm, text='Browse video', command=self.pick_infer_source).grid(row=1, column=2)

        ttk.Label(frm, text='Image size').grid(row=2, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.infer_imgsz_var).grid(row=2, column=1, sticky='we')

        ttk.Label(frm, text='Conf').grid(row=3, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.infer_conf_var).grid(row=3, column=1, sticky='we')

        ttk.Label(frm, text='Device').grid(row=4, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.infer_device_var).grid(row=4, column=1, sticky='we')

        ttk.Label(frm, text='View width (max)').grid(row=5, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.view_w_var).grid(row=5, column=1, sticky='we')

        ttk.Label(frm, text='View height (max)').grid(row=6, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.view_h_var).grid(row=6, column=1, sticky='we')

        ttk.Label(frm, text='Camera width (source=0)').grid(row=7, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.cam_w_var).grid(row=7, column=1, sticky='we')

        ttk.Label(frm, text='Camera height (source=0)').grid(row=8, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.cam_h_var).grid(row=8, column=1, sticky='we')

        ttk.Checkbutton(frm, text='Save overlay video', variable=self.save_video_var).grid(row=9, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.save_path_var, width=70).grid(row=9, column=1, sticky='we')
        ttk.Button(frm, text='Save As', command=self.pick_save_video).grid(row=9, column=2)

        ttk.Checkbutton(frm, text='Log instance count in terminal', variable=self.count_log_var).grid(row=10, column=0, sticky='w')
        ttk.Label(frm, text='Log every N frames').grid(row=10, column=1, sticky='e')
        ttk.Entry(frm, textvariable=self.count_log_every_var, width=8).grid(row=10, column=2, sticky='w')

        ttk.Checkbutton(frm, text='Enable human arm joint tracking', variable=self.human_joints_var).grid(row=11, column=0, sticky='w')
        ttk.Label(frm, text='Pose model').grid(row=11, column=1, sticky='e')
        ttk.Entry(frm, textvariable=self.human_model_var, width=24).grid(row=11, column=2, sticky='w')

        ttk.Label(frm, text='Human conf').grid(row=12, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.human_conf_var, width=8).grid(row=12, column=1, sticky='w')
        ttk.Label(frm, text='Human alpha').grid(row=12, column=1, sticky='e')
        ttk.Entry(frm, textvariable=self.human_alpha_var, width=8).grid(row=12, column=2, sticky='w')

        ttk.Button(frm, text='Run overlay inference', command=self.infer).grid(row=13, column=0, pady=8)
        frm.columnconfigure(1, weight=1)

    def pick_video(self):
        p = filedialog.askopenfilename(title='Select video')
        if p:
            self.video_var.set(p)

    def pick_weights(self):
        p = filedialog.askopenfilename(title='Select weights', filetypes=[('PyTorch', '*.pt'), ('All files', '*.*')])
        if p:
            self.weights_var.set(p)

    def pick_infer_source(self):
        p = filedialog.askopenfilename(title='Select inference video source')
        if p:
            self.source_var.set(p)

    def pick_train_model(self):
        p = filedialog.askopenfilename(title='Select model/weights for training', filetypes=[('PyTorch', '*.pt'), ('All files', '*.*')])
        if p:
            self.model_var.set(p)

    def pick_ddp_model(self):
        p = filedialog.askopenfilename(title='Select model/weights for DDP training', filetypes=[('PyTorch', '*.pt'), ('All files', '*.*')])
        if p:
            self.ddp_model_var.set(p)

    def _parse_ddp_hosts(self):
        raw = self.ddp_hosts_var.get().strip()
        if not raw:
            return []
        return [h.strip() for h in raw.split(',') if h.strip()]

    def check_ddp_connections(self):
        hosts = self._parse_ddp_hosts()
        if not hosts:
            self.ddp_status_var.set('Connection status: no hosts listed')
            self.log_line('DDP: no hosts listed to check.')
            return

        ok = 0
        for h in hosts:
            if os.name == 'nt':
                cmd = ['ping', '-n', '1', '-w', '1200', h]
            else:
                cmd = ['ping', '-c', '1', '-W', '1', h]
            try:
                rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
                if rc == 0:
                    ok += 1
                    self.log_line(f'DDP host reachable: {h}')
                else:
                    self.log_line(f'DDP host NOT reachable: {h}')
            except Exception as e:
                self.log_line(f'DDP host check error for {h}: {e}')

        self.ddp_status_var.set(f'Connection status: {ok}/{len(hosts)} reachable')

    def _ddp_launch_cmd(self, rank):
        return [
            str(PY), '-m', 'torch.distributed.run',
            '--nnodes', self.ddp_nnodes_var.get(),
            '--nproc_per_node', self.ddp_nproc_var.get(),
            '--node_rank', str(rank),
            '--master_addr', self.ddp_master_addr_var.get(),
            '--master_port', self.ddp_master_port_var.get(),
            'scripts/train_yolo_seg_ddp.py',
            '--model', self.ddp_model_var.get(),
            '--epochs', self.ddp_epochs_var.get(),
            '--imgsz', self.ddp_imgsz_var.get(),
            '--batch', self.ddp_batch_var.get(),
            '--workers', self.ddp_workers_var.get(),
        ]

    def show_ddp_commands(self):
        try:
            nnodes = int(self.ddp_nnodes_var.get())
        except Exception:
            self.log_line('DDP: invalid nnodes value.')
            return
        self.log_line('DDP launch commands (run on each corresponding node):')
        for r in range(max(1, nnodes)):
            cmd = self._ddp_launch_cmd(r)
            self.log_line(f'RANK {r}: ' + ' '.join(map(str, cmd)))

    def start_ddp_local(self):
        cmd = self._ddp_launch_cmd(self.ddp_node_rank_var.get())
        self.run_cmd(cmd)

    def pick_save_video(self):
        p = filedialog.asksaveasfilename(title='Save overlay video as', defaultextension='.mp4', filetypes=[('MP4 video', '*.mp4'), ('All files', '*.*')])
        if p:
            self.save_path_var.set(p)
            self.save_video_var.set(True)

    def pick_bg_dir(self):
        p = filedialog.askdirectory(title='Select background images folder')
        if p:
            self.bg_dir_var.set(p)

    def pick_synth_multi_bg_dir(self):
        p = filedialog.askdirectory(title='Select background images folder for multi-instance synth')
        if p:
            self.synth_multi_bg_dir_var.set(p)

    def pick_synth_all_bg_dir(self):
        p = filedialog.askdirectory(title='Select background images folder (all synth runner)')
        if p:
            self.synth_all_bg_dir_var.set(p)

    def pick_synth_all_obs_dir(self):
        p = filedialog.askdirectory(title='Select obstruction folder (all synth runner)')
        if p:
            self.synth_all_obs_dir_var.set(p)

    def pick_cutout_image(self):
        p = filedialog.askopenfilename(title='Select source image for masked object sample')
        if p:
            self.cutout_image_var.set(p)

    def pick_manual_video(self):
        p = filedialog.askopenfilename(title='Select video for manual review prep')
        if p:
            self.manual_video_var.set(p)

    def pick_manual_weights(self):
        p = filedialog.askopenfilename(title='Select init weights for manual prep', filetypes=[('PyTorch', '*.pt'), ('All files', '*.*')])
        if p:
            self.manual_init_weights_var.set(p)

    def pick_obs_dir(self):
        p = filedialog.askdirectory(title='Select obstruction folder')
        if p:
            self.obs_dir_var.set(p)

    def pick_obs_bg_dir(self):
        p = filedialog.askdirectory(title='Select background folder for obstruction generation')
        if p:
            self.obs_bg_dir_var.set(p)

    def autolabel(self):
        if not self.ensure_class_registered(self.class_var.get(), self.class_id_var.get()):
            return
        cmd = [
            str(PY), 'scripts/video_to_yoloseg_autolabel.py',
            '--video', self.video_var.get(),
            '--class-name', self.class_var.get(),
            '--class-id', self.class_id_var.get(),
            '--num-samples', self.num_samples_var.get(),
            '--every', self.every_var.get(),
            '--max-frames', self.max_var.get(),
            '--aug-per-frame', self.aug_var.get(),
            '--mask-quality', self.mask_quality_var.get(),
        ]
        self.run_cmd(cmd)
        self.after(1000, self.refresh_class_options)

    def _synth_cmd_base(self):
        cmd = [
            str(PY), 'scripts/synthesize_cutpaste_backgrounds.py',
            '--class-name', self.synth_class_var.get(),
            '--class-id', self.synth_class_id_var.get(),
            '--background-dir', self.bg_dir_var.get(),
            '--num-synthetic', self.synth_n_var.get(),
            '--min-scale', self.synth_min_scale_var.get(),
            '--max-scale', self.synth_max_scale_var.get(),
            '--max-rotation', self.synth_rot_var.get(),
            '--brightness-min', self.synth_bri_min_var.get(),
            '--brightness-max', self.synth_bri_max_var.get(),
            '--object-brightness-min', self.synth_obj_bri_min_var.get(),
            '--object-brightness-max', self.synth_obj_bri_max_var.get(),
        ]
        if self.synth_run_var.get().strip():
            cmd.extend(['--run-name', self.synth_run_var.get().strip()])
        return cmd

    def preview_synth(self):
        if not self.ensure_class_registered(self.synth_class_var.get(), self.synth_class_id_var.get()):
            return
        if not self.bg_dir_var.get().strip():
            self.log_line('Please select a background folder first.')
            return
        cmd = self._synth_cmd_base() + ['--preview-only', '--preview-window', '--preview-count', self.synth_preview_count_var.get()]
        self.run_cmd(cmd)

    def generate_synth(self):
        if not self.ensure_class_registered(self.synth_class_var.get(), self.synth_class_id_var.get()):
            return
        if not self.bg_dir_var.get().strip():
            self.log_line('Please select a background folder first.')
            return
        self.run_cmd(self._synth_cmd_base())

    def _synth_multi_cmd_base(self):
        cmd = [
            str(PY), 'scripts/synthesize_multi_instance.py',
            '--class-name', self.synth_multi_class_var.get(),
            '--class-id', self.synth_multi_class_id_var.get(),
            '--background-dir', self.synth_multi_bg_dir_var.get(),
            '--num-synthetic', self.synth_multi_n_var.get(),
            '--min-objects', self.synth_multi_min_obj_var.get(),
            '--max-objects', self.synth_multi_max_obj_var.get(),
            '--overlap-prob', self.synth_multi_overlap_var.get(),
            '--max-overlap-ratio', self.synth_multi_max_overlap_ratio_var.get(),
            '--cluster-distance-factor', self.synth_multi_cluster_dist_var.get(),
            '--min-scale', self.synth_multi_min_scale_var.get(),
            '--max-scale', self.synth_multi_max_scale_var.get(),
            '--max-rotation', '360',
            '--overlap-spread', self.synth_multi_overlap_spread_var.get(),
            '--brightness-min', self.synth_multi_bg_bri_min_var.get(),
            '--brightness-max', self.synth_multi_bg_bri_max_var.get(),
            '--object-brightness-min', self.synth_multi_obj_bri_min_var.get(),
            '--object-brightness-max', self.synth_multi_obj_bri_max_var.get(),
        ]
        if self.synth_multi_run_var.get().strip():
            cmd.extend(['--run-name', self.synth_multi_run_var.get().strip()])
        return cmd

    def preview_synth_multi(self):
        if not self.ensure_class_registered(self.synth_multi_class_var.get(), self.synth_multi_class_id_var.get()):
            return
        if not self.synth_multi_bg_dir_var.get().strip():
            self.log_line('Please select a background folder first (multi-instance).')
            return
        cmd = self._synth_multi_cmd_base() + ['--preview-only', '--preview-window', '--preview-count', self.synth_multi_preview_count_var.get()]
        self.run_cmd(cmd)

    def generate_synth_multi(self):
        if not self.ensure_class_registered(self.synth_multi_class_var.get(), self.synth_multi_class_id_var.get()):
            return
        if not self.synth_multi_bg_dir_var.get().strip():
            self.log_line('Please select a background folder first (multi-instance).')
            return
        self.run_cmd(self._synth_multi_cmd_base())

    def create_cutout_sample(self):
        if not self.ensure_class_registered(self.cutout_class_var.get(), self.cutout_class_id_var.get()):
            return
        if not self.cutout_image_var.get().strip():
            self.log_line('Please select a source image first.')
            return
        cmd = [
            str(PY), 'scripts/add_masked_object_from_box.py',
            '--image', self.cutout_image_var.get().strip(),
            '--class-name', self.cutout_class_var.get(),
            '--class-id', self.cutout_class_id_var.get(),
            '--prefix', self.cutout_prefix_var.get().strip() or 'cutout',
        ]
        self.run_cmd(cmd)

    def preview_combo_bg(self):
        if not self.ensure_class_registered(self.synth_all_class_var.get(), self.synth_all_class_id_var.get()):
            return
        if not self.synth_all_bg_dir_var.get().strip():
            self.log_line('COMBO RUN preview: select background folder first.')
            return
        base = self.synth_all_run_var.get().strip() or 'combo01'
        cmd = [
            str(PY), 'scripts/synthesize_cutpaste_backgrounds.py',
            '--class-name', self.synth_all_class_var.get(),
            '--class-id', self.synth_all_class_id_var.get(),
            '--background-dir', self.synth_all_bg_dir_var.get().strip(),
            '--num-synthetic', '1',
            '--min-scale', self.synth_all_min_scale_var.get(),
            '--max-scale', self.synth_all_max_scale_var.get(),
            '--max-rotation', '360',
            '--brightness-min', self.synth_all_bg_bri_min_var.get(),
            '--brightness-max', self.synth_all_bg_bri_max_var.get(),
            '--object-brightness-min', self.synth_all_obj_bri_min_var.get(),
            '--object-brightness-max', self.synth_all_obj_bri_max_var.get(),
            '--run-name', f'{base}_bg_preview',
            '--preview-only', '--preview-window', '--preview-count', self.synth_all_preview_count_var.get(),
        ]
        self.run_cmd(cmd)

    def preview_combo_multi(self):
        if not self.ensure_class_registered(self.synth_all_class_var.get(), self.synth_all_class_id_var.get()):
            return
        if not self.synth_all_bg_dir_var.get().strip():
            self.log_line('COMBO RUN preview: select background folder first.')
            return
        base = self.synth_all_run_var.get().strip() or 'combo01'
        cmd = [
            str(PY), 'scripts/synthesize_multi_instance.py',
            '--class-name', self.synth_all_class_var.get(),
            '--class-id', self.synth_all_class_id_var.get(),
            '--background-dir', self.synth_all_bg_dir_var.get().strip(),
            '--num-synthetic', '1',
            '--min-objects', self.synth_all_multi_min_obj_var.get(),
            '--max-objects', self.synth_all_multi_max_obj_var.get(),
            '--overlap-prob', self.synth_all_multi_overlap_var.get(),
            '--max-overlap-ratio', self.synth_all_multi_max_overlap_ratio_var.get(),
            '--cluster-distance-factor', self.synth_all_multi_cluster_dist_var.get(),
            '--overlap-spread', self.synth_all_multi_spread_var.get(),
            '--min-scale', self.synth_all_multi_min_scale_var.get(),
            '--max-scale', self.synth_all_multi_max_scale_var.get(),
            '--max-rotation', '360',
            '--brightness-min', self.synth_all_bg_bri_min_var.get(),
            '--brightness-max', self.synth_all_bg_bri_max_var.get(),
            '--object-brightness-min', self.synth_all_obj_bri_min_var.get(),
            '--object-brightness-max', self.synth_all_obj_bri_max_var.get(),
            '--run-name', f'{base}_multi_preview',
            '--preview-only', '--preview-window', '--preview-count', self.synth_all_preview_count_var.get(),
        ]
        self.run_cmd(cmd)

    def preview_combo_obs(self):
        if not self.ensure_class_registered(self.synth_all_class_var.get(), self.synth_all_class_id_var.get()):
            return
        if not self.synth_all_bg_dir_var.get().strip() or not self.synth_all_obs_dir_var.get().strip():
            self.log_line('COMBO RUN obstruction preview: select both background and obstruction folder first.')
            return
        base = self.synth_all_run_var.get().strip() or 'combo01'
        cmd = [
            str(PY), 'scripts/synthesize_with_obstructions.py',
            '--class-name', self.synth_all_class_var.get(),
            '--class-id', self.synth_all_class_id_var.get(),
            '--obstruction-dir', self.synth_all_obs_dir_var.get().strip(),
            '--background-dir', self.synth_all_bg_dir_var.get().strip(),
            '--num-synthetic', '1',
            '--entry-angle-min', self.synth_all_obs_angle_min_var.get(),
            '--entry-angle-max', self.synth_all_obs_angle_max_var.get(),
            '--rotation-deviation', self.synth_all_obs_rot_dev_var.get(),
            '--overlap-min', self.synth_all_obs_overlap_min_var.get(),
            '--overlap-max', self.synth_all_obs_overlap_max_var.get(),
            '--obstruction-scale-min', self.synth_all_obs_scale_min_var.get(),
            '--obstruction-scale-max', self.synth_all_obs_scale_max_var.get(),
            '--white-bg-prob', self.synth_all_obs_white_prob_var.get(),
            '--run-name', f'{base}_obs_preview',
            '--preview-only', '--preview-window', '--preview-count', self.synth_all_preview_count_var.get(),
        ]
        self.run_cmd(cmd)

    def copy_combo_base_to_run_filter(self):
        base = (self.synth_all_run_var.get() or '').strip()
        if not base:
            self.log_line('COMBO RUN base is empty; cannot copy to run filter.')
            return
        self.split_run_var.set(base)
        self.log_line(f'Copied COMBO RUN base to Data Prep run filter: {base}')

    def run_all_synth(self):
        if not self.ensure_class_registered(self.synth_all_class_var.get(), self.synth_all_class_id_var.get()):
            return
        if not self.synth_all_bg_dir_var.get().strip():
            self.log_line('Please select a background folder for COMBO RUN.')
            return

        base = self.synth_all_run_var.get().strip() or 'combo01'
        cls = self.synth_all_class_var.get()
        cid = self.synth_all_class_id_var.get()
        bg = self.synth_all_bg_dir_var.get().strip()
        obs = self.synth_all_obs_dir_var.get().strip()

        n_bg = int(self.synth_all_n_bg_var.get() or '0')
        n_multi = int(self.synth_all_n_multi_var.get() or '0')
        n_obs = int(self.synth_all_n_obs_var.get() or '0')

        if self.synth_all_use_bg_var.get() and n_bg > 0:
            cmd_bg = [
                str(PY), 'scripts/synthesize_cutpaste_backgrounds.py',
                '--class-name', cls, '--class-id', cid,
                '--background-dir', bg,
                '--num-synthetic', str(n_bg),
                '--min-scale', self.synth_all_min_scale_var.get(),
                '--max-scale', self.synth_all_max_scale_var.get(),
                '--max-rotation', '360',
                '--brightness-min', self.synth_all_bg_bri_min_var.get(),
                '--brightness-max', self.synth_all_bg_bri_max_var.get(),
                '--object-brightness-min', self.synth_all_obj_bri_min_var.get(),
                '--object-brightness-max', self.synth_all_obj_bri_max_var.get(),
                '--run-name', f'{base}_bg',
            ]
            self.run_cmd(cmd_bg)

        if self.synth_all_use_multi_var.get() and n_multi > 0:
            cmd_multi = [
                str(PY), 'scripts/synthesize_multi_instance.py',
                '--class-name', cls, '--class-id', cid,
                '--background-dir', bg,
                '--num-synthetic', str(n_multi),
                '--min-objects', self.synth_all_multi_min_obj_var.get(),
                '--max-objects', self.synth_all_multi_max_obj_var.get(),
                '--overlap-prob', self.synth_all_multi_overlap_var.get(),
                '--max-overlap-ratio', self.synth_all_multi_max_overlap_ratio_var.get(),
                '--cluster-distance-factor', self.synth_all_multi_cluster_dist_var.get(),
                '--overlap-spread', self.synth_all_multi_spread_var.get(),
                '--min-scale', self.synth_all_multi_min_scale_var.get(),
                '--max-scale', self.synth_all_multi_max_scale_var.get(),
                '--max-rotation', '360',
                '--brightness-min', self.synth_all_bg_bri_min_var.get(),
                '--brightness-max', self.synth_all_bg_bri_max_var.get(),
                '--object-brightness-min', self.synth_all_obj_bri_min_var.get(),
                '--object-brightness-max', self.synth_all_obj_bri_max_var.get(),
                '--run-name', f'{base}_multi',
            ]
            self.run_cmd(cmd_multi)

        if self.synth_all_use_obs_var.get() and n_obs > 0:
            if not obs:
                self.log_line('COMBO RUN: Obstruction enabled but no obstruction folder selected.')
            else:
                cmd_obs = [
                    str(PY), 'scripts/synthesize_with_obstructions.py',
                    '--class-name', cls, '--class-id', cid,
                    '--obstruction-dir', obs,
                    '--background-dir', bg,
                    '--num-synthetic', str(n_obs),
                    '--entry-angle-min', self.synth_all_obs_angle_min_var.get(),
                    '--entry-angle-max', self.synth_all_obs_angle_max_var.get(),
                    '--rotation-deviation', self.synth_all_obs_rot_dev_var.get(),
                    '--overlap-min', self.synth_all_obs_overlap_min_var.get(),
                    '--overlap-max', self.synth_all_obs_overlap_max_var.get(),
                    '--obstruction-scale-min', self.synth_all_obs_scale_min_var.get(),
                    '--obstruction-scale-max', self.synth_all_obs_scale_max_var.get(),
                    '--white-bg-prob', self.synth_all_obs_white_prob_var.get(),
                    '--run-name', f'{base}_obs',
                ]
                self.run_cmd(cmd_obs)

        self.split_run_var.set(base)
        self.log_line(f'COMBO RUN started with base: {base}. Data Prep run filter auto-set to "{base}". Build split, then train.')

    def prepare_manual(self):
        if not self.ensure_class_registered(self.manual_class_var.get(), self.manual_class_id_var.get()):
            return
        if not self.manual_video_var.get().strip():
            self.log_line('Please select a video file first.')
            return
        cmd = [
            str(PY), 'scripts/prepare_manual_review_from_video.py',
            '--video', self.manual_video_var.get(),
            '--class-name', self.manual_class_var.get(),
            '--class-id', self.manual_class_id_var.get(),
            '--num-samples', self.manual_samples_var.get(),
            '--prefix', self.manual_prefix_var.get(),
            '--init-source', self.manual_init_source_var.get(),
            '--init-conf', self.manual_init_conf_var.get(),
            '--imgsz', self.manual_init_imgsz_var.get(),
            '--device', self.manual_init_device_var.get(),
        ]
        if self.manual_init_source_var.get() == 'yolo' and self.manual_init_weights_var.get().strip():
            cmd.extend(['--weights', self.manual_init_weights_var.get().strip()])
        self.run_cmd(cmd)

    def _obstruction_cmd_base(self):
        return [
            str(PY), 'scripts/synthesize_with_obstructions.py',
            '--class-name', self.obs_class_var.get(),
            '--class-id', self.obs_class_id_var.get(),
            '--obstruction-dir', self.obs_dir_var.get(),
            '--background-dir', self.obs_bg_dir_var.get(),
            '--white-bg-prob', self.obs_white_prob_var.get(),
            '--entry-angle-min', self.obs_angle_min_var.get(),
            '--entry-angle-max', self.obs_angle_max_var.get(),
            '--rotation-deviation', self.obs_rot_dev_var.get(),
            '--overlap-level', self.obs_overlap_var.get(),
            '--obstruction-scale', self.obs_scale_var.get(),
        ] + (['--run-name', self.obs_run_var.get().strip()] if self.obs_run_var.get().strip() else [])

    def preview_obstruction(self):
        if not self.ensure_class_registered(self.obs_class_var.get(), self.obs_class_id_var.get()):
            return
        if not self.obs_dir_var.get().strip():
            self.log_line('Please select obstruction folder first.')
            return
        if not self.obs_bg_dir_var.get().strip():
            self.log_line('Please select background folder first.')
            return
        cmd = self._obstruction_cmd_base() + ['--preview-only', '--preview-window', '--preview-count', self.obs_preview_count_var.get()]
        self.run_cmd(cmd)

    def generate_obstruction(self):
        if not self.ensure_class_registered(self.obs_class_var.get(), self.obs_class_id_var.get()):
            return
        if not self.obs_dir_var.get().strip():
            self.log_line('Please select obstruction folder first.')
            return
        if not self.obs_bg_dir_var.get().strip():
            self.log_line('Please select background folder first.')
            return
        cmd = self._obstruction_cmd_base() + ['--num-synthetic', self.obs_num_var.get()]
        self.run_cmd(cmd)

    def open_manual_reviewer(self):
        cmd = [
            str(PY), 'scripts/manual_mask_reviewer.py',
            '--class-name', self.manual_class_var.get(),
            '--class-id', self.manual_class_id_var.get(),
            '--contains', f"{self.manual_prefix_var.get()}_",
        ]
        self.run_cmd(cmd)

    def build_split(self):
        cmd = [str(PY), 'scripts/build_dataset_split.py', '--mode', self.split_mode_var.get()]
        sel = self.split_class_var.get().strip()
        if sel and sel != 'ALL classes':
            if ':' in sel:
                class_name = sel.split(':', 1)[1].strip()
            else:
                class_name = sel.strip()
            if class_name:
                cmd.extend(['--class-name', class_name])
        run_filter = self.split_run_var.get().strip()
        if run_filter:
            cmd.extend(['--run-name', run_filter])
        self.run_cmd(cmd)

    def update_yaml(self):
        classes = self.classes_var.get().strip().split()
        if not classes:
            self.log_line('No classes provided')
            return
        self.run_cmd([str(PY), 'scripts/update_dataset_yaml.py', '--classes', *classes])
        self.after(500, self.refresh_class_options)

    def sync_yaml_from_folders(self):
        data_images = ROOT / 'data' / 'images'
        folder_classes = sorted([p.name for p in data_images.iterdir() if p.is_dir()]) if data_images.exists() else []
        if not folder_classes:
            self.log_line('No class folders found in data/images to sync.')
            return

        dataset_yaml = ROOT / 'dataset.yaml'
        data = {
            'path': 'data/yolo_dataset',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {},
        }

        if dataset_yaml.exists():
            try:
                loaded = yaml.safe_load(dataset_yaml.read_text(encoding='utf-8')) or {}
                if isinstance(loaded, dict):
                    data.update({k: v for k, v in loaded.items() if k in ['path', 'train', 'val', 'test', 'names']})
            except Exception as e:
                self.log_line(f'Warning: could not parse existing dataset.yaml, rebuilding: {e}')

        names = data.get('names', {})
        if isinstance(names, list):
            id_to_name = {int(i): str(n) for i, n in enumerate(names)}
        elif isinstance(names, dict):
            id_to_name = {int(k): str(v) for k, v in names.items()}
        else:
            id_to_name = {}

        existing_names = set(id_to_name.values())
        added = []
        for cname in folder_classes:
            if cname in existing_names:
                continue
            nid = 0
            used = set(id_to_name.keys())
            while nid in used:
                nid += 1
            id_to_name[nid] = cname
            existing_names.add(cname)
            added.append((nid, cname))

        data['path'] = data.get('path') or 'data/yolo_dataset'
        data['train'] = data.get('train') or 'images/train'
        data['val'] = data.get('val') or 'images/val'
        data['test'] = data.get('test') or 'images/test'
        data['names'] = {int(k): id_to_name[k] for k in sorted(id_to_name.keys())}

        dataset_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
        if added:
            self.log_line('Auto-sync added classes: ' + ', '.join([f'{cid}:{cn}' for cid, cn in added]))
        else:
            self.log_line('Auto-sync complete: no new classes needed.')

        ordered = [data['names'][k] for k in sorted(data['names'].keys())]
        self.classes_var.set(' '.join(ordered))
        self.refresh_class_options()

    def train(self):
        cmd = [
            str(PY), 'scripts/train_yolo_seg.py',
            '--model', self.model_var.get(),
            '--epochs', self.epochs_var.get(),
            '--imgsz', self.imgsz_var.get(),
            '--batch', self.batch_var.get(),
            '--device', self.device_var.get(),
            '--workers', self.workers_var.get(),
        ]
        self.run_cmd(cmd)

    def infer(self):
        cmd = [
            str(PY), 'scripts/run_inference_overlay.py',
            '--weights', self.weights_var.get(),
            '--source', self.source_var.get(),
            '--imgsz', self.infer_imgsz_var.get(),
            '--conf', self.infer_conf_var.get(),
            '--device', self.infer_device_var.get(),
            '--view-width', self.view_w_var.get(),
            '--view-height', self.view_h_var.get(),
            '--cam-width', self.cam_w_var.get(),
            '--cam-height', self.cam_h_var.get(),
            '--count-log-every', self.count_log_every_var.get(),
            '--human-model', self.human_model_var.get(),
            '--human-conf', self.human_conf_var.get(),
            '--human-alpha', self.human_alpha_var.get(),
        ]
        if self.count_log_var.get():
            cmd.append('--count-log')
        if self.human_joints_var.get():
            cmd.append('--human-joints')
        if self.save_video_var.get():
            cmd.append('--save-video')
            if self.save_path_var.get().strip():
                cmd.extend(['--save-path', self.save_path_var.get().strip()])
        self.run_cmd(cmd)


if __name__ == '__main__':
    app = App()
    app.mainloop()
