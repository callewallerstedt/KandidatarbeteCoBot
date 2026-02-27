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

        self.log = tk.Text(self, wrap='word')
        self.log.pack(side='bottom', fill='both', expand=True)

        notebook = ttk.Notebook(self)
        notebook.pack(fill='x')

        self.tab_instructions = ttk.Frame(notebook)
        self.tab_data = ttk.Frame(notebook)
        self.tab_synth = ttk.Frame(notebook)
        self.tab_manual = ttk.Frame(notebook)
        self.tab_obstruction = ttk.Frame(notebook)
        self.tab_train = ttk.Frame(notebook)
        self.tab_ddp = ttk.Frame(notebook)
        self.tab_infer = ttk.Frame(notebook)
        notebook.add(self.tab_instructions, text='0) Instructions')
        notebook.add(self.tab_data, text='1) Data Prep')
        notebook.add(self.tab_synth, text='2) Synthetic BG')
        notebook.add(self.tab_obstruction, text='3) Obstruction Data')
        notebook.add(self.tab_manual, text='4) Manual Real Data')
        notebook.add(self.tab_train, text='5) Train')
        notebook.add(self.tab_ddp, text='6) DDP Multi-PC')
        notebook.add(self.tab_infer, text='7) Inference')

        self.class_id_choices = [str(i) for i in range(0, 25)]

        self.build_instructions_tab()
        self.build_data_tab()
        self.build_synth_tab()
        self.build_obstruction_tab()
        self.build_manual_tab()
        self.build_train_tab()
        self.build_ddp_tab()
        self.build_infer_tab()
        self.refresh_class_options()
        self.sync_yaml_from_folders()
        self.auto_assign_class_id(self.class_var, self.class_id_var)
        self.auto_assign_class_id(self.synth_class_var, self.synth_class_id_var)
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

        for cb_name in ['data_class_cb', 'synth_class_cb', 'obs_class_cb', 'manual_class_cb']:
            cb = getattr(self, cb_name, None)
            if cb is not None:
                cb['values'] = self.class_choices

        for cb_name in ['data_class_id_cb', 'synth_class_id_cb', 'obs_class_id_cb', 'manual_class_id_cb']:
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

        ttk.Label(frm, text='all = all data | real = no synth/obs | synth = only *_synth_* | obs = only *_obs_*').grid(row=11, column=0, columnspan=3, sticky='w')
        ttk.Button(frm, text='Build train/val/test split', command=self.build_split).grid(row=12, column=0, pady=8, sticky='w')

        ttk.Separator(frm, orient='horizontal').grid(row=13, column=0, columnspan=3, sticky='we', pady=8)

        ttk.Label(frm, text='All classes (space-separated, in class-id order)').grid(row=14, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.classes_var, width=70).grid(row=14, column=1, sticky='we')
        ttk.Button(frm, text='Update dataset.yaml', command=self.update_yaml).grid(row=14, column=2)
        ttk.Button(frm, text='Auto-sync dataset.yaml from data folders', command=self.sync_yaml_from_folders).grid(row=15, column=0, pady=6, sticky='w')

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

        ttk.Button(frm, text='Generate synthetic cut-paste set', command=self.generate_synth).grid(row=7, column=0, pady=8)
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

        ttk.Button(frm, text='Preview 1 obstruction sample', command=self.preview_obstruction).grid(row=11, column=0, pady=8)
        ttk.Button(frm, text='Generate obstruction synthetic set', command=self.generate_obstruction).grid(row=11, column=1, pady=8, sticky='w')
        ttk.Label(frm, text='Preview shows debug: yellow=center, magenta=hand vector (bottom→top), cyan=top→center target.').grid(row=12, column=0, columnspan=3, sticky='w')
        frm.columnconfigure(1, weight=1)

    def build_manual_tab(self):
        frm = self.tab_manual
        self.manual_video_var = tk.StringVar()
        self.manual_class_var = tk.StringVar(value='object_name')
        self.manual_class_id_var = tk.StringVar(value='0')
        self.manual_samples_var = tk.StringVar(value='80')
        self.manual_prefix_var = tk.StringVar(value='manual')
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

        ttk.Button(frm, text='Prepare frames + initial masks', command=self.prepare_manual).grid(row=5, column=0, pady=8)
        ttk.Button(frm, text='Open manual mask reviewer', command=self.open_manual_reviewer).grid(row=5, column=1, pady=8, sticky='w')

        ttk.Label(frm, text='Reviewer hotkeys: mouse draw | a add | e erase | s save | n/p next/prev | +/- brush | q quit').grid(row=6, column=0, columnspan=3, sticky='w')
        frm.columnconfigure(1, weight=1)

    def build_train_tab(self):
        frm = self.tab_train
        self.model_var = tk.StringVar(value='yolo11n-seg.pt')
        self.epochs_var = tk.StringVar(value='80')
        self.imgsz_var = tk.StringVar(value='640')
        self.batch_var = tk.StringVar(value='16')
        self.device_var = tk.StringVar(value='0')

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

        ttk.Button(frm, text='Start training', command=self.train).grid(row=6, column=0, pady=8)
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
        self.infer_imgsz_var = tk.StringVar(value='640')
        self.infer_conf_var = tk.StringVar(value='0.25')
        self.infer_device_var = tk.StringVar(value='0')
        self.view_w_var = tk.StringVar(value='1280')
        self.view_h_var = tk.StringVar(value='720')
        self.save_video_var = tk.BooleanVar(value=False)
        self.save_path_var = tk.StringVar(value='')

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

        ttk.Checkbutton(frm, text='Save overlay video', variable=self.save_video_var).grid(row=7, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.save_path_var, width=70).grid(row=7, column=1, sticky='we')
        ttk.Button(frm, text='Save As', command=self.pick_save_video).grid(row=7, column=2)

        ttk.Button(frm, text='Run overlay inference', command=self.infer).grid(row=8, column=0, pady=8)
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

    def pick_manual_video(self):
        p = filedialog.askopenfilename(title='Select video for manual review prep')
        if p:
            self.manual_video_var.set(p)

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

    def generate_synth(self):
        if not self.ensure_class_registered(self.synth_class_var.get(), self.synth_class_id_var.get()):
            return
        if not self.bg_dir_var.get().strip():
            self.log_line('Please select a background folder first.')
            return
        cmd = [
            str(PY), 'scripts/synthesize_cutpaste_backgrounds.py',
            '--class-name', self.synth_class_var.get(),
            '--class-id', self.synth_class_id_var.get(),
            '--background-dir', self.bg_dir_var.get(),
            '--num-synthetic', self.synth_n_var.get(),
            '--min-scale', self.synth_min_scale_var.get(),
            '--max-scale', self.synth_max_scale_var.get(),
            '--max-rotation', self.synth_rot_var.get(),
        ]
        self.run_cmd(cmd)

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
        ]
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
        ]

    def preview_obstruction(self):
        if not self.ensure_class_registered(self.obs_class_var.get(), self.obs_class_id_var.get()):
            return
        if not self.obs_dir_var.get().strip():
            self.log_line('Please select obstruction folder first.')
            return
        if not self.obs_bg_dir_var.get().strip():
            self.log_line('Please select background folder first.')
            return
        cmd = self._obstruction_cmd_base() + ['--num-synthetic', '1', '--preview-only', '--preview-window']
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
        ]
        if self.save_video_var.get():
            cmd.append('--save-video')
            if self.save_path_var.get().strip():
                cmd.extend(['--save-path', self.save_path_var.get().strip()])
        self.run_cmd(cmd)


if __name__ == '__main__':
    app = App()
    app.mainloop()
