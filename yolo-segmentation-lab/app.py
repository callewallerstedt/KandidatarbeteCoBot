#!/usr/bin/env python3
import os
import sys
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog

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

        self.tab_data = ttk.Frame(notebook)
        self.tab_train = ttk.Frame(notebook)
        self.tab_infer = ttk.Frame(notebook)
        notebook.add(self.tab_data, text='1) Data Prep')
        notebook.add(self.tab_train, text='2) Train')
        notebook.add(self.tab_infer, text='3) Inference')

        self.build_data_tab()
        self.build_train_tab()
        self.build_infer_tab()

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

    def build_data_tab(self):
        frm = self.tab_data

        self.video_var = tk.StringVar()
        self.class_var = tk.StringVar(value='object_name')
        self.class_id_var = tk.StringVar(value='0')
        self.every_var = tk.StringVar(value='8')
        self.num_samples_var = tk.StringVar(value='120')
        self.max_var = tk.StringVar(value='300')
        self.aug_var = tk.StringVar(value='1')
        self.classes_var = tk.StringVar(value='object_name')

        ttk.Label(frm, text='Video file').grid(row=0, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.video_var, width=70).grid(row=0, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_video).grid(row=0, column=2)

        ttk.Label(frm, text='Class name').grid(row=1, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.class_var).grid(row=1, column=1, sticky='we')

        ttk.Label(frm, text='Class id').grid(row=2, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.class_id_var).grid(row=2, column=1, sticky='we')

        ttk.Label(frm, text='Target samples (evenly across video)').grid(row=3, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.num_samples_var).grid(row=3, column=1, sticky='we')

        ttk.Label(frm, text='Fallback: every N frames (if target=0)').grid(row=4, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.every_var).grid(row=4, column=1, sticky='we')

        ttk.Label(frm, text='Max frames').grid(row=5, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.max_var).grid(row=5, column=1, sticky='we')

        ttk.Label(frm, text='Aug per frame').grid(row=6, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.aug_var).grid(row=6, column=1, sticky='we')

        ttk.Button(frm, text='Auto-label from video', command=self.autolabel).grid(row=7, column=0, pady=8)
        ttk.Button(frm, text='Build train/val/test split', command=self.build_split).grid(row=7, column=1, pady=8, sticky='w')

        ttk.Separator(frm, orient='horizontal').grid(row=8, column=0, columnspan=3, sticky='we', pady=8)

        ttk.Label(frm, text='All classes (space-separated, in class-id order)').grid(row=9, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.classes_var, width=70).grid(row=9, column=1, sticky='we')
        ttk.Button(frm, text='Update dataset.yaml', command=self.update_yaml).grid(row=9, column=2)

        frm.columnconfigure(1, weight=1)

    def build_train_tab(self):
        frm = self.tab_train
        self.model_var = tk.StringVar(value='yolo11n-seg.pt')
        self.epochs_var = tk.StringVar(value='80')
        self.imgsz_var = tk.StringVar(value='640')
        self.batch_var = tk.StringVar(value='16')
        self.device_var = tk.StringVar(value='0')

        ttk.Label(frm, text='Model').grid(row=0, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.model_var).grid(row=0, column=1, sticky='we')

        ttk.Label(frm, text='Epochs').grid(row=1, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.epochs_var).grid(row=1, column=1, sticky='we')

        ttk.Label(frm, text='Image size').grid(row=2, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.imgsz_var).grid(row=2, column=1, sticky='we')

        ttk.Label(frm, text='Batch').grid(row=3, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.batch_var).grid(row=3, column=1, sticky='we')

        ttk.Label(frm, text='Device').grid(row=4, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.device_var).grid(row=4, column=1, sticky='we')

        ttk.Button(frm, text='Start training', command=self.train).grid(row=5, column=0, pady=8)
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

        ttk.Label(frm, text='Weights').grid(row=0, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.weights_var, width=70).grid(row=0, column=1, sticky='we')
        ttk.Button(frm, text='Browse', command=self.pick_weights).grid(row=0, column=2)

        ttk.Label(frm, text='Source (0 webcam or video path)').grid(row=1, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.source_var).grid(row=1, column=1, sticky='we')

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

        ttk.Button(frm, text='Run overlay inference', command=self.infer).grid(row=7, column=0, pady=8)
        frm.columnconfigure(1, weight=1)

    def pick_video(self):
        p = filedialog.askopenfilename(title='Select video')
        if p:
            self.video_var.set(p)

    def pick_weights(self):
        p = filedialog.askopenfilename(title='Select weights', filetypes=[('PyTorch', '*.pt'), ('All files', '*.*')])
        if p:
            self.weights_var.set(p)

    def autolabel(self):
        cmd = [
            str(PY), 'scripts/video_to_yoloseg_autolabel.py',
            '--video', self.video_var.get(),
            '--class-name', self.class_var.get(),
            '--class-id', self.class_id_var.get(),
            '--num-samples', self.num_samples_var.get(),
            '--every', self.every_var.get(),
            '--max-frames', self.max_var.get(),
            '--aug-per-frame', self.aug_var.get(),
        ]
        self.run_cmd(cmd)

    def build_split(self):
        self.run_cmd([str(PY), 'scripts/build_dataset_split.py'])

    def update_yaml(self):
        classes = self.classes_var.get().strip().split()
        if not classes:
            self.log_line('No classes provided')
            return
        self.run_cmd([str(PY), 'scripts/update_dataset_yaml.py', '--classes', *classes])

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
        self.run_cmd(cmd)


if __name__ == '__main__':
    app = App()
    app.mainloop()
