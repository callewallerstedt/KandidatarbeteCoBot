#!/usr/bin/env python3
import os
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('LCC → PLY / PLY → Mesh Toolkit')
        self.geometry('980x700')

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value='Ready')

        # PLY->Mesh tab vars
        self.mesh_in_var = tk.StringVar()
        self.mesh_out_var = tk.StringVar()
        self.voxel_var = tk.StringVar(value='0.01')
        self.normal_radius_var = tk.StringVar(value='0.05')
        self.normal_nn_var = tk.StringVar(value='30')
        self.depth_var = tk.StringVar(value='9')
        self.trim_var = tk.StringVar(value='0.02')

        self.proc = None

        self._build_ui()

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill='both', expand=True)

        tabs = ttk.Notebook(root)
        tabs.pack(fill='both', expand=True)

        self.tab_lcc = ttk.Frame(tabs)
        self.tab_mesh = ttk.Frame(tabs)
        tabs.add(self.tab_lcc, text='LCC → PLY')
        tabs.add(self.tab_mesh, text='PLY → Mesh (Open3D)')

        # --- LCC tab ---
        frm = self.tab_lcc
        ttk.Label(frm, text='Input LCC file').grid(row=0, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.input_var, width=95).grid(row=1, column=0, sticky='we')
        ttk.Button(frm, text='Browse…', command=self.pick_input).grid(row=1, column=1, padx=(8, 0), sticky='w')

        ttk.Label(frm, text='Output PLY file').grid(row=2, column=0, pady=(12, 0), sticky='w')
        ttk.Entry(frm, textvariable=self.output_var, width=95).grid(row=3, column=0, sticky='we')
        ttk.Button(frm, text='Browse…', command=self.pick_output).grid(row=3, column=1, padx=(8, 0), sticky='w')

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=2, pady=(14, 8), sticky='w')
        self.run_btn = ttk.Button(btns, text='Convert LCC → PLY', command=self.start_convert)
        self.run_btn.pack(side='left')
        self.stop_btn = ttk.Button(btns, text='Stop', command=self.stop_convert, state='disabled')
        self.stop_btn.pack(side='left', padx=(8, 0))

        # --- Mesh tab ---
        mf = self.tab_mesh
        ttk.Label(mf, text='Input PLY point cloud').grid(row=0, column=0, sticky='w')
        ttk.Entry(mf, textvariable=self.mesh_in_var, width=95).grid(row=1, column=0, sticky='we')
        ttk.Button(mf, text='Browse…', command=self.pick_mesh_input).grid(row=1, column=1, padx=(8, 0), sticky='w')

        ttk.Label(mf, text='Output mesh file (.ply/.obj)').grid(row=2, column=0, pady=(12, 0), sticky='w')
        ttk.Entry(mf, textvariable=self.mesh_out_var, width=95).grid(row=3, column=0, sticky='we')
        ttk.Button(mf, text='Browse…', command=self.pick_mesh_output).grid(row=3, column=1, padx=(8, 0), sticky='w')

        params = ttk.Frame(mf)
        params.grid(row=4, column=0, columnspan=2, sticky='w', pady=(10, 4))
        ttk.Label(params, text='voxel').grid(row=0, column=0, sticky='w')
        ttk.Entry(params, textvariable=self.voxel_var, width=8).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(params, text='normal_radius').grid(row=0, column=2, sticky='w')
        ttk.Entry(params, textvariable=self.normal_radius_var, width=8).grid(row=0, column=3, padx=(4, 12))
        ttk.Label(params, text='normal_nn').grid(row=0, column=4, sticky='w')
        ttk.Entry(params, textvariable=self.normal_nn_var, width=8).grid(row=0, column=5, padx=(4, 12))
        ttk.Label(params, text='poisson_depth').grid(row=1, column=0, sticky='w')
        ttk.Entry(params, textvariable=self.depth_var, width=8).grid(row=1, column=1, padx=(4, 12))
        ttk.Label(params, text='trim_quantile').grid(row=1, column=2, sticky='w')
        ttk.Entry(params, textvariable=self.trim_var, width=8).grid(row=1, column=3, padx=(4, 12))

        self.mesh_btn = ttk.Button(mf, text='Convert PLY → Mesh', command=self.start_mesh_convert)
        self.mesh_btn.grid(row=5, column=0, pady=(8, 6), sticky='w')

        self.progress = ttk.Progressbar(root, mode='indeterminate')
        self.progress.pack(fill='x', pady=(8, 8))

        ttk.Label(root, text='Log').pack(anchor='w')
        self.log = tk.Text(root, height=16, wrap='word')
        self.log.pack(fill='both', expand=True)

        status = ttk.Label(root, textvariable=self.status_var)
        status.pack(anchor='w', pady=(8, 0))

        frm.columnconfigure(0, weight=1)
        mf.columnconfigure(0, weight=1)

        self.log_line('Tip: Keep all companion LCC files (data.bin/index.bin/assets) in same folder as .lcc')

    def log_line(self, text):
        self.log.insert('end', text + '\n')
        self.log.see('end')

    def pick_input(self):
        p = filedialog.askopenfilename(title='Select .lcc input', filetypes=[('LCC files', '*.lcc'), ('All files', '*.*')])
        if p:
            self.input_var.set(p)
            if not self.output_var.get().strip():
                base, _ = os.path.splitext(p)
                self.output_var.set(base + '_out.ply')

    def pick_output(self):
        p = filedialog.asksaveasfilename(title='Select output .ply', defaultextension='.ply', filetypes=[('PLY files', '*.ply'), ('All files', '*.*')])
        if p:
            self.output_var.set(p)

    def pick_mesh_input(self):
        p = filedialog.askopenfilename(title='Select input point cloud .ply', filetypes=[('PLY files', '*.ply'), ('All files', '*.*')])
        if p:
            self.mesh_in_var.set(p)
            if not self.mesh_out_var.get().strip():
                base, _ = os.path.splitext(p)
                self.mesh_out_var.set(base + '_mesh.ply')

    def pick_mesh_output(self):
        p = filedialog.asksaveasfilename(title='Select output mesh', defaultextension='.ply', filetypes=[('PLY mesh', '*.ply'), ('OBJ mesh', '*.obj'), ('All files', '*.*')])
        if p:
            self.mesh_out_var.set(p)

    def start_convert(self):
        if self.proc is not None:
            return

        in_path = self.input_var.get().strip().strip('"')
        out_path = self.output_var.get().strip().strip('"')
        if not in_path or not os.path.isfile(in_path):
            messagebox.showerror('Missing input', 'Please select a valid .lcc input file.')
            return
        if not out_path:
            messagebox.showerror('Missing output', 'Please select an output .ply path.')
            return

        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        cmd = ['npx', '@playcanvas/splat-transform', '-w', in_path, out_path]
        self.log_line('> ' + ' '.join(f'"{c}"' if ' ' in c else c for c in cmd))
        self.status_var.set('Converting...')
        self.run_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress.start(10)

        def worker():
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                )
                for line in self.proc.stdout:
                    self.log_line(line.rstrip())
                rc = self.proc.wait()
                self.log_line(f'[exit {rc}]')
                if rc == 0:
                    self.status_var.set('Done ✅')
                    self.log_line(f'Output: {out_path}')
                else:
                    self.status_var.set('Failed ❌')
            except Exception as e:
                self.log_line(f'Error: {e}')
                self.status_var.set('Failed ❌')
            finally:
                self.proc = None
                self.progress.stop()
                self.run_btn.config(state='normal')
                self.stop_btn.config(state='disabled')

        threading.Thread(target=worker, daemon=True).start()

    def start_mesh_convert(self):
        if self.proc is not None:
            return

        in_path = self.mesh_in_var.get().strip().strip('"')
        out_path = self.mesh_out_var.get().strip().strip('"')
        if not in_path or not os.path.isfile(in_path):
            messagebox.showerror('Missing input', 'Please select a valid input PLY point cloud.')
            return
        if not out_path:
            messagebox.showerror('Missing output', 'Please select an output mesh path.')
            return

        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        cmd = [
            'python', 'ply_to_mesh_open3d.py',
            '--input', in_path,
            '--output', out_path,
            '--voxel', self.voxel_var.get(),
            '--normal-radius', self.normal_radius_var.get(),
            '--normal-nn', self.normal_nn_var.get(),
            '--poisson-depth', self.depth_var.get(),
            '--trim-quantile', self.trim_var.get(),
        ]
        self.log_line('> ' + ' '.join(f'"{c}"' if ' ' in c else c for c in cmd))
        self.status_var.set('Meshing...')
        self.run_btn.config(state='disabled')
        self.mesh_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress.start(10)

        def worker():
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=os.path.dirname(__file__) or None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                )
                for line in self.proc.stdout:
                    self.log_line(line.rstrip())
                rc = self.proc.wait()
                self.log_line(f'[exit {rc}]')
                if rc == 0:
                    self.status_var.set('Done ✅')
                    self.log_line(f'Output: {out_path}')
                else:
                    self.status_var.set('Failed ❌')
            except Exception as e:
                self.log_line(f'Error: {e}')
                self.status_var.set('Failed ❌')
            finally:
                self.proc = None
                self.progress.stop()
                self.run_btn.config(state='normal')
                self.mesh_btn.config(state='normal')
                self.stop_btn.config(state='disabled')

        threading.Thread(target=worker, daemon=True).start()

    def stop_convert(self):
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.log_line('Termination requested...')
            self.status_var.set('Stopping...')
        except Exception as e:
            self.log_line(f'Stop error: {e}')


if __name__ == '__main__':
    App().mainloop()
