#!/usr/bin/env python3
import os
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('LCC → PLY Converter (splat-transform)')
        self.geometry('920x620')

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value='Ready')

        self.proc = None

        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill='both', expand=True)

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

        self.progress = ttk.Progressbar(frm, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=2, sticky='we', pady=(4, 8))

        ttk.Label(frm, text='Log').grid(row=6, column=0, sticky='w')
        self.log = tk.Text(frm, height=22, wrap='word')
        self.log.grid(row=7, column=0, columnspan=2, sticky='nsew')

        status = ttk.Label(frm, textvariable=self.status_var)
        status.grid(row=8, column=0, columnspan=2, sticky='w', pady=(8, 0))

        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(7, weight=1)

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
