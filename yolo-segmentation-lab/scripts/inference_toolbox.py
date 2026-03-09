#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import tkinter as tk


def write_command(cmd_path: Path, seq: int, command: str):
    cmd_path.write_text(json.dumps({'seq': seq, 'command': command}), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--command-file', required=True)
    ap.add_argument('--status-file', required=True)
    args = ap.parse_args()

    cmd_path = Path(args.command_file)
    status_path = Path(args.status_file)
    seq = 0

    root = tk.Tk()
    root.title('Inference Toolbox')
    root.geometry('280x170')
    root.attributes('-topmost', True)

    status_var = tk.StringVar(value='Running')

    def send(command):
        nonlocal seq
        seq += 1
        write_command(cmd_path, seq, command)

    frm = tk.Frame(root, padx=10, pady=10)
    frm.pack(fill='both', expand=True)
    tk.Button(frm, text='Pause / Resume', width=26, command=lambda: send('toggle_pause')).pack(pady=4)
    tk.Button(frm, text='Choose Object', width=26, command=lambda: send('choose_object')).pack(pady=4)
    tk.Button(frm, text='Clear Selection', width=26, command=lambda: send('clear_selection')).pack(pady=4)
    tk.Label(frm, textvariable=status_var, anchor='w', justify='left').pack(fill='x', pady=(8, 0))

    def refresh_status():
        try:
            if status_path.exists():
                data = json.loads(status_path.read_text(encoding='utf-8'))
                txt = str(data.get('status', '')).strip()
                if txt:
                    status_var.set(txt)
        except Exception:
            pass
        root.after(150, refresh_status)

    refresh_status()
    root.mainloop()


if __name__ == '__main__':
    main()
