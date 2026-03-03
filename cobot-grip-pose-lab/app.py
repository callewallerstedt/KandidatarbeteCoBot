#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog
import subprocess, threading, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PY=ROOT/'.venv'/('Scripts/python.exe' if sys.platform.startswith('win') else 'bin/python')
if not PY.exists(): PY=Path(sys.executable)

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('Cobot Grip Pose Lab'); self.geometry('980x700')
        nb=ttk.Notebook(self); nb.pack(fill='both',expand=True)
        self.t_data=ttk.Frame(nb); self.t_train=ttk.Frame(nb); self.t_infer=ttk.Frame(nb)
        nb.add(self.t_data,text='1) Unity -> Dataset'); nb.add(self.t_train,text='2) Train Pose'); nb.add(self.t_infer,text='3) Inference')
        self.log=tk.Text(self,height=10); self.log.pack(fill='x')
        self.build_data(); self.build_train(); self.build_infer()

    def run(self,cmd):
        self.log.insert('end','> '+' '.join(map(str,cmd))+'\n'); self.log.see('end')
        def t():
            p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for ln in p.stdout: self.log.insert('end',ln); self.log.see('end')
            self.log.insert('end',f'[exit {p.wait()}]\n')
        threading.Thread(target=t,daemon=True).start()

    def build_data(self):
        f=self.t_data
        self.unity=tk.StringVar(); self.train_ratio=tk.StringVar(value='0.9')
        ttk.Label(f,text='Unity export folder (RGB + annotations)').grid(row=0,column=0,sticky='w')
        ttk.Entry(f,textvariable=self.unity,width=80).grid(row=0,column=1,sticky='we')
        ttk.Button(f,text='Browse',command=lambda:self.unity.set(filedialog.askdirectory() or self.unity.get())).grid(row=0,column=2)
        ttk.Label(f,text='Train ratio').grid(row=1,column=0,sticky='w')
        ttk.Entry(f,textvariable=self.train_ratio,width=10).grid(row=1,column=1,sticky='w')
        ttk.Button(f,text='Convert to YOLO Pose dataset',command=self.convert).grid(row=2,column=0,pady=8,sticky='w')
        f.columnconfigure(1,weight=1)

    def build_train(self):
        f=self.t_train
        self.model=tk.StringVar(value='yolo11s-pose.pt'); self.epochs=tk.StringVar(value='100'); self.imgsz=tk.StringVar(value='960'); self.batch=tk.StringVar(value='16'); self.device=tk.StringVar(value='0')
        for i,(k,v) in enumerate([('Model',self.model),('Epochs',self.epochs),('Image size',self.imgsz),('Batch',self.batch),('Device',self.device)]):
            ttk.Label(f,text=k).grid(row=i,column=0,sticky='w'); ttk.Entry(f,textvariable=v).grid(row=i,column=1,sticky='we')
        ttk.Button(f,text='Start pose training',command=self.train).grid(row=5,column=0,pady=8,sticky='w')
        f.columnconfigure(1,weight=1)

    def build_infer(self):
        f=self.t_infer
        self.weights=tk.StringVar(value=str(ROOT/'runs/pose/train/weights/best.pt')); self.source=tk.StringVar(value='0'); self.ic=tk.StringVar(value='0.25'); self.ii=tk.StringVar(value='960'); self.id=tk.StringVar(value='0')
        ttk.Label(f,text='Weights').grid(row=0,column=0,sticky='w'); ttk.Entry(f,textvariable=self.weights,width=80).grid(row=0,column=1,sticky='we')
        ttk.Button(f,text='Browse',command=lambda:self.weights.set(filedialog.askopenfilename() or self.weights.get())).grid(row=0,column=2)
        ttk.Label(f,text='Source').grid(row=1,column=0,sticky='w'); ttk.Entry(f,textvariable=self.source).grid(row=1,column=1,sticky='we')
        ttk.Label(f,text='Conf').grid(row=2,column=0,sticky='w'); ttk.Entry(f,textvariable=self.ic).grid(row=2,column=1,sticky='we')
        ttk.Label(f,text='Image size').grid(row=3,column=0,sticky='w'); ttk.Entry(f,textvariable=self.ii).grid(row=3,column=1,sticky='we')
        ttk.Label(f,text='Device').grid(row=4,column=0,sticky='w'); ttk.Entry(f,textvariable=self.id).grid(row=4,column=1,sticky='we')
        ttk.Button(f,text='Run grip inference',command=self.infer).grid(row=5,column=0,pady=8,sticky='w')
        f.columnconfigure(1,weight=1)

    def convert(self):
        self.run([str(PY),'scripts/convert_unity_pose_json.py','--unity-dir',self.unity.get(),'--out-dir','dataset','--train-ratio',self.train_ratio.get()])
    def train(self):
        self.run([str(PY),'scripts/train_yolo_pose.py','--model',self.model.get(),'--epochs',self.epochs.get(),'--imgsz',self.imgsz.get(),'--batch',self.batch.get(),'--device',self.device.get()])
    def infer(self):
        self.run([str(PY),'scripts/run_inference_pose_overlay.py','--weights',self.weights.get(),'--source',self.source.get(),'--conf',self.ic.get(),'--imgsz',self.ii.get(),'--device',self.id.get()])

if __name__=='__main__':
    App().mainloop()
