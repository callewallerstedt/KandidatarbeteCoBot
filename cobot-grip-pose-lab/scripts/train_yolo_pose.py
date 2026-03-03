#!/usr/bin/env python3
import argparse
from ultralytics import YOLO

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='yolo11s-pose.pt')
    ap.add_argument('--data',default='dataset/dataset.yaml')
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--imgsz',type=int,default=960)
    ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--device',default='0')
    ap.add_argument('--project',default='runs/pose')
    ap.add_argument('--name',default='train')
    args=ap.parse_args()
    YOLO(args.model).train(data=args.data,epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,device=args.device,project=args.project,name=args.name,task='pose')

if __name__=='__main__': main()
