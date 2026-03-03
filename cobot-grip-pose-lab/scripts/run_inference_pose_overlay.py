#!/usr/bin/env python3
import argparse, math
from ultralytics import YOLO
import cv2


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights',required=True)
    ap.add_argument('--source',default='0')
    ap.add_argument('--imgsz',type=int,default=960)
    ap.add_argument('--conf',type=float,default=0.25)
    ap.add_argument('--device',default='0')
    args=ap.parse_args()

    model=YOLO(args.weights)
    source=0 if args.source=='0' else args.source
    for r in model.predict(source=source,stream=True,imgsz=args.imgsz,conf=args.conf,device=args.device):
        frame=r.orig_img.copy()
        if r.keypoints is not None:
            k=r.keypoints.xy.cpu().numpy()
            for p in k:
                if len(p)<3: continue
                c=tuple(map(int,p[0])); a=tuple(map(int,p[1])); b=tuple(map(int,p[2]))
                cv2.line(frame,a,b,(0,255,255),2,cv2.LINE_AA)
                cv2.circle(frame,c,4,(255,255,255),-1)
                cv2.circle(frame,a,5,(0,255,0),-1)
                cv2.circle(frame,b,5,(0,0,255),-1)
                ang=math.degrees(math.atan2(b[1]-a[1],b[0]-a[0]))
                cv2.putText(frame,f'grip_angle={ang:.1f}',(c[0]+8,c[1]-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('Grip Pose (q quit)',frame)
        if (cv2.waitKey(1)&0xFF)==ord('q'):
            break
    cv2.destroyAllWindows()

if __name__=='__main__': main()
