#!/usr/bin/env python3
import argparse, json, random, shutil
from pathlib import Path

IMG_EXTS={'.jpg','.jpeg','.png','.webp','.bmp'}

def norm(v,m): return max(0.0,min(1.0,float(v)/max(1.0,float(m))))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--unity-dir',required=True)
    ap.add_argument('--out-dir',default='dataset')
    ap.add_argument('--train-ratio',type=float,default=0.9)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()

    u=Path(args.unity_dir)
    rgb=u/'RGB'; ann=u/'annotations'
    if not rgb.exists() or not ann.exists():
        raise RuntimeError('unity-dir must contain RGB/ and annotations/')

    frames=[]
    for jp in sorted(ann.glob('*.json')):
        d=json.loads(jp.read_text(encoding='utf-8'))
        img=(rgb/d['image'])
        if not img.exists():
            continue
        frames.append((img,d))

    if not frames: raise RuntimeError('No frame annotations found')
    random.seed(args.seed); random.shuffle(frames)
    ntr=int(len(frames)*args.train_ratio)
    splits={'train':frames[:ntr],'val':frames[ntr:]}

    out=Path(args.out_dir)
    for s,items in splits.items():
        (out/'images'/s).mkdir(parents=True,exist_ok=True)
        (out/'labels'/s).mkdir(parents=True,exist_ok=True)
        for i,(img,d) in enumerate(items,1):
            name=f'frame_{i:06d}'
            shutil.copy2(img,out/'images'/s/f'{name}{img.suffix.lower()}')
            w,h=d['width'],d['height']
            lines=[]
            for o in d.get('objects',[]):
                c=int(o.get('class_id',0))
                x1,y1,x2,y2=o['bbox_xyxy']
                cx,cy=(x1+x2)/2,(y1+y2)/2
                bw,bh=(x2-x1),(y2-y1)
                k1=o['center']; k2=o['grip_a']; k3=o['grip_b']
                vals=[c,norm(cx,w),norm(cy,h),norm(bw,w),norm(bh,h),
                      norm(k1[0],w),norm(k1[1],h),int(k1[2]),
                      norm(k2[0],w),norm(k2[1],h),int(k2[2]),
                      norm(k3[0],w),norm(k3[1],h),int(k3[2])]
                lines.append(' '.join(str(v) for v in vals))
            (out/'labels'/s/f'{name}.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    yml=out/'dataset.yaml'
    yml.write_text(
"""path: dataset
train: images/train
val: images/val
kpt_shape: [3, 3]
flip_idx: [0, 2, 1]
names:
  0: object
""",encoding='utf-8')
    print(f'Converted {len(frames)} frames -> {out}')

if __name__=='__main__': main()
