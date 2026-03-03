#!/usr/bin/env python3
import argparse, json
from pathlib import Path

IMG_EXTS={'.jpg','.jpeg','.png','.webp','.bmp'}


def pick_dirs(root: Path):
    rgb=mask=ann=None
    for d in root.iterdir():
        if not d.is_dir(): continue
        n=d.name.lower()
        if rgb is None and n in {'rgb','images','image','color','render','renders','camera','frames'}: rgb=d
        if mask is None and n in {'mask','masks','seg','segmentation','label','labels','redmask','red_masks'}: mask=d
        if ann is None and n in {'annotations','annotation','ann','json'}: ann=d
    return rgb,mask,ann


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--unity-dir',required=True)
    args=ap.parse_args()
    u=Path(args.unity_dir)
    rgb,mask,ann=pick_dirs(u)
    if not u.exists(): raise RuntimeError('unity-dir not found')
    print(f'RGB dir: {rgb}')
    print(f'MASK dir: {mask}')
    print(f'ANN dir: {ann}')
    rgb_n=mask_n=ann_n=0
    if rgb and rgb.exists(): rgb_n=sum(1 for p in rgb.rglob('*') if p.suffix.lower() in IMG_EXTS)
    if mask and mask.exists(): mask_n=sum(1 for p in mask.rglob('*') if p.suffix.lower() in IMG_EXTS)
    if ann and ann.exists(): ann_n=len(list(ann.glob('*.json')))
    print(f'RGB files: {rgb_n}')
    print(f'MASK files: {mask_n}')
    print(f'ANN json files: {ann_n}')
    sample=None
    if ann and ann.exists():
        js=sorted(ann.glob('*.json'))
        if js:
            sample=js[0]
            d=json.loads(sample.read_text(encoding='utf-8'))
            print(f'Sample json: {sample.name}')
            print(f'Keys: {list(d.keys())}')
            objs=d.get('objects',[])
            if objs:
                print(f'First object keys: {list(objs[0].keys())}')

if __name__=='__main__': main()
