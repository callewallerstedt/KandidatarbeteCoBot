#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-yaml', default=str(Path(__file__).resolve().parents[1] / 'dataset.yaml'))
    ap.add_argument('--classes', nargs='+', required=True)
    args = ap.parse_args()

    p = Path(args.dataset_yaml)
    data = {
        'path': 'data/yolo_dataset',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: c for i, c in enumerate(args.classes)},
    }
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    print(f'Updated {p} with classes: {args.classes}')


if __name__ == '__main__':
    main()
