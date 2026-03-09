#!/usr/bin/env python3
import numpy as np


def select_pickable_instance(boxes_xyxy, frame_shape):
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None, []

    height, width = frame_shape[:2]
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    infos = []
    areas = np.maximum(1.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    max_area = float(np.max(areas)) if len(areas) else 1.0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        area = areas[i]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        border_dist = min(x1, y1, max(0.0, width - x2), max(0.0, height - y2))
        edge_margin = float(np.clip(border_dist / max(12.0, 0.10 * max(width, height)), 0.0, 1.0))

        crowding = 0.0
        contact_count = 0
        for j, other in enumerate(boxes):
            if i == j:
                continue
            ox1, oy1, ox2, oy2 = [float(v) for v in other]
            inter_w = max(0.0, min(x2, ox2) - max(x1, ox1))
            inter_h = max(0.0, min(y2, oy2) - max(y1, oy1))
            inter = inter_w * inter_h
            if inter <= 0.0:
                continue
            crowding += inter / area
            contact_count += 1

        area_score = min(1.0, area / max_area)
        isolation = 1.0 / (1.0 + 2.5 * crowding + 0.40 * contact_count)
        score = 1.40 * isolation + 0.45 * edge_margin + 0.15 * area_score
        infos.append({
            'score': float(score),
            'center': (int(round(cx)), int(round(cy))),
            'contact_count': int(contact_count),
            'edge_margin': float(edge_margin),
            'area': float(area),
        })

    best_idx = int(max(range(len(infos)), key=lambda idx: infos[idx]['score']))
    return best_idx, infos
