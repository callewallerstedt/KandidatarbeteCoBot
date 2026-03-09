#!/usr/bin/env python3
import math


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / max(1.0, area_a + area_b - inter)


def _box_center(box):
    x1, y1, x2, y2 = box
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def update_tracks(tracks, next_track_id, detections, frame_shape, max_missed=2):
    height, width = frame_shape[:2]
    diag = math.hypot(width, height)
    assignments = {}
    used_tracks = set()
    used_dets = set()

    candidates = []
    for det_idx, det in enumerate(detections):
        dbox = det['box']
        dcx, dcy = _box_center(dbox)
        for track_id, track in tracks.items():
            tbox = track['box']
            iou = _box_iou(dbox, tbox)
            tcx, tcy = _box_center(tbox)
            dist = math.hypot(dcx - tcx, dcy - tcy) / max(1.0, diag)
            score = iou - 0.35 * dist
            if iou >= 0.08 or dist <= 0.08:
                candidates.append((score, det_idx, track_id))

    for score, det_idx, track_id in sorted(candidates, key=lambda x: x[0], reverse=True):
        if det_idx in used_dets or track_id in used_tracks:
            continue
        assignments[det_idx] = track_id
        used_dets.add(det_idx)
        used_tracks.add(track_id)

    new_tracks = {}
    for det_idx, det in enumerate(detections):
        track_id = assignments.get(det_idx)
        if track_id is None:
            track_id = next_track_id
            next_track_id += 1
        prev = tracks.get(track_id)
        if prev is not None:
            pbox = prev['box']
            box = (
                0.45 * pbox[0] + 0.55 * det['box'][0],
                0.45 * pbox[1] + 0.55 * det['box'][1],
                0.45 * pbox[2] + 0.55 * det['box'][2],
                0.45 * pbox[3] + 0.55 * det['box'][3],
            )
        else:
            box = det['box']
        new_tracks[track_id] = {
            'id': track_id,
            'box': box,
            'poly': det.get('poly'),
            'ttl': max_missed,
            'matched': True,
        }

    for track_id, track in tracks.items():
        if track_id in new_tracks:
            continue
        ttl = int(track.get('ttl', max_missed)) - 1
        if ttl >= 0:
            new_tracks[track_id] = {
                'id': track_id,
                'box': track['box'],
                'poly': track.get('poly'),
                'ttl': ttl,
                'matched': False,
            }

    ordered = sorted(new_tracks.items(), key=lambda item: (not item[1]['matched'], item[0]))
    return dict(ordered), next_track_id
