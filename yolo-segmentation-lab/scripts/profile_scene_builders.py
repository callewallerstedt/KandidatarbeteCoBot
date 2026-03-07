#!/usr/bin/env python3
import json
import re
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
_VARIANT_CACHE = {}
_VARIANT_CACHE_MAX = 2048


def next_output_index(out_dir: Path, stem_prefix: str):
    if not out_dir.exists():
        return 1
    pat = re.compile(rf'^{re.escape(stem_prefix)}_(\d+)$', re.IGNORECASE)
    max_idx = 0
    for path in out_dir.iterdir():
        if not path.is_file():
            continue
        m = pat.match(path.stem)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def load_profile(profile_path: Path):
    if not profile_path.exists():
        raise RuntimeError(f'Profile not found: {profile_path}')
    data = json.loads(profile_path.read_text(encoding='utf-8-sig'))
    items = data.get('items')
    if not isinstance(items, dict):
        raise RuntimeError('Invalid placement profile: missing items object')
    return data


def resolve_profile_item(profile_data, background_dir: Path, bg_path: Path):
    items = profile_data.get('items', {})
    rel_key = None
    abs_key = str(bg_path).replace('\\', '/')
    try:
        rel_key = str(bg_path.relative_to(background_dir)).replace('\\', '/')
    except Exception:
        pass

    if rel_key and rel_key in items:
        return items[rel_key]
    if abs_key in items:
        return items[abs_key]
    if bg_path.name in items:
        return items[bg_path.name]

    rel_low = rel_key.lower() if rel_key else None
    abs_low = abs_key.lower()
    name_low = bg_path.name.lower()
    for key, value in items.items():
        norm = str(key).replace('\\', '/').lower()
        if rel_low and (norm == rel_low or norm.endswith('/' + rel_low)):
            return value
        if norm == abs_low or norm.endswith('/' + name_low):
            return value
    return None


def parse_first_polygon(label_path: Path, width: int, height: int):
    text = label_path.read_text(encoding='utf-8', errors='ignore').strip()
    if not text:
        return None
    for line in text.splitlines():
        vals = line.strip().split()
        if len(vals) < 7 or (len(vals) - 1) % 2 != 0:
            continue
        pts = np.array([float(x) for x in vals[1:]], dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= width
        pts[:, 1] *= height
        return pts
    return None


def polygon_to_mask(poly: np.ndarray, width: int, height: int):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask


def rotate_image_mask(image, mask, angle_deg):
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    mat[0, 2] += (nw / 2.0) - cx
    mat[1, 2] += (nh / 2.0) - cy
    rot_img = cv2.warpAffine(image, mat, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rot_mask = cv2.warpAffine(mask, mat, (nw, nh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rot_img, rot_mask


def contour_from_mask(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    contour = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.0008 * peri, True)
    if len(approx) < 3:
        return None
    return approx.reshape(-1, 2)


def point_in_poly(x, y, poly_pts):
    if poly_pts is None or len(poly_pts) < 3:
        return True
    contour = np.array(poly_pts, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def bbox_fully_inside_poly(px, py, ow, oh, poly_pts):
    if poly_pts is None or len(poly_pts) < 3:
        return True
    corners = [(px, py), (px + ow, py), (px, py + oh), (px + ow, py + oh)]
    return all(point_in_poly(cx, cy, poly_pts) for cx, cy in corners)


def load_realism_preferences(data_root: Path, class_name: str):
    pref_path = data_root / 'preferences' / f'{class_name}_realism.json'
    if not pref_path.exists():
        return {}
    try:
        loaded = json.loads(pref_path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    scores = loaded.get('scores', {})
    return scores if isinstance(scores, dict) else {}


def realism_score_to_weight(score):
    try:
        return float(np.clip(1.0 + 0.35 * float(score), 0.15, 6.0))
    except Exception:
        return 1.0


def weighted_choice(items):
    if not items:
        raise RuntimeError('weighted_choice called with empty items')
    weights = [max(0.001, float(item.get('weight', 1.0))) for item in items]
    return random.choices(items, weights=weights, k=1)[0]


def collect_cutouts(data_root: Path, class_name: str):
    image_root = data_root / 'images' / class_name
    label_root = data_root / 'labels' / class_name
    if not image_root.exists() or not label_root.exists():
        raise RuntimeError(f'Missing source data for class={class_name}')

    preference_scores = load_realism_preferences(data_root, class_name)
    cutouts = []
    for image_path in sorted(image_root.rglob('*')):
        if image_path.suffix.lower() not in IMG_EXTS:
            continue
        rel = image_path.relative_to(image_root)
        if any(tag in rel.parts for tag in ['synth_runs', 'obs_runs', 'synth_multi_runs']):
            continue
        label_path = (label_root / rel).with_suffix('.txt')
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        poly = parse_first_polygon(label_path, w, h)
        if poly is None:
            continue
        mask = polygon_to_mask(poly, w, h)
        x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
        crop = image[y:y + bh, x:x + bw].copy()
        crop_mask = mask[y:y + bh, x:x + bw].copy()
        if crop.size == 0 or int((crop_mask > 0).sum()) < 20:
            continue
        rel_key = str(rel).replace('\\', '/')
        cutouts.append({
            'image': crop,
            'mask': crop_mask,
            'long_side': float(max(crop.shape[0], crop.shape[1])),
            'source': image_path.name,
            'source_rel': rel_key,
            'weight': realism_score_to_weight(preference_scores.get(rel_key, 0.0)),
            'cache_id': rel_key,
        })

    if not cutouts:
        raise RuntimeError(f'No valid cutouts found for class={class_name}')
    return cutouts


def _cache_put(key, value):
    if len(_VARIANT_CACHE) >= _VARIANT_CACHE_MAX:
        try:
            _VARIANT_CACHE.pop(next(iter(_VARIANT_CACHE)))
        except Exception:
            _VARIANT_CACHE.clear()
    _VARIANT_CACHE[key] = value


def prepare_cutout_variant(cutout, normalized_scale, angle):
    tw = max(8, int(cutout['image'].shape[1] * normalized_scale))
    th = max(8, int(cutout['image'].shape[0] * normalized_scale))
    angle_q = round(float(angle) / 6.0) * 6.0
    key = (cutout.get('cache_id', cutout.get('source', '')), tw, th, angle_q)
    cached = _VARIANT_CACHE.get(key)
    if cached is not None:
        return cached

    obj_img = cv2.resize(cutout['image'], (tw, th), interpolation=cv2.INTER_LINEAR)
    obj_mask = cv2.resize(cutout['mask'], (tw, th), interpolation=cv2.INTER_NEAREST)
    obj_img, obj_mask = rotate_image_mask(obj_img, obj_mask, angle_q)
    ys, xs = np.where(obj_mask > 0)
    if len(xs) < 20:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    obj_img = obj_img[y1:y2 + 1, x1:x2 + 1]
    obj_mask = obj_mask[y1:y2 + 1, x1:x2 + 1]
    variant = {
        'image': obj_img,
        'mask': obj_mask,
        'mask_bin': (obj_mask > 0),
        'area': int((obj_mask > 0).sum()),
        'shape': obj_mask.shape[:2],
    }
    _cache_put(key, variant)
    return variant


def choose_effective_settings(item, class_name: str):
    min_scale = float(item.get('min_scale', 0.55))
    max_scale = float(item.get('max_scale', 1.25))
    bg_min = float(item.get('bg_brightness_min', 0.0))
    bg_max = float(item.get('bg_brightness_max', 0.0))
    obj_min = float(item.get('obj_brightness_min', 0.0))
    obj_max = float(item.get('obj_brightness_max', 0.0))

    cset = item.get('class_settings', {})
    if class_name and isinstance(cset, dict):
        cls = cset.get(class_name)
        if cls is None:
            want = class_name.strip().lower()
            for key, value in cset.items():
                if str(key).strip().lower() == want:
                    cls = value
                    break
        if isinstance(cls, dict):
            min_scale = float(cls.get('min_scale', min_scale))
            max_scale = float(cls.get('max_scale', max_scale))
            bg_min = float(cls.get('bg_brightness_min', bg_min))
            bg_max = float(cls.get('bg_brightness_max', bg_max))
            obj_min = float(cls.get('obj_brightness_min', obj_min))
            obj_max = float(cls.get('obj_brightness_max', obj_max))

    poly_norm = None
    poly = item.get('poly')
    if isinstance(poly, list) and len(poly) >= 3:
        poly_norm = poly

    return {
        'min_scale': min_scale,
        'max_scale': max_scale,
        'bg_brightness_min': bg_min,
        'bg_brightness_max': bg_max,
        'obj_brightness_min': obj_min,
        'obj_brightness_max': obj_max,
        'poly': poly_norm,
    }


def pick_random_background(bg_files):
    bg_path = random.choice(bg_files)
    bg = cv2.imread(str(bg_path))
    return bg, bg_path


def choose_requested_scale(preview_mode: str, effective):
    if preview_mode == 'min_scale':
        return min(effective['min_scale'], effective['max_scale'])
    if preview_mode == 'max_scale':
        return max(effective['min_scale'], effective['max_scale'])
    return random.uniform(min(effective['min_scale'], effective['max_scale']), max(effective['min_scale'], effective['max_scale']))


def choose_background_beta(preview_mode: str, effective):
    if preview_mode == 'bg_bri_min':
        return effective['bg_brightness_min']
    if preview_mode == 'bg_bri_max':
        return effective['bg_brightness_max']
    return random.uniform(effective['bg_brightness_min'], effective['bg_brightness_max'])


def sample_position_in_polygon(w, h, ow, oh, poly_px, tries=80):
    if poly_px is None or len(poly_px) < 3:
        return random.randint(0, max(0, w - ow)), random.randint(0, max(0, h - oh))

    xs = [p[0] for p in poly_px]
    ys = [p[1] for p in poly_px]
    min_px = max(0, min(xs))
    min_py = max(0, min(ys))
    max_px = min(w - ow, max(xs))
    max_py = min(h - oh, max(ys))
    if max_px < min_px or max_py < min_py:
        return None

    for _ in range(max(1, tries)):
        px = random.randint(min_px, max_px)
        py = random.randint(min_py, max_py)
        if bbox_fully_inside_poly(px, py, ow, oh, poly_px):
            return px, py
    return None


def sample_scattered_position(w, h, ow, oh, poly_px, centers, tries=80):
    if not centers:
        return sample_position_in_polygon(w, h, ow, oh, poly_px, tries=tries)

    best = None
    best_score = -1.0
    for _ in range(max(1, tries)):
        pos = sample_position_in_polygon(w, h, ow, oh, poly_px, tries=4)
        if pos is None:
            continue
        px, py = pos
        cx_new = px + ow // 2
        cy_new = py + oh // 2
        nearest = min((((cx_new - cx) ** 2 + (cy_new - cy) ** 2) ** 0.5) for cx, cy in centers)
        edge_margin = min(cx_new, cy_new, w - cx_new, h - cy_new)
        score = nearest + 0.08 * edge_margin
        if score > best_score:
            best_score = score
            best = (px, py)
    return best


def _clip_uint8(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def _masked_mean_std(img, alpha):
    flat_img = img.reshape(-1, img.shape[2]).astype(np.float32)
    flat_w = np.clip(alpha.reshape(-1, 1).astype(np.float32), 0.0, 1.0)
    total = float(flat_w.sum())
    if total <= 1e-6:
        mean = flat_img.mean(axis=0)
        std = flat_img.std(axis=0)
        return mean.astype(np.float32), np.maximum(std, 1.0).astype(np.float32)
    mean = (flat_img * flat_w).sum(axis=0) / total
    var = ((flat_img - mean) ** 2 * flat_w).sum(axis=0) / total
    return mean.astype(np.float32), np.sqrt(np.maximum(var, 1.0)).astype(np.float32)


def _sample_background_patch(canvas, px, py, ow, oh):
    pad_x = max(6, ow // 8)
    pad_y = max(6, oh // 8)
    x1 = max(0, px - pad_x)
    y1 = max(0, py - pad_y)
    x2 = min(canvas.shape[1], px + ow + pad_x)
    y2 = min(canvas.shape[0], py + oh + pad_y)
    return canvas[y1:y2, x1:x2].copy()


def _scene_illumination_stats(canvas):
    canvas_f = canvas.astype(np.float32)
    mean_bgr = canvas_f.reshape(-1, 3).mean(axis=0)
    lab = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_lab = lab.reshape(-1, 3).mean(axis=0)
    std_lab = np.maximum(lab.reshape(-1, 3).std(axis=0), np.array([6.0, 2.0, 2.0], dtype=np.float32))
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV).astype(np.float32)
    mean_sat = float(hsv[:, :, 1].mean())
    return {
        'mean_bgr': mean_bgr.astype(np.float32),
        'mean_lab': mean_lab.astype(np.float32),
        'std_lab': std_lab.astype(np.float32),
        'mean_sat': mean_sat,
    }


def _object_neutrality(obj_img, obj_mask):
    alpha = np.clip(obj_mask.astype(np.float32) / 255.0, 0.0, 1.0)
    hsv = cv2.cvtColor(obj_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    flat_s = hsv[:, :, 1].reshape(-1)
    flat_w = alpha.reshape(-1)
    total = float(flat_w.sum())
    if total <= 1e-6:
        sat = float(flat_s.mean())
    else:
        sat = float((flat_s * flat_w).sum() / total)
    return 1.0 - np.clip(sat / 90.0, 0.0, 1.0)


def _apply_temperature_variation(obj_img, obj_mask, bias, variance):
    neutrality = _object_neutrality(obj_img, obj_mask)
    t = float(bias) + random.uniform(-float(variance), float(variance))
    t *= 0.60 + 0.40 * neutrality

    out = obj_img.astype(np.float32)
    # Primary temperature move in BGR for a clear visible difference.
    out[:, :, 2] *= 1.0 + 0.22 * t
    out[:, :, 0] *= 1.0 - 0.22 * t
    out[:, :, 1] *= 1.0 + 0.05 * t
    out = _clip_uint8(out)

    # Secondary move in Lab b-channel to shift blue-yellow temperature perceptually.
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 2] += 18.0 * t
    if neutrality > 0.45:
        lab[:, :, 1] += 5.0 * t
    return cv2.cvtColor(_clip_uint8(lab), cv2.COLOR_LAB2BGR)


def _apply_partial_object_shading(obj_img, obj_mask, shade_prob, shade_strength):
    if random.random() >= float(shade_prob):
        return obj_img

    h, w = obj_mask.shape[:2]
    if h < 6 or w < 6:
        return obj_img

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = xx / max(1.0, float(w - 1))
    yy = yy / max(1.0, float(h - 1))

    angle = random.uniform(-0.55 * np.pi, 0.55 * np.pi)
    direction = np.cos(angle) * xx + np.sin(angle) * yy
    direction = (direction - direction.min()) / max(1e-6, float(direction.max() - direction.min()))

    if random.random() < 0.35:
        cx = random.uniform(0.25, 0.75)
        cy = random.uniform(0.25, 0.75)
        radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        radial = radial / max(1e-6, float(radial.max()))
        direction = 0.7 * direction + 0.3 * radial

    soft = cv2.GaussianBlur(direction.astype(np.float32), (0, 0), sigmaX=max(2.0, 0.18 * w), sigmaY=max(2.0, 0.18 * h))
    soft = (soft - soft.min()) / max(1e-6, float(soft.max() - soft.min()))
    strength = random.uniform(0.45, 1.0) * float(shade_strength)
    shadow = 1.0 - strength * soft

    alpha = np.clip(obj_mask.astype(np.float32) / 255.0, 0.0, 1.0)
    shadow = 1.0 - (1.0 - shadow) * alpha

    lab = cv2.cvtColor(obj_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] *= shadow
    return cv2.cvtColor(_clip_uint8(lab), cv2.COLOR_LAB2BGR)


def _match_object_to_background(obj_img, obj_mask, scene_stats, bg_patch, beta_min, beta_max):
    alpha = np.clip(obj_mask.astype(np.float32) / 255.0, 0.0, 1.0)
    if bg_patch.size == 0:
        bg_patch = obj_img
    neutrality = _object_neutrality(obj_img, obj_mask)

    obj_lab = cv2.cvtColor(obj_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    obj_mean_lab, obj_std_lab = _masked_mean_std(obj_lab, alpha)
    patch_lab = cv2.cvtColor(bg_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    patch_mean_lab = patch_lab.reshape(-1, 3).mean(axis=0).astype(np.float32)

    target_l_mean = 0.78 * scene_stats['mean_lab'][0] + 0.22 * patch_mean_lab[0]
    target_l_std = 0.70 * scene_stats['std_lab'][0] + 0.30 * max(float(patch_lab.reshape(-1, 3).std(axis=0)[0]), 6.0)

    scene_ab = scene_stats['mean_lab'][1:3]
    patch_ab = patch_mean_lab[1:3]
    chroma_pull = 0.10 + (1.0 - neutrality) * 0.12
    target_ab = obj_mean_lab[1:3] * (1.0 - chroma_pull) + (0.75 * scene_ab + 0.25 * patch_ab) * chroma_pull

    out_lab = obj_lab.copy()
    out_lab[:, :, 0] = (out_lab[:, :, 0] - obj_mean_lab[0]) * (0.72 * target_l_std / max(obj_std_lab[0], 6.0)) + target_l_mean
    out_lab[:, :, 0] += random.uniform(beta_min, beta_max) * 0.45
    out_lab[:, :, 1] = (out_lab[:, :, 1] - obj_mean_lab[1]) * (1.0 - 0.10 * neutrality) + target_ab[0]
    out_lab[:, :, 2] = (out_lab[:, :, 2] - obj_mean_lab[2]) * (1.0 - 0.10 * neutrality) + target_ab[1]

    out = cv2.cvtColor(_clip_uint8(out_lab), cv2.COLOR_LAB2BGR).astype(np.float32)
    wb = scene_stats['mean_bgr'] / max(float(scene_stats['mean_bgr'].mean()), 1.0)
    out *= (0.92 + 0.08 * wb)[None, None, :]
    out *= random.uniform(0.96, 1.04)

    hsv = cv2.cvtColor(_clip_uint8(out), cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_scale = 0.96 + min(0.08, scene_stats['mean_sat'] / 2550.0)
    hsv[:, :, 1] *= sat_scale * random.uniform(0.97, 1.03)
    hsv[:, :, 2] *= random.uniform(0.97, 1.04)
    return cv2.cvtColor(_clip_uint8(hsv), cv2.COLOR_HSV2BGR)


def _apply_floor_bounce_and_wrap(obj_img, obj_mask, bg_patch):
    if bg_patch.size == 0:
        return obj_img
    out = obj_img.astype(np.float32)
    alpha = np.clip(obj_mask.astype(np.float32) / 255.0, 0.0, 1.0)
    bg_mean = bg_patch.reshape(-1, 3).mean(axis=0).astype(np.float32)

    h, w = obj_mask.shape[:2]
    vertical = np.repeat(np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None], w, axis=1)
    bounce = (vertical ** 1.8) * alpha
    out += bounce[:, :, None] * (bg_mean[None, None, :] * random.uniform(0.06, 0.16))

    edge = cv2.GaussianBlur(alpha, (5, 5), 0) - cv2.GaussianBlur(alpha, (13, 13), 0)
    edge = np.clip(edge, 0.0, 1.0)
    out += edge[:, :, None] * bg_mean[None, None, :] * (random.uniform(6.0, 16.0) / 255.0)
    return _clip_uint8(out)


def apply_object_appearance_variation(obj_img, obj_mask, canvas, px, py, beta_min, beta_max, scene_depth=0.5, scene_stats=None, temp_bias=0.35, temp_variance=0.18, shade_prob=0.45, shade_strength=0.22):
    bg_patch = _sample_background_patch(canvas, px, py, obj_img.shape[1], obj_img.shape[0])
    if scene_stats is None:
        scene_stats = _scene_illumination_stats(canvas)
    out = _match_object_to_background(obj_img, obj_mask, scene_stats, bg_patch, beta_min, beta_max)
    out = _apply_temperature_variation(out, obj_mask, temp_bias, temp_variance)
    out = _apply_partial_object_shading(out, obj_mask, shade_prob, shade_strength)
    out = _apply_floor_bounce_and_wrap(out, obj_mask, bg_patch)

    if random.random() < 0.22:
        out = cv2.GaussianBlur(out, (random.choice([3, 5]), random.choice([3, 5])), 0)

    if random.random() < 0.35:
        noise = np.random.normal(0.0, random.uniform(1.5, 5.0), out.shape).astype(np.float32)
        out = _clip_uint8(out.astype(np.float32) + noise)

    if random.random() < 0.18:
        shade = np.ones(out.shape[:2], dtype=np.float32)
        axis = np.linspace(random.uniform(0.86, 0.98), random.uniform(0.98, 1.04), out.shape[1], dtype=np.float32)
        shade *= axis[None, :]
        out = _clip_uint8(out.astype(np.float32) * shade[:, :, None])

    if scene_depth > 0.62 and random.random() < 0.28:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def apply_scene_camera_variation(canvas):
    out = canvas.copy()
    if random.random() < 0.24:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    if random.random() < 0.40:
        noise = np.random.normal(0.0, random.uniform(1.0, 4.0), out.shape).astype(np.float32)
        out = _clip_uint8(out.astype(np.float32) + noise)

    if random.random() < 0.26:
        quality = int(random.uniform(68, 92))
        ok, buf = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if decoded is not None:
                out = decoded

    if random.random() < 0.38:
        gains = np.array([
            random.uniform(0.95, 1.05),
            random.uniform(0.96, 1.04),
            random.uniform(0.95, 1.05),
        ], dtype=np.float32)
        out = _clip_uint8(out.astype(np.float32) * gains[None, None, :])

    if random.random() < 0.18:
        ksize = random.choice([5, 7, 9])
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        if random.random() < 0.5:
            kernel[ksize // 2, :] = 1.0 / ksize
        else:
            kernel[:, ksize // 2] = 1.0 / ksize
        out = cv2.filter2D(out, -1, kernel)

    if random.random() < 0.35:
        h, w = out.shape[:2]
        yy = np.linspace(0.94, 1.04, h, dtype=np.float32)[:, None]
        xx = np.linspace(random.uniform(0.97, 1.01), random.uniform(0.99, 1.03), w, dtype=np.float32)[None, :]
        out = _clip_uint8(out.astype(np.float32) * (yy * xx)[:, :, None])
    return out


def soften_mask(mask):
    if mask is None:
        return None
    out = mask.copy()
    if random.random() < 0.40:
        k = np.ones((3, 3), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if random.random() < 0.50:
        blur_k = random.choice([3, 5, 7])
        out = cv2.GaussianBlur(out, (blur_k, blur_k), 0)
    return out


def alpha_from_mask(mask):
    soft = soften_mask(mask)
    alpha = np.clip(soft.astype(np.float32) / 255.0, 0.0, 1.0)
    if random.random() < 0.35:
        alpha = np.clip(alpha * random.uniform(0.90, 1.0), 0.0, 1.0)
    return alpha


def add_object_shadow(canvas, alpha_mask, px, py):
    oh, ow = alpha_mask.shape[:2]
    shadow = np.zeros(canvas.shape[:2], dtype=np.float32)

    shift_x = int(random.uniform(0.02, 0.10) * ow) * random.choice([-1, 1])
    shift_y = int(random.uniform(0.04, 0.12) * oh)
    x1 = max(0, px + shift_x)
    y1 = max(0, py + shift_y)
    x2 = min(canvas.shape[1], x1 + ow)
    y2 = min(canvas.shape[0], y1 + oh)
    sx1 = max(0, -(px + shift_x))
    sy1 = max(0, -(py + shift_y))
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    if x2 <= x1 or y2 <= y1:
        return canvas

    shadow[y1:y2, x1:x2] = np.maximum(shadow[y1:y2, x1:x2], alpha_mask[sy1:sy2, sx1:sx2])
    blur_k = random.choice([9, 13, 17, 21])
    shadow = cv2.GaussianBlur(shadow, (blur_k, blur_k), 0)
    strength = random.uniform(0.10, 0.28)

    out = canvas.astype(np.float32)
    shadow3 = shadow[:, :, None] * strength
    out = out * (1.0 - shadow3)
    return _clip_uint8(out)


def add_contact_shadow(canvas, alpha_mask, px, py):
    oh, ow = alpha_mask.shape[:2]
    shadow = np.zeros(canvas.shape[:2], dtype=np.float32)
    contact = np.clip(alpha_mask.astype(np.float32), 0.0, 1.0)
    contact *= np.linspace(0.35, 1.0, oh, dtype=np.float32)[:, None]
    contact = cv2.GaussianBlur(contact, (11, 11), 0)
    x1 = max(0, px)
    y1 = max(0, py)
    x2 = min(canvas.shape[1], px + ow)
    y2 = min(canvas.shape[0], py + oh)
    if x2 <= x1 or y2 <= y1:
        return canvas
    shadow[y1:y2, x1:x2] = np.maximum(shadow[y1:y2, x1:x2], contact[: y2 - y1, : x2 - x1])
    out = canvas.astype(np.float32)
    out *= (1.0 - shadow[:, :, None] * random.uniform(0.12, 0.24))
    return _clip_uint8(out)


def composite_object(canvas, obj_img, obj_mask, px, py, scene_depth=0.5):
    oh, ow = obj_mask.shape[:2]
    alpha = alpha_from_mask(obj_mask)
    canvas = add_contact_shadow(canvas, alpha, px, py)
    canvas = add_object_shadow(canvas, alpha, px, py)

    roi = canvas[py:py + oh, px:px + ow].astype(np.float32)
    objf = obj_img.astype(np.float32)
    alpha3 = alpha[:, :, None]
    roi = roi * (1.0 - alpha3) + objf * alpha3
    if scene_depth > 0.60:
        floor_mix = np.linspace(0.0, min(0.10, (scene_depth - 0.60) * 0.35), oh, dtype=np.float32)[:, None, None]
        roi = roi * (1.0 - floor_mix) + canvas[py:py + oh, px:px + ow].astype(np.float32) * floor_mix
    canvas[py:py + oh, px:px + ow] = _clip_uint8(roi)
    return canvas


def _sample_heap_anchor(w, h, poly_px):
    if poly_px is None:
        return int(w * random.uniform(0.35, 0.65)), int(h * random.uniform(0.55, 0.82))
    xs = [p[0] for p in poly_px]
    ys = [p[1] for p in poly_px]
    min_px = max(0, min(xs))
    max_px = min(w - 1, max(xs))
    min_py = max(0, min(ys))
    max_py = min(h - 1, max(ys))
    best = None
    best_score = -1.0
    for _ in range(80):
        x = random.randint(min_px, max_px)
        y = random.randint(min_py, max_py)
        if point_in_poly(x, y, poly_px):
            score = (y / max(1, h)) + random.uniform(-0.05, 0.05)
            if score > best_score:
                best_score = score
                best = (x, y)
    return best


def _sample_heap_position(w, h, ow, oh, poly_px, centers, heap_anchor, support_masks, cluster_distance_factor, overlap_spread):
    if not centers:
        return sample_position_in_polygon(w, h, ow, oh, poly_px, tries=80)
    anchor_x, anchor_y = heap_anchor if heap_anchor is not None else centers[random.randrange(len(centers))]
    best = None
    best_score = -1e9
    reach = max(16, int((0.6 + cluster_distance_factor) * max(ow, oh)))
    for _ in range(90):
        ref_x, ref_y = centers[random.randrange(len(centers))]
        mix = random.uniform(0.35, 0.85)
        base_x = int(ref_x * mix + anchor_x * (1.0 - mix))
        base_y = int(ref_y * mix + anchor_y * (1.0 - mix))
        jitter_x = int(random.uniform(-1.0, 1.0) * overlap_spread * reach)
        jitter_y = int(random.uniform(-0.35, 1.0) * overlap_spread * reach)
        px = int(np.clip(base_x - ow // 2 + jitter_x, 0, w - ow))
        py = int(np.clip(base_y - oh // 2 + jitter_y, 0, h - oh))
        if poly_px is not None and not bbox_fully_inside_poly(px, py, ow, oh, poly_px):
            continue
        cx_new = px + ow // 2
        cy_new = py + oh // 2
        nearest = min((((cx_new - cx) ** 2 + (cy_new - cy) ** 2) ** 0.5) for cx, cy in centers)
        score = -abs(nearest - max(10.0, 0.55 * max(ow, oh)))
        score += 2.5 * (py / max(1, h))
        score -= 0.35 * abs(cx_new - anchor_x) / max(1, w)
        if support_masks:
            probe_y1 = max(py + int(oh * 0.62), py)
            probe_y2 = min(py + oh, h)
            support = 0
            if probe_y2 > probe_y1:
                for prev in support_masks:
                    support += int((prev[probe_y1:probe_y2, px:px + ow] > 0).sum())
            score += min(600, support) / 200.0
        if score > best_score:
            best_score = score
            best = (px, py)
    return best


def _scene_header_overlay(canvas, visible_masks, target_objects, preview_mode, effective, bg_name, source_name):
    overlay = canvas.copy()
    colors = [
        (80, 220, 120),
        (120, 140, 255),
        (255, 190, 90),
        (220, 120, 255),
        (120, 255, 255),
    ]
    for idx, mask in enumerate(visible_masks):
        color = colors[idx % len(colors)]
        tint = np.zeros_like(overlay)
        tint[:] = color
        alpha = mask > 0
        overlay[alpha] = cv2.addWeighted(overlay[alpha], 0.55, tint[alpha], 0.45, 0)
        contour = contour_from_mask(mask)
        if contour is not None:
            cv2.drawContours(overlay, [contour.astype(np.int32).reshape(-1, 1, 2)], -1, color, 2)
    cv2.putText(overlay, f'instances={len(visible_masks)} target={target_objects} mode={preview_mode}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, f'profile scale={effective["min_scale"]:.2f}..{effective["max_scale"]:.2f}', (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, f'bg={bg_name} source={source_name}', (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def build_multi_scene(args, bg_files, profile_data, cutouts, reference_long_side):
    for _ in range(200):
        bg, bg_path = pick_random_background(bg_files)
        if bg is None:
            continue
        entry = resolve_profile_item(profile_data, Path(args.background_dir), bg_path)
        if not isinstance(entry, dict):
            continue

        h, w = bg.shape[:2]
        effective = choose_effective_settings(entry, args.class_name)
        poly_px = None
        if effective['poly'] is not None:
            poly_px = [(int(pt[0] * w), int(pt[1] * h)) for pt in effective['poly']]

        bg_beta = choose_background_beta(getattr(args, 'preview_mode', 'random'), effective)
        canvas = cv2.convertScaleAbs(bg.copy(), alpha=1.0, beta=bg_beta)
        scene_stats = _scene_illumination_stats(canvas)

        target_objects = random.randint(args.min_objects, args.max_objects)
        visible_masks = []
        visible_areas = []
        centers = []
        placed_instances = []
        heap_anchor = _sample_heap_anchor(w, h, poly_px)
        attempts_budget = max(120, target_objects * 120)

        for _ in range(attempts_budget):
            if len(placed_instances) >= target_objects:
                break

            cutout = weighted_choice(cutouts)
            requested_scale = choose_requested_scale(getattr(args, 'preview_mode', 'random'), effective)
            normalized_scale = requested_scale * (reference_long_side / max(1.0, cutout['long_side']))
            angle = random.uniform(-args.max_rotation, args.max_rotation)
            variant = prepare_cutout_variant(cutout, normalized_scale, angle)
            if variant is None:
                continue
            obj_img = variant['image']
            obj_mask = variant['mask']
            obj_mask_bin = variant['mask_bin']
            cand_area = max(1, int(variant['area']))
            oh, ow = variant['shape']
            if ow >= w or oh >= h:
                continue

            placed = False
            new_mask = None
            for _ in range(60):
                if centers:
                    if random.random() < max(0.55, args.overlap_prob):
                        pos = _sample_heap_position(
                            w, h, ow, oh, poly_px, centers, heap_anchor, visible_masks,
                            args.cluster_distance_factor, args.overlap_spread
                        )
                        if pos is None:
                            continue
                        px, py = pos
                    else:
                        pos = sample_scattered_position(w, h, ow, oh, poly_px, centers, tries=40)
                        if pos is None:
                            continue
                        px, py = pos
                else:
                    pos = sample_position_in_polygon(w, h, ow, oh, poly_px, tries=40)
                    if pos is None:
                        continue
                    px, py = pos

                if poly_px is not None and not bbox_fully_inside_poly(px, py, ow, oh, poly_px):
                    continue

                overlap_ok = True
                for prev, prev_area in zip(visible_masks, visible_areas):
                    prev_roi = prev[py:py + oh, px:px + ow] > 0
                    inter = int((prev_roi & obj_mask_bin).sum())
                    if (inter / cand_area) > args.max_overlap_ratio or (inter / prev_area) > args.max_overlap_ratio:
                        overlap_ok = False
                        break

                if overlap_ok and centers and random.random() < max(0.0, 1.0 - args.overlap_prob):
                    cx_new = px + ow // 2
                    cy_new = py + oh // 2
                    nearest = min((((cx_new - cx) ** 2 + (cy_new - cy) ** 2) ** 0.5) for cx, cy in centers)
                    max_scatter = (args.cluster_distance_factor + 1.6) * max(ow, oh)
                    if nearest < 0.55 * max(ow, oh):
                        overlap_ok = False
                    elif nearest > max_scatter:
                        overlap_ok = False

                if not overlap_ok:
                    continue

                scene_depth = py / max(1, h - oh)
                obj_img_var = apply_object_appearance_variation(
                    obj_img,
                    obj_mask,
                    canvas,
                    px,
                    py,
                    effective['obj_brightness_min'],
                    effective['obj_brightness_max'],
                    scene_depth=scene_depth,
                    scene_stats=scene_stats,
                    temp_bias=getattr(args, 'object_temp_bias', 0.35),
                    temp_variance=getattr(args, 'object_temp_variance', 0.18),
                    shade_prob=getattr(args, 'object_shade_prob', 0.45),
                    shade_strength=getattr(args, 'object_shade_strength', 0.22),
                )
                new_mask = np.zeros((h, w), dtype=np.uint8)
                new_mask[py:py + oh, px:px + ow][obj_mask_bin] = 255
                canvas = composite_object(canvas, obj_img_var, obj_mask, px, py, scene_depth=scene_depth)
                placed = True
                break

            if not placed or new_mask is None:
                continue

            updated_masks = []
            updated_areas = []
            for prev in visible_masks:
                vis = ((prev > 0) & (new_mask == 0)).astype(np.uint8) * 255
                area = int((vis > 0).sum())
                if area > 20:
                    updated_masks.append(vis)
                    updated_areas.append(area)
            visible_masks = updated_masks
            visible_areas = updated_areas
            visible_masks.append(new_mask)
            visible_areas.append(int((new_mask > 0).sum()))

            ys2, xs2 = np.where(new_mask > 0)
            centers.append((int(xs2.mean()), int(ys2.mean())))
            placed_instances.append({'mask': new_mask, 'source': cutout['source']})

        if len(visible_masks) < args.min_objects:
            continue

        canvas = apply_scene_camera_variation(canvas)

        overlay = _scene_header_overlay(
            canvas,
            visible_masks,
            target_objects,
            getattr(args, 'preview_mode', 'random'),
            effective,
            bg_path.name,
            placed_instances[-1]['source'],
        )
        if poly_px is not None:
            cv2.polylines(overlay, [np.array(poly_px, dtype=np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2, cv2.LINE_AA)
        return {
            'image': canvas,
            'visible_masks': visible_masks,
            'overlay': overlay,
            'background': bg_path.name,
            'effective': effective,
            'target_objects': target_objects,
        }
    return None


def build_single_scene(args, bg_files, profile_data, cutouts, reference_long_side):
    for _ in range(200):
        bg, bg_path = pick_random_background(bg_files)
        if bg is None:
            continue
        entry = resolve_profile_item(profile_data, Path(args.background_dir), bg_path)
        if not isinstance(entry, dict):
            continue

        h, w = bg.shape[:2]
        effective = choose_effective_settings(entry, args.class_name)
        poly_px = None
        if effective['poly'] is not None:
            poly_px = [(int(pt[0] * w), int(pt[1] * h)) for pt in effective['poly']]

        requested_scale = choose_requested_scale(getattr(args, 'preview_mode', 'random'), effective)
        bg_beta = choose_background_beta(getattr(args, 'preview_mode', 'random'), effective)
        canvas = cv2.convertScaleAbs(bg.copy(), alpha=1.0, beta=bg_beta)
        scene_stats = _scene_illumination_stats(canvas)

        cutout = weighted_choice(cutouts)
        normalized_scale = requested_scale * (reference_long_side / max(1.0, cutout['long_side']))
        angle = random.uniform(-args.max_rotation, args.max_rotation)
        variant = prepare_cutout_variant(cutout, normalized_scale, angle)
        if variant is None:
            continue
        obj_img = variant['image']
        obj_mask = variant['mask']
        obj_mask_bin = variant['mask_bin']
        oh, ow = variant['shape']
        if ow >= w or oh >= h:
            continue

        px = None
        py = None
        for _ in range(60):
            tx = random.randint(0, w - ow)
            ty = random.randint(0, h - oh)
            if poly_px is None or bbox_fully_inside_poly(tx, ty, ow, oh, poly_px):
                px, py = tx, ty
                break
        if px is None or py is None:
            continue

        scene_depth = py / max(1, h - oh)
        obj_img = apply_object_appearance_variation(
            obj_img,
            obj_mask,
            canvas,
            px,
            py,
            effective['obj_brightness_min'],
            effective['obj_brightness_max'],
            scene_depth=scene_depth,
            scene_stats=scene_stats,
            temp_bias=getattr(args, 'object_temp_bias', 0.35),
            temp_variance=getattr(args, 'object_temp_variance', 0.18),
            shade_prob=getattr(args, 'object_shade_prob', 0.45),
            shade_strength=getattr(args, 'object_shade_strength', 0.22),
        )

        canvas = composite_object(canvas, obj_img, obj_mask, px, py, scene_depth=scene_depth)

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[py:py + oh, px:px + ow][obj_mask_bin] = 255
        canvas = apply_scene_camera_variation(canvas)
        overlay = _scene_header_overlay(
            canvas,
            [full_mask],
            1,
            getattr(args, 'preview_mode', 'random'),
            effective,
            bg_path.name,
            cutout['source'],
        )
        if poly_px is not None:
            cv2.polylines(overlay, [np.array(poly_px, dtype=np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2, cv2.LINE_AA)
        return {
            'image': canvas,
            'mask': full_mask,
            'overlay': overlay,
            'background': bg_path.name,
            'effective': effective,
        }
    return None
