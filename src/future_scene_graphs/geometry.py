import math
import numpy as np

def depth_to_meters(depth, max_val=1000.0):
    depth = depth.astype(np.float32)
    if depth.max() > 100.0:
        depth = depth / max_val
    return depth

def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a.astype(np.float32)
    bx1, by1, bx2, by2 = b.astype(np.float32)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0

def crop_box(box, scale, width, height):
    x1, y1, x2, y2 = box.astype(np.float32)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    nx1 = max(0, int(round(cx - w / 2)))
    ny1 = max(0, int(round(cy - h / 2)))
    nx2 = min(width - 1, int(round(cx + w / 2)))
    ny2 = min(height - 1, int(round(cy + h / 2)))
    return np.array([nx1, ny1, nx2, ny2], dtype=np.int32)

def backproject_depth_region(depth_m, intr, box, pose, min_depth_m, max_depth_m, crop_scale=0.6, mask=None):
    h, w = depth_m.shape[:2]
    box = crop_box(box, crop_scale, w, h)
    x1, y1, x2, y2 = box.tolist()
    if x2 <= x1 or y2 <= y1:
        return None, None
    region = depth_m[y1:y2, x1:x2]
    ys, xs = np.mgrid[y1:y2, x1:x2]
    valid = (region >= min_depth_m) & (region <= max_depth_m)
    if mask is not None:
        valid &= (mask[y1:y2, x1:x2] > 0)
    if valid.sum() < 8:
        return None, None
    z = region[valid]
    x = xs[valid].astype(np.float32)
    y = ys[valid].astype(np.float32)
    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    xc = (x - cx) * z / fx
    yc = (y - cy) * z / fy
    pts_cam = np.stack([xc, yc, z], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
    pts_world = (pose @ pts_h.T).T[:, :3]
    centroid = np.median(pts_world, axis=0)
    size = np.percentile(pts_world, 90, axis=0) - np.percentile(pts_world, 10, axis=0)
    return centroid.astype(np.float32), size.astype(np.float32)

def pose_to_delta(prev_pose, pose):
    rel = np.linalg.inv(prev_pose) @ pose
    t = rel[:3, 3]
    r = rel[:3, :3]
    yaw = math.atan2(r[1, 0], r[0, 0])
    pitch = math.atan2(-r[2, 0], math.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
    roll = math.atan2(r[2, 1], r[2, 2])
    return np.array([t[0], t[1], t[2], roll, pitch, yaw], dtype=np.float32)

def distance3(a, b):
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))

def build_edges(nodes, near_thresh_m, front_thresh_m):
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a = nodes[i]
            b = nodes[j]
            if a['visible'] == 0 and b['visible'] == 0:
                continue
            pa = np.asarray(a['centroid'], dtype=np.float32)
            pb = np.asarray(b['centroid'], dtype=np.float32)
            d = np.linalg.norm(pa - pb)
            if d <= near_thresh_m:
                s = float(max(0.0, 1.0 - d / near_thresh_m))
                edges.append({'src': a['track_id'], 'dst': b['track_id'], 'type': 'near', 'score': s})
                edges.append({'src': b['track_id'], 'dst': a['track_id'], 'type': 'near', 'score': s})
            dz = pa[2] - pb[2]
            if abs(dz) >= front_thresh_m:
                s = float(min(1.0, abs(dz) / (front_thresh_m * 2)))
                if dz < 0:
                    edges.append({'src': a['track_id'], 'dst': b['track_id'], 'type': 'in_front_of', 'score': s})
                else:
                    edges.append({'src': b['track_id'], 'dst': a['track_id'], 'type': 'in_front_of', 'score': s})
    return edges
