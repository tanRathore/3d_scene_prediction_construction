from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .geometry import backproject_depth_region, build_edges, depth_to_meters, pose_to_delta
from .io_utils import (
    list_stems,
    load_json,
    read_depth,
    read_image,
    read_intrinsics_for_stem,
    read_pose,
    save_json,
    save_jsonl,
)
from .tracking import SimpleTracker

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _img_path(rgb_dir, stem, ext):
    for p in [
        rgb_dir / f"{stem}.{ext}",
        rgb_dir / f"{stem}.jpg",
        rgb_dir / f"{stem}.png",
        rgb_dir / f"{stem}.jpeg",
    ]:
        if p.exists():
            return p
    return None


def _depth_path(depth_dir, stem, ext):
    for p in [
        depth_dir / f"{stem}.{ext}",
        depth_dir / f"{stem}.png",
        depth_dir / f"{stem}.npy",
    ]:
        if p.exists():
            return p
    return None


def _normalize_segmentation(segmentation):
    if segmentation is None:
        return None

    if not isinstance(segmentation, (list, tuple)) or len(segmentation) == 0:
        return None

    first = segmentation[0]
    if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[0], (int, float)):
        segmentation = [segmentation]

    polys = []
    for poly in segmentation:
        arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] < 3:
            continue
        polys.append(arr.tolist())

    return polys or None


def _segmentation_to_mask(shape_hw, segmentation):
    polys = _normalize_segmentation(segmentation)
    if polys is None:
        return None

    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    draw = []
    for poly in polys:
        arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
        arr = np.round(arr).astype(np.int32)
        if arr.shape[0] >= 3:
            draw.append(arr)

    if not draw:
        return None

    cv2.fillPoly(mask, draw, 1)
    return mask


def _bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    aw = max(0.0, ax2 - ax1)
    ah = max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1)
    bh = max(0.0, by2 - by1)

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


class Detector:
    def __init__(
        self,
        mode,
        model_name,
        conf,
        iou,
        max_det,
        classes=None,
        seg_model_name=None,
        seg_conf=None,
        seg_iou_match=0.35,
    ):
        self.mode = mode
        self.model_name = model_name
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.classes = classes or []
        self.seg_model_name = seg_model_name
        self.seg_conf = conf if seg_conf is None else seg_conf
        self.seg_iou_match = seg_iou_match

        self.model = None
        self.det_model = None
        self.seg_model = None

        if YOLO is None:
            raise ImportError("ultralytics not found")

        if mode == "ultralytics":
            self.model = YOLO(model_name)
        elif mode == "hybrid_ultralytics":
            self.det_model = YOLO(model_name)
            self.seg_model = YOLO(seg_model_name)
        else:
            raise ValueError(f"unsupported detector mode: {mode}")

    def _predict_single(self, model, image, conf):
        res = model.predict(
            image,
            conf=conf,
            iou=self.iou,
            max_det=self.max_det,
            classes=self.classes or None,
            verbose=False,
        )

        out = []
        if not res:
            return out

        r = res[0]
        names = r.names
        boxes = r.boxes

        mask_xy = None
        if getattr(r, "masks", None) is not None and getattr(r.masks, "xy", None) is not None:
            mask_xy = r.masks.xy

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(np.float32)
            cls_id = int(boxes.cls[i].item())
            score = float(boxes.conf[i].item())

            row = {
                "bbox": xyxy.tolist(),
                "label_id": cls_id,
                "label": str(names[cls_id]),
                "score": score,
            }

            if mask_xy is not None and i < len(mask_xy):
                poly = np.asarray(mask_xy[i], dtype=np.float32).reshape(-1, 2)
                if poly.shape[0] >= 3:
                    row["segmentation"] = [poly.tolist()]

            out.append(row)

        return out

    def _merge_hybrid(self, dets, segs):
        out = []
        used = set()

        for det in dets:
            best_j = -1
            best_iou = 0.0

            for j, seg in enumerate(segs):
                if j in used:
                    continue
                if int(seg["label_id"]) != int(det["label_id"]):
                    continue
                if "segmentation" not in seg:
                    continue

                iou = _bbox_iou_xyxy(det["bbox"], seg["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            row = dict(det)
            if best_j >= 0 and best_iou >= self.seg_iou_match:
                used.add(best_j)
                row["segmentation"] = segs[best_j]["segmentation"]
                row["seg_score"] = float(segs[best_j]["score"])
                row["seg_iou"] = float(best_iou)

            out.append(row)

        return out

    def predict(self, image, det_path=None):
        if det_path is not None and det_path.exists():
            return load_json(det_path)

        if self.mode == "ultralytics":
            out = self._predict_single(self.model, image, self.conf)
        elif self.mode == "hybrid_ultralytics":
            dets = self._predict_single(self.det_model, image, self.conf)
            segs = self._predict_single(self.seg_model, image, self.seg_conf)
            out = self._merge_hybrid(dets, segs)
        else:
            raise ValueError(f"unsupported detector mode: {self.mode}")

        if det_path is not None:
            save_json(out, det_path)

        return out


def build_graphs_for_sequence(seq_dir, cfg):
    seq_dir = Path(seq_dir)
    data_cfg = cfg["data"]
    ex_cfg = cfg["extract"]

    rgb_dir = seq_dir / data_cfg["rgb_subdir"]
    depth_dir = seq_dir / data_cfg["depth_subdir"]
    pose_dir = seq_dir / data_cfg["pose_subdir"]
    det_dir = seq_dir / data_cfg["detections_subdir"]
    det_dir.mkdir(parents=True, exist_ok=True)

    stems = list_stems(rgb_dir, (data_cfg["image_ext"], "png", "jpg", "jpeg"))

    detector = Detector(
        ex_cfg["detector"],
        ex_cfg["detector_model"],
        ex_cfg["conf"],
        ex_cfg["iou"],
        ex_cfg["max_det"],
        ex_cfg.get("classes"),
        seg_model_name=ex_cfg.get("seg_model"),
        seg_conf=ex_cfg.get("seg_conf"),
        seg_iou_match=float(ex_cfg.get("seg_iou_match", 0.35)),
    )

    tracker = SimpleTracker(
        ex_cfg["track_max_missed"],
        ex_cfg["match_3d_weight"],
        ex_cfg["match_iou_weight"],
        ex_cfg["match_label_penalty"],
        ex_cfg["match_threshold"],
    )

    use_masks = bool(ex_cfg.get("use_masks", False))
    fallback_to_box = bool(ex_cfg.get("fallback_to_box", True))
    crop_scale = float(ex_cfg.get("center_crop", 1.0))

    graphs = []
    prev_pose = None

    for frame_idx, stem in enumerate(tqdm(stems, desc=seq_dir.name)):
        img_path = _img_path(rgb_dir, stem, data_cfg["image_ext"])
        depth_path = _depth_path(depth_dir, stem, data_cfg["depth_ext"])
        pose_path = pose_dir / f"{stem}.txt"

        if img_path is None or depth_path is None or not pose_path.exists():
            continue

        image = read_image(img_path)
        depth = depth_to_meters(read_depth(depth_path))
        pose = read_pose(pose_path)
        intr = read_intrinsics_for_stem(
            seq_dir,
            stem=stem,
            intrinsics_subdir=data_cfg.get("intrinsics_subdir", "intrinsics"),
            intrinsics_file=data_cfg["intrinsics_file"],
        )

        raw = detector.predict(image, det_dir / f"{stem}.json")

        detections = []
        for det in raw:
            box = np.asarray(det["bbox"], dtype=np.float32)

            mask = None
            if use_masks:
                mask = _segmentation_to_mask(image.shape[:2], det.get("segmentation"))

            centroid = None
            size = None
            lift_mode = "box"

            if mask is not None:
                centroid, size = backproject_depth_region(
                    depth,
                    intr,
                    box,
                    pose,
                    data_cfg["min_depth_m"],
                    data_cfg["max_depth_m"],
                    crop_scale=crop_scale,
                    mask=mask,
                )
                if centroid is not None:
                    lift_mode = "mask"

            if centroid is None and fallback_to_box:
                centroid, size = backproject_depth_region(
                    depth,
                    intr,
                    box,
                    pose,
                    data_cfg["min_depth_m"],
                    data_cfg["max_depth_m"],
                    crop_scale=crop_scale,
                    mask=None,
                )
                if centroid is not None:
                    lift_mode = "box_fallback" if mask is not None else "box"

            if centroid is None:
                continue

            row = {
                "bbox": box,
                "label_id": int(det["label_id"]),
                "label": str(det["label"]),
                "score": float(det["score"]),
                "centroid": centroid,
                "size": size,
                "lift_mode": lift_mode,
            }

            if mask is not None:
                row["mask_pixels"] = int(mask.sum())
            if "seg_iou" in det:
                row["seg_iou"] = float(det["seg_iou"])
            if "seg_score" in det:
                row["seg_score"] = float(det["seg_score"])

            detections.append(row)

        nodes = tracker.update(detections)
        edges = build_edges(nodes, ex_cfg["near_thresh_m"], ex_cfg["front_thresh_m"])
        cam_delta = np.zeros(6, dtype=np.float32) if prev_pose is None else pose_to_delta(prev_pose, pose)
        prev_pose = pose

        graphs.append(
            {
                "sequence_id": seq_dir.name,
                "frame_idx": int(frame_idx),
                "frame_stem": stem,
                "camera_pose": pose.astype(np.float32).tolist(),
                "camera_delta": cam_delta.astype(np.float32).tolist(),
                "nodes": nodes,
                "edges": edges,
            }
        )

    return graphs


def build_graphs(sequences_root, output_dir, cfg):
    sequences_root = Path(sequences_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq_dir in sorted([p for p in sequences_root.iterdir() if p.is_dir()]):
        graphs = build_graphs_for_sequence(seq_dir, cfg)
        if graphs:
            save_jsonl(graphs, output_dir / f"{seq_dir.name}.jsonl")
