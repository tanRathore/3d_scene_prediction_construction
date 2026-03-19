from dataclasses import dataclass, field
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from .geometry import box_iou_xyxy, distance3

@dataclass
class Track:
    track_id: int
    label: str
    label_id: int
    centroid: np.ndarray
    size: np.ndarray
    bbox: np.ndarray
    score: float
    visible: int = 1
    missed: int = 0
    age: int = 1
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    history: list = field(default_factory=list)
    def update(self, det):
        new_centroid = det['centroid'].astype(np.float32)
        self.vel = new_centroid - self.centroid
        self.centroid = new_centroid
        self.size = det['size'].astype(np.float32)
        self.bbox = det['bbox'].astype(np.float32)
        self.score = float(det['score'])
        self.visible = 1
        self.missed = 0
        self.age += 1
        self.history.append(self.centroid.copy())
        self.history = self.history[-8:]
    def mark_missed(self):
        self.visible = 0
        self.missed += 1
        self.age += 1
        self.centroid = self.centroid + self.vel
        self.history.append(self.centroid.copy())
        self.history = self.history[-8:]

class SimpleTracker:
    def __init__(self, max_missed=10, match_3d_weight=1.0, match_iou_weight=0.35, match_label_penalty=4.0, match_threshold=2.0):
        self.max_missed = max_missed
        self.match_3d_weight = match_3d_weight
        self.match_iou_weight = match_iou_weight
        self.match_label_penalty = match_label_penalty
        self.match_threshold = match_threshold
        self.tracks = []
        self._counter = itertools.count(1)
    def _cost(self, track, det):
        c3 = distance3(track.centroid, det['centroid']) * self.match_3d_weight
        iou = box_iou_xyxy(track.bbox, det['bbox'])
        ciou = (1.0 - iou) * self.match_iou_weight
        clabel = 0.0 if track.label_id == det['label_id'] else self.match_label_penalty
        return float(c3 + ciou + clabel)
    def _new_track(self, det):
        self.tracks.append(Track(next(self._counter), det['label'], int(det['label_id']), det['centroid'].astype(np.float32), det['size'].astype(np.float32), det['bbox'].astype(np.float32), float(det['score']), history=[det['centroid'].astype(np.float32)]))
    def _trim(self):
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
    def update(self, detections):
        if not self.tracks:
            for det in detections:
                self._new_track(det)
            return self.as_nodes()
        if not detections:
            for tr in self.tracks:
                tr.mark_missed()
            self._trim()
            return self.as_nodes()
        cost = np.full((len(self.tracks), len(detections)), 1e6, dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            for j, det in enumerate(detections):
                cost[i, j] = self._cost(tr, det)
        rows, cols = linear_sum_assignment(cost)
        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] <= self.match_threshold:
                self.tracks[r].update(detections[c])
                used_rows.add(r)
                used_cols.add(c)
        for i, tr in enumerate(self.tracks):
            if i not in used_rows:
                tr.mark_missed()
        for j, det in enumerate(detections):
            if j not in used_cols:
                self._new_track(det)
        self._trim()
        return self.as_nodes()
    def as_nodes(self):
        out = []
        for tr in self.tracks:
            out.append({'track_id': int(tr.track_id), 'label': tr.label, 'label_id': int(tr.label_id), 'centroid': tr.centroid.astype(np.float32).tolist(), 'size': tr.size.astype(np.float32).tolist(), 'bbox': tr.bbox.astype(np.float32).tolist(), 'score': float(tr.score), 'visible': int(tr.visible), 'missed': int(tr.missed), 'vel': tr.vel.astype(np.float32).tolist()})
        return out
