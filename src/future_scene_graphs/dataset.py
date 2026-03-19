from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .io_utils import load_jsonl

FEATURE_DIM = 13

def _frame_node_map(frame):
    return {int(n['track_id']): n for n in frame['nodes']}

def _pick_track_ids(history, target, max_nodes):
    cnt = Counter()
    last_seen = {}
    for t, frame in enumerate(history):
        for node in frame['nodes']:
            tid = int(node['track_id'])
            cnt[tid] += 3 if int(node['visible']) == 1 else 1
            last_seen[tid] = t
    for node in target['nodes']:
        tid = int(node['track_id'])
        cnt[tid] += 2
        last_seen[tid] = len(history)
    ranked = sorted(cnt.keys(), key=lambda k: (cnt[k], last_seen.get(k, -1), -k), reverse=True)
    return ranked[:max_nodes]

def _node_feature(node):
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    if node is None:
        return feat
    feat[0:3] = np.asarray(node['centroid'], dtype=np.float32)
    feat[3:6] = np.asarray(node.get('vel', [0, 0, 0]), dtype=np.float32)
    feat[6] = float(node.get('visible', 0))
    feat[7] = 1.0
    feat[8:11] = np.asarray(node.get('size', [0, 0, 0]), dtype=np.float32)
    feat[11] = float(node.get('label_id', 0))
    feat[12] = float(node.get('score', 0.0))
    return feat

def _adj_from_edges(frame, track_ids, max_nodes=None):
    idx = {tid: i for i, tid in enumerate(track_ids)}
    n = max_nodes if max_nodes is not None else len(track_ids)
    adj = np.zeros((n, n), dtype=np.float32)
    for e in frame['edges']:
        if e['type'] != 'near':
            continue
        s = int(e['src'])
        d = int(e['dst'])
        if s in idx and d in idx:
            adj[idx[s], idx[d]] = 1.0
    return adj

def build_samples_from_graphs(graphs, history, horizon, max_nodes):
    samples = []
    if len(graphs) < history + horizon:
        return samples

    node_maps = [_frame_node_map(g) for g in graphs]

    for end in range(history - 1, len(graphs) - horizon):
        hist = graphs[end - history + 1:end + 1]
        target = graphs[end + horizon]
        live_track_ids = _pick_track_ids(hist, target, max_nodes)

        track_ids = np.zeros((max_nodes,), dtype=np.int64)
        track_ids[:len(live_track_ids)] = np.asarray(live_track_ids, dtype=np.int64)

        x = np.zeros((history, max_nodes, FEATURE_DIM), dtype=np.float32)
        cam = np.zeros((history, 6), dtype=np.float32)

        for t, frame in enumerate(hist):
            m = node_maps[end - history + 1 + t]
            cam[t] = np.asarray(frame.get('camera_delta', [0, 0, 0, 0, 0, 0]), dtype=np.float32)
            for j, tid in enumerate(live_track_ids):
                x[t, j] = _node_feature(m.get(tid))

        target_map = _frame_node_map(target)
        y_pos = np.zeros((max_nodes, 3), dtype=np.float32)
        y_vis = np.zeros((max_nodes,), dtype=np.float32)
        y_present = np.zeros((max_nodes,), dtype=np.float32)
        y_size = np.zeros((max_nodes, 3), dtype=np.float32)
        labels = np.zeros((max_nodes,), dtype=np.int64)

        for j, tid in enumerate(live_track_ids):
            node = target_map.get(tid)
            if node is None:
                continue
            y_pos[j] = np.asarray(node['centroid'], dtype=np.float32)
            y_vis[j] = float(node.get('visible', 0))
            y_present[j] = 1.0
            y_size[j] = np.asarray(node.get('size', [0, 0, 0]), dtype=np.float32)
            labels[j] = int(node.get('label_id', 0))

        sample = {
            'sequence_id': hist[-1]['sequence_id'],
            'frame_stem': target['frame_stem'],
            'track_ids': track_ids,
            'x': x,
            'camera': cam,
            'adj': _adj_from_edges(hist[-1], live_track_ids, max_nodes=max_nodes),
            'y_pos': y_pos,
            'y_vis': y_vis,
            'y_present': y_present,
            'y_size': y_size,
            'y_adj': _adj_from_edges(target, live_track_ids, max_nodes=max_nodes),
            'labels': labels,
        }
        samples.append(sample)

    return samples

def build_window_file(graph_dir, out_path, history, horizon, max_nodes):
    graph_dir = Path(graph_dir)
    out_path = Path(out_path)
    all_samples = []
    for path in sorted(graph_dir.glob('*.jsonl')):
        all_samples.extend(build_samples_from_graphs(load_jsonl(path), history, horizon, max_nodes))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_samples, out_path)

class GraphWindowDataset(Dataset):
    def __init__(self, path):
        self.samples = torch.load(path, weights_only=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'sequence_id': s['sequence_id'],
            'frame_stem': s['frame_stem'],
            'track_ids': torch.tensor(s['track_ids'], dtype=torch.long),
            'x': torch.tensor(s['x'], dtype=torch.float32),
            'camera': torch.tensor(s['camera'], dtype=torch.float32),
            'adj': torch.tensor(s['adj'], dtype=torch.float32),
            'y_pos': torch.tensor(s['y_pos'], dtype=torch.float32),
            'y_vis': torch.tensor(s['y_vis'], dtype=torch.float32),
            'y_present': torch.tensor(s['y_present'], dtype=torch.float32),
            'y_size': torch.tensor(s['y_size'], dtype=torch.float32),
            'y_adj': torch.tensor(s['y_adj'], dtype=torch.float32),
            'labels': torch.tensor(s['labels'], dtype=torch.long),
        }
