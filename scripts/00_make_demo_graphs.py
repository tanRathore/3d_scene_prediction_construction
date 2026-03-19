from pathlib import Path
import argparse
import math
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from future_scene_graphs.io_utils import save_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/graphs/demo_seq.jsonl')
    ap.add_argument('--frames', type=int, default=80)
    args = ap.parse_args()
    rows = []
    for t in range(args.frames):
        a = [0.3 + 0.02 * t, 0.0, 2.0 + 0.05 * math.sin(t / 5)]
        b = [1.5, 0.0, 2.8 - 0.015 * t]
        c = [2.2 - 0.01 * t, 0.0, 1.7 + 0.03 * math.cos(t / 8)]
        visible_c = 1 if t % 11 not in {7, 8} else 0
        nodes = [
            {'track_id': 1, 'label': 'chair', 'label_id': 56, 'centroid': a, 'size': [0.5, 0.8, 0.5], 'bbox': [100, 140, 180, 260], 'score': 0.9, 'visible': 1, 'missed': 0, 'vel': [0.02, 0.0, 0.0]},
            {'track_id': 2, 'label': 'table', 'label_id': 60, 'centroid': b, 'size': [1.2, 0.8, 0.8], 'bbox': [240, 120, 360, 250], 'score': 0.95, 'visible': 1, 'missed': 0, 'vel': [0.0, 0.0, -0.015]},
            {'track_id': 3, 'label': 'cabinet', 'label_id': 72, 'centroid': c, 'size': [1.0, 1.6, 0.5], 'bbox': [400, 80, 520, 300], 'score': 0.88, 'visible': visible_c, 'missed': 0 if visible_c else 1, 'vel': [-0.01, 0.0, 0.0]},
        ]
        live = [n for n in nodes if n['visible'] == 1]
        edges = []
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                da = sum((live[i]['centroid'][k] - live[j]['centroid'][k]) ** 2 for k in range(3)) ** 0.5
                if da < 1.4:
                    s = 1.0 - da / 1.4
                    edges.append({'src': live[i]['track_id'], 'dst': live[j]['track_id'], 'type': 'near', 'score': s})
                    edges.append({'src': live[j]['track_id'], 'dst': live[i]['track_id'], 'type': 'near', 'score': s})
        rows.append({'sequence_id': 'demo_seq', 'frame_idx': t, 'frame_stem': f'{t:06d}', 'camera_pose': [[1,0,0,0.01*t],[0,1,0,0],[0,0,1,0],[0,0,0,1]], 'camera_delta': [0.01,0,0,0,0,0], 'nodes': nodes, 'edges': edges})
    save_jsonl(rows, args.out)
    print('saved', args.out)

if __name__ == '__main__':
    main()
