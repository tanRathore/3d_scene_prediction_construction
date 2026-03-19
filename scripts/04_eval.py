from pathlib import Path
import argparse
import json
import sys
import torch
from torch.utils.data import DataLoader
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from future_scene_graphs.config import load_config
from future_scene_graphs.dataset import GraphWindowDataset
from future_scene_graphs.evaluate import evaluate_baseline, evaluate_model
from future_scene_graphs.models import SceneGraphForecaster
from future_scene_graphs.train import split_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--windows', default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    windows = args.windows or str(Path(cfg['data']['window_dir']) / 'train.pt')
    ckpt = args.ckpt or str(Path(cfg['work_dir']) / 'best.pt')
    out = args.out or str(Path(cfg['work_dir']) / 'eval.json')
    ds = GraphWindowDataset(windows)
    _, va_ds = split_dataset(ds, cfg['train']['val_ratio'], cfg['seed'])
    loader = DataLoader(va_ds, batch_size=cfg['train']['batch_size'], shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SceneGraphForecaster(cfg['train']['hidden_dim'], cfg['train']['dropout'], cfg['train']['use_camera_motion']).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model'])
    rows = {'model': evaluate_model(model, loader, device), 'copy_last': evaluate_baseline('copy_last', loader, device), 'constant_velocity': evaluate_baseline('constant_velocity', loader, device)}
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    main()
