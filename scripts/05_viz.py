from pathlib import Path
import argparse
import sys
import torch
from torch.utils.data import DataLoader, Subset
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from future_scene_graphs.config import load_config
from future_scene_graphs.dataset import GraphWindowDataset
from future_scene_graphs.models import SceneGraphForecaster
from future_scene_graphs.train import split_dataset
from future_scene_graphs.viz import save_triptych

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--windows', default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--idx', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    windows = args.windows or str(Path(cfg['data']['window_dir']) / 'train.pt')
    ckpt = args.ckpt or str(Path(cfg['work_dir']) / 'best.pt')
    out = args.out or str(Path(cfg['work_dir']) / f'viz_{args.idx}.png')
    ds = GraphWindowDataset(windows)
    _, va_ds = split_dataset(ds, cfg['train']['val_ratio'], cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SceneGraphForecaster(cfg['train']['hidden_dim'], cfg['train']['dropout'], cfg['train']['use_camera_motion']).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model'])
    batch = next(iter(DataLoader(Subset(va_ds, [args.idx]), batch_size=1)))
    save_triptych(model, batch, out, device)
    print('saved', out)

if __name__ == '__main__':
    main()
