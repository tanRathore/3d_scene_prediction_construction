from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from future_scene_graphs.config import load_config, set_seed
from future_scene_graphs.train import run_training

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--windows', default=None)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    run_training(cfg, args.windows or str(Path(cfg['data']['window_dir']) / 'train.pt'), args.out_dir or cfg['work_dir'])

if __name__ == '__main__':
    main()
