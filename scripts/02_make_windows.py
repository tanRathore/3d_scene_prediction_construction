from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from future_scene_graphs.config import load_config
from future_scene_graphs.dataset import build_window_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--graph-dir', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = args.out or str(Path(cfg['data']['window_dir']) / 'train.pt')
    build_window_file(args.graph_dir or cfg['data']['graph_dir'], out, cfg['windows']['history'], cfg['windows']['horizon'], cfg['windows']['max_nodes'])
    print('saved', out)

if __name__ == '__main__':
    main()
