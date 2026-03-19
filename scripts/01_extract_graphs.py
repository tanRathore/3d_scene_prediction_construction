from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from future_scene_graphs.config import load_config, set_seed
from future_scene_graphs.graph_builder import build_graphs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--sequences-root', default=None)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    build_graphs(args.sequences_root or cfg['data']['sequences_root'], args.out_dir or cfg['data']['graph_dir'], cfg)
    print('saved', args.out_dir or cfg['data']['graph_dir'])

if __name__ == '__main__':
    main()
