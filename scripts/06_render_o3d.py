from pathlib import Path
import argparse
import sys
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from future_scene_graphs.config import load_config
from future_scene_graphs.dataset import GraphWindowDataset
from future_scene_graphs.models import SceneGraphForecaster
from future_scene_graphs.render3d import (
    build_scene_static_geoms,
    make_states,
    save_overlay_o3d,
    save_triptych_o3d,
)


def scene_mode_flags(mode):
    mode = mode.lower()
    if mode == 'mesh':
        return True, False
    if mode == 'pcd':
        return False, True
    if mode == 'both':
        return True, True
    raise ValueError(f'bad scene mode: {mode}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--windows', default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--sequences-root', default='data/sequences')
    ap.add_argument('--idx', type=int, default=0)
    ap.add_argument('--out-dir', default='runs/o3d')
    ap.add_argument('--scene-mesh', default=None)
    ap.add_argument('--scene-pcd', default=None)
    ap.add_argument('--scene-mode', default='both', choices=['mesh', 'pcd', 'both'])
    ap.add_argument('--zoom', type=float, default=-1.0)
    ap.add_argument('--visible', type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.zoom > 0:
        cfg['render']['zoom'] = float(args.zoom)

    windows = args.windows or str(Path(cfg['data']['window_dir']) / 'train.pt')
    ckpt = args.ckpt or str(Path(cfg['work_dir']) / 'best.pt')

    ds = GraphWindowDataset(windows)
    sample = ds[args.idx]

    batch = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0)
        else:
            batch[k] = v

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SceneGraphForecaster(
        cfg['train']['hidden_dim'],
        cfg['train']['dropout'],
        cfg['train']['use_camera_motion'],
    ).to(device)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model'])

    include_mesh, include_pcd = scene_mode_flags(args.scene_mode)

    seq_dir = Path(args.sequences_root) / sample['sequence_id']
    scene_geoms = build_scene_static_geoms(
        seq_dir,
        cfg,
        center_stem=sample['frame_stem'],
        mesh_override=args.scene_mesh,
        pcd_override=args.scene_pcd,
        include_mesh=include_mesh,
        include_pcd=include_pcd,
    ) if seq_dir.exists() else []

    hist, pred, true = make_states(model, batch, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trip = out_dir / 'triptych.png'
    over = out_dir / 'overlay.png'

    save_triptych_o3d(scene_geoms, hist, pred, true, cfg, trip, visible=bool(args.visible))
    save_overlay_o3d(scene_geoms, hist, pred, true, cfg, over, visible=bool(args.visible))

    print('saved', trip)
    print('saved', over)


if __name__ == '__main__':
    main()
