from pathlib import Path
import argparse
import sys
import imageio.v2 as imageio
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from future_scene_graphs.config import load_config
from future_scene_graphs.dataset import GraphWindowDataset
from future_scene_graphs.models import SceneGraphForecaster
from future_scene_graphs.render3d import build_scene_static_geoms, make_states, save_overlay_o3d


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
    ap.add_argument('--sequence-id', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--stop', type=int, default=-1)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--out-dir', default='runs/anim')
    ap.add_argument('--name', default='scene_forecast.mp4')
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
    picks = [i for i in range(len(ds)) if ds.samples[i]['sequence_id'] == args.sequence_id]

    if args.stop > 0:
        picks = picks[args.start:args.stop:args.step]
    else:
        picks = picks[args.start::args.step]

    if not picks:
        raise RuntimeError('no samples for this sequence')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SceneGraphForecaster(
        cfg['train']['hidden_dim'],
        cfg['train']['dropout'],
        cfg['train']['use_camera_motion'],
    ).to(device)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model'])

    include_mesh, include_pcd = scene_mode_flags(args.scene_mode)

    seq_dir = Path(args.sequences_root) / args.sequence_id
    scene_geoms = build_scene_static_geoms(
        seq_dir,
        cfg,
        mesh_override=args.scene_mesh,
        pcd_override=args.scene_pcd,
        include_mesh=include_mesh,
        include_pcd=include_pcd,
    ) if seq_dir.exists() else []

    out_dir = Path(args.out_dir)
    frame_dir = out_dir / 'frames'
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []

    for c, idx in enumerate(picks):
        sample = ds[idx]
        batch = {}
        for k, v in sample.items():
            if torch.is_tensor(v):
                batch[k] = v.unsqueeze(0)
            else:
                batch[k] = v

        hist, pred, true = make_states(model, batch, device)

        frame_path = frame_dir / f'frame_{c:04d}.png'
        save_overlay_o3d(scene_geoms, hist, pred, true, cfg, frame_path, visible=bool(args.visible))
        frame_paths.append(frame_path)
        print(f'frame {c + 1}/{len(picks)}')

    out_path = out_dir / args.name
    writer = imageio.get_writer(out_path, fps=cfg['render']['fps'])

    for frame_path in frame_paths:
        writer.append_data(imageio.imread(frame_path))

    writer.close()
    print('saved', out_path)


if __name__ == '__main__':
    main()
