from pathlib import Path
import argparse
import shutil
import cv2
import numpy as np


def find_dir(root, name):
    cand = root / name
    if cand.exists() and cand.is_dir():
        return cand
    for p in root.rglob(name):
        if p.is_dir():
            return p
    return None


def find_file(root, name):
    cand = root / name
    if cand.exists() and cand.is_file():
        return cand
    for p in root.rglob(name):
        if p.is_file():
            return p
    return None


def read_pincam(path):
    vals = Path(path).read_text().strip().split()
    if len(vals) < 6:
        raise RuntimeError(f'bad pincam: {path}')
    return np.array(
        [float(vals[2]), float(vals[3]), float(vals[4]), float(vals[5])],
        dtype=np.float32,
    )


def pose_from_axis_angle(rx, ry, rz, tx, ty, tz):
    rvec = np.array([rx, ry, rz], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return T


def read_traj(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        vals = line.strip().split()
        if len(vals) < 7:
            continue
        ts = float(vals[0])
        rx, ry, rz = map(float, vals[1:4])
        tx, ty, tz = map(float, vals[4:7])
        rows.append((ts, pose_from_axis_angle(rx, ry, rz, tx, ty, tz)))
    if not rows:
        raise RuntimeError(f'no traj rows in {path}')
    return rows


def to_float_stem(path):
    try:
        return float(path.stem)
    except Exception:
        return None


def nearest_by_stem(stem_val, items):
    vals = np.array([x[0] for x in items], dtype=np.float64)
    idx = int(np.argmin(np.abs(vals - stem_val)))
    return items[idx]


def link_or_copy(src, dst, copy_mode=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_mode:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def write_intr_txt(path, vals):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(vals, dtype=np.float32).reshape(1, 4), fmt='%.8f')


def load_raw_maps(src):
    rgb_dir = find_dir(src, 'lowres_wide')
    depth_dir = find_dir(src, 'lowres_depth')
    intr_dir = find_dir(src, 'lowres_wide_intrinsics')
    traj_path = find_file(src, 'lowres_wide.traj')

    if rgb_dir is None or depth_dir is None or intr_dir is None or traj_path is None:
        raise RuntimeError('missing lowres_wide / lowres_depth / lowres_wide_intrinsics / lowres_wide.traj')

    rgb_files = sorted([p for p in rgb_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    depth_files = sorted([p for p in depth_dir.iterdir() if p.suffix.lower() in {'.png', '.npy'}])
    intr_files = sorted([
        p for p in intr_dir.iterdir()
        if p.suffix.lower() in {'.pincam', '.txt'} or p.name.endswith('.pincam')
    ])

    if not rgb_files or not depth_files or not intr_files:
        raise RuntimeError('missing frames')

    depth_map = []
    for p in depth_files:
        v = to_float_stem(p)
        if v is not None:
            depth_map.append((v, p))

    intr_map = []
    for p in intr_files:
        v = to_float_stem(p)
        if v is not None:
            intr_map.append((v, p))

    traj_rows = read_traj(traj_path)
    return rgb_files, depth_map, intr_map, traj_rows


def update_intrinsics_only(src, dst):
    src = Path(src)
    dst = Path(dst)

    rgb_files, _, intr_map, _ = load_raw_maps(src)

    dst_rgb = dst / 'rgb'
    if not dst_rgb.exists():
        raise RuntimeError(f'missing {dst_rgb}')

    dst_intr_dir = dst / 'intrinsics'
    dst_intr_dir.mkdir(parents=True, exist_ok=True)

    imported_rgb = sorted([p for p in dst_rgb.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    if not imported_rgb:
        raise RuntimeError('no imported rgb frames found')

    all_intr = []
    n = 0
    for rgb_path in imported_rgb:
        sv = to_float_stem(rgb_path)
        if sv is None:
            continue
        _, intr_path = nearest_by_stem(sv, intr_map)
        intr = read_pincam(intr_path)
        write_intr_txt(dst_intr_dir / f'{rgb_path.stem}.txt', intr)
        all_intr.append(intr)
        n += 1

    if not all_intr:
        raise RuntimeError('no intrinsics written')

    all_intr = np.stack(all_intr, axis=0)
    fallback = np.median(all_intr, axis=0).astype(np.float32)
    write_intr_txt(dst / 'intrinsics.txt', fallback)

    print('intrinsics_only', n, dst_intr_dir)
    print('fallback', fallback.tolist())


def full_import(src, dst, stride, limit, copy_mode):
    src = Path(src)
    dst = Path(dst)

    rgb_files, depth_map, intr_map, traj_rows = load_raw_maps(src)

    out_rgb = dst / 'rgb'
    out_depth = dst / 'depth'
    out_poses = dst / 'poses'
    out_intr = dst / 'intrinsics'
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)
    out_poses.mkdir(parents=True, exist_ok=True)
    out_intr.mkdir(parents=True, exist_ok=True)

    picked = rgb_files[::max(1, stride)]
    if limit > 0:
        picked = picked[:limit]

    if not picked:
        raise RuntimeError('no frames picked')

    intr_rows = []
    n = 0

    for rgb_path in picked:
        stem = rgb_path.stem
        sv = to_float_stem(rgb_path)
        if sv is None:
            continue

        _, depth_path = nearest_by_stem(sv, depth_map)
        _, intr_path = nearest_by_stem(sv, intr_map)
        _, pose = nearest_by_stem(sv, traj_rows)

        intr = read_pincam(intr_path)

        link_or_copy(rgb_path, out_rgb / f'{stem}{rgb_path.suffix.lower()}', copy_mode)
        link_or_copy(depth_path, out_depth / f'{stem}{depth_path.suffix.lower()}', copy_mode)
        np.savetxt(out_poses / f'{stem}.txt', pose, fmt='%.8f')
        write_intr_txt(out_intr / f'{stem}.txt', intr)

        intr_rows.append(intr)
        n += 1

    if not intr_rows:
        raise RuntimeError('no frames imported')

    intr_rows = np.stack(intr_rows, axis=0)
    fallback = np.median(intr_rows, axis=0).astype(np.float32)
    write_intr_txt(dst / 'intrinsics.txt', fallback)

    mesh_candidates = list(src.rglob('*.ply')) + list(src.rglob('*.obj'))
    if mesh_candidates:
        mesh_src = mesh_candidates[0]
        mesh_dst = dst / f'scene_mesh{mesh_src.suffix.lower()}'
        if mesh_dst.exists() or mesh_dst.is_symlink():
            mesh_dst.unlink()
        if copy_mode:
            shutil.copy2(mesh_src, mesh_dst)
        else:
            mesh_dst.symlink_to(mesh_src.resolve())

    print('done', n, dst)
    print('fallback', fallback.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--limit', type=int, default=120)
    ap.add_argument('--copy', action='store_true')
    ap.add_argument('--intrinsics-only', action='store_true')
    args = ap.parse_args()

    if args.intrinsics_only:
        update_intrinsics_only(args.src, args.dst)
    else:
        full_import(args.src, args.dst, args.stride, args.limit, args.copy)


if __name__ == '__main__':
    main()
