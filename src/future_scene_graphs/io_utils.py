from pathlib import Path
import json
import cv2
import numpy as np


def list_stems(folder, exts):
    folder = Path(folder)
    stems = []
    for ext in exts:
        stems.extend([p.stem for p in folder.glob(f'*.{ext}')])
    return sorted(set(stems))


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_depth(path):
    path = Path(path)
    if path.suffix == '.npy':
        depth = np.load(path)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(path)
    return depth.astype(np.float32)


def read_pose(path):
    return np.loadtxt(path, dtype=np.float32).reshape(4, 4)


def _intrinsics_from_array(vals):
    vals = np.asarray(vals, dtype=np.float32).reshape(-1)
    if vals.size < 4:
        raise RuntimeError('bad intrinsics')
    fx, fy, cx, cy = vals[:4].tolist()
    return {
        'fx': float(fx),
        'fy': float(fy),
        'cx': float(cx),
        'cy': float(cy),
    }


def read_intrinsics(path):
    vals = np.loadtxt(path, dtype=np.float32)
    return _intrinsics_from_array(vals)


def write_intrinsics(path, intr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = np.array(
        [intr['fx'], intr['fy'], intr['cx'], intr['cy']],
        dtype=np.float32,
    ).reshape(1, 4)
    np.savetxt(path, vals, fmt='%.8f')


def read_intrinsics_for_stem(seq_dir, stem=None, intrinsics_subdir='intrinsics', intrinsics_file='intrinsics.txt'):
    seq_dir = Path(seq_dir)

    if stem is not None:
        frame_path = seq_dir / intrinsics_subdir / f'{stem}.txt'
        if frame_path.exists():
            return read_intrinsics(frame_path)

    fallback = seq_dir / intrinsics_file
    if fallback.exists():
        return read_intrinsics(fallback)

    raise FileNotFoundError(f'no intrinsics found for stem={stem} in {seq_dir}')


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
