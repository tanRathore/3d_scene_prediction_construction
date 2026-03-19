from pathlib import Path
import copy
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from .evaluate import move_batch
from .geometry import depth_to_meters
from .io_utils import list_stems, read_depth, read_image, read_intrinsics_for_stem, read_pose

def _img_path(rgb_dir, stem, ext):
    for p in [
        rgb_dir / f'{stem}.{ext}',
        rgb_dir / f'{stem}.jpg',
        rgb_dir / f'{stem}.png',
        rgb_dir / f'{stem}.jpeg',
    ]:
        if p.exists():
            return p
    return None

def _depth_path(depth_dir, stem, ext):
    for p in [
        depth_dir / f'{stem}.{ext}',
        depth_dir / f'{stem}.png',
        depth_dir / f'{stem}.npy',
    ]:
        if p.exists():
            return p
    return None

def _pick_stems(stems, center_stem=None, count=8):
    if not stems:
        return []
    if center_stem is None or center_stem not in stems:
        if len(stems) <= count:
            return stems
        idx = np.linspace(0, len(stems) - 1, count).astype(int)
        return [stems[i] for i in idx.tolist()]
    c = stems.index(center_stem)
    half = count // 2
    lo = max(0, c - half)
    hi = min(len(stems), lo + count)
    lo = max(0, hi - count)
    return stems[lo:hi]

def _backproject_full(depth_m, image, intr, pose, step, min_depth_m, max_depth_m):
    h, w = depth_m.shape[:2]
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    z = depth_m[ys, xs]
    valid = (z >= min_depth_m) & (z <= max_depth_m)

    if valid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    x = xs[valid].astype(np.float32)
    y = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    xc = (x - cx) * z / fx
    yc = (y - cy) * z / fy

    pts_cam = np.stack([xc, yc, z], axis=1)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
    pts_world = (pose @ pts_h.T).T[:, :3]

    cols = image[ys, xs][valid].astype(np.float32) / 255.0
    return pts_world.astype(np.float32), cols.astype(np.float32)


def build_scene_pointcloud(seq_dir, cfg, center_stem=None):
    seq_dir = Path(seq_dir)
    data_cfg = cfg['data']
    render_cfg = cfg['render']

    rgb_dir = seq_dir / data_cfg['rgb_subdir']
    depth_dir = seq_dir / data_cfg['depth_subdir']
    pose_dir = seq_dir / data_cfg['pose_subdir']

    if not rgb_dir.exists() or not depth_dir.exists() or not pose_dir.exists():
        return o3d.geometry.PointCloud()

    stems = list_stems(rgb_dir, (data_cfg['image_ext'], 'png', 'jpg', 'jpeg'))
    stems = _pick_stems(stems, center_stem=center_stem, count=render_cfg['point_frames'])

    pts_all = []
    col_all = []

    for stem in stems:
        img_path = _img_path(rgb_dir, stem, data_cfg['image_ext'])
        dep_path = _depth_path(depth_dir, stem, data_cfg['depth_ext'])
        pose_path = pose_dir / f'{stem}.txt'
        if img_path is None or dep_path is None or not pose_path.exists():
            continue

        image = read_image(img_path)
        depth = depth_to_meters(read_depth(dep_path))
        pose = read_pose(pose_path)
        intr = read_intrinsics_for_stem(
            seq_dir,
            stem=stem,
            intrinsics_subdir=data_cfg.get('intrinsics_subdir', 'intrinsics'),
            intrinsics_file=data_cfg['intrinsics_file'],
        )

        pts, cols = _backproject_full(
            depth,
            image,
            intr,
            pose,
            render_cfg['point_stride'],
            data_cfg['min_depth_m'],
            data_cfg['max_depth_m'],
        )

        if len(pts):
            pts_all.append(pts)
            col_all.append(cols)

    pcd = o3d.geometry.PointCloud()
    if not pts_all:
        return pcd

    pts = np.concatenate(pts_all, axis=0)
    cols = np.concatenate(col_all, axis=0)

    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

    if render_cfg['voxel'] > 0:
        pcd = pcd.voxel_down_sample(render_cfg['voxel'])

    return pcd


    pts = np.concatenate(pts_all, axis=0)
    cols = np.concatenate(col_all, axis=0)

    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

    if render_cfg['voxel'] > 0:
        pcd = pcd.voxel_down_sample(render_cfg['voxel'])

    return pcd



def build_scene_static_geoms(
    seq_dir,
    cfg,
    center_stem=None,
    mesh_override=None,
    pcd_override=None,
    include_mesh=True,
    include_pcd=True,
):
    seq_dir = Path(seq_dir)
    geoms = []

    if include_mesh:
        mesh_path = None
        mesh_candidates = []

        if mesh_override is not None:
            mesh_candidates.append(Path(mesh_override))

        mesh_candidates.extend([
            seq_dir / 'scene_mesh_fused.ply',
            seq_dir / 'scene_mesh_fused.obj',
            seq_dir / 'scene_mesh.ply',
            seq_dir / 'scene_mesh.obj',
        ])

        for p in mesh_candidates:
            if p.exists():
                mesh_path = p
                break

        if mesh_path is not None:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if len(mesh.vertices):
                mesh.compute_vertex_normals()
                if not mesh.has_vertex_colors():
                    mesh.paint_uniform_color([0.55, 0.58, 0.62])
                geoms.append(mesh)

    if include_pcd:
        pcd = o3d.geometry.PointCloud()
        pcd_candidates = []

        if pcd_override is not None:
            pcd_candidates.append(Path(pcd_override))

        pcd_candidates.extend([
            seq_dir / 'scene_points_fused.ply',
        ])

        chosen_pcd = None
        for p in pcd_candidates:
            if p.exists():
                chosen_pcd = p
                break

        if chosen_pcd is not None:
            pcd = o3d.io.read_point_cloud(str(chosen_pcd))
        else:
            pcd = build_scene_pointcloud(seq_dir, cfg, center_stem=center_stem)

        if len(pcd.points):
            geoms.append(pcd)

    return geoms



def _box_corners(center, size):
    cx, cy, cz = center.tolist()
    sx, sy, sz = size.tolist()
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    return np.array([
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ], dtype=np.float64)

def make_cuboid(center, size, color):
    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)

    corners = _box_corners(center, size)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1)))
    return ls

def make_sphere(center, radius, color):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    s.translate(np.asarray(center, dtype=np.float64))
    return s

def make_cylinder(p0, p1, radius, color):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-6:
        return None

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L)
    cyl.compute_vertex_normals()
    cyl.paint_uniform_color(color)

    z = np.array([0.0, 0.0, 1.0])
    v = d / L
    axis = np.cross(z, v)
    angle = np.arccos(np.clip(np.dot(z, v), -1.0, 1.0))

    if np.linalg.norm(axis) > 1e-8:
        axis = axis / np.linalg.norm(axis)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cyl.rotate(R, center=np.zeros(3))
    elif np.dot(z, v) < 0:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
        cyl.rotate(R, center=np.zeros(3))

    cyl.translate((p0 + p1) / 2.0 - np.array([0.0, 0.0, L / 2.0]))
    return cyl

def make_edge_lines(pos, live, adj, color):
    geoms = []
    ids = np.where(live)[0].tolist()
    for i in ids:
        for j in ids:
            if j > i and adj[i, j]:
                c = make_cylinder(pos[i], pos[j], 0.01, color)
                if c is not None:
                    geoms.append(c)
    return geoms

def crop_scene_geoms(scene_geoms, center, radius):
    out = []
    center = np.asarray(center, dtype=np.float64)
    rmin = center - radius
    rmax = center + radius
    bbox = o3d.geometry.AxisAlignedBoundingBox(rmin, rmax)

    for g in scene_geoms:
        if isinstance(g, o3d.geometry.PointCloud):
            cg = g.crop(bbox)
            if len(cg.points):
                out.append(cg)
        elif isinstance(g, o3d.geometry.TriangleMesh):
            try:
                cg = g.crop(bbox)
                if len(cg.vertices):
                    out.append(cg)
            except Exception:
                out.append(g)
        else:
            out.append(g)
    return out

def make_states(model, batch, device):
    batch = move_batch(batch, device)
    model.eval()
    with torch.no_grad():
        out = model(batch)

    hist = {
        'pos': batch['x'][0, -1, :, 0:3].detach().cpu().numpy(),
        'size': batch['x'][0, -1, :, 8:11].detach().cpu().numpy(),
        'live': ((batch['x'][0, -1, :, 6] > 0.5) & (batch['x'][0, -1, :, 7] > 0.5)).detach().cpu().numpy(),
        'adj': (batch['adj'][0] > 0.5).detach().cpu().numpy(),
    }

    pred = {
        'pos': out['pos'][0].detach().cpu().numpy(),
        'size': out['size'][0].detach().cpu().numpy(),
        'live': ((torch.sigmoid(out['vis_logits'][0]) > 0.5) & (torch.sigmoid(out['present_logits'][0]) > 0.5)).detach().cpu().numpy(),
        'adj': (torch.sigmoid(out['near_logits'][0]) > 0.5).detach().cpu().numpy(),
    }

    true = {
        'pos': batch['y_pos'][0].detach().cpu().numpy(),
        'size': batch['y_size'][0].detach().cpu().numpy(),
        'live': ((batch['y_vis'][0] > 0.5) & (batch['y_present'][0] > 0.5)).detach().cpu().numpy(),
        'adj': (batch['y_adj'][0] > 0.5).detach().cpu().numpy(),
    }

    return hist, pred, true

def _state_geoms(state, box_color, edge_color, sphere_radius=0.05):
    geoms = []
    idx = np.where(state['live'])[0]
    for j in idx:
        size = np.clip(state['size'][j], 0.05, None)
        geoms.append(make_cuboid(state['pos'][j], size, box_color))
        geoms.append(make_sphere(state['pos'][j], sphere_radius, box_color))
    geoms.extend(make_edge_lines(state['pos'], state['live'], state['adj'], edge_color))
    return geoms

def _lookat(scene_geoms, states):
    pts = []
    for state in states:
        idx = np.where(state['live'])[0]
        if len(idx):
            pts.append(state['pos'][idx].mean(axis=0).astype(np.float64))
    if pts:
        return np.mean(np.stack(pts, axis=0), axis=0)

    for g in scene_geoms:
        try:
            pts.append(np.asarray(g.get_center(), dtype=np.float64))
        except Exception:
            pass
    if pts:
        return np.mean(np.stack(pts, axis=0), axis=0)
    return np.zeros(3, dtype=np.float64)

def _render_image(geoms, cfg, out_path, lookat, visible=False):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=cfg['render']['width'],
        height=cfg['render']['height'],
        visible=visible,
    )

    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.asarray(cfg['render']['background'], dtype=np.float64)
    opt.point_size = float(cfg['render']['point_size'])

    ctr = vis.get_view_control()
    ctr.set_front(np.asarray(cfg['render']['front'], dtype=np.float64))
    ctr.set_up(np.asarray(cfg['render']['up'], dtype=np.float64))
    ctr.set_zoom(float(cfg['render']['zoom']) * 1.45)
    ctr.set_lookat(np.asarray(lookat, dtype=np.float64))

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(out_path))
    vis.destroy_window()

def save_triptych_o3d(scene_geoms, hist, pred, true, cfg, out_path, visible=False):
    out_path = Path(out_path)
    tmp_dir = out_path.parent / f'.{out_path.stem}_parts'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    lookat = _lookat(scene_geoms, [hist, pred, true])

    active_pts = []
    for state in [hist, pred, true]:
        idx = np.where(state['live'])[0]
        if len(idx):
            active_pts.append(state['pos'][idx])
    if active_pts:
        active_pts = np.concatenate(active_pts, axis=0)
        center = active_pts.mean(axis=0)
        radius = max(1.2, float(np.linalg.norm(active_pts.max(axis=0) - active_pts.min(axis=0)) * 1.2))
        scene_geoms = crop_scene_geoms(scene_geoms, center, radius)

    states = [
        ('history', hist, cfg['render']['history_color'], cfg['render']['edge_history_color']),
        ('pred', pred, cfg['render']['pred_color'], cfg['render']['edge_pred_color']),
        ('true', true, cfg['render']['true_color'], cfg['render']['edge_true_color']),
    ]

    panel_paths = []

    for name, state, box_color, edge_color in states:
        geoms = [copy.deepcopy(g) for g in scene_geoms]
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(cfg['render']['axis_size'])))
        geoms.extend(_state_geoms(state, box_color, edge_color))

        panel_path = tmp_dir / f'{name}.png'
        _render_image(geoms, cfg, panel_path, lookat, visible=visible)
        panel_paths.append(panel_path)

    imgs = [imageio.imread(p) for p in panel_paths]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, name in zip(axes, imgs, ['history', 'pred', 'true']):
        ax.imshow(img)
        ax.set_title(name)
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def save_overlay_o3d(scene_geoms, hist, pred, true, cfg, out_path, visible=False):
    out_path = Path(out_path)
    lookat = _lookat(scene_geoms, [hist, pred, true])

    active_pts = []
    for state in [hist, pred, true]:
        idx = np.where(state['live'])[0]
        if len(idx):
            active_pts.append(state['pos'][idx])
    if active_pts:
        active_pts = np.concatenate(active_pts, axis=0)
        center = active_pts.mean(axis=0)
        radius = max(1.2, float(np.linalg.norm(active_pts.max(axis=0) - active_pts.min(axis=0)) * 1.2))
        scene_geoms = crop_scene_geoms(scene_geoms, center, radius)

    geoms = [copy.deepcopy(g) for g in scene_geoms]
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(cfg['render']['axis_size'])))
    geoms.extend(_state_geoms(hist, cfg['render']['history_color'], cfg['render']['edge_history_color']))
    geoms.extend(_state_geoms(pred, cfg['render']['pred_color'], cfg['render']['edge_pred_color']))
    geoms.extend(_state_geoms(true, cfg['render']['true_color'], cfg['render']['edge_true_color']))

    _render_image(geoms, cfg, out_path, lookat, visible=visible)
