from pathlib import Path
import argparse
import copy
import sys

import numpy as np
import open3d as o3d
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from future_scene_graphs.config import load_config
from future_scene_graphs.dataset import GraphWindowDataset
from future_scene_graphs.models import SceneGraphForecaster
from future_scene_graphs.render3d import make_states


HIST_COLOR = [0.20, 0.52, 0.96]
PRED_COLOR = [1.00, 0.52, 0.10]
MOTION_COLOR = [0.98, 0.75, 0.26]
BG = np.asarray([0.97, 0.98, 0.99], dtype=np.float64)


def keep_mesh(mesh):
    mesh = copy.deepcopy(mesh)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return mesh
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.76, 0.78, 0.81])
    return mesh


def _box_corners(center, size):
    cx, cy, cz = [float(x) for x in center]
    sx, sy, sz = [float(x) for x in size]
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
    s = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
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

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=float(radius), height=float(L))
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


def make_state_geoms(state, color, sphere_radius):
    geoms = []
    ids = np.where(state["live"])[0].tolist()
    for j in ids:
        size = np.clip(state["size"][j], 0.05, None)
        geoms.append(make_cuboid(state["pos"][j], size, color))
        geoms.append(make_sphere(state["pos"][j], sphere_radius, color))
    return geoms


def make_motion_trails(hist, pred):
    geoms = []
    ids = np.where(hist["live"] & pred["live"])[0].tolist()
    for j in ids:
        p0 = hist["pos"][j]
        p1 = pred["pos"][j]
        if np.linalg.norm(p1 - p0) < 0.04:
            continue
        c = make_cylinder(p0, p1, 0.012, MOTION_COLOR)
        if c is not None:
            geoms.append(c)
    return geoms


def render_one(geoms, pinhole_path, out_path, width, height):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="render", width=width, height=height, visible=False)

    opt = vis.get_render_option()
    opt.background_color = BG
    opt.mesh_show_back_face = True
    opt.light_on = True

    for g in geoms:
        vis.add_geometry(g)

    vis.poll_events()
    vis.update_renderer()

    params = o3d.io.read_pinhole_camera_parameters(str(pinhole_path))
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(out_path))
    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--scene-mesh", required=True)
    ap.add_argument("--camera-pinhole", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=2800)
    ap.add_argument("--height", type=int, default=1800)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    ds = GraphWindowDataset(args.windows)
    sample = ds[args.idx]

    batch = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0)
        else:
            batch[k] = v

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SceneGraphForecaster(
        cfg["train"]["hidden_dim"],
        cfg["train"]["dropout"],
        cfg["train"]["use_camera_motion"],
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    hist, pred, true_state = make_states(model, batch, device)

    mesh = o3d.io.read_triangle_mesh(str(args.scene_mesh))
    mesh = keep_mesh(mesh)

    room_geoms = [copy.deepcopy(mesh)]
    hist_geoms = [copy.deepcopy(mesh)] + make_state_geoms(hist, HIST_COLOR, 0.050)
    pred_geoms = [copy.deepcopy(mesh)] + make_state_geoms(hist, HIST_COLOR, 0.040) + make_state_geoms(pred, PRED_COLOR, 0.065) + make_motion_trails(hist, pred)

    render_one(room_geoms, args.camera_pinhole, out_dir / "room_only_raw.png", args.width, args.height)
    render_one(hist_geoms, args.camera_pinhole, out_dir / "room_history_raw.png", args.width, args.height)
    render_one(pred_geoms, args.camera_pinhole, out_dir / "room_pred_raw.png", args.width, args.height)

    print("saved", out_dir / "room_only_raw.png")
    print("saved", out_dir / "room_history_raw.png")
    print("saved", out_dir / "room_pred_raw.png")


if __name__ == "__main__":
    main()
