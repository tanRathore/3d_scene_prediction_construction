from pathlib import Path
import argparse
import copy
import json
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from future_scene_graphs.config import load_config
from future_scene_graphs.dataset import GraphWindowDataset
from future_scene_graphs.models import SceneGraphForecaster
from future_scene_graphs.render3d import make_states, _render_image


BG_DARK = (10, 12, 18)
WHITE = (243, 245, 248)
MUTED = (165, 171, 182)

ROOM_BG = [0.965, 0.972, 0.985]
HIST_COLOR = [0.20, 0.52, 0.96]
PRED_COLOR = [1.00, 0.52, 0.10]
ROI_COLOR = [0.24, 0.82, 0.98]
MOTION_COLOR = [0.98, 0.75, 0.26]


def put_text(img, text, x, y, scale=1.0, color=WHITE, thick=2):
    cv2.putText(
        img,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thick),
        lineType=cv2.LINE_AA,
    )


def save_rgb(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_contain(img, out_w, out_h, bg=(0, 0, 0)):
    h, w = img.shape[:2]
    scale = min(out_w / max(w, 1), out_h / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    rs = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((out_h, out_w, 3), bg, dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = rs
    return canvas


def estimate_bg(img, patch=36):
    h, w = img.shape[:2]
    p = min(patch, max(8, h // 6), max(8, w // 6))
    crops = [
        img[:p, :p],
        img[:p, w - p:w],
        img[h - p:h, :p],
        img[h - p:h, w - p:w],
    ]
    vals = np.concatenate([c.reshape(-1, 3) for c in crops], axis=0)
    return np.median(vals, axis=0)


def crop_uniform_bg(img, tol=16, pad=18):
    bg = estimate_bg(img)
    dist = np.linalg.norm(img.astype(np.float32) - bg.reshape(1, 1, 3).astype(np.float32), axis=2)
    mask = dist > tol
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return img
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(img.shape[0], int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(img.shape[1], int(xs.max()) + pad)
    return img[y0:y1, x0:x1]


def crop_nonblack(img, thr=10, pad=18):
    gray = img.max(axis=2)
    ys, xs = np.where(gray > thr)
    if len(xs) == 0 or len(ys) == 0:
        return img
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(img.shape[0], int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(img.shape[1], int(xs.max()) + pad)
    return img[y0:y1, x0:x1]


def annotate(img, title, subtitle):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 108), BG_DARK, -1)
    put_text(out, title, 44, 58, scale=1.18, color=WHITE, thick=3)
    put_text(out, subtitle, 46, 92, scale=0.72, color=MUTED, thick=2)
    return out


def crossfade(a, b, steps):
    frames = []
    for i in range(1, steps + 1):
        t = i / steps
        frames.append(cv2.addWeighted(a, 1.0 - t, b, t, 0))
    return frames


def enhance_mesh_colors(mesh):
    if mesh.has_vertex_colors():
        cols = np.asarray(mesh.vertex_colors).copy()
        cols = np.clip(cols * 1.08 + 0.02, 0.0, 1.0)
        mesh.vertex_colors = o3d.utility.Vector3dVector(cols)
    else:
        mesh.paint_uniform_color([0.76, 0.78, 0.81])
    return mesh


def beautify_mesh(mesh, min_component_tris=3000, smooth_iters=8, simplify_ratio=0.985):
    mesh = copy.deepcopy(mesh)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return mesh

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    tri_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if len(cluster_n_triangles):
        keep = cluster_n_triangles[tri_clusters] >= int(min_component_tris)
        if keep.sum() == 0:
            biggest = int(np.argmax(cluster_n_triangles))
            keep = tri_clusters == biggest
        mesh.remove_triangles_by_mask(~keep)
        mesh.remove_unreferenced_vertices()

    target = int(len(mesh.triangles) * float(simplify_ratio))
    if 5000 < target < len(mesh.triangles):
        mesh = mesh.simplify_quadric_decimation(target)

    mesh = mesh.filter_smooth_taubin(number_of_iterations=int(smooth_iters))
    mesh.compute_vertex_normals()
    mesh = enhance_mesh_colors(mesh)
    return mesh


def style_mesh(mesh):
    mesh = copy.deepcopy(mesh)
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


def make_edge_geoms(state, color, radius=0.010):
    geoms = []
    ids = np.where(state["live"])[0].tolist()
    for i in ids:
        for j in ids:
            if j > i and state["adj"][i, j]:
                c = make_cylinder(state["pos"][i], state["pos"][j], radius, color)
                if c is not None:
                    geoms.append(c)
    return geoms


def make_hist_geoms(hist):
    geoms = []
    ids = np.where(hist["live"])[0].tolist()
    for j in ids:
        size = np.clip(hist["size"][j], 0.05, None)
        geoms.append(make_cuboid(hist["pos"][j], size, HIST_COLOR))
        geoms.append(make_sphere(hist["pos"][j], 0.028, HIST_COLOR))
    return geoms


def make_pred_geoms(pred):
    geoms = []
    ids = np.where(pred["live"])[0].tolist()
    for j in ids:
        size = np.clip(pred["size"][j], 0.05, None)
        geoms.append(make_cuboid(pred["pos"][j], size, PRED_COLOR))
        geoms.append(make_sphere(pred["pos"][j], 0.040, PRED_COLOR))
    geoms.extend(make_edge_geoms(pred, PRED_COLOR, radius=0.011))
    return geoms


def make_motion_trails(hist, pred):
    geoms = []
    ids = np.where(hist["live"] & pred["live"])[0].tolist()
    for j in ids:
        p0 = hist["pos"][j]
        p1 = pred["pos"][j]
        if np.linalg.norm(p1 - p0) < 0.04:
            continue
        c = make_cylinder(p0, p1, 0.0065, MOTION_COLOR)
        if c is not None:
            geoms.append(c)
    return geoms


def make_roi_box(hist, pred, true_state=None):
    pts = []
    for state in [hist, pred] if true_state is None else [hist, pred, true_state]:
        idx = np.where(state["live"])[0]
        if len(idx):
            pts.append(state["pos"][idx])

    if not pts:
        return []

    pts = np.concatenate(pts, axis=0)
    lo = pts.min(axis=0) - np.array([0.28, 0.18, 0.28], dtype=np.float32)
    hi = pts.max(axis=0) + np.array([0.28, 0.18, 0.28], dtype=np.float32)
    center = (lo + hi) / 2.0
    size = np.maximum(hi - lo, np.array([0.25, 0.25, 0.25], dtype=np.float32))
    return [make_cuboid(center, size, ROI_COLOR)]


def render_geoms(geoms, cfg, out_path, lookat, front, up, zoom, visible=False, width=1920, height=1080):
    cfg2 = copy.deepcopy(cfg)
    cfg2["render"]["width"] = int(width)
    cfg2["render"]["height"] = int(height)
    cfg2["render"]["zoom"] = float(zoom)
    cfg2["render"]["front"] = [float(front[0]), float(front[1]), float(front[2])]
    cfg2["render"]["up"] = [float(up[0]), float(up[1]), float(up[2])]
    cfg2["render"]["background"] = [float(ROOM_BG[0]), float(ROOM_BG[1]), float(ROOM_BG[2])]
    cfg2["render"]["axis_size"] = 0.0
    cfg2["render"]["point_size"] = 2.0
    _render_image(geoms, cfg2, out_path, lookat, visible=visible)


def choose_lookat(mesh, pred):
    bbox = mesh.get_axis_aligned_bounding_box()
    scene_ctr = bbox.get_center()
    idx = np.where(pred["live"])[0]
    if len(idx):
        pred_ctr = pred["pos"][idx].mean(axis=0).astype(np.float64)
        return 0.72 * scene_ctr + 0.28 * pred_ctr
    return scene_ctr


def make_mesh_slice(mesh, axis, thresh):
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    if len(v) == 0 or len(t) == 0:
        return o3d.geometry.TriangleMesh()

    keep_tri = np.all(v[t][:, :, axis] <= thresh, axis=1)
    tris = t[keep_tri]
    if len(tris) == 0:
        return o3d.geometry.TriangleMesh()

    used = np.unique(tris.reshape(-1))
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    new_v = v[used]
    new_t = remap[tris]

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(new_v)
    out.triangles = o3d.utility.Vector3iVector(new_t)

    if mesh.has_vertex_colors():
        cols = np.asarray(mesh.vertex_colors)
        out.vertex_colors = o3d.utility.Vector3dVector(cols[used])

    out.compute_vertex_normals()
    return out


def make_board(hero_img, room_img, overlay_img, triptych_img, eval_json, out_path):
    W, H = 2560, 1440
    canvas = np.full((H, W, 3), BG_DARK, dtype=np.uint8)

    put_text(canvas, "Future 3D Semantic Scene Graph Prediction", 56, 82, scale=1.42, color=WHITE, thick=3)
    put_text(canvas, "Room-scale scene context + forecasted future graph overlay", 58, 124, scale=0.76, color=MUTED, thick=2)

    hero = resize_contain(hero_img, 1600, 820, bg=(245, 247, 250))
    room = resize_contain(room_img, 760, 360, bg=(245, 247, 250))
    overlay = resize_contain(crop_nonblack(overlay_img, thr=10, pad=22), 760, 360, bg=(0, 0, 0))
    trip = resize_contain(crop_uniform_bg(triptych_img, tol=10, pad=8), 760, 250, bg=(245, 245, 245))

    canvas[160:160 + hero.shape[0], 48:48 + hero.shape[1]] = hero
    canvas[160:160 + room.shape[0], 1750:1750 + room.shape[1]] = room
    canvas[580:580 + overlay.shape[0], 1750:1750 + overlay.shape[1]] = overlay
    canvas[1040:1040 + trip.shape[0], 48:48 + trip.shape[1]] = trip

    cv2.rectangle(canvas, (48, 160), (1648, 980), (90, 170, 255), 2)
    cv2.rectangle(canvas, (1750, 160), (2510, 520), (130, 130, 145), 2)
    cv2.rectangle(canvas, (1750, 580), (2510, 940), (255, 170, 90), 2)
    cv2.rectangle(canvas, (48, 1040), (808, 1290), (110, 215, 130), 2)

    put_text(canvas, "Full-room forecast hero", 58, 148, scale=0.64, color=WHITE, thick=2)
    put_text(canvas, "Room mesh only", 1760, 148, scale=0.60, color=WHITE, thick=2)
    put_text(canvas, "Local forecast overlay", 1760, 568, scale=0.60, color=WHITE, thick=2)
    put_text(canvas, "History / prediction / ground truth", 58, 1028, scale=0.60, color=WHITE, thick=2)

    put_text(canvas, "What this is showing", 890, 1060, scale=0.84, color=WHITE, thick=2)
    put_text(canvas, "Blue boxes = recent object-centric state.", 892, 1118, scale=0.60, color=MUTED, thick=2)
    put_text(canvas, "Orange boxes/edges = forecasted future graph.", 892, 1156, scale=0.60, color=MUTED, thick=2)
    put_text(canvas, "Yellow links = motion from history to forecast.", 892, 1194, scale=0.60, color=MUTED, thick=2)
    put_text(canvas, "The room mesh is context, not predicted geometry.", 892, 1232, scale=0.60, color=MUTED, thick=2)

    if eval_json is not None:
        put_text(canvas, "H=5 metrics", 1760, 1022, scale=0.84, color=WHITE, thick=2)
        put_text(canvas, f"Model L2       {eval_json['model']['l2']:.3f}", 1762, 1082, scale=0.64, color=(255, 190, 100), thick=2)
        put_text(canvas, f"Copy-last L2   {eval_json['copy_last']['l2']:.3f}", 1762, 1122, scale=0.56, color=MUTED, thick=2)
        put_text(canvas, f"Const-vel L2   {eval_json['constant_velocity']['l2']:.3f}", 1762, 1158, scale=0.56, color=MUTED, thick=2)
        put_text(canvas, f"Model Vis F1   {eval_json['model']['vis_f1']:.3f}", 1762, 1218, scale=0.64, color=(110, 210, 255), thick=2)
        put_text(canvas, f"Model Edge F1  {eval_json['model']['edge_f1']:.3f}", 1762, 1258, scale=0.64, color=(120, 235, 140), thick=2)

    save_rgb(out_path, canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--scene-mesh", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--local-overlay", default=None)
    ap.add_argument("--triptych", default=None)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--zoom", type=float, default=0.92)
    ap.add_argument("--front", nargs=3, type=float, default=[0.34, -0.12, -0.93])
    ap.add_argument("--up", nargs=3, type=float, default=[0.0, 1.0, 0.0])
    ap.add_argument("--build-steps", type=int, default=22)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--visible", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    tmp_dir = out_dir / "_tmp"
    build_dir = out_dir / "build_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

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
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"empty mesh: {args.scene_mesh}")

    mesh = beautify_mesh(mesh)
    mesh = style_mesh(mesh)
    beautified_path = out_dir / "scene_mesh_beautified_showcase.ply"
    o3d.io.write_triangle_mesh(str(beautified_path), mesh)

    lookat = choose_lookat(mesh, pred)

    room_raw = tmp_dir / "room_raw.png"
    hist_raw = tmp_dir / "hist_raw.png"
    hero_raw = tmp_dir / "hero_raw.png"

    render_geoms(
        [mesh],
        cfg,
        room_raw,
        lookat=lookat,
        front=args.front,
        up=args.up,
        zoom=args.zoom,
        visible=bool(args.visible),
    )

    hist_geoms = [copy.deepcopy(mesh)]
    hist_geoms.extend(make_roi_box(hist, pred, true_state))
    hist_geoms.extend(make_hist_geoms(hist))
    render_geoms(
        hist_geoms,
        cfg,
        hist_raw,
        lookat=lookat,
        front=args.front,
        up=args.up,
        zoom=args.zoom,
        visible=bool(args.visible),
    )

    hero_geoms = [copy.deepcopy(mesh)]
    hero_geoms.extend(make_roi_box(hist, pred, true_state))
    hero_geoms.extend(make_hist_geoms(hist))
    hero_geoms.extend(make_motion_trails(hist, pred))
    hero_geoms.extend(make_pred_geoms(pred))
    render_geoms(
        hero_geoms,
        cfg,
        hero_raw,
        lookat=lookat,
        front=args.front,
        up=args.up,
        zoom=args.zoom,
        visible=bool(args.visible),
    )

    room_img = annotate(
        resize_contain(crop_uniform_bg(load_rgb(room_raw), tol=14, pad=30), 1920, 1080, bg=(245, 247, 250)),
        "Future 3D Scene Graph Prediction",
        "Room-scale reconstructed scene",
    )
    hist_img = annotate(
        resize_contain(crop_uniform_bg(load_rgb(hist_raw), tol=14, pad=30), 1920, 1080, bg=(245, 247, 250)),
        "Future 3D Scene Graph Prediction",
        "Recent scene graph state inside room context",
    )
    hero_img = annotate(
        resize_contain(crop_uniform_bg(load_rgb(hero_raw), tol=14, pad=30), 1920, 1080, bg=(245, 247, 250)),
        "Future 3D Scene Graph Prediction",
        "Forecasted future graph inside room context",
    )

    room_out = out_dir / "room_only_tight.png"
    hist_out = out_dir / "room_history_tight.png"
    hero_out = out_dir / "room_pred_tight.png"
    save_rgb(room_out, room_img)
    save_rgb(hist_out, hist_img)
    save_rgb(hero_out, hero_img)

    bbox = mesh.get_axis_aligned_bounding_box()
    y_min = float(bbox.min_bound[1])
    y_max = float(bbox.max_bound[1])
    y_vals = np.linspace(y_min + 0.02 * (y_max - y_min), y_max, int(args.build_steps))

    build_frames = []
    for i, yv in enumerate(y_vals):
        part = make_mesh_slice(mesh, axis=1, thresh=float(yv))
        if len(part.vertices) == 0 or len(part.triangles) == 0:
            continue
        frame_path = build_dir / f"build_{i:03d}.png"
        render_geoms(
            [part],
            cfg,
            frame_path,
            lookat=lookat,
            front=args.front,
            up=args.up,
            zoom=args.zoom,
            visible=bool(args.visible),
        )
        frame = annotate(
            resize_contain(crop_uniform_bg(load_rgb(frame_path), tol=14, pad=30), 1920, 1080, bg=(245, 247, 250)),
            "Future 3D Scene Graph Prediction",
            "Room mesh build-up from the ground up",
        )
        build_frames.append(frame)

    frames = []
    frames.extend(build_frames)
    frames.extend([room_img] * 8)
    frames.extend(crossfade(room_img, hist_img, 8))
    frames.extend([hist_img] * 8)
    frames.extend(crossfade(hist_img, hero_img, 8))
    frames.extend([hero_img] * 12)

    gif_path = out_dir / "room_pred_build.gif"
    imageio.mimsave(gif_path, frames, fps=int(args.fps))

    mp4_path = out_dir / "room_pred_build.mp4"
    writer = imageio.get_writer(mp4_path, fps=int(args.fps), macro_block_size=1)
    for f in frames:
        writer.append_data(f)
    writer.close()

    if args.local_overlay and args.triptych:
        overlay = load_rgb(args.local_overlay)
        trip = load_rgb(args.triptych)
        eval_json = None
        if args.eval:
            with open(args.eval, "r", encoding="utf-8") as f:
                eval_json = json.load(f)
        board_path = out_dir / "room_pred_showcase_board.png"
        make_board(hero_img, room_img, overlay, trip, eval_json, board_path)
        print("saved", board_path)

    print("saved", beautified_path)
    print("saved", room_out)
    print("saved", hist_out)
    print("saved", hero_out)
    print("saved", gif_path)
    print("saved", mp4_path)


if __name__ == "__main__":
    main()
