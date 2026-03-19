from pathlib import Path
import argparse
import copy
import math
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from future_scene_graphs.config import load_config
from future_scene_graphs.render3d import _render_image


BG_DARK = (10, 12, 18)
WHITE = (240, 242, 247)
MUTED = (165, 170, 180)


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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


def resize_contain(img, out_w, out_h, bg=(0, 0, 0)):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((out_h, out_w, 3), bg, dtype=np.uint8)

    scale = min(out_w / w, out_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((out_h, out_w, 3), bg, dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def save_rgb(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def annotate_frame(img, title, subtitle):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 120), BG_DARK, -1)
    put_text(out, title, 50, 60, scale=1.25, color=WHITE, thick=3)
    put_text(out, subtitle, 52, 96, scale=0.80, color=MUTED, thick=2)
    return out


def match_frame_size(img, out_w=1920, out_h=1080, bg=BG_DARK):
    return resize_contain(img, out_w, out_h, bg=bg)


def crossfade(a, b, steps):
    if a.shape != b.shape:
        raise ValueError(f"crossfade shape mismatch: {a.shape} vs {b.shape}")
    frames = []
    for i in range(1, steps + 1):
        t = i / steps
        frames.append(cv2.addWeighted(a, 1.0 - t, b, t, 0))
    return frames


def beautify_mesh(mesh, min_component_tris=3000, smooth_iters=6, simplify_ratio=0.98):
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
    if target > 5000 and target < len(mesh.triangles):
        mesh = mesh.simplify_quadric_decimation(target)

    mesh = mesh.filter_smooth_taubin(number_of_iterations=int(smooth_iters))
    mesh.compute_vertex_normals()
    return mesh


def make_floor(mesh):
    bbox = mesh.get_axis_aligned_bounding_box()
    ext = bbox.get_extent()
    ctr = bbox.get_center()

    sx = max(2.0, float(ext[0] * 1.4))
    sz = max(2.0, float(ext[2] * 1.4))
    sy = 0.01

    floor = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)
    floor.paint_uniform_color([0.92, 0.93, 0.96])

    tx = ctr[0] - sx / 2.0
    ty = bbox.min_bound[1] - 0.03
    tz = ctr[2] - sz / 2.0
    floor.translate([tx, ty, tz])
    floor.compute_vertex_normals()
    return floor


def render_mesh_frame(mesh_geoms, cfg, lookat, front, up, zoom, out_path, visible=False):
    cfg2 = copy.deepcopy(cfg)
    cfg2["render"]["width"] = 1920
    cfg2["render"]["height"] = 1080
    cfg2["render"]["zoom"] = float(zoom)
    cfg2["render"]["front"] = [float(front[0]), float(front[1]), float(front[2])]
    cfg2["render"]["up"] = [float(up[0]), float(up[1]), float(up[2])]
    cfg2["render"]["background"] = [0.97, 0.98, 1.0]
    cfg2["render"]["axis_size"] = 0.0

    _render_image(mesh_geoms, cfg2, out_path, lookat, visible=visible)


def make_room_board(scene_img, overlay_img, trip_img, title, subtitle, out_path):
    W, H = 1920, 1080
    canvas = np.full((H, W, 3), BG_DARK, dtype=np.uint8)

    put_text(canvas, title, 50, 60, scale=1.35, color=WHITE, thick=3)
    put_text(canvas, subtitle, 52, 98, scale=0.80, color=MUTED, thick=2)

    # big room panel
    room = resize_contain(scene_img, 1820, 560, bg=(245, 247, 250))
    canvas[140:140 + room.shape[0], 50:50 + room.shape[1]] = room
    cv2.rectangle(canvas, (50, 140), (1870, 700), (90, 170, 255), 2)

    # bottom left overlay
    over = resize_contain(overlay_img, 900, 300, bg=(0, 0, 0))
    canvas[740:740 + over.shape[0], 50:50 + over.shape[1]] = over
    cv2.rectangle(canvas, (50, 740), (950, 1040), (255, 170, 90), 2)

    # bottom right triptych
    trip = resize_contain(trip_img, 860, 300, bg=(245, 245, 245))
    canvas[740:740 + trip.shape[0], 1010:1010 + trip.shape[1]] = trip
    cv2.rectangle(canvas, (1010, 740), (1870, 1040), (110, 215, 130), 2)

    put_text(canvas, "Room-scale mesh context", 60, 730, scale=0.70, color=WHITE, thick=2)
    put_text(canvas, "Predicted future graph overlay", 60, 1070, scale=0.68, color=MUTED, thick=2)
    put_text(canvas, "History / prediction / ground truth", 1015, 1070, scale=0.68, color=MUTED, thick=2)

    save_rgb(out_path, canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hybrid_mask.yaml")
    ap.add_argument("--scene-mesh", required=True)
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--triptych", default=None)
    ap.add_argument("--title", default="Future 3D Scene Graph Prediction")
    ap.add_argument("--subtitle", default="Room-scale scene mesh + future graph overlay")
    ap.add_argument("--out-dir", default="runs/cinematic_recon")
    ap.add_argument("--visible", type=int, default=0)
    ap.add_argument("--zoom", type=float, default=0.72)
    ap.add_argument("--front", nargs=3, type=float, default=[0.35, -0.12, -0.93])
    ap.add_argument("--up", nargs=3, type=float, default=[0.0, 1.0, 0.0])
    ap.add_argument("--add-floor", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(args.scene_mesh))
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"empty mesh: {args.scene_mesh}")

    mesh = beautify_mesh(mesh, min_component_tris=3000, smooth_iters=6, simplify_ratio=0.98)
    o3d.io.write_triangle_mesh(str(out_dir / "scene_mesh_beautified.ply"), mesh)

    geoms = [mesh]
    if int(args.add_floor):
        geoms = [make_floor(mesh), mesh]

    bbox = mesh.get_axis_aligned_bounding_box()
    lookat = bbox.get_center()

    still_front = out_dir / "scene_front.png"
    still_left = out_dir / "scene_left.png"
    still_right = out_dir / "scene_right.png"

    render_mesh_frame(
        geoms, cfg, lookat,
        front=args.front,
        up=args.up,
        zoom=args.zoom,
        out_path=still_front,
        visible=bool(args.visible),
    )
    render_mesh_frame(
        geoms, cfg, lookat,
        front=[0.72, -0.10, -0.69],
        up=args.up,
        zoom=args.zoom,
        out_path=still_left,
        visible=bool(args.visible),
    )
    render_mesh_frame(
        geoms, cfg, lookat,
        front=[-0.62, -0.10, -0.78],
        up=args.up,
        zoom=args.zoom,
        out_path=still_right,
        visible=bool(args.visible),
    )

    orbit_mp4 = out_dir / "scene_orbit.mp4"
    writer = imageio.get_writer(orbit_mp4, fps=12, macro_block_size=1)

    n = 96
    for i in range(n):
        theta = -1.0 + (2.0 * i / max(1, n - 1))
        front = np.array([math.sin(theta) * 0.78, -0.10, -math.cos(theta) * 0.95], dtype=np.float32)
        front = front / max(1e-8, np.linalg.norm(front))
        p = frames_dir / f"orbit_{i:04d}.png"
        render_mesh_frame(geoms, cfg, lookat, front=front, up=args.up, zoom=args.zoom, out_path=p, visible=False)
        img = load_rgb(p)
        img = annotate_frame(img, args.title, "Room-scale scene mesh")
        img = match_frame_size(img, 1920, 1080, bg=BG_DARK)
        writer.append_data(img)
    writer.close()

    if args.overlay and args.triptych:
        overlay = load_rgb(args.overlay)
        trip = load_rgb(args.triptych)
        front_img = load_rgb(still_front)

        front_img = annotate_frame(front_img, args.title, "Room-scale scene mesh")
        overlay = annotate_frame(overlay, args.title, "Predicted future graph in local reconstructed scene")
        trip = annotate_frame(trip, args.title, "History / prediction / ground truth")

        front_img = match_frame_size(front_img, 1920, 1080, bg=BG_DARK)
        overlay = match_frame_size(overlay, 1920, 1080, bg=BG_DARK)
        trip = match_frame_size(trip, 1920, 1080, bg=BG_DARK)

        teaser_mp4 = out_dir / "scene_reveal.mp4"
        writer = imageio.get_writer(teaser_mp4, fps=12, macro_block_size=1)

        hold = 20
        for _ in range(hold):
            writer.append_data(front_img)
        for f in crossfade(front_img, overlay, 12):
            writer.append_data(f)
        for _ in range(hold):
            writer.append_data(overlay)
        for f in crossfade(overlay, trip, 12):
            writer.append_data(f)
        for _ in range(hold):
            writer.append_data(trip)
        writer.close()

        make_room_board(front_img, overlay, trip, args.title, args.subtitle, out_dir / "room_board.png")
        print("saved", out_dir / "scene_reveal.mp4")
        print("saved", out_dir / "room_board.png")

    print("saved", still_front)
    print("saved", still_left)
    print("saved", still_right)
    print("saved", orbit_mp4)


if __name__ == "__main__":
    main()
