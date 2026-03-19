from pathlib import Path
import argparse
import copy
import json

import numpy as np
import open3d as o3d


def light_cleanup(mesh):
    mesh = copy.deepcopy(mesh)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return mesh

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.76, 0.78, 0.81])

    return mesh


def load_mesh(path, mesh_style):
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"empty mesh: {path}")

    if mesh_style == "keep":
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        if not mesh.has_vertex_colors():
            mesh.paint_uniform_color([0.76, 0.78, 0.81])
        return mesh

    return light_cleanup(mesh)


def apply_camera_json(vis, cam):
    ctr = vis.get_view_control()
    ctr.set_front(np.asarray(cam["front"], dtype=np.float64))
    ctr.set_up(np.asarray(cam["up"], dtype=np.float64))
    ctr.set_lookat(np.asarray(cam["lookat"], dtype=np.float64))
    ctr.set_zoom(float(cam["zoom"]))


def save_camera_bundle(vis, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()

    pinhole_path = out_dir / "room_camera_pinhole.json"
    preview_path = out_dir / "room_camera_preview.png"
    matrix_path = out_dir / "room_camera_matrices.json"

    o3d.io.write_pinhole_camera_parameters(str(pinhole_path), params)

    bundle = {
        "extrinsic": np.asarray(params.extrinsic).tolist(),
        "intrinsic_matrix": np.asarray(params.intrinsic.intrinsic_matrix).tolist(),
        "width": int(params.intrinsic.width),
        "height": int(params.intrinsic.height),
    }
    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(preview_path))

    print("saved", pinhole_path)
    print("saved", matrix_path)
    print("saved", preview_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-mesh", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--camera-json", default=None)
    ap.add_argument("--mesh-style", choices=["keep", "light_cleanup"], default="keep")
    ap.add_argument("--width", type=int, default=1800)
    ap.add_argument("--height", type=int, default=1100)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(args.scene_mesh, args.mesh_style)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Tune room camera", width=args.width, height=args.height, visible=True)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.97, 0.98, 0.99], dtype=np.float64)
    opt.mesh_show_back_face = True
    opt.light_on = True

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front(np.asarray([0.35, -0.18, -0.92], dtype=np.float64))
    ctr.set_up(np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
    ctr.set_zoom(0.82)

    if args.camera_json:
        cam = json.load(open(args.camera_json, "r", encoding="utf-8"))
        apply_camera_json(vis, cam)

    def save_current(v):
        save_camera_bundle(v, out_dir)
        return False

    vis.register_key_callback(ord("P"), save_current)
    vis.register_key_callback(ord("S"), save_current)

    print()
    print("Rotate to the exact room angle you want.")
    print("Press P to save camera params + screenshot.")
    print("Close the window when done.")
    print()

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
