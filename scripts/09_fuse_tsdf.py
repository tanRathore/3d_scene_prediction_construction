from pathlib import Path
import argparse
import sys

import cv2
import numpy as np
import open3d as o3d

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from future_scene_graphs.config import load_config
from future_scene_graphs.geometry import depth_to_meters
from future_scene_graphs.io_utils import (
    list_stems,
    read_depth,
    read_image,
    read_pose,
    read_intrinsics_for_stem,
)


def _img_path(rgb_dir, stem, ext):
    for p in [
        rgb_dir / f"{stem}.{ext}",
        rgb_dir / f"{stem}.jpg",
        rgb_dir / f"{stem}.png",
        rgb_dir / f"{stem}.jpeg",
    ]:
        if p.exists():
            return p
    return None


def _depth_path(depth_dir, stem, ext):
    for p in [
        depth_dir / f"{stem}.{ext}",
        depth_dir / f"{stem}.png",
        depth_dir / f"{stem}.npy",
    ]:
        if p.exists():
            return p
    return None


def _pick_stride_limit(stems, start=0, stop=-1, stride=1, limit=-1):
    if stop > 0:
        picked = stems[start:stop:stride]
    else:
        picked = stems[start::stride]
    if limit > 0:
        picked = picked[:limit]
    return picked


def _pick_local(stems, center_stem, frame_radius):
    if center_stem is None or frame_radius < 0:
        return stems
    if center_stem in stems:
        idx = stems.index(center_stem)
    else:
        vals = np.array([float(s.split('_')[-1]) if '_' in s else float(s) for s in stems], dtype=np.float64)
        target = float(center_stem.split('_')[-1]) if '_' in center_stem else float(center_stem)
        idx = int(np.argmin(np.abs(vals - target)))
    lo = max(0, idx - frame_radius)
    hi = min(len(stems), idx + frame_radius + 1)
    return stems[lo:hi]


def _resize_depth_to_rgb(depth_m, rgb_shape):
    h, w = rgb_shape[:2]
    if depth_m.shape[:2] == (h, w):
        return depth_m
    return cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)


def _make_intrinsic(width, height, intr):
    return o3d.camera.PinholeCameraIntrinsic(
        int(width),
        int(height),
        float(intr["fx"]),
        float(intr["fy"]),
        float(intr["cx"]),
        float(intr["cy"]),
    )


def _clean_mesh(mesh, min_component_triangles=2000):
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
        keep = cluster_n_triangles[tri_clusters] >= int(min_component_triangles)
        if keep.sum() == 0:
            biggest = int(np.argmax(cluster_n_triangles))
            keep = tri_clusters == biggest
        mesh.remove_triangles_by_mask(~keep)
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    return mesh


def _default_out_paths(seq_dir, center_stem):
    if center_stem is None:
        return (
            seq_dir / "scene_mesh_fused.ply",
            seq_dir / "scene_points_fused.ply",
        )
    safe = center_stem.replace("/", "_").replace(".", "_")
    return (
        seq_dir / f"scene_mesh_fused_local_{safe}.ply",
        seq_dir / f"scene_points_fused_local_{safe}.ply",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sequence-dir", default=None)
    ap.add_argument("--sequences-root", default="data/sequences")
    ap.add_argument("--sequence-id", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--center-stem", default=None)
    ap.add_argument("--frame-radius", type=int, default=-1)
    ap.add_argument("--voxel-length", type=float, default=0.03)
    ap.add_argument("--sdf-trunc", type=float, default=0.12)
    ap.add_argument("--depth-trunc", type=float, default=None)
    ap.add_argument("--pcd-voxel", type=float, default=0.02)
    ap.add_argument("--min-component-triangles", type=int, default=2000)
    ap.add_argument("--out-mesh", default=None)
    ap.add_argument("--out-pcd", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    if args.sequence_dir is not None:
        seq_dir = Path(args.sequence_dir)
    else:
        if not args.sequence_id:
            raise ValueError("pass --sequence-dir or --sequence-id")
        seq_dir = Path(args.sequences_root) / args.sequence_id

    rgb_dir = seq_dir / data_cfg["rgb_subdir"]
    depth_dir = seq_dir / data_cfg["depth_subdir"]
    pose_dir = seq_dir / data_cfg["pose_subdir"]

    if not rgb_dir.exists() or not depth_dir.exists() or not pose_dir.exists():
        raise FileNotFoundError(f"bad sequence dir: {seq_dir}")

    stems = list_stems(rgb_dir, (data_cfg["image_ext"], "png", "jpg", "jpeg"))
    stems = _pick_stride_limit(
        stems,
        start=args.start,
        stop=args.stop,
        stride=max(1, args.stride),
        limit=args.limit,
    )
    stems = _pick_local(stems, args.center_stem, args.frame_radius)

    if not stems:
        raise RuntimeError("no frames selected")

    depth_trunc = float(args.depth_trunc) if args.depth_trunc is not None else float(data_cfg["max_depth_m"])

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(args.voxel_length),
        sdf_trunc=float(args.sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    fused = 0

    print("frames", len(stems))
    print("first", stems[0])
    print("last", stems[-1])

    for i, stem in enumerate(stems):
        img_path = _img_path(rgb_dir, stem, data_cfg["image_ext"])
        dep_path = _depth_path(depth_dir, stem, data_cfg["depth_ext"])
        pose_path = pose_dir / f"{stem}.txt"

        if img_path is None or dep_path is None or not pose_path.exists():
            continue

        image = read_image(img_path)
        depth_m = depth_to_meters(read_depth(dep_path))
        depth_m = _resize_depth_to_rgb(depth_m, image.shape)

        valid = (depth_m >= float(data_cfg["min_depth_m"])) & (depth_m <= depth_trunc)
        depth_m = np.where(valid, depth_m, 0.0).astype(np.float32)

        intr = read_intrinsics_for_stem(
            seq_dir,
            stem=stem,
            intrinsics_subdir=data_cfg.get("intrinsics_subdir", "intrinsics"),
            intrinsics_file=data_cfg["intrinsics_file"],
        )
        pose_c2w = read_pose(pose_path)
        extrinsic_w2c = np.linalg.inv(pose_c2w).astype(np.float64)

        h, w = image.shape[:2]
        intrinsic = _make_intrinsic(w, h, intr)

        color_o3d = o3d.geometry.Image(np.ascontiguousarray(image.astype(np.uint8)))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_m.astype(np.float32)))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        volume.integrate(rgbd, intrinsic, extrinsic_w2c)
        fused += 1

        if (i + 1) % 10 == 0 or i == len(stems) - 1:
            print(f"fused {i + 1}/{len(stems)}")

    if fused == 0:
        raise RuntimeError("no frames fused")

    mesh = volume.extract_triangle_mesh()
    mesh = _clean_mesh(mesh, min_component_triangles=args.min_component_triangles)

    pcd = volume.extract_point_cloud()
    if args.pcd_voxel > 0 and len(pcd.points):
        pcd = pcd.voxel_down_sample(float(args.pcd_voxel))

    default_mesh, default_pcd = _default_out_paths(seq_dir, args.center_stem)
    out_mesh = Path(args.out_mesh) if args.out_mesh else default_mesh
    out_pcd = Path(args.out_pcd) if args.out_pcd else default_pcd
    out_mesh.parent.mkdir(parents=True, exist_ok=True)
    out_pcd.parent.mkdir(parents=True, exist_ok=True)

    ok_mesh = o3d.io.write_triangle_mesh(str(out_mesh), mesh)
    ok_pcd = o3d.io.write_point_cloud(str(out_pcd), pcd)

    print("mesh", out_mesh, ok_mesh)
    print("pcd", out_pcd, ok_pcd)
    print("verts", len(mesh.vertices), "tris", len(mesh.triangles), "pts", len(pcd.points))


if __name__ == "__main__":
    main()
