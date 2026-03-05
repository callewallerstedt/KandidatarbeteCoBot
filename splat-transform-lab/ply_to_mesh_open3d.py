#!/usr/bin/env python3
import argparse
import sys
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print('ERROR: open3d is not installed for this Python environment.', flush=True)
    print(f'Python in use: {sys.executable}', flush=True)
    print('Install with: python -m pip install open3d numpy', flush=True)
    raise


def main():
    ap = argparse.ArgumentParser(description='Convert PLY point cloud to mesh using Open3D Poisson.')
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--voxel', type=float, default=0.01)
    ap.add_argument('--normal-radius', type=float, default=0.05)
    ap.add_argument('--normal-nn', type=int, default=30)
    ap.add_argument('--poisson-depth', type=int, default=9)
    ap.add_argument('--trim-quantile', type=float, default=0.02)
    args = ap.parse_args()

    print(f'[mesh] reading: {args.input}', flush=True)
    pcd = o3d.io.read_point_cloud(args.input)
    if pcd.is_empty():
        raise RuntimeError('Input point cloud is empty or invalid.')

    print(f'[mesh] points before downsample: {len(pcd.points)}', flush=True)
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel)
    print(f'[mesh] points after downsample: {len(pcd.points)}', flush=True)

    print('[mesh] estimating normals...', flush=True)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.normal_radius,
            max_nn=args.normal_nn,
        )
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    print('[mesh] running poisson reconstruction...', flush=True)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=args.poisson_depth,
    )

    d = np.asarray(densities)
    q = max(0.0, min(0.95, args.trim_quantile))
    thresh = np.quantile(d, q)
    remove_mask = d < thresh
    mesh.remove_vertices_by_mask(remove_mask)

    print('[mesh] cleaning mesh...', flush=True)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    print(f'[mesh] writing: {args.output}', flush=True)
    ok = o3d.io.write_triangle_mesh(args.output, mesh)
    if not ok:
        raise RuntimeError('Failed writing output mesh')

    print(f'[mesh] done. vertices={len(mesh.vertices)} triangles={len(mesh.triangles)}', flush=True)


if __name__ == '__main__':
    main()
