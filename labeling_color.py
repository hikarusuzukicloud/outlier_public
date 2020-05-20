import os
import pickle
import time
import sys

import numpy as np
import open3d as o3d

def makeset(points_np):
    s = set()
    for p in points_np:
        #p = np.round(p, 2)
        s.add(tuple(p))
    return s

def set2pcd(set_true, set_false):
    points_false = np.array(list(set_false))
    normals_false = np.zeros((points_false.shape[0], 3))
    points_true = np.array(list(set_true))
    normals_true = np.ones((points_true.shape[0], 3))

    points_base = np.concatenate([points_false, points_true])
    normals_base = np.concatenate([normals_false, normals_true])

    points_base = np.round(points_base, 1)

    pcd_fromnp = o3d.geometry.PointCloud()
    pcd_fromnp.points = o3d.utility.Vector3dVector(points_base)
    pcd_fromnp.normals = o3d.utility.Vector3dVector(normals_base)

    return pcd_fromnp

def main():
    s=time.time()
    pcd = o3d.io.read_point_cloud("cat_raw_icp_0005_round1.ply")
    pcd_true = o3d.io.read_point_cloud("cat_raw_ccicp_BHremove_icp_0005_round1.ply")
    #pcd = o3d.io.read_point_cloud("cat_raw_all_ccicp.ply")
    #pcd_true = o3d.io.read_point_cloud("cat_raw_ccicp_b_remove_b195_handremove.ply")
    print(time.time()-s)

    set_true = makeset(np.array(pcd_true.points, dtype=np.float64))
    set_all = makeset(np.array(pcd.points, dtype=np.float64))
    print(time.time()-s)
    print(len(set_all))
    print(len(set_true))

    set_zero = set_true - set_all

    set_true = set_true - set_zero
    set_all = set_all - set_zero

    set_false = set_all - set_true

    print(len(set_zero))
    print(len(set_all))
    print(len(set_true))
    print(len(set_false))
    print(time.time()-s)

    pcd_labeled = set2pcd(set_true, set_false)
    print(time.time()-s)
    o3d.io.write_point_cloud("cat_raw_icp_labeled_BHremove_round1.ply", pcd_labeled)
    print(time.time()-s)


if __name__ == "__main__":
    main()
