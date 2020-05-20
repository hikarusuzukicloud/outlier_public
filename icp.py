# examples/Python/Basic/icp_registration.py

import open3d as o3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == "__main__":
    source = o3d.io.read_point_cloud("./cat_raw_filter_remove_outlier_ccicp.ply")
    #source = o3d.io.read_point_cloud("../cat_test_result_batch_repeat_repeat30k_round1.ply")
    #source = o3d.io.read_point_cloud("./filter_combine.ply")
    target = o3d.io.read_point_cloud("./cat_ground-truth.ply")
    threshold = 0.005
    """
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)
    """

    trans_init = np.asarray([[1,0,0,0],[0,1,0,0],
                             [0,0,1,0], [0.0, 0.0, 0.0, 1.0]])
    criteria = o3d.registration.ICPConvergenceCriteria(relative_fitness = 1e-6,
                                                       relative_rmse = 1e-6,
                                                       max_iteration = 50)
    print("Apply point-to-point ICP")
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint(),
        criteria=criteria)
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    #source.transform(reg_p2p.transformation)
    #o3d.io.write_point_cloud("cat_raw_ccicp_BHremove_icp_0005.ply", source)
    #draw_registration_result(source, target, reg_p2p.transformation)
    """
    print("Apply point-to-plane ICP")
    reg_p2l = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    print("")
    draw_registration_result(source, target, reg_p2l.transformation)
    """
