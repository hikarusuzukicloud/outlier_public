import os
import sys

import numpy as np
import open3d as o3d

def refine():
    pcd = o3d.io.read_point_cloud("cat_raw_icp_0005_round1_unique.ply")
    pcd_labeled = o3d.io.read_point_cloud("cat_raw_icp_labeled_BHremove_round1.ply")
    vol = o3d.visualization.read_selection_polygon_volume("cat_crop_test.json")
    mode = "test"
    savedir_true = "/local_disk/hikaru/pointnet2/catdata/crop32/labelingcolor/almostTrue_{}".format(mode)
    savedir_false = "/local_disk/hikaru/pointnet2/catdata/crop32/labelingcolor/almostFalse_{}".format(mode)
    savedir_true_label = "/local_disk/hikaru/pointnet2/catdata/crop32/labelingcolor/almostTrue_label_{}".format(mode)
    savedir_false_label = "/local_disk/hikaru/pointnet2/catdata/crop32/labelingcolor/almostFalse_label_{}".format(mode)
    os.makedirs(savedir_true, exist_ok=True)
    os.makedirs(savedir_false, exist_ok=True)
    os.makedirs(savedir_true_label, exist_ok=True)
    os.makedirs(savedir_false_label, exist_ok=True)

    crop_size = 32
    stride = 16
    if mode == "train":
        index_file = "crop_before_raw_icp_BHremove_round1_C64_L256_cate0.txt"
    elif mode == "test":
        index_file = "crop_before_raw_icp_BHremove_round1_C64_L256_cate1.txt"
    else:
        sys.exit(0)
    gt_index = np.loadtxt(index_file)
    save_index = []
    count = 0
    print(len(gt_index))
    for x1_crop16, _, z1_crop16, _, y1_crop16, _ in gt_index:
        if (-163 < x1_crop16 < 160) and (-118 < y1_crop16 < 205) and (-139 < z1_crop16 < 145):
            label = True
        else:
            label = False
        for i in range(3):
            x1 = x1_crop16 + stride * i
            x2 = x1 + crop_size - 1e-2
            for j in range(3):
                z1 = z1_crop16 + stride * j
                z2 = z1 + crop_size - 1e-2
                vol.bounding_polygon[0] = [x1, 0.0, z1]
                vol.bounding_polygon[1] = [x1, 0.0, z2]
                vol.bounding_polygon[2] = [x2, 0.0, z2]
                vol.bounding_polygon[3] = [x2, 0.0, z1]
                for k in range(3):
                    vol.axis_min = y1_crop16 + stride * k
                    vol.axis_max = vol.axis_min + crop_size - 1e-2

                    crop_part = vol.crop_point_cloud(pcd_labeled)
                    point_num = (np.asarray(crop_part.points)).shape[0]
                    if point_num < 512:
                        continue
                    crop_part_raw = vol.crop_point_cloud(pcd)

                    point_num_raw = (np.array(crop_part_raw.points)).shape[0]

                    if point_num != point_num_raw:
                        print(i, j, k)
                        continue
                    filename = "cat_raw_icp0005_C32L512_{x1}_{z1}_{y1}.ply".format(
                            x1=int(x1), z1=int(z1), y1=int(vol.axis_min))
                    if label:
                        o3d.io.write_point_cloud(os.path.join(savedir_true, filename), 
                                                crop_part_raw)
                        o3d.io.write_point_cloud(os.path.join(savedir_true_label, filename),
                                                crop_part)
                    else:
                        o3d.io.write_point_cloud(os.path.join(savedir_false, filename), 
                                                crop_part_raw)
                        o3d.io.write_point_cloud(os.path.join(savedir_false_label,filename),
                                                crop_part)

                    save_index.append(
                        "{x1} {x2} {z1} {z2} {y1} {y2}".format(
                            x1=float(x1), x2=float(x2), z1=float(z1), z2=float(z2), 
                            y1=float(vol.axis_min), y2=float(vol.axis_max)))
        count += 1
        if count % 10 == 0:
            print(count)

    if mode == "train":
        savefile = "crop_after_BHremove_round1_C64L256_C32S16L512_cate0.txt"
    elif mode == "test":
        savefile = "crop_after_BHremove_round1_C64L256_C32S16L512_cate1.txt"
    with open(savefile, "w") as f:
        f.write("\n".join(save_index))

if __name__ == "__main__":
    refine()
