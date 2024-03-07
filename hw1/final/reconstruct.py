from itertools import count
from json import load
import open3d as o3d
import os
import numpy as np

from mpmath import cot
import cv2


def depth_image_to_point_cloud(path_img, pcd_path):
    focal = (512/2 * cot(np.pi/2/2))
    count = int(len([name for name in os.listdir(path_img)
                     if os.path.isfile(os.path.join(path_img, name))])/2)
    for num in range(count):
        img = cv2.imread(path_img + 'front_rgb_img_' + str(num+1) + '.png', 1)
        depth_img = cv2.imread(
            path_img + 'front_depth_img_' + str(num+1) + '.png', 1)
        pixel_rgb = []
        pixel_xyz = []
        for i in range(512):
            for j in range(512):
                pixel_rgb.append([img[j, i, 2]/255,
                                  img[j, i, 1]/255, img[j, i, 0]/255])
                pixel_xyz.append([-(j-256) * depth_img[j, i, 2]/focal / 25.5,
                                  (i-256) * depth_img[j, i, 1]/focal / 25.5, depth_img[j, i, 0] / 25.5])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pixel_xyz)
        pcd.colors = o3d.utility.Vector3dVector(pixel_rgb)
        o3d.io.write_point_cloud(
            pcd_path + "picture" + str(num+1) + ".pcd", pcd)
        print("{} th pcd is made".format(num+1))
    return count


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(target, voxel_size, num_con):
    source = o3d.io.read_point_cloud(
        './pcd_data/picture' + str(num_con+2) + '.pcd')
    source.estimate_normals()
    target.estimate_normals()
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    source.transform(np.identity(4))
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    source.estimate_normals()
    target.estimate_normals()
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def get_new_dcp(source, target_final, transformation, num_con):
    print("epoch :{}".format(num_con+1))
    source.transform(transformation)
    target_final.append(source)
    return source


def store_relative_pos(colors, camera_points, transformation):
    temp = np.array([0, 0, 0, 1])
    temp = transformation @ temp
    temp = temp[0:3].tolist()
    camera_points.append(temp)
    colors.append([1, 0, 0])


def local_icp_algorithm(camera_points, colors):
    target_final = []
    x = o3d.io.read_point_cloud('picture1.pcd')
    target_final.append(x)
    target_tp = x
    for num_con in range(count-1):
        voxel_size = 0.05  # means 5cm for this dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(target_tp,
                                                                                             voxel_size, num_con)
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                         voxel_size, result_ransac)
        target_tp = get_new_dcp(source, target_final,
                                result_icp.transformation, num_con)
        store_relative_pos(colors, camera_points, result_icp.transformation)

    o3d.visualization.draw_geometries(target_final)
    final_pcd = o3d.geometry.PointCloud()

    for i in range(count):
        final_pcd = final_pcd + target_final[i]

    o3d.io.write_point_cloud("./final_pcd.pcd", final_pcd)
    print("reconstruct by ICP is done!!!!")
    return final_pcd


def initial_estimate():
    camera_points = []
    camera_points.append([0, 0, 0])
    colors = []
    lines = []
    for i in range(count-1):
        temp_line = [i, i+1]
        lines.append(temp_line)
    return camera_points, colors, lines


def get_posit():
    posit_data = []
    with open("posit_data.txt") as f:
        for line in f.readlines():
            line = line.split(' ')
            line = line[0:3]
            posit_data.append(line)
    for i in range(count):
        for j in range(3):
            posit_data[i][j] = float(posit_data[i][j])
    return posit_data


def ground_truth():
    posit_data = get_posit()
    base = [posit_data[0][0], posit_data[0][1], posit_data[0][2]]
    ground_truth_posit = []
    ground_truth_posit.append(base)
    ground_truth_color = [[0, 0, 0]]
    for i in range(1, count):
        gt_temp = [posit_data[i][k] - base[k] for k in range(3)]
        gt_temp[2] = -gt_temp[2]
        ground_truth_posit.append(gt_temp)
        ground_truth_color.append([0, 0, 0])
    return ground_truth_posit, ground_truth_color


#####################################################################################################
if not (os.path.exists("./pcd_data")):
    os.makedirs('pcd_data')
path_img = "./images/"
pcd_path = "./pcd_data/"
count = depth_image_to_point_cloud(path_img, pcd_path)
camera_points, colors, lines = initial_estimate()
final_pcd = local_icp_algorithm(camera_points, colors)
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(camera_points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([final_pcd, line_set])
# o3d.io.write_point_cloud("./estimate_pcd.pcd", final_pcd)
print("estimate is done!!!!")

ground_truth_posit, ground_truth_color = ground_truth()
line_gt = o3d.geometry.LineSet()
line_gt.points = o3d.utility.Vector3dVector(ground_truth_posit)
line_gt.lines = o3d.utility.Vector2iVector(lines)
line_gt.colors = o3d.utility.Vector3dVector(ground_truth_color)
o3d.visualization.draw_geometries([final_pcd, line_gt, line_set])
o3d.io.write_point_cloud("./ground_truth.pcd", final_pcd)
print("ground_truth is done!!!!")
