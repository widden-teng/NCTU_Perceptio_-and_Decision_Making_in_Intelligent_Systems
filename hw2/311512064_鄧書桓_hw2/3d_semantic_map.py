import argparse
import json
import os
import numpy as np
from PIL import Image  # for image processing
import cv2
import open3d as o3d
import habitat_sim
import habitat_sim.agent
import habitat_sim.bindings as hsim
from habitat_sim.agent import AgentState
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.utils.common import d3_40_colors_rgb
from mpmath import cot
# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###


# 將資料存入後經此function將2D轉成3D
def depth_image_to_point_cloud(img_path, pcd_path, dept_path):
    focal = (512/2 * cot(np.pi/2/2))
    # 用於儲存有幾筆資料
    count = int(len([name for name in os.listdir(img_path)
                     if os.path.isfile(os.path.join(img_path, name))]))
    print("There are {} pcds need to be made".format(count))
    for num in range(count):
        # 讀取資料
        img = cv2.imread(img_path + '/apartment_0_' + str(num+1) + '.png', 1)
        depth_img = cv2.imread(
            dept_path + '/apartment_0_' + str(num+1) + '.png', 1)
        pixel_rgb = []
        pixel_xyz = []
        # 將資料由2D轉成3D，將各個pixelu,v乘上z/f以得到以那張照片為frame的x,y,z
        # RGB則是正規化到0~1
        for i in range(512):
            for j in range(512):
                # 限制y的大小，以此來去除天花板
                if (((i-256) * depth_img[i, j, 1]/focal / 25.5) > -0.65):
                    pixel_rgb.append([img[i, j, 2]/255,
                                      img[i, j, 1]/255, img[i, j, 0]/255])
                    pixel_xyz.append([(j-256) * depth_img[i, j, 2]/focal / 25.5,
                                      (i-256) * depth_img[i, j, 1]/focal / 25.5, depth_img[i, j, 0] / 25.5])
        # 將資料以點雲方式呈現
        # 初始化點雲
        pcd = o3d.geometry.PointCloud()
        # 將位置與rgb寫入
        pcd.points = o3d.utility.Vector3dVector(pixel_xyz)
        pcd.colors = o3d.utility.Vector3dVector(pixel_rgb)
        # 將所轉換成的pcd儲存至指定資料夾
        o3d.io.write_point_cloud(
            pcd_path + "picture" + str(num+1) + ".pcd", pcd)
        print("{} th pcd is made..............{}%".format(
            num+1, int((num+1)/count * 100)))
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

# 準備資料(提取feature-FPFH)


def prepare_dataset(target, voxel_size, num_con):
    source = o3d.io.read_point_cloud(
        './pcd_data/picture' + str(num_con+2) + '.pcd')
    # 尋找source點雲與target點雲的法向向量(以便做feature mapping)
    source.estimate_normals()
    target.estimate_normals()
    # 初始化轉移矩陣(沒啥功能)
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # 將source pcd轉換到target pcd frame
    source.transform(trans_init)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

# 使用RANSAC做registration(於附近33 維 FPFH 特徵空間中的最近鄰來檢測的)


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

# 這裡做point-to-plane ICP(用來細化點雲，可使點雲看起來更完整)


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

# 用來做source dcp最終的轉換疊圖，並將結果存入terget_final(用來儲存所有相對於target pcd的疊圖pcd)


def get_new_dcp(source, target_final, transformation, num_con):
    print("epoch :{}..............{}%".format(
        num_con+1, int((num_con+1)/(count-1) * 100)))
    source.transform(transformation)
    target_final.append(source)
    return source

# 用來儲存estimate點的位置與之後要連線線的顏色


def store_relative_pos(colors, camera_points, transformation):
    # 最後加上1是方便做矩陣乘法
    temp = np.array([0, 0, 0, 1])
    # 將每個位置的點與該trasformation相乘，以轉乘target pcd的frame
    temp = transformation @ temp
    temp = temp[0:3].tolist()
    camera_points.append(temp)
    colors.append([1, 0, 0])

# 整合後的icp


def local_icp_algorithm(camera_points, colors):
    target_final = []
    # 將第一張圖視為target dcp，並存入target_final(用來儲存所有相對於target pcd的疊圖pcd)
    x = o3d.io.read_point_cloud('./pcd_data/picture1.pcd')
    target_final.append(x)
    target_tp = x
    # 開始疊圖
    for num_con in range(count-1):
        # 自己設的voxel_size
        voxel_size = 0.05
        # 將資料做預處理
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(target_tp,
                                                                                             voxel_size, num_con)
        # 將資料做global registration
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        # 將資料做point-to-plane ICP(用來細化點雲，可使點雲看起來更完整)
        result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                         voxel_size, result_ransac)
        # 用來做source dcp最終的轉換疊圖，並將結果存入terget_final(用來儲存所有相對於target pcd的疊圖pcd)
        target_tp = get_new_dcp(source_down, target_final,
                                result_icp.transformation, num_con)
        # 用來儲存estimate點的位置與之後要連線線的顏色
        store_relative_pos(colors, camera_points, result_icp.transformation)

    # 全部做完後可視化成果
    o3d.visualization.draw_geometries(target_final)
    # 將每個疊完圖的成果儲存至final_pcd
    final_pcd = o3d.geometry.PointCloud()
    for i in range(count):
        final_pcd = final_pcd + target_final[i]

    o3d.io.write_point_cloud("./final_pcd.pcd", final_pcd)
    print("reconstruct by ICP is done!!!!")
    return final_pcd


# 初始化estimate點的位置與之後要連線線的顏色與標記要將哪幾個點相連
def initial_estimate():
    camera_points = []
    camera_points.append([0, 0, 0])
    colors = []
    lines = []
    # 將1與2相連,3與4相連...依此類推
    for i in range(count-1):
        temp_line = [i, i+1]
        lines.append(temp_line)
    return camera_points, colors, lines


pcd_path = "./pcd_data/"
img_path = "./ckpt/apartment0/result"
dept_path = "./data/apartment0_new/depth"
if not (os.path.exists(pcd_path)):
    os.makedirs(pcd_path)
count = depth_image_to_point_cloud(img_path, pcd_path, dept_path)
# 初始化estimate點的位置與之後要連線線的顏色與標記要將哪幾個點相連
camera_points, colors, lines = initial_estimate()
# 使用整合後的icp進行疊圖
final_pcd = local_icp_algorithm(camera_points, colors)


# create bounding box for custom voxel down
def custom_voxel_down(final_pcd, voxel):
    # initialize box and other list
    custom_point = []
    custom_color = []
    num = 1
    box = final_pcd.get_axis_aligned_bounding_box()
    box.color = (1, 1, 1)
    # get array (num_of_point * 3)
    point_all = np.asarray(final_pcd.points)
    color_all = np.asarray(final_pcd.colors)
    # get center, extent of box (box is bounding final pcd)
    box_center = box.get_center()
    box_margin = box.get_extent()
    # get the start, end of x, y, z
    x_start = round(box_center[0] - box_margin[0]/2, 5)
    x_end = round(box_center[0] + box_margin[0]/2, 5)
    y_start = round(box_center[1] - box_margin[1]/2, 5)
    y_end = round(box_center[1] + box_margin[1]/2, 5)
    z_start = round(box_center[2] - box_margin[2]/2, 5)
    z_end = round(box_center[2] + box_margin[2]/2, 5)
    voxel_down_pcd = o3d.geometry.PointCloud()
    total_iteration = int((x_end-x_start)/voxel) * \
        int((y_end-y_start)/voxel) * int((z_end-z_start)/voxel)
    for i in range(int(x_start * (10**5)), int(x_end * (10**5)), int(voxel * (10**5))):
        for j in range(int(y_start * (10**5)), int(y_end * (10**5)), int(voxel * (10**5))):
            for k in range(int(z_start * (10**5)), int(z_end * (10**5)), int(voxel * (10**5))):
                print("iteration for {} ........{}%:".format(
                    str(num), round(num/total_iteration * 100, 2)))
                num = num + 1

                # slice the box in x direction
                slice_x = point_all[np.where((point_all[:, 0] >= (i/(10**5))) &
                                             (point_all[:, 0] <= (i/(10**5)+voxel)))]
                # slice the remaining box in y direction
                slice_y = slice_x[np.where(
                    (slice_x[:, 1] >= (j/(10**5))) & (slice_x[:, 1] <= (j/(10**5)+voxel)))]
                # slice the remaining box in z direction and get the corresponding color
                slice_z = slice_y[np.where(
                    (slice_y[:, 2] >= (k/(10**5))) & (slice_y[:, 2] <= (k/(10**5)+voxel)))]

                # get the color with corresponding slices
                # get the color in x slice
                slice_color = color_all[np.where(
                    (point_all[:, 0] >= (i/(10**5))) & (point_all[:, 0] <= (i/(10**5)+voxel)))]
                # get the color in xy slice
                slice_color = slice_color[np.where(
                    (slice_x[:, 1] >= (j/(10**5))) & (slice_x[:, 1] <= (j/(10**5)+voxel)))]
                # get the color in xyz slice
                slice_color = slice_color[np.where(
                    (slice_y[:, 2] >= (k/(10**5))) & (slice_y[:, 2] <= (k/(10**5)+voxel)))]

                #計算出現次數最多的顏色,並新增到point, color
                # 若沒做四捨五入,會有太多種顏色
                slice_color = np.around(slice_color, 2)
                # return the numbers of each different data
                unique, counts = np.unique(
                    slice_color, axis=0, return_counts=True)
                if counts != []:
                    # find the max number of color and use index to get it's value
                    major_color = unique[counts.tolist().index(max(counts))]
                    # get the center of that voxel
                    custom_point.append(
                        [(2*(i/(10**5))+voxel)/2, (2*(j/(10**5))+voxel)/2, (2*(k/(10**5))+voxel)/2])
                    custom_color.append(major_color)

    voxel_down_pcd.points = o3d.utility.Vector3dVector(custom_point)
    voxel_down_pcd.colors = o3d.utility.Vector3dVector(custom_color)
    return voxel_down_pcd, box


# for custom_voxel_down
final_pcd = o3d.io.read_point_cloud('./final_pcd.pcd')
voxel = 0.1
voxel_down_pcd, box = custom_voxel_down(final_pcd, voxel)
o3d.visualization.draw_geometries([voxel_down_pcd, box])
o3d.io.write_point_cloud('pcd_after_voxeldown.pcd', voxel_down_pcd)
