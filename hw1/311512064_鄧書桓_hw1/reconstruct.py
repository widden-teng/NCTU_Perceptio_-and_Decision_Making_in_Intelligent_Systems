from ftplib import error_temp
from itertools import count
from json import load
import open3d as o3d
import os
import numpy as np
import math
from mpmath import cot
import cv2


# 將資料存入後經此function將2D轉成3D
def depth_image_to_point_cloud(path_img, pcd_path):
    focal = (512/2 * cot(np.pi/2/2))
    # 用於儲存有幾筆資料
    count = int(len([name for name in os.listdir(path_img)
                     if os.path.isfile(os.path.join(path_img, name))])/2)
    print("There are {} pcds need to be made".format(count))
    for num in range(count):
        # 讀取資料
        img = cv2.imread(path_img + 'front_rgb_img_' + str(num+1) + '.png', 1)
        depth_img = cv2.imread(
            path_img + 'front_depth_img_' + str(num+1) + '.png', 1)
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

# 將點雲做前處理(使pcd降維)


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
        target_tp = get_new_dcp(source, target_final,
                                result_icp.transformation, num_con)
        # 用來儲存estimate點的位置與之後要連線線的顏色
        store_relative_pos(colors, camera_points, result_icp.transformation)

    # 全部做完後可視化成果
    o3d.visualization.draw_geometries(target_final)
    # 將每個疊完圖的成果儲存至final_pcd
    final_pcd = o3d.geometry.PointCloud()
    for i in range(count):
        final_pcd = final_pcd + target_final[i]

    # o3d.io.write_point_cloud("./final_pcd.pcd", final_pcd)
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

# 將load.py儲存的位置(.txt)寫入並調整型態(變成float)


def get_posit():
    posit_data = []
    # 開啟檔案
    with open("posit_data.txt") as f:
        for line in f.readlines():
            # 以空白當成分割線
            line = line.split(' ')
            # 保存所需資料
            line = line[0:3]
            posit_data.append(line)
    # 轉成float型態
    for i in range(count):
        for j in range(3):
            posit_data[i][j] = float(posit_data[i][j])
    return posit_data

# 用來計算ground truth的點與儲存之後要連線線的顏色與標記要將哪幾個點相連


def ground_truth():
    # 將load.py儲存的位置(.txt)寫入並調整型態(變成float)
    posit_data = get_posit()
    # 把第一筆資料當成base
    base = [posit_data[0][0], posit_data[0][1], posit_data[0][2]]
    ground_truth_posit = []
    ground_truth_color = [[0, 0, 0]]
    for i in range(count):
        # 將每筆資料與base相減，以達成相對於第一張圖的座標
        gt_temp = [posit_data[i][k] - base[k] for k in range(3)]
        # 因為我們是往-z方向前進，故z須加個-號
        gt_temp[2] = -gt_temp[2]
        ground_truth_posit.append(gt_temp)
        ground_truth_color.append([0, 0, 0])
    print(ground_truth_posit)
    return ground_truth_posit, ground_truth_color


# 計算ground truth 與 estimate 的L2 distance
def error_count():
    err = 0
    for i in range(count):
        # 使第i個點ground truth 與 estimate的x, y, z相減
        error_temp = [ground_truth_posit[i][k] - camera_points[i][k]
                      for k in range(3)]
        #將x, y, z差的值做平方開更號
        err = err + math.sqrt(error_temp[0] **
                              2 + error_temp[1]**2 + error_temp[2]**2)
    err = err / count
    return err


#####################################################################################################
# 製造資料夾
if not (os.path.exists("./pcd_data")):
    os.makedirs('pcd_data')
# 初始化路徑
path_img = "./images/"
pcd_path = "./pcd_data/"
# 判斷有多少資料
count = depth_image_to_point_cloud(path_img, pcd_path)
# 初始化estimate點的位置與之後要連線線的顏色與標記要將哪幾個點相連
camera_points, colors, lines = initial_estimate()
# 使用整合後的icp進行疊圖
final_pcd = local_icp_algorithm(camera_points, colors)

# 製造estimate的線
line_set = o3d.geometry.LineSet()
# 將estimate的點存入o3d.geometry.LineSet
line_set.points = o3d.utility.Vector3dVector(camera_points)
# 將estimate的線存入o3d.geometry.LineSet
line_set.lines = o3d.utility.Vector2iVector(lines)
# 將estimate的顏色存入o3d.geometry.LineSet
line_set.colors = o3d.utility.Vector3dVector(colors)from ftplib import error_temp


# 將資料存入後經此function將2D轉成3D
def depth_image_to_point_cloud(path_img, pcd_path):
    focal = (512/2 * cot(np.pi/2/2))
    # 用於儲存有幾筆資料
    count = int(len([name for name in os.listdir(path_img)
                     if os.path.isfile(os.path.join(path_img, name))])/2)
    print("There are {} pcds need to be made".format(count))
    for num in range(count):
        # 讀取資料
        img = cv2.imread(path_img + 'front_rgb_img_' + str(num+1) + '.png', 1)
        depth_img = cv2.imread(
            path_img + 'front_depth_img_' + str(num+1) + '.png', 1)
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

# 將點雲做前處理(使pcd降維)


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
        target_tp = get_new_dcp(source, target_final,
                                result_icp.transformation, num_con)
        # 用來儲存estimate點的位置與之後要連線線的顏色
        store_relative_pos(colors, camera_points, result_icp.transformation)

    # 全部做完後可視化成果
    o3d.visualization.draw_geometries(target_final)
    # 將每個疊完圖的成果儲存至final_pcd
    final_pcd = o3d.geometry.PointCloud()
    for i in range(count):
        final_pcd = final_pcd + target_final[i]

    # o3d.io.write_point_cloud("./final_pcd.pcd", final_pcd)
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

# 將load.py儲存的位置(.txt)寫入並調整型態(變成float)


def get_posit():
    posit_data = []
    # 開啟檔案
    with open("posit_data.txt") as f:
        for line in f.readlines():
            # 以空白當成分割線
            line = line.split(' ')
            # 保存所需資料
            line = line[0:3]
            posit_data.append(line)
    # 轉成float型態
    for i in range(count):
        for j in range(3):
            posit_data[i][j] = float(posit_data[i][j])
    return posit_data

# 用來計算ground truth的點與儲存之後要連線線的顏色與標記要將哪幾個點相連


def ground_truth():
    # 將load.py儲存的位置(.txt)寫入並調整型態(變成float)
    posit_data = get_posit()
    # 把第一筆資料當成base
    base = [posit_data[0][0], posit_data[0][1], posit_data[0][2]]
    ground_truth_posit = []
    ground_truth_color = [[0, 0, 0]]
    for i in range(count):
        # 將每筆資料與base相減，以達成相對於第一張圖的座標
        gt_temp = [posit_data[i][k] - base[k] for k in range(3)]
        # 因為我們是往-z方向前進，故z須加個-號
        gt_temp[2] = -gt_temp[2]
        ground_truth_posit.append(gt_temp)
        ground_truth_color.append([0, 0, 0])
    print(ground_truth_posit)
    return ground_truth_posit, ground_truth_color


# 計算ground truth 與 estimate 的L2 distance
def error_count():
    err = 0
    for i in range(count):
        # 使第i個點ground truth 與 estimate的x, y, z相減
        error_temp = [ground_truth_posit[i][k] - camera_points[i][k]
                      for k in range(3)]
        #將x, y, z差的值做平方開更號
        err = err + math.sqrt(error_temp[0] **
                              2 + error_temp[1]**2 + error_temp[2]**2)
    err = err / count
    return err


#####################################################################################################
# 製造資料夾
if not (os.path.exists("./pcd_data")):
    os.makedirs('pcd_data')
# 初始化路徑
path_img = "./images/"
pcd_path = "./pcd_data/"
# 判斷有多少資料
count = depth_image_to_point_cloud(path_img, pcd_path)
# 初始化estimate點的位置與之後要連線線的顏色與標記要將哪幾個點相連
camera_points, colors, lines = initial_estimate()
# 使用整合後的icp進行疊圖
final_pcd = local_icp_algorithm(camera_points, colors)

# 製造estimate的線
line_set = o3d.geometry.LineSet()
# 將estimate的點存入o3d.geometry.LineSet
line_set.points = o3d.utility.Vector3dVector(camera_points)
# 將estimate的線存入o3d.geometry.LineSet
line_set.lines = o3d.utility.Vector2iVector(lines)
# 將estimate的顏色存入o3d.geometry.LineSet
line_set.colors = o3d.utility.Vector3dVector(colors)
# 將estimate與先前疊圖的pcd一同呈現
o3d.visualization.draw_geometries([final_pcd, line_set])
# o3d.io.write_point_cloud("./estimate_pcd.pcd", final_pcd)
print("estimate is done!!!!")

# 製造ground truth的線
ground_truth_posit, ground_truth_color = ground_truth()
line_gt = o3d.geometry.LineSet()
# 將ground truth的點存入o3d.geometry.LineSet
line_gt.points = o3d.utility.Vector3dVector(ground_truth_posit)
# 將ground truth的線存入o3d.geometry.LineSet
line_gt.lines = o3d.utility.Vector2iVector(lines)
# 將ground truth的顏色存入o3d.geometry.LineSet
line_gt.colors = o3d.utility.Vector3dVector(ground_truth_color)
# 將ground truth, estimate與先前疊圖的pcd一同呈現
o3d.visualization.draw_geometries([final_pcd, line_gt, line_set])
# o3d.io.write_point_cloud("./ground_truth.pcd", final_pcd)
print("ground_truth is done!!!!")

# 計算ground truth 與 estimate 的L2 distance
L2_distance = error_count()
print("The L2 distance between estimated camera poses and groundtruth camera poses is :{}".format(L2_distance))

# 將estimate與先前疊圖的pcd一同呈現
o3d.visualization.draw_geometries([final_pcd, line_set])
# o3d.io.write_point_cloud("./estimate_pcd.pcd", final_pcd)
print("estimate is done!!!!")

# 製造ground truth的線
ground_truth_posit, ground_truth_color = ground_truth()
line_gt = o3d.geometry.LineSet()
# 將ground truth的點存入o3d.geometry.LineSet
line_gt.points = o3d.utility.Vector3dVector(ground_truth_posit)
# 將ground truth的線存入o3d.geometry.LineSet
line_gt.lines = o3d.utility.Vector2iVector(lines)
# 將ground truth的顏色存入o3d.geometry.LineSet
line_gt.colors = o3d.utility.Vector3dVector(ground_truth_color)
# 將ground truth, estimate與先前疊圖的pcd一同呈現
o3d.visualization.draw_geometries([final_pcd, line_gt, line_set])
# o3d.io.write_point_cloud("./ground_truth.pcd", final_pcd)
print("ground_truth is done!!!!")

# 計算ground truth 與 estimate 的L2 distance
L2_distance = error_count()
print("The L2 distance between estimated camera poses and groundtruth camera poses is :{}".format(L2_distance))
