import open3d as o3d
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


pcd_data_path = "./semantic_3d_pointcloud/"
# 初始化點雲

pcd = o3d.geometry.PointCloud()
# 將位置與rgb寫入
temp_points = np.load(pcd_data_path + "point.npy")
temp_colors = np.load(pcd_data_path + "color01.npy")
temp_points = temp_points * 10000. / 255
pcd_points = np.empty((0, 3), float)
pcd_colors = np.empty((0, 3), float)
# min = -1.775325843388430  max = 2.365105159246803
flour = -1.35  # -1.35
ceiling = -0.1  # -0.1
for i in range(temp_colors.shape[0]):
    if (temp_points[i][1] > flour and temp_points[i][1] < ceiling):
        pcd_points = np.append(pcd_points, [temp_points[i]], axis=0)
        pcd_colors = np.append(pcd_colors, [temp_colors[i]], axis=0)
pcd.points = o3d.utility.Vector3dVector(pcd_points)
pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
o3d.io.write_point_cloud("./after_filtering.pcd", pcd)
plt.scatter(pcd_points[:, 2], pcd_points[:, 0], s=5, c=pcd_colors,  alpha=1)
plt.xlim(-6, 11)
plt.ylim(-4, 7)
plt.axis("off")
plt.savefig("map.png", bbox_inches='tight', pad_inches=0)
plt.show()
