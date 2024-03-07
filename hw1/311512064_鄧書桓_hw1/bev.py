import cv2
import numpy as np
from mpmath import cot

points = []


class Projection(object):

    def __init__(self, image_path, points):  # 初始化 ; image_path 為front_rgb
        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=np.pi/2):
        self.focal = (self.height/2 * cot(fov/2))
        print(self.focal)
        trans_mat = [[1, 0, 0, 0],
                     [0, 0, -1, 0],
                     [0, 1, 0, -1.3],
                     [0, 0, 0, 1]
                     ]
        print("points are : ", points)
        num = len(points)
        wpoints_BEV = np.multiply(
            (np.array(points)-256).tolist(), (1.5/self.focal))
        wpoints_BEV_mat = [[1.5 for col in range(num)] for row in range(4)]
        for i in range(num):
            wpoints_BEV_mat[3][i] = 1
            for j in range(2):
                wpoints_BEV_mat[j][i] = wpoints_BEV[i][j]
        print("points in BEV world view (matrix) is : ", wpoints_BEV_mat)
        wpoints_front_mat = np.dot(trans_mat, wpoints_BEV_mat)
        print("points in front world view (matrix) is : ", wpoints_front_mat)
        new_pixels = [[0 for col in range(2)] for row in range(num)]
        for i in range(num):
            for j in range(2):
                new_pixels[i][j] = int(wpoints_front_mat[j][i] *
                                       (self.focal / wpoints_front_mat[2][i]))
                if j == 0:
                    new_pixels[i][j] = -new_pixels[i][j]
        new_pixels = (np.array(new_pixels)+256).tolist()

        print("points in front pixel view (matrix) are : ", new_pixels)
        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90
    front_rgb = "./images_for_projection/front_img.png"
    top_rgb = "./images_for_projection/BEV_img.png"
    front_depth = "./images_for_projection/Depth_img.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    depth_img = cv2.imread(front_depth, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)  # 鼠标在图上做标记
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    projection = Projection(front_rgb, points)  # points 為鼠標點的位置; 這行用於初始化
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)
