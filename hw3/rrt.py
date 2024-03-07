import cv2
import numpy as np
import math
import random
import argparse
import os


class Nodes:
    """Class to store the RRT graph"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent_x = []
        self.parent_y = []

# check collision


def collision(x1, y1, x2, y2):
    color = []
    # 在兩點間直線取100間隔去判斷是否有碰到障礙物
    x = list(np.arange(x1, x2, (x2-x1)/100))
    y = list(((y2-y1)/(x2-x1))*(x-x1) + y1)
    # print("collision", x, y)
    for i in range(len(x)):
        # 因為code中的x, y方向與img相反, 所以會相反
        color.append(img[int(y[i]), int(x[i])])
    if (0 in color):  # 若有一個全黑
        return True  # collision
    else:
        return False  # no-collision


# check the  collision with obstacle and trim
# stepSize 這邊才會用到
def check_collision(x1, y1, x2, y2):
    _, theta = dist_and_angle(x2, y2, x1, y1)
    x = x2 + stepSize*np.cos(theta)
    y = y2 + stepSize*np.sin(theta)
    # print(x2, y2, x1, y1)
    # print("theta", theta)
    # print("check_collision", x, y)

    # TODO: trim the branch if its going out of image area
    # print("Image shape",img.shape)
    hy, hx = img.shape
    directCon = False
    nodeCon = False
    if y < 0 or y > hy or x < 0 or x > hx:
        # print("Point out of image bound")
        {}
    else:
        # check connection between two nodes
        if collision(x, y, x2, y2):
            nodeCon = False
        else:
            nodeCon = True
            # 是否該node(與給定點多一個step size的點）與終點相連
            if collision(x, y, end[0], end[1]):
                directCon = False
            else:
                directCon = True

    return(x, y, directCon, nodeCon)


# return dist and angle b/w new point and nearest node
def dist_and_angle(x1, y1, x2, y2):
    dist = math.sqrt(((x1-x2)**2)+((y1-y2)**2))
    angle = math.atan2(y2-y1, x2-x1)
    return(dist, angle)


# return the neaerst node index
def nearest_node(x, y):
    temp_dist = []
    for i in range(len(node_list)):
        # _為忽略回傳值
        dist, _ = dist_and_angle(x, y, node_list[i].x, node_list[i].y)
        temp_dist.append(dist)
    return temp_dist.index(min(temp_dist))

# generate a random point in the image space

# 回傳 0~369  0~496 隨機值


def rnd_point(h, l):
    new_y = random.randint(0, h)
    new_x = random.randint(0, l)
    return (new_x, new_y)


def RRT(img, img2, start, end, stepSize):
    h, l = img.shape  # dim of the loaded image
    # print(img.shape) # (384, 683)
    # print(h,l)

    # insert the starting point in the node class
    # node_list = [0] # 裡面存有所有node的位置資訊
    node_list[0] = Nodes(start[0], start[1])
    node_list[0].parent_x.append(start[0])
    node_list[0].parent_y.append(start[1])

    # display start and end , 都為藍色圈圈
    cv2.circle(img2, (start[0], start[1]), 5,
               (0, 0, 255), thickness=3, lineType=8)
    cv2.circle(img2, (end[0], end[1]), 5, (0, 0, 255), thickness=3, lineType=8)

    i = 1
    num_nodes = 1
    pathFound = False
    while pathFound == False:
        nx, ny = rnd_point(h, l)
        # print("Random points:", nx, ny)

        nearest_ind = nearest_node(nx, ny)
        nearest_x = node_list[nearest_ind].x
        nearest_y = node_list[nearest_ind].y
        # print("Nearest node coordinates:", nearest_x, nearest_y)

        # check direct connection
        # 會往前走一個step tx, ty
        tx, ty, directCon, nodeCon = check_collision(
            nx, ny, nearest_x, nearest_y)
        # print("Check collision:", tx, ty, directCon, nodeCon)

        # 都true表示可以直接與終點相連
        if directCon and nodeCon:
            # print("Node can connect directly with end")
            node_list.append(i)  # 用於初始化node_list[i] ,沒其他功能
            node_list[i] = Nodes(tx, ty)
            # list.copy() 用於list內容的複製，而非記憶體複製
            # 此處將最近點的路徑都複製進新走一步的parent list 中
            for temp in range(i):
                if len(node_list[temp].parent_x) == 0:
                    print("144  fuck!!!!!!!!!!!!!!!!!!!!!!")
                    print(temp)
            node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
            node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)

            cv2.circle(img2, (int(tx), int(ty)), 2,
                       (0, 0, 255), thickness=3, lineType=8)
            # 將最近點與剛走得step以綠色相連
            cv2.line(img2, (int(tx), int(ty)), (int(node_list[nearest_ind].x), int(
                node_list[nearest_ind].y)), (0, 255, 0), thickness=1, lineType=8)
            cv2.line(img2, (int(tx), int(ty)),
                     (end[0], end[1]), (255, 0, 0), thickness=2, lineType=8)

            # print("Path has been found")
            # print("parent_x",node_list[i].parent_x)
            # 最佳相連的路徑變藍色
            for j in range(len(node_list[i].parent_x)-1):
                cv2.line(img2, (int(node_list[i].parent_x[j]), int(node_list[i].parent_y[j])), (int(
                    node_list[i].parent_x[j+1]), int(node_list[i].parent_y[j+1])), (255, 0, 0), thickness=2, lineType=8)
            # cv2.waitKey(1)
            cv2.imwrite("media/"+str(i)+".jpg", img2)
            cv2.imwrite("out.jpg", img2)
            cv2.imshow("out.jpg", img2)
            cv2.waitKey()
            break

        # 表示新的step不會撞牆, 但還沒到終點
        elif nodeCon:
            # print("Nodes connected")
            node_list.append(i)
            node_list[i] = Nodes(tx, ty)
            node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
            node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
            # print(i)
            # print(node_list[nearest_ind].parent_y)
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)
            i = i+1
            print("There are {} of node had ybeen predict".format(str(num_nodes)))
            num_nodes = num_nodes + 1

            # display
            cv2.circle(img2, (int(tx), int(ty)), 2,
                       (0, 0, 255), thickness=3, lineType=8)
            # 將最近點與剛走得step以綠色相連
            cv2.line(img2, (int(tx), int(ty)), (int(node_list[nearest_ind].x), int(
                node_list[nearest_ind].y)), (0, 255, 0), thickness=1, lineType=8)
            cv2.imwrite("media/"+str(i)+".jpg", img2)
            cv2.imshow("sdc", img2)
            cv2.waitKey(1)
            continue

        else:
            # print("No direct con. and no node con. :( Generating new rnd numbers")
            continue
    node_list[i].parent_x.append(end[0])
    node_list[i].parent_y.append(end[1])
    temp_x = node_list[i].parent_x
    temp_y = node_list[i].parent_y
    temp_x = np.float64(temp_x)*(17/l) - 6
    temp_y = 7 - (np.float64(temp_y)*(11/h))
    final_nodes = np.empty((0, 2), float)
    for j in range(temp_x.shape[0]):
        final_nodes = np.append(final_nodes, [[temp_y[j], temp_x[j]]], axis=0)
    np.save("path_nodes", final_nodes)
    print(final_nodes[0])


def draw_circle(event, x, y, flags, param):
    global coordinates
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img2, (x, y), 5, (255, 0, 0), -1)
        coordinates.append(x)
        coordinates.append(y)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Below are the params:')
    parser.add_argument('-p', type=str, default='map.png', metavar='ImagePath', action='store', dest='imagePath',
                        help='Path of the image containing mazes')
    parser.add_argument('-s', type=int, default=10, metavar='Stepsize', action='store', dest='stepSize',
                        help='Step-size to be used for RRT branches')
    parser.add_argument('-start', type=int, default=[20, 20], metavar='startCoord', dest='start', nargs='+',
                        help='Starting position in the maze')
    parser.add_argument('-stop', type=int, default=[450, 250], metavar='stopCoord', dest='stop', nargs='+',
                        help='End position in the maze')
    parser.add_argument(
        '-selectPoint', help='Select start and end points from figure', action='store_true')

    args = parser.parse_args()

    # 刪除media資料夾
    try:
        os.system("rm -rf media")
    except:
        {
            # print("Dir already clean")
        }
    os.mkdir("media")

    img = cv2.imread(args.imagePath, 0)  # load grayscale maze image
    # 將灰階圖轉黑白圖
    thresh = 230
    img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((7, 7), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    cv2.imshow("erosion", img)
    cv2.waitKey()
    img2 = cv2.imread(args.imagePath)  # 全彩圖
    start = tuple(args.start)  # (20,20) # starting coordinate
    end = tuple(args.stop)  # (450,250) # target coordinate
    stepSize = args.stepSize  # stepsize for RRT
    node_list = [0]  # list to store all the node points 順便初始化

    coordinates = []
    # 判斷點哪裡
    if args.selectPoint:
        print("Select start and end points by double clicking, press 'escape' to exit")
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_circle)
        while(1):
            cv2.imshow('image', img2)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        # print(coordinates)
        start = (coordinates[0], coordinates[1])
        end = (coordinates[2], coordinates[3])

    # run the RRT algorithm
    RRT(img, img2, start, end, stepSize)
