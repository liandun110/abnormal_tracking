"""
图像对齐代码。但是仅在exp10能取得较好效果。受错误关键点的干扰太大，没法在其他图像对上取得较好结果。
"""
import numpy as np
from scipy.interpolate import Rbf
import cv2
import matplotlib.pyplot as plt

from xfeat.xfeat import XFeat

def draw_matches(im1, im2, mkpts_0, mkpts_1):
    """
    绘制两幅图像并排显示，并连接关键点。
    :param im1: 第一幅图像 (ndarray)
    :param im2: 第二幅图像 (ndarray)
    :param mkpts_0: 第一幅图像的关键点 (ndarray)
    :param mkpts_1: 第二幅图像的关键点 (ndarray)
    """
    # 确保输入的图像是彩色的
    if len(im1.shape) == 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    if len(im2.shape) == 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)

    # 获取两幅图像的高度和宽度
    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    # 创建并排的空白图像
    canvas_height = max(h1, h2)
    canvas_width = w1 + w2
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # 将两幅图像拼接到画布上
    canvas[:h1, :w1, :] = im1
    canvas[:h2, w1:w1 + w2, :] = im2

    # 偏移量，用于将第二幅图像的关键点坐标偏移到并排图像的正确位置
    offset = np.array([w1, 0])

    # 绘制关键点和连线
    for pt1, pt2 in zip(mkpts_0, mkpts_1):
        # 计算偏移后的第二幅图像关键点位置
        pt2_offset = pt2 + offset

        # 绘制关键点
        pt1 = tuple(map(int, pt1))
        pt2_offset = tuple(map(int, pt2_offset))
        cv2.circle(canvas, pt1, radius=5, color=(0, 255, 0), thickness=-1)  # 第一幅图关键点
        cv2.circle(canvas, pt2_offset, radius=5, color=(0, 255, 0), thickness=-1)  # 第二幅图关键点

        # 绘制连线
        cv2.line(canvas, pt1, pt2_offset, color=(0, 255, 0), thickness=2)

    # 使用 Matplotlib 显示
    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Matched Keypoints')
    plt.show()

def image_align(fixed_img_path, moving_img_path):
    # 从 *_img_path 中读取图像
    xfeat = XFeat()
    im1 = cv2.imread(fixed_img_path)
    im2 = cv2.imread(moving_img_path)

    # 缩放图片尺寸为(1662, 682)
    # im1 = cv2.resize(im1, (1662, 682))
    # im2 = cv2.resize(im2, (1662, 682))

    mkpts_0, mkpts_1 = xfeat.match_xfeat(im1, im2, top_k=256, min_cossim=0.85)

    draw_matches(im1, im2, mkpts_0, mkpts_1)

    w = im1.shape[1]
    h = im1.shape[0]
    interval_w = w / 10
    interval_h = h / 10
    points = []
    for i in range(11):
        # 上边 (从左到右)
        points.append((i * interval_w, 0))
        # 右边 (从上到下，排除右上角的重复点)
        if i > 0:
            points.append((w, i * interval_h))
        # 下边 (从右到左，排除右下角的重复点)
        if i > 0:
            points.append((w - i * interval_w, h))
        # 左边 (从下到上，排除左下角和左上角的重复点)
        if i > 0 and i < 10:
            points.append((0, h - i * interval_h))
    boundary_points = np.array(points)
    # 将边界坐标添加到原始 ndarray
    key_points1 = np.vstack((mkpts_0, boundary_points))
    key_points2 = np.vstack((mkpts_1, boundary_points))
    # 使用 RBF 插值进行非刚性变换
    rbf = Rbf(key_points1[:, 0], key_points1[:, 1], key_points2[:, 0], function='thin_plate')

    # 创建映射网格
    h, w = im2.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))

    # 计算目标图像的 x 和 y 坐标
    dst_x = rbf(map_x, map_y)
    rbf = Rbf(key_points1[:, 0], key_points1[:, 1], key_points2[:, 1], function='thin_plate')
    dst_y = rbf(map_x, map_y)

    # 使用 remap 进行非刚性变形
    remapped_img = cv2.remap(im2, dst_x.astype(np.float32), dst_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    # 把 moving_img 保存到 moving_img_path 中
    cv2.imwrite(moving_img_path, remapped_img)


    return


if __name__ == '__main__':
    image_align(fixed_img_path='/home/suma/PycharmProjects/abnormal_tracking/dataset/exp10/00001.jpg',
                moving_img_path='/home/suma/PycharmProjects/abnormal_tracking/dataset/exp10/00002.jpg')
