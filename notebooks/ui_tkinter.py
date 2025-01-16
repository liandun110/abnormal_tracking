import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import os
from notebooks.main import abnormal_tracking
# 图像处理函数
def process_images(img1_path):
    # 假设 abnormal_tracking 返回三个图像的路径
    img3_path, img4_path, img5_path = abnormal_tracking(img1_path)
    return img3_path, img4_path, img5_path

# 创建应用窗口
root = tk.Tk()
root.title("图像处理应用")

# 创建Tab控件
tab_control = tk.ttk.Notebook(root)

# 创建Tab1, Tab2, Tab3
tab1 = tk.Frame(tab_control)
tab2 = tk.Frame(tab_control)
tab3 = tk.Frame(tab_control)

tab_control.add(tab1, text="Tab 1: 加载图片")
tab_control.add(tab2, text="Tab 2: 展示 img3")
tab_control.add(tab3, text="Tab 3: 展示 img4 和 img5")
tab_control.pack(expand=1, fill="both")

# 初始化图像路径变量
img1_path = None
img2_path = None
img3_path = None
img4_path = None
img5_path = None

# 更新标签显示图像
def update_image_label(image_path, label):
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path)
        img_resized = img.resize((1900, 550))  # 调整图像大小
        img_tk = ImageTk.PhotoImage(img_resized)
        label.config(image=img_tk)
        label.image = img_tk  # 避免图像被回收

# 读取并展示图片
def load_image1():
    global img1_path
    img1_path = '/home/suma/PycharmProjects/abnormal_tracking/dataset/exp10/original_images/00001.jpg'
    update_image_label(img1_path, img1_label)

def load_image2():
    global img2_path
    img2_path = '/home/suma/PycharmProjects/abnormal_tracking/dataset/exp10/original_images/00002.jpg'
    update_image_label(img2_path, img2_label)

# 处理图片并生成img3, img4, img5
def process_and_switch_tab():
    global img1_path, img2_path, img3_path, img4_path, img5_path
    if img1_path and img2_path:
        img3_path, img4_path, img5_path = process_images(img1_path)
        update_image_label(img3_path, img3_label)
        tab_control.select(tab2)  # 切换到Tab2
    else:
        print("请先加载 img1 和 img2！")

# 在Tab1中创建按钮和图像显示区域
img1_label = tk.Label(tab1)
img1_label.pack(pady=10)

load_btn1 = tk.Button(tab1, text="加载并展示 img1", command=load_image1)
load_btn1.pack(pady=5)

img2_label = tk.Label(tab1)
img2_label.pack(pady=10)

load_btn2 = tk.Button(tab1, text="加载并展示 img2", command=load_image2)
load_btn2.pack(pady=5)

process_btn = tk.Button(tab1, text="执行函数生成图片并跳转到 Tab 2", command=process_and_switch_tab)
process_btn.pack(pady=10)

# 在Tab2中展示img3
img3_label = tk.Label(tab2)
img3_label.pack(pady=10)

# 在Tab3中展示img4和img5
img4_label = tk.Label(tab3)
img4_label.pack(pady=10)

img5_label = tk.Label(tab3)
img5_label.pack(pady=10)

# 更新图像并展示
def show_img4_and_img5():
    global img4_path, img5_path
    update_image_label(img4_path, img4_label)
    update_image_label(img5_path, img5_label)

# 添加按钮到Tab3
show_btn = tk.Button(tab3, text="展示 img4 和 img5", command=show_img4_and_img5)
show_btn.pack(pady=10)

# 运行应用
root.mainloop()