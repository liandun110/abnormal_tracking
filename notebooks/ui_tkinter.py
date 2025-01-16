import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from notebooks.main import abnormal_tracking

import Demo
import Ice
import sys
import threading
# 全局变量
plate_number = None

class ImageTransferI(Demo.ImageTransfer):
    def __init__(self, condition):
        self.condition = condition

    def printString(self, s, current=None):
        global plate_number
        plate_number = s  # 更新全局变量
        print(f"Received plate number: {plate_number}")
        with self.condition:
            self.condition.notify()  # 通知等待线程

# 创建条件变量
condition = threading.Condition()

# 初始化 Ice 通信
with Ice.initialize(sys.argv) as communicator:
    adapter = communicator.createObjectAdapterWithEndpoints("SimplePrinterAdapter", "default -p 10000")
    object = ImageTransferI(condition)
    adapter.add(object, communicator.stringToIdentity("SimplePrinter"))
    adapter.activate()
    print("Server is running... Waiting for data.")

    with condition:
        condition.wait()  # 等待条件变量被通知
        print("Plate number received. Shutting down...")

# 继续执行后续操作，例如弹出窗口
print("继续执行后续操作...")

# 图像处理函数
def process_images(query_img_path, ref_img_path):
    """
    query_img_path: 现场采集的车底图像
    ref_img_path: 历史车底图像
    """
    # 假设 abnormal_tracking 返回三个图像的路径
    img3_path, img4_path, img5_path = abnormal_tracking(query_img_path, ref_img_path)
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
img1_path = plate_number
plate_number_base = os.path.basename(plate_number)
plate_number_id = plate_number_base.split('_')[0]
gallery_dir = "/home/suma/PycharmProjects/abnormal_tracking/dataset/gallery"
for filename in os.listdir(gallery_dir):
    if filename.startswith(plate_number_id):
        img2_path = os.path.join(gallery_dir, filename)
        print(f"匹配的图像路径: {img2_path}")
        break
else:
    print(f"未找到匹配的图像路径，ID: {plate_number_id}")



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
    img1_path = img1_path
    update_image_label(img1_path, img1_label)

def load_image2():
    global img2_path
    img2_path = img2_path
    update_image_label(img2_path, img2_label)

# 处理图片并生成img3, img4, img5
def process_and_switch_tab():
    global img1_path, img2_path, img3_path, img4_path, img5_path
    if img1_path and img2_path:
        img3_path, img4_path, img5_path = process_images(img1_path, img2_path)
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