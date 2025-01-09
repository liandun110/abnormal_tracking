import cv2
import glob
import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch
import clip
from PIL import Image

def save_cropped_images(anns, original_image, output_dir):
    """
    根据分割信息裁剪图片并保存。
    :param anns: 包含分割信息的列表，每个元素包含 'bbox' 和 'segmentation'。
    :param original_image: 原始图片，格式为 numpy.ndarray。
    :param output_dir: 输出文件夹，用于保存裁剪的图像。
    """
    if len(anns) == 0:
        print("没有分割信息，跳过处理。")
        return

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建一个副本用于叠加 bbox 信息
    image_with_bboxes = original_image.copy()

    # 遍历每个对象，裁剪图像并保存
    for obj_id, ann in enumerate(anns):
        bbox = ann['bbox']  # bbox 格式为 [x, y, w, h]

        # 获取 bbox 的信息
        x, y, w, h = map(int, bbox)  # 转换为整数
        x_center = x + w // 2  # 计算中心点 x
        y_center = y + h // 2  # 计算中心点 y
        max_side = max(w, h)  # 获取 bbox 的最长边作为裁剪的正方形边长

        # 在原图上当前box框出来，指明ID。

        # 计算正方形裁剪区域
        crop_x1 = max(0, x_center - max_side // 2)
        crop_y1 = max(0, y_center - max_side // 2)
        crop_x2 = min(original_image.shape[1], x_center + max_side // 2)
        crop_y2 = min(original_image.shape[0], y_center + max_side // 2)

        # 裁剪图像
        cropped_image = original_image[crop_y1:crop_y2, crop_x1:crop_x2]

        # 保存裁剪的图像
        cropped_output_path = f"{output_dir}/{str(obj_id).zfill(2)}.png"
        cv2.imwrite(cropped_output_path, cropped_image)
        print(f"保存裁剪图像: {cropped_output_path}")

        # 在原图上绘制 bbox 框和 ID
        cv2.rectangle(image_with_bboxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_with_bboxes, f"ID: {obj_id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 保存叠加了 bbox 信息的原始图像
    image_with_bboxes_path = os.path.join(output_dir, "image_with_bboxes.png")
    cv2.imwrite(image_with_bboxes_path, image_with_bboxes)
    print(f"保存叠加了 bbox 的原始图像: {image_with_bboxes_path}")


def show_mask(mask, ax, obj_id=None, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

    # 获取 mask 的中心点作为 ID 的显示位置
    # 确保 mask 是二维
    if mask.ndim > 2:
        mask = mask.squeeze()
    coords = np.argwhere(mask)
    if len(coords) > 0:
        y, x = coords.mean(axis=0).astype(int)  # 计算中心点坐标
        ax.text(x, y, str(obj_id), color='white', fontsize=4, fontweight='bold', ha='center', va='center')


def img_seg(query_img, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sam2_checkpoint = "/home/suma/PycharmProjects/abnormal_tracking/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    image = Image.open(query_img)
    print(image.size)
    image = np.array(image.convert("RGB"))
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,  # 为什么由32改成64？如果不改，则exp2中的异物无法分割。
        points_per_batch=128,  # 修改为265或64，分割时间影响不大。
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        stability_score_offset=0.9,
        crop_n_layers=1,  # 如果改为0，exp1中只能识别出6/7个异常物。
        box_nms_thresh=0.5,  # 能够在很大程度上避免多个掩码重叠在一起的情况。
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25,
        use_m2m=False,  # 改成False，时间从15秒降低至7秒，且三个实验结果全对。
        multimask_output=True  # 如果改为False，exp1中只能识别出5/7个异常物。
    )
    print('开始分割')

    start_time = time.time()
    masks = mask_generator.generate(image)
    end_time = time.time()
    print('结束分割')
    print('分割耗时时间：', end_time - start_time)
    save_cropped_images(masks, image, output_dir)
    return masks


def extract_features_from_folder(folder_path):
    """
    从文件夹中加载所有图像，用 CLIP 提取特征。
    :param folder_path: 图像文件夹路径
    :param model: CLIP 模型
    :param preprocess: CLIP 的预处理函数
    :param device: 设备（"cuda" 或 "cpu"）
    :return: 提取的特征列表（张量形式）
    """
    features = []
    file_names = sorted(os.listdir(folder_path))


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)


    for file_name in file_names:
        image_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(image_path):  # 跳过非文件
            continue

        # 加载并预处理图像
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # 提取特征
        with torch.no_grad():
            feature = model.encode_image(image)
            features.append(feature)

    return torch.cat(features)  # 合并所有特征为一个张量

def compute_cosine_similarity(features1, features2):
    import torch.nn.functional as F
    features1 = F.normalize(features1, p=2, dim=1)  # [N1, 512]
    features2 = F.normalize(features2, p=2, dim=1)  # [N2, 512]
    return torch.matmul(features1, features2.T)  # [N1, N2]

def main():
    # 对于00001.jpg，运行SAM得到每个分割区域的bbox。
    # 对于00002.jpg，运行SAM得到每个分割区域的bbox。
    # 对于00001.jpg，根据每个分割区域的bbox，裁剪正方形小图像，保存到 outputs/clip_compare/00001 文件夹。
    # 对于00002.jpg，根据每个分割区域的bbox，裁剪正方形小图像，保存到 outputs/clip_compare/00002 文件夹。
    # 对 outputs/clip_compare/00001 文件夹中的正方形小图像用clip提特征，保存到变量 features1 中。
    # 对 outputs/clip_compare/00002 文件夹中的正方形小图像用clip提特征，保存到变量 features2 中。
    # 对于 features1中的每个特征 feature1 与 features2 中每个特征的相似性的最大值，看异常物的特征是否
    # 设置图像路径
    video_dir = '../dataset/exp1'

    # 把图像缩放至(1662, 682)。因为常见轿车长宽比为2.5:1。缩放后覆盖原始图像文件（这样能避免图像分割和视频追踪代码读入图像尺寸不一致的问题），必须实时保存。
    # 必须进行缩放，否则会爆显存。
    original_images = sorted(glob.glob(os.path.join(video_dir, 'original_images/*.jpg')))
    for original_image_path in original_images:
        image_name = original_image_path.split('/')[-1]
        image = Image.open(original_image_path)
        scaled_image = image.resize((1662, 682))
        save_image_path = os.path.join(video_dir, image_name)
        scaled_image.save(save_image_path)

    output_dir1='/home/suma/PycharmProjects/abnormal_tracking/outputs/clip_compare/00001'
    output_dir2='/home/suma/PycharmProjects/abnormal_tracking/outputs/clip_compare/00002'
    # 得到图像分割的结果
    query_img = os.path.join(video_dir, '00001.jpg')
    masks = img_seg(query_img, output_dir1)

    query_img = os.path.join(video_dir, '00002.jpg')
    masks = img_seg(query_img, output_dir2)



    features1 = extract_features_from_folder(output_dir1)
    print(f"提取了 {len(features1)} 个特征 (来自 {output_dir2})") #[特征数，512]

    # 提取 00002 文件夹中的特征
    features2 = extract_features_from_folder(output_dir2)
    print(f"提取了 {len(features2)} 个特征 (来自 {output_dir2})")  #[特征数，512]

    # 计算余弦相似性矩阵
    similarity_matrix = compute_cosine_similarity(features1, features2)

    # 找到每个特征的最大相似性
    max_similarities, max_indices = torch.max(similarity_matrix, dim=1)

    # 打印每个特征的最大相似性
    for i, (max_sim, max_idx) in enumerate(zip(max_similarities, max_indices)):
        print(f"Feature1[{i}] 与 Feature2[{max_idx}] 的最大相似性: {max_sim.item():.4f}")

    # 检查低于阈值的异常特征
    threshold = 0.3
    for i, max_sim in enumerate(max_similarities):
        if max_sim < threshold:
            print(f"Feature1[{i}] 是异常特征，相似性: {max_sim.item():.4f}")



if __name__ == '__main__':
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    main()
