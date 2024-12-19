import cv2
import glob
import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor


def show_img_mask(anns, original_image, output_dir, borders=True):
    if len(anns) == 0:
        return

    # 确保原始图像是浮点类型，方便叠加处理
    original_image = original_image.astype(np.float32) / 255.0

    # 遍历每个对象并绘制
    for obj_id in range(len(anns)):
        ann = anns[obj_id]
        m = ann['segmentation']  # 二值 mask
        bbox = ann['bbox']  # x1y1x2y2
        img_overlay = original_image.copy()  # 复制原始图像

        # 随机生成颜色掩码
        color_mask = np.random.random(3)  # 随机 RGB 颜色
        alpha = 0.5  # 掩码透明度

        # 将 mask 叠加到图像上
        for c in range(3):  # 针对 RGB 三个通道
            img_overlay[:, :, c] = np.where(m,
                                            img_overlay[:, :, c] * (1 - alpha) + color_mask[c] * alpha,
                                            img_overlay[:, :, c])

        # 绘制bbox和obj_id
        x1, y1, w, h = [int(item_) for item_ in bbox]  # 获取 bbox 坐标
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(img_overlay, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)  # 绿色矩形框

        # 在 bbox 左上角绘制 obj_id
        text_position = (x1, y1 - 5)  # 文本位置（bbox 上方）
        cv2.putText(img_overlay, f"ID: {obj_id}", text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # 绿色文本


        # 保存结果图像
        output_path = f"{output_dir}/mask_{obj_id}.png"
        img_to_save = (img_overlay * 255).astype(np.uint8)  # 恢复到 0-255 范围
        cv2.imwrite(output_path, img_to_save)
        print(f"Saved: {output_path}")


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
    show_img_mask(masks, image, output_dir)
    return masks


def track(masks, video_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    frame_names = sorted([
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ])
    sam2_checkpoint = "/home/suma/pycharmprojects/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=video_dir)
    ann_frame_idx = 0

    # 循环添加mask
    obj_num = len(masks)
    print('初始帧共有{}个目标。'.format(obj_num))
    track_obj_id = 0
    for ann_obj_id in range(obj_num):
        mask = masks[ann_obj_id]['segmentation']
        # 如果mask符合特定大小，就跟踪它。我们没有能力跟踪极端小的物体。再强的算法也没有这种能力。跟踪过程对于极端小的物体效果也不好。
        if 300 < np.sum(mask) < 4000:
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=track_obj_id,
                mask=mask,
            )
            track_obj_id += 1

    print('开始追踪')
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # 获得每个mask的pred_iou
    obj_iou_per_frame = {}
    out_obj_ids = inference_state['obj_ids'] # list
    for out_obj_id in out_obj_ids:  # 遍历每个物体
        obj_iou_per_frame[out_obj_id] = {}
        non_cond_frame_outputs = inference_state['output_dict_per_obj'][out_obj_id]['non_cond_frame_outputs']
        for out_frame_idx, value in non_cond_frame_outputs.items():
            ious = value['ious']
            obj_iou_per_frame[out_obj_id][out_frame_idx] = ious
    print('追踪完成')

    # render the segmentation results every few frames
    vis_frame_stride = 1
    # 循环每帧：其实就是2帧：query_frame 和 reference_frame.
    query_obj = []  # 现场车底中的目标。第0帧。
    reference_obj = []  # 参考帧中跟踪到的目标。第1帧。一定是首帧目标的子集，因为是跟踪的结果，不可能出现新物体。
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):  # 遍历每帧
        output_path = os.path.join(output_dir, f'frame_{out_frame_idx:04d}.png')
        plt.figure(figsize=(6, 4))
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # 如果当前帧是初始帧，则绘制mask
            if out_frame_idx == 0:
                query_obj.append(out_obj_id)
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            elif torch.max(obj_iou_per_frame[out_obj_id][out_frame_idx]) > 0.6:
                # 如果不是初始帧：如果一个mask的pred_iou大于阈值，则绘制该mask。
                reference_obj.append(out_obj_id)
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    # 计算两帧之间的目标差异。第二帧一定是首帧目标的子集，因为是跟踪的结果，不可能出现新物体。
    diff_objs = set(query_obj) - set(reference_obj)
    print(diff_objs)


def main():
    # 设置图像路径
    video_dir = '/home/suma/PycharmProjects/abnormal_tracking/exp1'
    output_dir = os.path.join(video_dir, 'outputs/img_seg_result')

    # 把图像缩放至(1024, 410)。因为常见轿车长宽比为2.5:1。缩放后覆盖原始图像文件（这样能避免图像分割和视频追踪代码读入图像尺寸不一致的问题），必须实时保存。
    # 必须进行缩放，否则会爆显存。
    original_images = sorted(glob.glob(os.path.join(video_dir, 'original_images/*.jpg')))
    for original_image_path in original_images:
        image_name = original_image_path.split('/')[-1]
        image = Image.open(original_image_path)
        scaled_image = image.resize((1662, 682))
        save_image_path = os.path.join(video_dir, image_name)
        scaled_image.save(save_image_path)

    # 得到图像分割的结果
    query_img = os.path.join(video_dir, '00001.jpg')
    masks = img_seg(query_img, output_dir)

    # 跟踪
    track(masks, video_dir, output_dir)

if __name__ == '__main__':
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    main()
