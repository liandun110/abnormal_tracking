import os
import time
import torch

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor

def show_anns(anns, output_path, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

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

def img_seg(query_img, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sam2_checkpoint = "/home/suma/PycharmProjects/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    image = Image.open(query_img)
    image = np.array(image.convert("RGB"))
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.9,
        stability_score_offset=0.9,
        crop_n_layers=1,
        box_nms_thresh=0.9,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25,
        use_m2m=True
    )
    print('开始分割')

    start_time = time.time()
    masks = mask_generator.generate(image)
    end_time = time.time()
    print('结束分割')
    print('分割耗时时间：', end_time - start_time)

    output_path = os.path.join(output_dir, 'segmentation_result.png')
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks, output_path)

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
    for ann_obj_id in range(len(masks)):
        mask = masks[ann_obj_id]['segmentation']
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask=mask,
        )

    print('开始追踪')
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    print('追踪完成')

    # render the segmentation results every few frames
    vis_frame_stride = 1
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        output_path = os.path.join(output_dir, f'frame_{out_frame_idx:04d}.png')
        plt.figure(figsize=(6, 4))
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # 设置图像路径
    video_dir = '/home/suma/PycharmProjects/sam2/exp1'
    query_img = os.path.join(video_dir, '00001.jpg')
    output_dir = '/home/suma/PycharmProjects/sam2/outputs'

    # 得到图像分割的结果
    masks = img_seg(query_img, output_dir)

    # 跟踪
    track(masks, video_dir, output_dir)

if __name__ == '__main__':
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    main()
