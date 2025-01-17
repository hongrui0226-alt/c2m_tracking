
# Channel and Spatial Reliability Tracker

import cv2
import time
import os
from utils.common import natural_sort_key, resize_frame, binarize_image,background_subtraction, get_image_files


# 指定图像帧所在目录
frame_path = "./data/104/zhaopian_big"
background_path = "./data/104/frame_id195.jpg"

# 设置帧率（每秒播放的帧数）
fps = 1  # 可以根据需要调整，数值越小播放越慢
frame_delay = 1000 // fps  # 计算每帧之间的延迟时间（毫秒）
scale = 0.5  # 缩小到原图的 50%
interval = 0    # 帧处理间隔，0 为不间隔处理


frame_files = get_image_files(frame_path)
background = cv2.imread(background_path)

# 创建多目标追踪器
multi_tracker = cv2.legacy.MultiTracker_create()

# 读取第一帧
first_frame = cv2.imread(frame_files[0])
if first_frame is None:
    print("无法读取第一帧")
    exit()
first_frame_resized = resize_frame(first_frame, scale)
first_frame_subtraction = background_subtraction(first_frame, background)
first_frame_binarized = binarize_image(first_frame_subtraction, method='global', threshold_value=1, color_mode='channel_max')

cv2.imshow('first_frame_binarized', resize_frame(first_frame_binarized, scale))

# 循环选择多个 ROI
while True:
    # 使用 selectROI 选择一个目标区域
    bbox_scaled = cv2.selectROI('Select object', first_frame_resized, fromCenter=False, showCrosshair=True)

    if (bbox_scaled[2]*bbox_scaled[3]) > 70000 * scale * scale:
        break

    bbox = (
        int(bbox_scaled[0] / scale),  # x
        int(bbox_scaled[1] / scale),  # y
        int(bbox_scaled[2] / scale),  # width
        int(bbox_scaled[3] / scale)   # height
    )
    print(f"bbox: {bbox}")

    # 为每个选中的目标区域添加追踪器
    # multi_tracker.add(cv2.legacy.TrackerKCF_create(), first_frame, bbox)
    # multi_tracker.add(cv2.legacy.TrackerMIL_create(), first_frame, bbox)
    # multi_tracker.add(cv2.legacy.TrackerTLD_create(), first_frame, bbox)
    # multi_tracker.add(cv2.legacy.TrackerMOSSE_create(), first_frame, bbox)
    multi_tracker.add(cv2.legacy.TrackerCSRT_create(), first_frame, bbox) # Nice
    # multi_tracker.add(cv2.legacy.TrackerMedianFlow_create(), first_frame, bbox)
    # multi_tracker.add(cv2.legacy.TrackerBoosting_create(), first_frame, bbox) #Nice

cv2.destroyWindow('Select object')  # 关闭选择窗口
cv2.destroyWindow('first_frame_binarized')  # 关闭选择窗口

i = 0

for frame_path in frame_files:
    i+=1
    if(i % (interval+1) != 0):
        continue

    frame = cv2.imread(frame_path)
    frame_subtraction = background_subtraction(frame, background)
    frame_binarized = binarize_image(frame_subtraction, method='global', threshold_value=1, color_mode='channel_max')
    cv2.imshow('frame_binarized', resize_frame(frame_binarized, scale))

    st = time.time()
    # 更新追踪器
    success, boxes = multi_tracker.update(frame)
    et = time.time()
    print(f"Time: {et-st}")
    
    print(f"success: {success}")
    print(f"boxes: {boxes}")
    print("--------------------")
    # 绘制追踪框
    for i, box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in box]
         # 跳过负坐标的追踪框
        if x < 0 or y < 0:
            print(f"跳过无效追踪器 {i}：坐标 ({x}, {y})")
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    frame_resized = resize_frame(frame, scale)
    cv2.imshow('MultiTracker', frame_resized)

    # 按 'q' 键退出，增加延迟控制播放速度
    key = cv2.waitKey(frame_delay)
    if key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


