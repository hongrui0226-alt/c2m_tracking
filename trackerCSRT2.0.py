# Channel and Spatial Reliability Tracker

import cv2
import time
import os
from utils.common import resize_frame, binarize_image,background_subtraction, get_image_files, generate_distinct_colors, save_bboxes,load_bboxes

def main(frame_path, background_path, bboxes_filename, frame_delay, scale, interval, tracker_colors_list):
    # 获取排序后的帧文件
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

    cv2.imshow('frame_binarized', resize_frame(first_frame_binarized, scale))
    bboxes = []
    multi_tracker = []
    tracker_colors = {}

    saved_bboxes = load_bboxes(bboxes_filename)

    if saved_bboxes:
        # 使用已保存的 bbox
        bboxes = saved_bboxes
    else:
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
            bboxes.append(bbox)
        # 保存选择的 bbox
        save_bboxes(bboxes, bboxes_filename)

    # 为每个 bbox 创建追踪器
    for bbox in bboxes:
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(first_frame, bbox)
        multi_tracker.append(tracker)
        
        current_color = tracker_colors_list[len(multi_tracker)]
        tracker_colors[tracker] = current_color

        # 为每个选中的目标区域添加追踪器
        # multi_tracker.add(cv2.legacy.TrackerKCF_create(), first_frame, bbox)
        # multi_tracker.add(cv2.legacy.TrackerMIL_create(), first_frame, bbox)
        # multi_tracker.add(cv2.legacy.TrackerTLD_create(), first_frame, bbox)
        # multi_tracker.add(cv2.legacy.TrackerMOSSE_create(), first_frame, bbox)
        # multi_tracker.add(cv2.legacy.TrackerCSRT_create(), first_frame, bbox) # Nice
        # multi_tracker.add(cv2.legacy.TrackerMedianFlow_create(), first_frame, bbox)
        # multi_tracker.add(cv2.legacy.TrackerBoosting_create(), first_frame, bbox) #Nice

    cv2.destroyWindow('Select object')  # 关闭选择窗口
    # cv2.destroyWindow('frame_binarized')  # 关闭选择窗口

    # 绘制追踪框
    for tracker, box in zip(multi_tracker, bboxes):
        (x, y, w, h) = [int(v) for v in box]
        color = tracker_colors.get(tracker, (0, 0, 0))  # 默认黑色
        cv2.rectangle(first_frame, (x, y), (x + w, y + h), color, 2)
    cv2.imshow('MultiTracker', resize_frame(first_frame, scale))
    key = cv2.waitKey(0)


    i = 0

    for frame_path in frame_files:
        i+=1
        if(i % (interval+1) != 0):
            continue

        frame = cv2.imread(frame_path)
        frame_subtraction = background_subtraction(frame, background)
        frame_binarized = binarize_image(frame_subtraction, method='global', threshold_value=1)
        cv2.imshow('frame_binarized', resize_frame(frame_binarized, scale))

        tracker_success = []
        boxes = []
        success = True
        
        st = time.time()
        for tracker in multi_tracker:
            # 对每个追踪器单独更新
            success, bbox = tracker.update(frame)
            boxes.append(bbox)
            tracker_success.append(success)
            
        et = time.time()
        print(f"Time: {et-st}")

        # 删除追踪失败的追踪器
        multi_tracker[:] = [tracker for tracker, success in zip(multi_tracker, tracker_success) if success]
        boxes[:] = [box for box, success in zip(boxes, tracker_success) if success]
            
        # 绘制追踪框
        for tracker, box in zip(multi_tracker, boxes):
            (x, y, w, h) = [int(v) for v in box]
            # 跳过负坐标的追踪框
            if x < 0 or y < 0:
                print(f"跳过无效追踪器 {i}：坐标 ({x}, {y})")
                continue
            color = tracker_colors.get(tracker, (0, 0, 0))  # 默认黑色
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        frame_resized = resize_frame(frame, scale)
        cv2.imshow('MultiTracker', frame_resized)

        # 按 'q' 键退出，增加延迟控制播放速度
        key = cv2.waitKey(frame_delay)
        if key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":

    # 指定图像帧所在目录
    # frame_path = "./data/104/zhaopian_big"
    # background_path = "./data/104/frame_id19.jpg"
    frame_path = "./data/20250117/60big_2"
    background_path = "./data/20250117/frame_id240.jpg"

    bboxes_filename = os.path.join(frame_path, 'saved_bboxes1.json') # ***必看参数***: bboxes 保存的文件，如果存在则直接使用，不会 selectROI

    # 设置帧率（每秒播放的帧数）
    fps = 1  # 可以根据需要调整，数值越小播放越慢
    frame_delay = 1000 // fps  # 计算每帧之间的延迟时间（毫秒）
    scale = 0.5  # 缩小到原图的 50%
    interval = 0    # 帧处理间隔，0 为不间隔处理

    tracker_colors_list = generate_distinct_colors(30)  # 生成30种框的颜色
    
    main(frame_path, background_path, bboxes_filename, frame_delay, scale, interval, tracker_colors_list)