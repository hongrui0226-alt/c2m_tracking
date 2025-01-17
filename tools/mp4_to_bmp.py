import cv2
import os

def video_to_frames(video_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 初始化帧计数器
    frame_count = 0

    while True:
        # 读取帧
        ret, frame = cap.read()

        # 如果没有更多帧，退出循环
        if not ret:
            break

        # 生成帧文件名（带前导零）
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.bmp')

        # 保存帧为 .bmp 文件
        cv2.imwrite(frame_filename, frame)

        # 增加帧计数器
        frame_count += 1

    # 释放视频捕获对象
    cap.release()

    print(f"Extracted {frame_count} frames to {output_dir}")

# 使用示例
video_path = './data/result_policy_best_1.mp4'
output_dir = './data/result_policy_best_1'  # 与之前的脚本保持一致

video_to_frames(video_path, output_dir)