import cv2
import numpy as np
import os
import time
from utils.common import natural_sort_key, resize_frame

# 指定图像帧所在目录
frame_path = "./data/test2"

# 设置帧率和缩放
fps = 5
frame_delay = 1000 // fps
scale = 0.5

# 获取排序后的帧文件
frame_files = sorted([
    os.path.join(frame_path, f) 
    for f in os.listdir(frame_path) 
    if f.endswith(('.bmp', '.png', '.jpg', '.jpeg'))
], key=natural_sort_key)

if not frame_files:
    print("没有找到图像帧")
    exit()

# 读取第一帧
first_frame = cv2.imread(frame_files[0])
first_frame_resized = resize_frame(first_frame, scale)
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# 可视化光流
def visualize_flow(flow, frame):
    # 将光流转换为极坐标
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 归一化
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag = mag.astype(np.uint8)
    
    # 色调表示方向，亮度表示大小
    hsv = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = mag
    
    # 转换为BGR
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

# 主处理循环
for frame_path in frame_files[1:]:
    # 读取帧
    frame = cv2.imread(frame_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 计算稠密光流
    st = time.time()
    flow = cv2.calcOpticalFlowFarneback(
        first_frame_gray, frame_gray, None, 
        pyr_scale=0.5,     # 图像金字塔缩放因子
        levels=3,          # 金字塔层数
        winsize=15,        # 窗口大小
        iterations=3,      # 迭代次数
        poly_n=5,          # 像素邻域大小
        poly_sigma=1.2,    # 高斯标准差
        flags=0            # 标志
    )
    et = time.time()
    print(f"光流计算时间: {et-st:.4f}s")
    
    # 可视化光流
    flow_visualization = visualize_flow(flow, frame)
    flow_visualization_resized = resize_frame(flow_visualization, scale)
    
    # 显示原始帧和光流
    frame_resized = resize_frame(frame, scale)
    cv2.imshow('Original Frame', frame_resized)
    cv2.imshow('Optical Flow', flow_visualization_resized)
    
    # 按键控制
    key = cv2.waitKey(frame_delay)
    if key & 0xFF == ord('q'):
        break
    
    # 更新前一帧
    first_frame_gray = frame_gray

cv2.destroyAllWindows()