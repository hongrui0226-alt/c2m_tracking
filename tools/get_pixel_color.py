from turtle import back
import cv2
import numpy as np
import os, sys
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)
from utils.common import resize_frame, binarize_image, background_subtraction

def get_pixel_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 获取点击位置的BGR值
        bgr = frame[y, x]
        
        # BGR转RGB
        rgb = bgr[::-1]
        
        # 打印坐标和颜色值
        print(f"坐标 (x, y): ({x}, {y})")
        # print(f"BGR值: {bgr}")
        print(f"RGB值: {rgb}")
        print("---")

# 读取图像
image_path = "../data/test2/65.jpg"  # 替换为你的图像路径
background_path = "../data/test2/219.jpg"
frame = cv2.imread(image_path)
background = cv2.imread(background_path)

# frame = resize_frame(frame, scale=0.5)
# background = resize_frame(background, scale=0.5)

# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# frame = filter_background(frame, threshold=30)
frame = background_subtraction(frame, background)
frame = binarize_image(frame, method='global', threshold_value=1, color_mode='channel_max')
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

frame = resize_frame(frame, scale=0.5)

if frame is None:
    print("无法读取图像")
    exit()

# 创建窗口并设置鼠标回调
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', get_pixel_color)

# 显示图像
while True:
    cv2.imshow('Image', frame)
    
    # 按 'q' 键退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()