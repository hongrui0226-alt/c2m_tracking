import cv2
import numpy as np
import os
import sys
import time

from utils.common import resize_frame, natural_sort_key, get_image_files

def correlation_track(template, search_region):
    """
    使用模板匹配进行目标追踪
    
    参数:
    - template: 目标模板
    - search_region: 搜索区域
    
    返回:
    - 匹配位置的左上角坐标
    """
    # 模板匹配
    result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    return max_loc, max_val

def main():
    # 读取图像序列
    frame_path = "./data/test2"
    fps = 5  # 可以根据需要调整，数值越小播放越慢
    frame_delay = 1000 // fps  # 计算每帧之间的延迟时间（毫秒）
    scale = 0.5  # 缩小到原图的 50%

    frame_files = get_image_files(frame_path)

    # 读取第一帧
    first_frame = cv2.imread(frame_files[0])
    first_frame_resized = resize_frame(first_frame, scale)

    # 选择初始追踪目标
    bbox = cv2.selectROI('Select Target', first_frame_resized, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow('Select Target')

    # 调整坐标到原始图像
    x, y, w, h = [int(v / scale) for v in bbox]
    
    # 提取初始模板
    template = first_frame[y:y+h, x:x+w]
    
    # 处理后续帧
    for frame_path in frame_files[1:]:
        # 读取帧
        frame = cv2.imread(frame_path)
        st = time.time()
        # 进行模板匹配
        (new_x, new_y), confidence = correlation_track(template, frame)
        et = time.time()
        print(f"Time: {et-st}")
        # 绘制追踪框
        cv2.rectangle(
            frame, 
            (new_x, new_y), 
            (new_x + w, new_y + h), 
            (0, 255, 0), 
            2
        )
        
        # 显示置信度
        cv2.putText(
            frame, 
            f'Confidence: {confidence:.2f}', 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), 
            2
        )
        
        # 显示帧
        frame_resized = resize_frame(frame, 0.5)
        cv2.imshow('Correlation Tracking', frame_resized)
        
        # 更新模板（可选）
        # template = frame[new_y:new_y+h, new_x:new_x+w]
        
        # 控制帧率
        key = cv2.waitKey(frame_delay)  # 200ms 延迟
        if key & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()