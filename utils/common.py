import cv2
import os
import re
import numpy as np
import colorsys
import json

def natural_sort_key(s):
    """
    将文件名转换为可以自然排序的键
    处理 '0', '1', '2', ..., '10', '11', ..., '101' 这类文件名
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', os.path.splitext(s)[0])]

# 缩放帧
def resize_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def get_image_files(frame_path, extensions=('.bmp', '.png', '.jpg', '.jpeg')):

    # 获取并排序图像文件
    frame_files = sorted([
        os.path.join(frame_path, f) 
        for f in os.listdir(frame_path) 
        if f.lower().endswith(extensions)
    ], key=natural_sort_key)
    
    # 检查是否找到图像文件
    if not frame_files:
        print(f"没有在 {frame_path} 目录下找到图像帧")
    
    return frame_files

def binarize_image(image, method='otsu', blur_kernel=(3,3), threshold_value=127, color_mode='and'):
    """
    图像二值化函数
    
    参数:
    - image: 输入图像
    - method: 二值化方法 
      - 'otsu': 大津法（自动选择阈值）
      - 'global': 全局阈值
    #   - 'adaptive_gaussian': 高斯自适应阈值
    #   - 'adaptive_mean': 均值自适应阈值
    - blur_kernel: 高斯模糊核大小，默认(5,5)
    - threshold_value: 全局阈值时使用的阈值
    
    返回:
    二值化后的图像
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 高斯模糊降噪
    # blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    # blurred = cv2.medianBlur(gray, 5)  # 3是核大小，必须是奇数
    blurred = gray

    # 根据方法选择二值化
    if method == 'otsu':
        # 大津法
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    elif method == 'global':
        # 全局阈值
        _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    
    elif method == 'adaptive_gaussian':
        # 高斯自适应阈值
        binary = cv2.adaptiveThreshold(
            blurred, 
            255,  
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  
            cv2.THRESH_BINARY,  
            11,  # 块大小
            2    # 常数
        )
    
    elif method == 'color_mode':
        # 对每个颜色通道分别二值化
        b, g, r = cv2.split(image)
        _, b_binary = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
        _, g_binary = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
        _, r_binary = cv2.threshold(r, 127, 255, cv2.THRESH_BINARY)

        # 根据color_mode组合通道
        if color_mode == 'and':
            binary = cv2.bitwise_and(b_binary, g_binary, r_binary)
        elif color_mode == 'or':
            binary = cv2.bitwise_or(b_binary, g_binary, r_binary)
        elif color_mode == 'channel_max':
            binary = cv2.max(cv2.max(b_binary, g_binary), r_binary)
        elif color_mode == 'channel_min':
            binary = cv2.min(cv2.min(b_binary, g_binary), r_binary)
        else:
            raise ValueError("Invalid color_mode")
        
    
    # elif method == 'adaptive_mean':
    #     # 均值自适应阈值
    #     binary = cv2.adaptiveThreshold(
    #         blurred, 
    #         255,  
    #         cv2.ADAPTIVE_THRESH_MEAN_C,  
    #         cv2.THRESH_BINARY,  
    #         11,  # 块大小
    #         2    # 常数
    #     )
    
    else:
        raise ValueError("Invalid binarization method")
    
    return binary


def background_subtraction(image, background, threshold=25):
    """
    背景减除函数
    
    参数:
    - image: 输入图像
    - background: 背景图像
    - threshold: 阈值
    
    返回:
    背景减除后的图像
    """
    # 确保图像大小一致
    if image.shape[:2] != background.shape[:2]:
        background = cv2.resize(background, (image.shape[1], image.shape[0]))
    
    # 处理彩色图
    if len(image.shape) == 3 and len(background.shape) == 3:
        # 分通道处理
        diff_channels = []
        for i in range(image.shape[2]):
            diff_channel = cv2.absdiff(image[:,:,i], background[:,:,i])
            diff_channels.append(diff_channel)
        
        # 合并通道差值
        diff = np.max(diff_channels, axis=0)
    
    # 处理灰度图
    elif len(image.shape) == 2 and len(background.shape) == 2:
        diff = cv2.absdiff(image, background)
    
    else:
        # 转换为灰度
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
            background_gray = background
        
        diff = cv2.absdiff(image_gray, background_gray)
    
    # 阈值化
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # 创建结果图像
    if len(image.shape) == 3:
        # 彩色图
        result = image.copy()
        result[mask == 0] = [0, 0, 0]  # 将背景设为黑色
    else:
        # 灰度图
        result = image.copy()
        result[mask == 0] = 0  # 将背景设为黑色
    
    return result


def generate_distinct_colors(n):
    """
    生成 n 个尽可能区分的颜色
    
    :param n: 需要生成的颜色数量
    :return: RGB颜色列表
    """
    colors = []
    for i in range(n):
        # 在色相空间均匀分布
        hue = i / n
        # 饱和度和亮度固定，可以调整
        saturation = 0.8
        value = 0.8
        
        # HSV 转 RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # 转换为 0-255 范围的整数
        rgb_255 = tuple(int(x * 255) for x in rgb)
        
        colors.append(rgb_255)
    
    return colors



# 保存 bbox 的函数
def save_bboxes(bboxes, filename='saved_bboxes.json'):
    """
    将边界框保存到 JSON 文件
    
    :param bboxes: 边界框列表，每个 bbox 是 (x, y, w, h) 格式
    :param filename: 保存文件名
    """
    with open(filename, 'w') as f:
        json.dump(bboxes, f)
    print(f"Bounding boxes saved to {filename}")

# 加载 bbox 的函数
def load_bboxes(filename='saved_bboxes.json'):
    """
    从 JSON 文件加载边界框
    
    :param filename: 加载文件名
    :return: 边界框列表
    """
    if not os.path.exists(filename):
        return []
    
    with open(filename, 'r') as f:
        bboxes = json.load(f)
    print(f"Loaded {len(bboxes)} bounding boxes from {filename}")
    return bboxes