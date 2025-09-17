import os
import re
import time
import cv2

def images_to_video(image_folder, output_path, fps=89, codec='XVID'):
    """
    将图片文件夹中的图片合成视频。
    
    参数:
        image_folder (str): 包含图片的文件夹路径。
        output_path (str): 输出视频文件路径（如 "output.mp4"）。
        fps (int): 视频帧率（如 89）。
        codec (str): 编解码器（如 'XVID' 或 'mp4v'）。
    """
    # 获取所有以 .png 结尾的文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    
    # 按数字排序（如 51.png, 52.png）
    image_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    
    if not image_files:
        print("未找到图片文件。")
        return

    # 读取第一张图片以确定尺寸
    first_image_path = os.path.join(image_folder, image_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"无法读取图片 {image_files[0]}。")
        return
    height, width, layers = frame.shape

    # 初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_path = os.path.join(image_folder, output_path)
    print(f"正在创建视频文件 {video_path}...")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print("无法初始化视频写入器，请检查编解码器或路径。")
        return

    # 逐帧写入视频
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path)
        if img is None:
            print(f"跳过无法读取的图片 {image_file}。")
            continue
        # 确保所有图片尺寸一致
        if img.shape != frame.shape:
            print(f"图片 {image_file} 的尺寸与首帧不同，正在调整尺寸。")
            img = cv2.resize(img, (width, height))
        video_writer.write(img)

    # 释放资源
    video_writer.release()
    print(f"视频已保存至 {video_path}")

if __name__ == '__main__':
    folders_path = "/data/Base_dataset/infer_data/0902_raw/3"
    for img_folder in os.listdir(folders_path):
        if len(img_folder) != 16:
            continue
        print(img_folder)
        folder = os.path.join(folders_path, img_folder)
        print("folder", folder)
        if os.path.isdir(folder):
            images_to_video(folder, img_folder + '.mp4')

# images_to_video('/data/small/10/20250814111658_1', 'small.mp4')