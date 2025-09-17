import cv2
import os

def video_to_images(video_path, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    frame_count = 0
    while True:
        # 读取视频的每一帧
        ret, frame = cap.read()

        # 如果读取成功
        if ret:
            # 保存当前帧为图像
            frame_filename = os.path.join(output_folder, f"{frame_count}.png")
            cv2.imwrite(frame_filename, frame)

            print(f"Saved {frame_filename}")

            frame_count += 1
        else:
            break

    # 释放视频资源
    cap.release()
    print("Video to images conversion complete.")

# 使用示例
video_path = '/data/Base_dataset/0909 error tracking dl/20000101114643_1.mp4'  # 替换为视频文件的路径
output_folder = '/data/Base_dataset/0909 error tracking dl/20000101114643_1'  # 替换为你想保存图像的文件夹路径
# video_path = '../test.avi'
# output_folder = '../test'
video_to_images(video_path, output_folder)

# img = cv2.imread('/data/small/10/20250814111658_1/58.png')
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()