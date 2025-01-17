import cv2
import os
import time
import numpy as np
from utils.common import natural_sort_key, resize_frame


# 在ROI中检测特征点
feature_params = dict(
    maxCorners=50,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# 光流法参数
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# 指定图像帧所在目录
frame_path = "./data/test2"

# 设置帧率（每秒播放的帧数）
fps = 5  # 可以根据需要调整，数值越小播放越慢
frame_delay = 1000 // fps  # 计算每帧之间的延迟时间（毫秒）
scale = 0.5  # 缩小到原图的 50%

frame_files = sorted([
    os.path.join(frame_path, f) 
    for f in os.listdir(frame_path) 
    if f.endswith(('.bmp', '.png', '.jpg', '.jpeg'))  # 添加 .bmp 格式
], key=natural_sort_key)

if not frame_files:
    print("没有找到图像帧")
    exit()

print(frame_files)

# 读取第一帧
first_frame = cv2.imread(frame_files[0])
if first_frame is None:
    print("无法读取第一帧")
    exit()

first_frame_resized = resize_frame(first_frame, scale)

# 存储多个目标的特征点
objects = []

while True:
    # 选择ROI
    bbox_scaled = cv2.selectROI('Select object', first_frame_resized, fromCenter=False, showCrosshair=True)
    x, y, w, h = [int(v / scale) for v in bbox_scaled]

    if (bbox_scaled[2]*bbox_scaled[3]) > 70000 * scale * scale:
        break

    # 创建ROI的mask
    roi = first_frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


    # 在ROI中检测特征点
    roi_points = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
    if roi_points is not None:
        # 将特征点坐标调整为全局坐标
        roi_points = roi_points + np.array([[x, y]], dtype=np.float32)
        objects.append(roi_points)
    else:
        print("No features found in ROI")
        exit()
cv2.destroyWindow('Select object')  # 关闭选择窗口

# 创建用于绘制的mask
mask = np.zeros_like(first_frame)
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

i = 0
for frame_path in frame_files:
    i+=1
    if(i % 1 != 0):
        continue
    
    
    # 读取每一帧
    frame = cv2.imread(frame_path)
    frame_resized = resize_frame(frame, scale)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    new_objects = []
    
    for i, roi_points in enumerate(objects):
        
        # 检查特征点是否为空
        if roi_points is None or len(roi_points) == 0:
            print(f"目标 {i} 的特征点为空")
            continue

        st = time.time()
        new_p, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, roi_points, None, **lk_params
        )
        et = time.time()
        print(f"Time: {et-st}")

        # 找到好的跟踪点
        good_new = new_p[status==1]
        good_old = roi_points[status==1]


        # 如果没有好的跟踪点，跳过
        if len(good_new) == 0:
            continue
        
         # 绘制跟踪轨迹
        for new, old in zip(good_new, good_old):
            a, b = new.ravel() * scale
            c, d = old.ravel() * scale
            
            # 绘制轨迹线
            cv2.line(frame_resized, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            cv2.circle(frame_resized, (int(a), int(b)), 5, (0, 0, 255), -1)
        
        # 更新特征点
        new_objects.append(good_new.reshape(-1, 1, 2))
    # 更新特征点列表
    objects = new_objects

    # 显示帧
    cv2.imshow('Multi-Target Optical Flow', frame_resized)
    
    # 按 'q' 键退出，增加延迟控制播放速度
    key = cv2.waitKey(frame_delay)
    if key & 0xFF == ord('q'):
        break
    
    # 更新前一帧
    old_gray = frame_gray.copy()

# 清理
cv2.destroyAllWindows()




    
#     if roi_points is not None and len(roi_points) > 0:
#         # 计算光流
#         new_points, status, error = cv2.calcOpticalFlowPyrLK(
#             old_gray, frame_gray, roi_points, None, **lk_params
#         )

#         if new_points is not None:
#             # 选择good points
#             good_new = new_points[status == 1]
#             good_old = roi_points[status == 1]

#             # 计算边界框
#             if len(good_new) > 0:
#                 x_min = np.min(good_new[:, 0])
#                 x_max = np.max(good_new[:, 0])
#                 y_min = np.min(good_new[:, 1])
#                 y_max = np.max(good_new[:, 1])
                
#                 # 绘制边界框
#                 cv2.rectangle(frame, 
#                             (int(x_min), int(y_min)), 
#                             (int(x_max), int(y_max)), 
#                             (0, 255, 0), 2)

#             # 绘制轨迹
#             for i, (new, old) in enumerate(zip(good_new, good_old)):
#                 a, b = new.ravel()
#                 c, d = old.ravel()
#                 mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
#                 frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
            
#             # 更新特征点
#             roi_points = good_new.reshape(-1, 1, 2)

#         # 显示光流轨迹
#         img = cv2.add(frame, mask)
#         cv2.imshow('Optical Flow Tracking', img)

#     else:
#         cv2.imshow('Optical Flow Tracking', frame)

#     # 更新前一帧
#     old_gray = frame_gray.copy()
    
#     # 按 'q' 键退出，增加延迟控制播放速度
#     key = cv2.waitKey(frame_delay)
#     if key & 0xFF == ord('q'):
#         break

# # 清理
# cv2.destroyAllWindows()
