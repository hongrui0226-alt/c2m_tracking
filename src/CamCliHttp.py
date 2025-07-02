import requests
import cv2
import numpy as np
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from proto.image_data_pb2 import Response

# 定义ImageData类，对应C++中的ImageData结构
class ImageData:
    def __init__(self):
        self.images = []
        self.names = []

def save_image(image_data, file_name):
    """将二进制图像数据保存为文件"""
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite(file_name, img)

def decode_map(images_data):
    res_data = ImageData()
    for image in images_data.images:
        nparr = np.frombuffer(image.image_data)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        res_data.images.append(img)
        res_data.names.append(image.name)
    return res_data

def parase_data(response):
    # 解析protobuf消息
    proto_response = Response()
    proto_response.ParseFromString(response.content)

    timestamp = proto_response.timestamp

    track_results = {}
    for id, images_data in proto_response.image_map.items():
        track_results[id] = decode_map(images_data)

    return track_results, timestamp

def get_tracking_results(url):
    response = requests.get(url)
    if response.status_code == 200:
        track_results, timestamp = parase_data(response)
        return track_results, timestamp
    else:
        print(f"{url} 请求失败")
        return None, None

def main():
    # 服务器URL
    url = "http://192.168.0.51:8086/get_data"
    url_lists = [
        "http://192.168.0.51:8086/get_data",
        "http://192.168.0.51:8085/get_data"
    ]
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(get_tracking_results, url) for url in url_lists]
        results = [future.result() for future in futures]

    for index, (track_resulsts, timestamp) in enumerate(results):
        # 创建保存目录
        if not os.path.exists(f"received_images_{index}"):
            os.makedirs(f"received_images_{index}")
        
        for key, pair in track_resulsts.items():
            print(f"键: {key}, 图像数量: {len(pair.images)}")
            if not os.path.exists(f"received_images_{index}/{key}"):
                os.makedirs(f"received_images_{index}/{key}")
            for i in range(len(pair.images)):
                image = pair.images[i]
                name = pair.names[i]
                # print(f"图像 {i+1}: 名称 = {name}, 大小 = {image.shape}")
                
                # 保存图像
                filename = f"received_images_{index}/{key}/{name}"
                cv2.imwrite(filename, image)

        if not os.path.exists(f"received_images_{index}/vis"):
            os.makedirs(f"received_images_{index}/vis")

if __name__ == "__main__":
    main()