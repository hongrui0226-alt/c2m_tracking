import requests
import cv2
import numpy as np
from proto.image_data_pb2 import Response  # 确保已经生成了protobuf的Python文件

def save_image(image_data, file_name):
    """将二进制图像数据保存为文件"""
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite(file_name, img)

def 

def main():
    # 服务器URL
    url = "http://192.168.0.51:8086/get_data"
    
    try:
        # 发送GET请求
        response = requests.get(url)
        
        if response.status_code == 200:
            # 解析protobuf消息
            proto_response = Response()
            proto_response.ParseFromString(response.content)
            
            print(f"收到数据，时间戳: {proto_response.timestamp}")
            
            # 处理图像数据
            for key, image_data in proto_response.image_map.items():
                print(f"处理key为{key}的图像数据")
                for idx, image in enumerate(image_data.images):
                    print(f"图像名称: {image.name}")
                    
                    # 保存图像
                    output_filename = f"received_{key}_{idx}_{image.name}"
                    save_image(image.image_data, output_filename)
                    print(f"图像已保存为: {output_filename}")
        else:
            print(f"请求失败，状态码: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"请求出错: {e}")
    except Exception as e:
        print(f"处理数据时出错: {e}")

if __name__ == "__main__":
    main()