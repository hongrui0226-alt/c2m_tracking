import socket
import struct
import cv2
import numpy as np
import os
import pickle  # 用于发送请求数据

# 定义ImageData类，对应C++中的ImageData结构
class ImageData:
    def __init__(self):
        self.images = []
        self.names = []

# 从字节流反序列化图像
def deserialize_mat(buffer):
    nparr = np.frombuffer(buffer, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

# 反序列化ImageData
def deserialize_image_data(data):
    result = ImageData()
    offset = 0
    
    # 反序列化names数组
    names_size = struct.unpack_from('I', data, offset)[0]
    offset += struct.calcsize('I')
    
    for i in range(names_size):
        name_size = struct.unpack_from('I', data, offset)[0]
        offset += struct.calcsize('I')
        
        name = data[offset:offset+name_size].decode('utf-8')
        offset += name_size
        
        result.names.append(name)
    
    # 反序列化images数组
    images_size = struct.unpack_from('I', data, offset)[0]
    offset += struct.calcsize('I')
    
    for i in range(images_size):
        img_size = struct.unpack_from('I', data, offset)[0]
        offset += struct.calcsize('I')
        
        img_data = data[offset:offset+img_size]
        offset += img_size
        
        result.images.append(deserialize_mat(img_data))
    
    return result

# 反序列化整个字典
def deserialize_map(data):
    result = {}
    offset = 0
    
    # 反序列化map大小
    map_size = struct.unpack_from('I', data, offset)[0]
    offset += struct.calcsize('I')
    
    # 反序列化每个键值对
    for i in range(map_size):
        # 反序列化键
        key = struct.unpack_from('I', data, offset)[0]
        offset += struct.calcsize('I')
        
        # 反序列化值的大小
        value_size = struct.unpack_from('I', data, offset)[0]
        offset += struct.calcsize('I')
        
        # 提取值数据
        value_data = data[offset:offset+value_size]
        offset += value_size
        
        # 反序列化ImageData
        img_data = deserialize_image_data(value_data)
        
        # 存储结果
        result[key] = img_data
    
    return result

# 通过socket发送请求并接收数据
def send_request_and_receive_data(server_ip, server_port, request_data):
    # 创建socket
    sockfd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sockfd is None:
        print("创建socket失败")
        return None
    
    try:
        # 连接到服务器
        sockfd.connect((server_ip, server_port))
        print(f"已连接到服务器 {server_ip}:{server_port}")
        
        # 发送请求数据（这里使用pickle序列化请求）
        request_packed = pickle.dumps(request_data)
        request_size = len(request_packed)
        
        # 发送请求大小
        sockfd.sendall(struct.pack('Q', request_size))
        # 发送请求数据
        sockfd.sendall(request_packed)
        print(f"已发送请求，大小: {request_size} 字节")
        
        # 接收数据大小
        data_size_packed = sockfd.recv(struct.calcsize('Q'))
        if len(data_size_packed) != struct.calcsize('Q'):
            print("接收数据大小失败")
            sockfd.close()
            return None
        
        data_size = struct.unpack('Q', data_size_packed)[0]
        print(f"即将接收数据，大小: {data_size} 字节")
        
        # 接收数据
        received_data = b''
        remaining = data_size
        while remaining > 0:
            chunk = sockfd.recv(min(4096, remaining))
            if not chunk:
                print("接收数据失败")
                sockfd.close()
                return None
            
            received_data += chunk
            remaining -= len(chunk)
        
        sockfd.close()
        return received_data
    
    except Exception as e:
        print(f"通信时发生错误: {e}")
        if 'sockfd' in locals():
            sockfd.close()
        return None

# 主函数
if __name__ == "__main__":
    # 服务器配置
    server_ip = "localhost"  # 服务器IP地址
    server_port = 8085       # 服务器端口
    
    # 构造请求数据（示例：请求获取跟踪结果）
    request = {"action": "get_track_results", "camera_id": 1}
    
    # 发送请求并接收数据
    received_data = send_request_and_receive_data(server_ip, server_port, request)
    if received_data is None:
        print("请求失败或未接收到数据")
        exit(-1)
    
    # 反序列化接收到的数据
    received_map = deserialize_map(received_data)
    
    if not received_map:
        print("反序列化数据失败或数据为空")
        exit(-1)
    
    print(f"接收数据成功，包含 {len(received_map)} 个键值对")
    
    # 创建保存目录
    if not os.path.exists("received_images"):
        os.makedirs("received_images")
    
    for key, pair in received_map.items():
        print(f"键: {key}, 图像数量: {len(pair.images)}")

        for i in range(len(pair.images)):
            image = pair.images[i]
            name = pair.names[i]
            print(f"图像 {i+1}: 名称 = {name}, 大小 = {image.shape}")
            
            # 保存图像
            filename = f"received_images/received_{key}_{name}"
            cv2.imwrite(filename, image)
    
    print("图像已成功保存到 received_images 目录")
