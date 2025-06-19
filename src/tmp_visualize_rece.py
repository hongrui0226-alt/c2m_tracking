# client.py
import socket
import struct
import numpy as np
import cv2

def receive_data(sock):
    # 1. 接收图像数量
    count_data = sock.recv(4)
    count = struct.unpack('I', count_data)[0]
    
    images = []
    names = []
    
    # 2. 循环接收每个图像和对应的名称
    for _ in range(count):
        # # 2.1 接收图像宽度、高度和通道数
        # width_data = sock.recv(4)
        # height_data = sock.recv(4)
        # channels_data = sock.recv(4)
        
        # width = struct.unpack('I', width_data)[0]
        # height = struct.unpack('I', height_data)[0]
        # channels = struct.unpack('I', channels_data)[0]
        
        # 2.2 接收图像数据大小
        image_size_data = sock.recv(8)
        image_size = struct.unpack('Q', image_size_data)[0]
        
        # 2.3 接收图像数据
        image_data = b''
        while len(image_data) < image_size:
            chunk = sock.recv(min(65536, image_size - len(image_data)))
            if not chunk:
                break
            image_data += chunk
        
        # 将字节数据转换为 numpy 数组
        img_array = np.frombuffer(image_data, dtype=np.uint8)
        img = img_array.reshape((480, 640, 3))
        
        # 2.4 接收图像名称长度
        name_length_data = sock.recv(4)
        name_length = struct.unpack('I', name_length_data)[0]
        
        # 2.5 接收图像名称
        name_data = sock.recv(name_length)
        name = name_data.decode('utf-8')
        
        images.append(img)
        names.append(name)
    
    return images, names

def main():
    # 创建套接字并连接到服务器
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('127.0.0.1', 8888)
    sock.connect(server_address)
    
    print("Connected to server!")
    
    # 接收数据
    images, names = receive_data(sock)
    
    print(f"Received {len(images)} images:")
    for i, (img, name) in enumerate(zip(images, names)):
        print(f"  Image {i+1}: {name}, shape: {img.shape}")
        # # 显示图像
        # cv2.imshow(name, img)
        # cv2.waitKey(1000)  # 等待1秒
    
    # cv2.destroyAllWindows()
    sock.close()

if __name__ == "__main__":
    main()
