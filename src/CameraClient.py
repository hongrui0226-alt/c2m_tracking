import socket
import struct
import cv2
import numpy as np
import os
import pickle  # 用于发送请求数据
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any, Optional, Tuple
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def send_request_and_receive_data(
    server_ip: str,
    server_port: int,
    request_data: Any,
    max_retries: int = 3,
    timeout: float = 15.0,
    buffer_size: int = 65536  # 64KB
) -> Optional[bytes]:
    """
    发送请求到服务器并接收响应数据，包含完整的错误处理和重试机制
    
    参数:
        server_ip: 服务器IP地址
        server_port: 服务器端口号
        request_data: 要发送的请求数据（支持任意可pickle序列化的对象）
        max_retries: 最大重试次数，默认为3次
        timeout: 连接和接收超时时间（秒），默认为15秒
        buffer_size: 接收数据的缓冲区大小（字节），默认为16KB
    
    返回:
        接收到的二进制数据，若失败则返回None
    """
    for attempt in range(max_retries):
        sockfd = None
        try:
            # 创建TCP socket
            sockfd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if not sockfd:
                logger.error("创建socket失败")
                return None
            
            # 配置socket选项
            sockfd.settimeout(timeout)  # 设置整体超时
            # 启用TCP保活机制，防止网络设备断开空闲连接
            sockfd.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sockfd.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)    # 60秒无数据则发送保活包
            sockfd.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)  # 保活包间隔10秒
            sockfd.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)     # 3次保活失败则断开
            
            # 连接到服务器
            logger.info(f"尝试连接到服务器 {server_ip}:{server_port} (尝试 {attempt+1}/{max_retries})")
            sockfd.connect((server_ip, server_port))
            logger.info(f"成功连接到服务器 {server_ip}:{server_port}")
            
            # 序列化请求数据
            try:
                request_packed = pickle.dumps(request_data)
            except Exception as e:
                logger.error(f"请求数据序列化失败: {e}")
                return None
            
            request_size = len(request_packed)
            
            # 发送请求大小和数据
            logger.debug(f"发送请求大小: {request_size} 字节")
            sockfd.sendall(struct.pack('Q', request_size))
            sockfd.sendall(request_packed)
            logger.info(f"已发送请求，大小: {request_size} 字节")
            
            # 接收时间戳
            timestamp = sockfd.recv(1024)
            timestamp = timestamp.decode('utf-8')
            logger.info(f"接收到时间戳: {timestamp}")

            # 接收响应数据大小
            logger.debug("等待接收数据大小...")
            data_size_packed = sockfd.recv(struct.calcsize('Q'))
            if len(data_size_packed) != struct.calcsize('Q'):
                logger.error("接收数据大小失败")
                return None
            
            # # 接收可视化数据
            # count_data = sockfd.recv(4)
            # count = struct.unpack('I', count_data)[0]
            
            images = []
            names = []
            
            # # 2. 循环接收每个图像和对应的名称
            # for _ in range(count):
                
            #     # 2.2 接收图像数据大小
            #     image_size_data = sockfd.recv(8)
            #     image_size = struct.unpack('Q', image_size_data)[0]
                
            #     # 2.3 接收图像数据
            #     image_data = b''
            #     while len(image_data) < image_size:
            #         chunk = sockfd.recv(min(buffer_size, image_size - len(image_data)))
            #         if not chunk:
            #             break
            #         image_data += chunk
                
            #     # 将字节数据转换为 numpy 数组
            #     img_array = np.frombuffer(image_data, dtype=np.uint8)
            #     img = img_array.reshape((480, 640, 3))
                
            #     # 2.4 接收图像名称长度
            #     name_length_data = sockfd.recv(4)
            #     name_length = struct.unpack('I', name_length_data)[0]
                
            #     # 2.5 接收图像名称
            #     name_data = sockfd.recv(name_length)
            #     name = name_data.decode('utf-8')
                
            #     images.append(img)
            #     names.append(name)

            data_size = struct.unpack('Q', data_size_packed)[0]
            logger.info(f"即将接收数据，大小: {data_size} 字节")
            
            # 接收完整数据
            track_res_data = b''
            remaining = data_size
            while remaining > 0:
                try:
                    # 调整接收超时，避免长时间阻塞
                    sockfd.settimeout(min(timeout, 5.0))
                    chunk = sockfd.recv(min(buffer_size, remaining))
                except socket.timeout:
                    logger.warning("接收超时，尝试继续接收...")
                    continue
                
                if not chunk:
                    # 处理服务器半关闭连接
                    if remaining > 0:
                        logger.warning(f"服务器提前关闭连接，剩余 {remaining} 字节未接收")
                        break
                    else:
                        break  # 数据已接收完毕
                
                track_res_data += chunk
                remaining -= len(chunk)
                logger.debug(f"已接收 {data_size - remaining}/{data_size} 字节")
            
            # 检查是否完整接收
            if remaining > 0:
                logger.error(f"数据接收不完整，仅接收 {data_size - remaining}/{data_size} 字节")
                return None
            
            logger.info(f"成功接收数据，总大小: {len(track_res_data)} 字节")
            return track_res_data, images, names, timestamp
            
        except (ConnectionResetError, BrokenPipeError) as e:
            logger.error(f"连接被服务器重置 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 0.5 * (2 ** attempt)  # 指数退避算法
                logger.info(f"将在 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
        except socket.timeout as e:
            logger.error(f"连接或接收超时 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"将在1秒后重试...")
                time.sleep(1)
        except socket.gaierror as e:
            logger.error(f"域名解析错误: {e}")
            break  # 域名错误无需重试
        except Exception as e:
            logger.error(f"发生未知错误 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt >= max_retries - 1:
                logger.exception(e)  # 记录详细异常信息
        finally:
            # 确保关闭socket
            if sockfd:
                try:
                    sockfd.shutdown(socket.SHUT_RDWR)
                    sockfd.close()
                except Exception:
                    pass  # 忽略关闭时的异常
    
    logger.error(f"所有 {max_retries} 次尝试均失败")
    return None

def get_tracking_results(server_ip, server_port):
    # 构造请求数据（示例：请求获取跟踪结果）
    request = {"action": "get_track_results", "camera_id": 1}
    
    # 发送请求并接收数据
    track_res_data, vis_imgs, vis_names, timestamp = send_request_and_receive_data(server_ip, server_port, request)
    if track_res_data is None:
        print("请求失败或未接收到数据")
        exit(-1)
    
    # 反序列化接收到的数据
    track_resulsts = deserialize_map(track_res_data)
    
    if not track_resulsts:
        print("反序列化数据失败或数据为空")
        exit(-1)
    
    print(f"接收数据成功，包含 {len(track_resulsts)} 个键值对")
    
    return track_resulsts, vis_imgs, vis_names, timestamp


# 主函数
if __name__ == "__main__":
    # 服务器配置
    ips = ["192.168.0.51", "192.168.0.51"]  # 目标IP列表
    ports = [8085, 8086]                       # 目标端口列表
    
    with ThreadPoolExecutor(max_workers=len(ips)) as executor:
        futures = [executor.submit(get_tracking_results, ip, port) for ip, port in zip(ips, ports)]
        results = [future.result() for future in futures]

    # results = [get_tracking_results(ips[0], ports[0])]
    
    
    for index, (track_resulsts, vis_imgs, vis_names, timestamp) in enumerate(results):

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

        for i, (img, name) in enumerate(zip(vis_imgs, vis_names)):
            print(f"可视化图像 {i+1}: 名称 = {name}, 大小 = {img.shape}")
            # 保存可视化图像
            vis_filename = f"received_images_{index}/vis/{name}"
            cv2.imwrite(vis_filename, img)
        
    print(f"图像已成功保存到 received_images_{index} 目录")

    
