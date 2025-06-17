#include "tracker.hpp"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

// 从字节流反序列化 cv::Mat
cv::Mat deserializeMat(const std::vector<uchar>& buffer) {
    return cv::imdecode(buffer, cv::IMREAD_COLOR);
}

// 反序列化 ImageData
ImageData deserializeImageData(const std::vector<uchar>& data, ImageData& result) {
    size_t offset = 0;
    
    // 反序列化 names 数组
    int namesSize;
    memcpy(&namesSize, &data[offset], sizeof(namesSize));
    offset += sizeof(namesSize);
    
    for (int i = 0; i < namesSize; ++i) {
        int nameSize;
        memcpy(&nameSize, &data[offset], sizeof(nameSize));
        offset += sizeof(nameSize);
        
        std::string name(data.begin() + offset, data.begin() + offset + nameSize);
        offset += nameSize;
        
        result.names.push_back(name);
    }
    
    // 反序列化 images 数组
    int imagesSize;
    memcpy(&imagesSize, &data[offset], sizeof(imagesSize));
    offset += sizeof(imagesSize);
    
    for (int i = 0; i < imagesSize; ++i) {
        int imgSize;
        memcpy(&imgSize, &data[offset], sizeof(imgSize));
        offset += sizeof(imgSize);
        
        std::vector<uchar> imgData(data.begin() + offset, data.begin() + offset + imgSize);
        offset += imgSize;
        
        result.images.push_back(deserializeMat(imgData));
    }
    
    return result;
}

// 反序列化整个 unordered_map
std::unordered_map<int, ImageData> deserializeMap(const std::vector<uchar>& data) {
    std::unordered_map<int, ImageData> result;
    size_t offset = 0;
    
    // 1. 反序列化 map 大小
    int mapSize;
    memcpy(&mapSize, &data[offset], sizeof(mapSize));
    offset += sizeof(mapSize);
    
    // 2. 反序列化每个键值对
    for (int i = 0; i < mapSize; ++i) {
        // 反序列化键
        int key;
        memcpy(&key, &data[offset], sizeof(key));
        offset += sizeof(key);
        
        // 反序列化值的大小
        int valueSize;
        memcpy(&valueSize, &data[offset], sizeof(valueSize));
        offset += sizeof(valueSize);
        
        // 提取值数据
        std::vector<uchar> valueData(&data[offset], &data[offset + valueSize]);
        offset += valueSize;  // 跳过整个valueData块
        
        // 反序列化ImageData
        ImageData imgData;
        deserializeImageData(valueData, imgData);
        
        // 存储结果
        result[key] = imgData;
    }
    
    return result;
}

bool receiveDataOverSocket(std::vector<uchar>& data, int port) {
    // 创建 socket
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        std::cerr << "创建 socket 失败" << std::endl;
        return false;
    }
    
    // 设置服务器地址
    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(port);
    
    // 绑定 socket
    if (bind(sockfd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "绑定失败" << std::endl;
        close(sockfd);
        return false;
    }
    
    // 监听连接
    if (listen(sockfd, 1) < 0) {
        std::cerr << "监听失败" << std::endl;
        close(sockfd);
        return false;
    }
    
    std::cout << "等待连接..." << std::endl;
    
    // 接受连接
    struct sockaddr_in clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);
    int clientSockfd = accept(sockfd, (struct sockaddr*)&clientAddr, &clientAddrLen);
    if (clientSockfd < 0) {
        std::cerr << "接受连接失败" << std::endl;
        close(sockfd);
        return false;
    }
    
    std::cout << "连接已建立" << std::endl;
    
    // 接收数据大小
    size_t dataSize;
    if (recv(clientSockfd, &dataSize, sizeof(dataSize), 0) != sizeof(dataSize)) {
        std::cerr << "接收数据大小失败" << std::endl;
        close(clientSockfd);
        close(sockfd);
        return false;
    }
    
    // 接收数据
    data.resize(dataSize);
    uchar* ptr = data.data();
    size_t remaining = dataSize;
    while (remaining > 0) {
        ssize_t received = recv(clientSockfd, ptr, remaining, 0);
        if (received <= 0) {
            std::cerr << "接收数据失败" << std::endl;
            close(clientSockfd);
            close(sockfd);
            return false;
        }
        ptr += received;
        remaining -= received;
    }
    
    close(clientSockfd);
    close(sockfd);
    return true;
}

// 接收示例
std::unordered_map<int, ImageData> receiveImageDataMap(int port) {
    std::vector<uchar> receivedData;
    if (!receiveDataOverSocket(receivedData, port)) {
        return std::unordered_map<int, ImageData>();
    }
    
    return deserializeMap(receivedData);
}


int main(int argc, char** argv) { 
    // 接收数据
    int port = 8085; // 监听端口
    std::unordered_map<int, ImageData> receivedMap = receiveImageDataMap(port);
    if (receivedMap.empty()) {
        std::cerr << "接收数据失败或数据为空" << std::endl;
        return -1;
    }
    std::cout << "接收数据成功，包含 " << receivedMap.size() << " 个键值对" << std::endl;
    for (const auto& pair : receivedMap) {
        std::cout << "键: " << pair.first << ", 图像数量: " << pair.second.images.size() << std::endl;
        for (size_t i = 0; i < pair.second.images.size(); ++i) {
            std::cout << "图像 " << i + 1 << ": 名称 = " << pair.second.names[i] << ", 大小 = "
                      << pair.second.images[i].size() << std::endl;
            cv::imwrite("received_" + std::to_string(pair.first) + "_" + pair.second.names[i], pair.second.images[i]);
        }
    }
    return 0;
}