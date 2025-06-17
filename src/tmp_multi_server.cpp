#include "tracker.hpp"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>

// 数据存储与线程安全
std::unordered_map<int, ImageData> globalTrackResults;
std::mutex mtx;
std::condition_variable condVar;
bool newDataAvailable = false;

// 将 cv::Mat 序列化为字节流
std::vector<uchar> serializeMat(const cv::Mat& mat) {
    std::vector<uchar> buffer;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
    cv::imencode(".jpg", mat, buffer, params);
    return buffer;
}

// 序列化 ImageData
std::vector<uchar> serializeImageData(const ImageData& data) {
    std::vector<uchar> result;
    
    // 序列化 names 数组
    int namesSize = data.names.size();
    result.insert(result.end(), (uchar*)&namesSize, (uchar*)&namesSize + sizeof(namesSize));
    
    for (const auto& name : data.names) {
        int nameSize = name.size();
        result.insert(result.end(), (uchar*)&nameSize, (uchar*)&nameSize + sizeof(nameSize));
        result.insert(result.end(), name.begin(), name.end());
    }
    
    // 序列化 images 数组
    int imagesSize = data.images.size();
    result.insert(result.end(), (uchar*)&imagesSize, (uchar*)&imagesSize + sizeof(imagesSize));
    
    for (const auto& image : data.images) {
        std::vector<uchar> imgData = serializeMat(image);
        int imgSize = imgData.size();
        
        result.insert(result.end(), (uchar*)&imgSize, (uchar*)&imgSize + sizeof(imgSize));
        result.insert(result.end(), imgData.begin(), imgData.end());
    }
    
    return result;
}

// 序列化整个 unordered_map
std::vector<uchar> serializeMap(const std::unordered_map<int, ImageData>& map) {
    std::vector<uchar> result;
    
    // 序列化 map 大小
    int mapSize = map.size();
    result.insert(result.end(), (uchar*)&mapSize, (uchar*)&mapSize + sizeof(mapSize));
    
    // 序列化每个键值对
    for (const auto& pair : map) {
        // 序列化键
        int key = pair.first;
        result.insert(result.end(), (uchar*)&key, (uchar*)&key + sizeof(key));
        
        // 序列化值
        std::vector<uchar> valueData = serializeImageData(pair.second);
        int valueSize = valueData.size();
        
        result.insert(result.end(), (uchar*)&valueSize, (uchar*)&valueSize + sizeof(valueSize));
        result.insert(result.end(), valueData.begin(), valueData.end());
    }
    
    return result;
}

// 处理客户端连接
void handleClient(int clientSockfd) {
    try {
        // 接收客户端请求
        const int BUFFER_SIZE = 4096;
        char buffer[BUFFER_SIZE];
        
        // 接收请求大小
        size_t requestSize;
        if (recv(clientSockfd, &requestSize, sizeof(requestSize), 0) != sizeof(requestSize)) {
            throw std::runtime_error("接收请求大小失败");
        }
        
        // 接收请求数据
        std::vector<uchar> requestData(requestSize);
        size_t remaining = requestSize;
        uchar* ptr = requestData.data();
        
        while (remaining > 0) {
            ssize_t received = recv(clientSockfd, ptr, remaining, 0);
            if (received <= 0) {
                throw std::runtime_error("接收请求数据失败");
            }
            ptr += received;
            remaining -= received;
        }
        
        // 解析请求数据（示例：简单打印请求内容）
        std::string requestStr(reinterpret_cast<char*>(requestData.data()), requestData.size());
        std::cout << "收到客户端请求: " << requestStr << std::endl;
        
        // 准备响应数据（从全局存储获取）
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<uchar> responseData = serializeMap(globalTrackResults);
        
        // 发送响应数据大小
        size_t responseSize = responseData.size();
        if (send(clientSockfd, &responseSize, sizeof(responseSize), 0) != sizeof(responseSize)) {
            throw std::runtime_error("发送响应大小失败");
        }
        
        // 发送响应数据
        ptr = responseData.data();
        remaining = responseSize;
        while (remaining > 0) {
            ssize_t sent = send(clientSockfd, ptr, remaining, 0);
            if (sent <= 0) {
                throw std::runtime_error("发送响应数据失败");
            }
            ptr += sent;
            remaining -= sent;
        }
        
        std::cout << "已向客户端发送 " << responseSize << " 字节数据" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "客户端处理错误: " << e.what() << std::endl;
        close(clientSockfd);
    } 
}

// 模拟数据更新线程
void dataUpdaterThread() {
    while (true) {
        // 模拟更新跟踪结果（实际应用中应来自跟踪算法）
        {
            std::lock_guard<std::mutex> lock(mtx);
            ImageData imgData;
            imgData.names = {"blue.jpg", "green.jpg"};
            
            // 生成测试图像
            cv::Mat frame1(224, 224, CV_8UC3, cv::Scalar(255, 0, 0)); // 蓝色图像
            cv::Mat frame2(224, 224, CV_8UC3, cv::Scalar(0, 255, 0)); // 绿色图像
            imgData.images.push_back(frame1);
            imgData.images.push_back(frame2);
            
            // 更新时间戳作为键
            time_t now = time(nullptr);
            globalTrackResults.clear();
            globalTrackResults.insert({1, imgData});
        }
        
        // 通知主线程数据已更新
        {
            std::lock_guard<std::mutex> lock(mtx);
            newDataAvailable = true;
        }
        condVar.notify_all();
        
        // 模拟数据更新频率
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

// 端口监听线程函数
void portListenerThread(int port) {
    // 创建 socket
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        std::cerr << "创建 socket 失败，端口: " << port << std::endl;
        return;
    }
    
    // 设置端口复用
    int opt = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "设置 socket 选项失败，端口: " << port << std::endl;
        close(sockfd);
        return;
    }
    
    // 设置服务器地址
    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(port); // 监听端口
    
    // 绑定 socket
    if (bind(sockfd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "绑定失败，端口: " << port << std::endl;
        close(sockfd);
        return;
    }
    
    // 监听连接
    if (listen(sockfd, 5) < 0) {
        std::cerr << "监听失败，端口: " << port << std::endl;
        close(sockfd);
        return;
    }
    
    std::cout << "服务器启动成功，正在监听端口 " << port << "..." << std::endl;
    
    // 主循环处理客户端连接
    while (true) {
        // 接受客户端连接
        struct sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        int clientSockfd = accept(sockfd, (struct sockaddr*)&clientAddr, &clientAddrLen);
        
        if (clientSockfd < 0) {
            std::cerr << "接受连接失败，端口: " << port << std::endl;
            continue;
        }
        
        // 打印客户端信息
        char clientIP[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &clientAddr.sin_addr, clientIP, sizeof(clientIP));
        std::cout << "新客户端连接，端口 " << port << ": " << clientIP << ":" << ntohs(clientAddr.sin_port) << std::endl;
        
        // 在新线程中处理客户端请求
        std::thread clientThread(handleClient, clientSockfd);
        clientThread.detach();
    }
    
    // 关闭 socket
    close(sockfd);
}

int main() {
    try {
        // 定义要监听的端口
        const int PORT1 = 8085;
        const int PORT2 = 8086;
        
        // 启动数据更新线程
        std::thread updaterThread(dataUpdaterThread);
        updaterThread.detach();
        
        // 启动端口监听线程
        std::thread listenerThread1(portListenerThread, PORT1);
        std::thread listenerThread2(portListenerThread, PORT2);
        
        // 等待端口监听线程结束（理论上不会结束）
        listenerThread1.join();
        listenerThread2.join();
        
    } catch (const std::exception& e) {
        std::cerr << "主程序异常: " << e.what() << std::endl;
    }
    
    std::cout << "服务器停止运行" << std::endl;
    return 0;
}