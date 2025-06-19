// server.cpp
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>

// 发送图像和名称的函数
bool send_data(int sockfd, const std::vector<cv::Mat>& images, const std::vector<std::string>& names) {
    // 检查图像和名称数量是否匹配
    if (images.size() != names.size()) {
        std::cerr << "Error: Number of images and names do not match!" << std::endl;
        return false;
    }

    // 1. 发送图像数量
    uint32_t count = images.size();
    if (send(sockfd, &count, sizeof(count), 0) != sizeof(count)) {
        std::cerr << "Error sending image count!" << std::endl;
        return false;
    }

    // 2. 循环发送每个图像和对应的名称
    for (size_t i = 0; i < count; ++i) {
        const cv::Mat& image = images[i];
        const std::string& name = names[i];

        // // 2.1 发送图像宽度、高度和通道数
        // uint32_t width = image.cols;
        // uint32_t height = image.rows;
        // uint32_t channels = image.channels();
        
        // send(sockfd, &width, sizeof(width), 0);
        // send(sockfd, &height, sizeof(height), 0);
        // send(sockfd, &channels, sizeof(channels), 0);

        // 2.2 发送图像数据大小
        size_t image_size = image.total() * image.elemSize();
        send(sockfd, &image_size, sizeof(image_size), 0);

        // 2.3 发送图像数据
        if (send(sockfd, image.data, image_size, 0) != static_cast<ssize_t>(image_size)) {
            std::cerr << "Error sending image data!" << std::endl;
            return false;
        }

        // 2.4 发送图像名称长度
        uint32_t name_length = name.length();
        send(sockfd, &name_length, sizeof(name_length), 0);

        // 2.5 发送图像名称
        if (send(sockfd, name.c_str(), name_length, 0) != static_cast<ssize_t>(name_length)) {
            std::cerr << "Error sending image name!" << std::endl;
            return false;
        }
    }

    return true;
}

int main() {
    // 创建示例数据
    std::vector<cv::Mat> visualize_images;
    std::vector<std::string> visualize_names;

    // 生成3张测试图像
    for (int i = 0; i < 80; ++i) {
        cv::Mat image(480, 640, CV_8UC3);
        // 用不同颜色填充每张图像
        image.setTo(cv::Scalar(i * 50, 100 + i * 30, 200 - i * 40));
        visualize_images.push_back(image);
        visualize_names.push_back("image_" + std::to_string(i));
    }

    // 创建套接字
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        std::cerr << "Error creating socket!" << std::endl;
        return -1;
    }

    // 设置套接字选项
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == -1) {
        std::cerr << "Error setting socket options!" << std::endl;
        close(server_fd);
        return -1;
    }

    // 绑定地址和端口
    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8888);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) == -1) {
        std::cerr << "Error binding socket!" << std::endl;
        close(server_fd);
        return -1;
    }

    // 监听连接
    if (listen(server_fd, 1) == -1) {
        std::cerr << "Error listening!" << std::endl;
        close(server_fd);
        return -1;
    }

    std::cout << "Server listening on port 8888..." << std::endl;

    // 接受客户端连接
    socklen_t addrlen = sizeof(address);
    int client_fd = accept(server_fd, (struct sockaddr *)&address, &addrlen);
    if (client_fd == -1) {
        std::cerr << "Error accepting connection!" << std::endl;
        close(server_fd);
        return -1;
    }

    std::cout << "Client connected!" << std::endl;

    // 发送数据
    if (send_data(client_fd, visualize_images, visualize_names)) {
        std::cout << "Data sent successfully!" << std::endl;
    } else {
        std::cerr << "Failed to send data!" << std::endl;
    }

    // 关闭连接
    close(client_fd);
    close(server_fd);

    return 0;
}
