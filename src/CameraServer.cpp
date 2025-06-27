#include "CameraServer.hpp"

// 服务器主循环
void CameraServer::create_serverloop(int port, httplib::Response& res) {

    vector<uchar> serializedData;
    vector<struct timeval> timevals;
    string timestamp;

    if (port == cam1_server_port && running) {
        cout << "Starting recording on Camera 1..." << endl;
        camera_manager.startRecording(camera_manager.camera1);
        cv::Mat bgr_img;
        bool over = false;
        int frame_count = 0;

        while (cam1_tracker.is_track_over() == false && !over) {
            if (camera_manager.getFrameQueueSize(camera_manager.camera1) > 0) {
                Frame frame = camera_manager.getFrameQueue(camera_manager.camera1);
                timevals.push_back(frame.timestamp);
                cv::Mat yuv_copy = frame.image.clone(); // 强制深拷贝YUV数据
                cv::Mat bgr_img;
                cv::cvtColor(yuv_copy, bgr_img, cv::COLOR_YUV2BGR_NV12);
                cv::Mat bgr_copy = bgr_img.clone(); // 再拷贝一次BGR数据
                
                // 使用深拷贝的数据进行跟踪和保存
                cam1_tracker.track(bgr_copy, true);

                frame_count++;
                if (frame_count + camera_manager.getFrameQueueSize(camera_manager.camera1) > 280) {
                    cout << "over" << endl;
                    over = true;
                }
            } else {
                this_thread::sleep_for(chrono::milliseconds(1));
            }
        }

        camera_manager.camera1.stopCapture();
        cout << "Stop recording on Camera 1..." << endl;

        while ((camera_manager.getFrameQueueSize(camera_manager.camera1) > 0) && (cam1_tracker.is_track_over() == false)) {
            Frame frame = camera_manager.getFrameQueue(camera_manager.camera1);
            timevals.push_back(frame.timestamp);
            cv::Mat yuv_copy = frame.image.clone();
            cv::cvtColor(yuv_copy, bgr_img, cv::COLOR_YUV2BGR_NV12); // Decode NV12 to BGR
            cv::Mat bgr_copy = bgr_img.clone(); // 再拷贝一次BGR数据
            // video_writer.write(bgr_img);
            // cv::imwrite(cv::format("%d.png", frame_count), bgr_img);
            cam1_tracker.track(bgr_copy, true);
            frame_count++;
        }

        cam1_tracker.post_process();

        // 回传上位机
        timestamp = generateTimestamp(node_index*2 - 1);
        std::string response2host = generateResponse(cam1_tracker.getTrackResults(), timestamp);
        res.set_content(response2host, "application/octet-stream");

        // 上传数据闭环
        DataLoopInfo info;
        info.project_id = "231";
        info.sample_id = timestamp.c_str();
        info.operator_name = "discover";
        info.type = "NonSequential";
        info.version = "1";
        uploadImages(cam1_tracker.getVisImages(), cam1_tracker.getVisNames(), info);

        cam1_tracker.reset(); // Reset tracker for next session
        camera_manager.camera1.frame_queue_.clear(); // Clear Camera Queue to reset

    } else if (port == cam2_server_port && running) {
        cout << "Starting recording on Camera 2..." << endl;
        camera_manager.startRecording(camera_manager.camera2);
        cv::Mat bgr_img;
        bool over = false;
        int frame_count = 0;

        while (cam2_tracker.is_track_over() == false && !over) {
            if (camera_manager.getFrameQueueSize(camera_manager.camera2) > 0) {
                Frame frame = camera_manager.getFrameQueue(camera_manager.camera2);
                timevals.push_back(frame.timestamp);
                cv::Mat yuv_copy = frame.image.clone(); // 强制深拷贝YUV数据
                cv::Mat bgr_img;
                cv::cvtColor(yuv_copy, bgr_img, cv::COLOR_YUV2BGR_NV12);
                cv::Mat bgr_copy = bgr_img.clone(); // 再拷贝一次BGR数据
                
                // 使用深拷贝的数据进行跟踪和保存
                cam2_tracker.track(bgr_copy, true);
                frame_count++;
                if (frame_count + camera_manager.getFrameQueueSize(camera_manager.camera2) > 280) {
                    cout << "over" << endl;
                    over = true;
                }
            } else {
                this_thread::sleep_for(chrono::milliseconds(1));
            }
        }

        camera_manager.camera2.stopCapture();
        cout << "Stop recording on Camera 2..." << endl;

        while ((camera_manager.getFrameQueueSize(camera_manager.camera2) > 0) && (cam2_tracker.is_track_over() == false)) {
            Frame frame = camera_manager.getFrameQueue(camera_manager.camera2);
            timevals.push_back(frame.timestamp);
            cv::Mat yuv_copy = frame.image.clone();
            cv::cvtColor(yuv_copy, bgr_img, cv::COLOR_YUV2BGR_NV12); // Decode NV12 to BGR
            cv::Mat bgr_copy = bgr_img.clone(); // 再拷贝一次BGR数据
            cam2_tracker.track(bgr_copy, true);
            frame_count++;
        }

        cam2_tracker.post_process();

        // 回传上位机
        timestamp = generateTimestamp(node_index*2);
        std::string response2host = generateResponse(cam2_tracker.getTrackResults(), timestamp);
        res.set_content(response2host, "application/octet-stream");

        // 上传数据闭环
        DataLoopInfo info;
        info.project_id = "231";
        info.sample_id = timestamp.c_str();
        info.operator_name = "discover";
        info.type = "NonSequential";
        info.version = "1";
        uploadImages(cam2_tracker.getVisImages(), cam2_tracker.getVisNames(), info);
        // write_timevals_to_binary_file(cv::format("%d_output.txt", node_index*2), timevals);

        cam2_tracker.reset(); // Reset tracker for next session
        camera_manager.camera2.frame_queue_.clear(); // Clear Camera Queue to reset

    }

}

void CameraServer::serverLoop(SocketInfo& socket_info) {

    // Running server logic
    while (running) {
        int client_socket = accept(socket_info.fd, (struct sockaddr *)&socket_info.address, (socklen_t*)&socket_info.addrlen);
        if (client_socket < 0) {
            if (running) cerr << "Accept error" << endl;
            continue;
        }

        // start capture
        char buffer[1024];
        int valread = read(client_socket, buffer, 1024);
        if (valread > 0) {
            vector<uchar> serializedData;
            vector<struct timeval> timevals;
            string timestamp;

            if (socket_info.port == cam1_server_port && running) {
                cout << "Starting recording on Camera 1..." << endl;
                camera_manager.startRecording(camera_manager.camera1);
                cv::Mat bgr_img;
                bool over = false;
                int frame_count = 0;

                while (cam1_tracker.is_track_over() == false && !over) {
                    if (camera_manager.getFrameQueueSize(camera_manager.camera1) > 0) {
                        Frame frame = camera_manager.getFrameQueue(camera_manager.camera1);
                        timevals.push_back(frame.timestamp);
                        cv::Mat yuv_copy = frame.image.clone(); // 强制深拷贝YUV数据
                        cv::Mat bgr_img;
                        cv::cvtColor(yuv_copy, bgr_img, cv::COLOR_YUV2BGR_NV12);
                        cv::Mat bgr_copy = bgr_img.clone(); // 再拷贝一次BGR数据
                        
                        // 使用深拷贝的数据进行跟踪和保存
                        cam1_tracker.track(bgr_copy, true);

                        frame_count++;
                        if (frame_count + camera_manager.getFrameQueueSize(camera_manager.camera1) > 280) {
                            cout << "over" << endl;
                            over = true;
                        }
                    } else {
                        this_thread::sleep_for(chrono::milliseconds(1));
                    }
                }

                camera_manager.camera1.stopCapture();
                cout << "Stop recording on Camera 1..." << endl;

                while ((camera_manager.getFrameQueueSize(camera_manager.camera1) > 0) && (cam1_tracker.is_track_over() == false)) {
                    Frame frame = camera_manager.getFrameQueue(camera_manager.camera1);
                    timevals.push_back(frame.timestamp);
                    cv::Mat yuv_copy = frame.image.clone();
                    cv::cvtColor(yuv_copy, bgr_img, cv::COLOR_YUV2BGR_NV12); // Decode NV12 to BGR
                    cv::Mat bgr_copy = bgr_img.clone(); // 再拷贝一次BGR数据
                    // video_writer.write(bgr_img);
                    // cv::imwrite(cv::format("%d.png", frame_count), bgr_img);
                    cam1_tracker.track(bgr_copy, true);
                    frame_count++;
                }

                cam1_tracker.post_process();

                // 回传上位机
                serializedData = serializeMap(cam1_tracker.getTrackResults());
                // send_visualize_data(client_socket, cam1_tracker.getVisImages(), cam1_tracker.getVisNames());

                // 上传数据闭环
                DataLoopInfo info;
                info.project_id = "231";
                info.sample_id = timestamp.c_str();
                info.operator_name = "discover";
                info.type = "NonSequential";
                info.version = "1";
                uploadImages(cam1_tracker.getVisImages(), cam1_tracker.getVisNames(), info);
                write_timevals_to_binary_file(cv::format("%d_output.txt", node_index*2-1), timevals);

                cam1_tracker.reset(); // Reset tracker for next session
                camera_manager.camera1.frame_queue_.clear(); // Clear Camera Queue to reset

                // cam1_tracker.setOutputFolder("/home/sunrise/qimeng3/dataset/tracking_images/debug");
                // cam1_tracker.save_results(true);

            } else if (socket_info.port == cam2_server_port && running) {
                cout << "Starting recording on Camera 2..." << endl;
                camera_manager.startRecording(camera_manager.camera2);
                cv::Mat bgr_img;
                bool over = false;
                int frame_count = 0;

                while (cam2_tracker.is_track_over() == false && !over) {
                    if (camera_manager.getFrameQueueSize(camera_manager.camera2) > 0) {
                        Frame frame = camera_manager.getFrameQueue(camera_manager.camera2);
                        timevals.push_back(frame.timestamp);
                        cv::Mat yuv_copy = frame.image.clone(); // 强制深拷贝YUV数据
                        cv::Mat bgr_img;
                        cv::cvtColor(yuv_copy, bgr_img, cv::COLOR_YUV2BGR_NV12);
                        cv::Mat bgr_copy = bgr_img.clone(); // 再拷贝一次BGR数据
                        
                        // 使用深拷贝的数据进行跟踪和保存
                        cam2_tracker.track(bgr_copy, true);
                        frame_count++;
                        if (frame_count + camera_manager.getFrameQueueSize(camera_manager.camera2) > 280) {
                            cout << "over" << endl;
                            over = true;
                        }
                    } else {
                        this_thread::sleep_for(chrono::milliseconds(1));
                    }
                }

                camera_manager.camera2.stopCapture();
                cout << "Stop recording on Camera 2..." << endl;

                while ((camera_manager.getFrameQueueSize(camera_manager.camera2) > 0) && (cam2_tracker.is_track_over() == false)) {
                    Frame frame = camera_manager.getFrameQueue(camera_manager.camera2);
                    timevals.push_back(frame.timestamp);
                    cv::Mat yuv_copy = frame.image.clone();
                    cv::cvtColor(yuv_copy, bgr_img, cv::COLOR_YUV2BGR_NV12); // Decode NV12 to BGR
                    cv::Mat bgr_copy = bgr_img.clone(); // 再拷贝一次BGR数据
                    cam2_tracker.track(bgr_copy, true);
                    frame_count++;
                }

                // video_writer.release();
                // std::cout << "Video saved to: " << output_video << std::endl;

                cam2_tracker.post_process();
                serializedData = serializeMap(cam2_tracker.getTrackResults());
                // 上传图像数据
                DataLoopInfo info;
                timestamp = generateTimestamp(node_index*2);
                info.project_id = "231";
                info.sample_id = timestamp.c_str();
                info.operator_name = "discover";
                info.type = "NonSequential";
                info.version = "1";
                uploadImages(cam2_tracker.getVisImages(), cam2_tracker.getVisNames(), info);
                write_timevals_to_binary_file(cv::format("%d_output.txt", node_index*2), timevals);

                cam2_tracker.reset(); // Reset tracker for next session
                camera_manager.camera2.frame_queue_.clear(); // Clear Camera Queue to reset

            }

            // send track results to client
            size_t dataSize = serializedData.size();

            try {
                // send data
                if (!sendDataToClient(client_socket, serializedData, timestamp, 5)) {
                    std::cerr << "数据发送失败，关闭连接" << std::endl;
                    close(client_socket);
                    return;
                }
                
                std::cout << "数据发送成功，大小: " << serializedData.size() << " 字节" << std::endl;
                this_thread::sleep_for(chrono::milliseconds(2000));
                close(client_socket);
            } catch (const std::exception& e) {
                std::cerr << "处理客户端请求时发生异常: " << e.what() << std::endl;
                close(client_socket);
            }
            
        }
        close(client_socket);
        cout << socket_info.port <<" close client socket" << endl;
    }

    cout << "Stop Running" << endl;
    
    close(socket_info.fd);
}

bool CameraServer::sendDataToClient(int client_socket, 
                                    const std::vector<uchar>& serializedData, 
                                    const std::string& timestamp,
                                    int timeoutSeconds = 30) {
    size_t dataSize = serializedData.size();
    
    // 设置套接字发送超时
    struct timeval tv;
    tv.tv_sec = timeoutSeconds;
    tv.tv_usec = 0;
    setsockopt(client_socket, SOL_SOCKET, SO_SNDTIMEO, (char*)&tv, sizeof(tv));
    
    // 发送时间戳
    if (send(client_socket, timestamp.c_str(), timestamp.size(), 0) != timestamp.size()) {
        int error = errno;
        if (error == EAGAIN || error == EWOULDBLOCK) {
            std::cerr << "发送时间戳超时" << std::endl;
        } else if (error == EPIPE || error == ECONNRESET) {
            std::cerr << "客户端已关闭连接" << std::endl;
        }
    }
    
    // 发送数据大小
    if (send(client_socket, &dataSize, sizeof(dataSize), 0) != sizeof(dataSize)) {
        int error = errno;
        if (error == EAGAIN || error == EWOULDBLOCK) {
            std::cerr << "发送数据大小超时" << std::endl;
        } else if (error == EPIPE || error == ECONNRESET) {
            std::cerr << "客户端已关闭连接" << std::endl;
        } else {
            std::cerr << "发送数据大小失败: " << strerror(error) << std::endl;
        }
        return false;
    }
    
    // 发送实际数据
    const uchar* ptr = serializedData.data();
    size_t remaining = dataSize;
    
    while (remaining > 0) {
        ssize_t sent = send(client_socket, ptr, remaining, 0);
        if (sent < 0) {
            int error = errno;
            if (error == EAGAIN || error == EWOULDBLOCK) {
                std::cerr << "发送数据超时" << std::endl;
                break;
            } else if (error == EPIPE || error == ECONNRESET) {
                std::cerr << "客户端已关闭连接" << std::endl;
                break;
            } else {
                std::cerr << "发送数据失败: " << strerror(error) << std::endl;
                break;
            }
        } else if (sent == 0) {
            // 对方已关闭连接
            std::cerr << "客户端已关闭连接" << std::endl;
            break;
        }
        
        ptr += sent;
        remaining -= sent;
    }
    
    return remaining == 0;
}

int CameraServer::connectToClient(const string& clientIp, int clientPort) {
    // 创建socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        cerr << "创建Port为 " << clientPort << " 的socket失败" << endl;
        return -1;
    }

    // 设置客户端地址
    sockaddr_in clientAddr;
    clientAddr.sin_family = AF_INET;
    clientAddr.sin_port = htons(clientPort);
    inet_pton(AF_INET, clientIp.c_str(), &clientAddr.sin_addr);

    // 连接到客户端
    if (connect(sock, (sockaddr*)&clientAddr, sizeof(clientAddr)) < 0) {
        cerr << "无法连接到客户端端口" << clientPort << endl;
        close(sock);
        return -1;
    }

    return sock;
}

void CameraServer::initServerSocket(SocketInfo& socket_info) {
    // Init socket server 
    int opt = 1;
    socket_info.addrlen = sizeof(socket_info.address);
    if ((socket_info.fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        cerr << "Socket creation error" << endl;
        return;
    }
    if (setsockopt(socket_info.fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        cerr << "Setsockopt error" << endl;
        return;
    }
    socket_info.address.sin_family = AF_INET;
    socket_info.address.sin_addr.s_addr = INADDR_ANY;
    socket_info.address.sin_port = htons(socket_info.port);
    if (bind(socket_info.fd, (struct sockaddr *)&socket_info.address, sizeof(socket_info.address)) < 0) {
        cerr << "Bind failed" << endl;
        return;
    }
    if (listen(socket_info.fd, 3) < 0) {
        cerr << "Listen error" << endl;
        return;
    }
    cout << "Server listening on server_port " << socket_info.port << endl;
}

void CameraServer::start() {
    if (running) return;

    running = true;
    // cam1_server_thread = thread(&CameraServer::serverLoop, this, ref(cam1_socket_info));
    // cam2_server_thread = thread(&CameraServer::serverLoop, this, ref(cam2_socket_info));
    // 使用线程启动各个服务器
    cam1_server_thread = thread([this]() {
        std::cout << "Server 1 started at http://localhost:8085\n";
        svr1.listen("localhost", 8085);
    });

    cam2_server_thread = thread([this]() {
        std::cout << "Server 2 started at http://localhost:8086\n";
        svr2.listen("localhost", 8086);
    });

    this_thread::sleep_for(chrono::nanoseconds(1000));  // 等待服务器启动
}

void CameraServer::stop() {
    {
        lock_guard<mutex> lock(mtx);
        if (!running) return;
    }
    {   
        lock_guard<mutex> lock(mtx);
        running = false;
    }

    svr1.stop();
    svr2.stop();
}
    
std::vector<uchar> CameraServer::serializeMat(const cv::Mat& mat) {
    std::vector<uchar> buffer;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
    cv::imencode(".jpg", mat, buffer, params);
    return buffer;
}

// 序列化 ImageData
std::vector<uchar> CameraServer::serializeImageData(const ImageData& data) {
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
std::vector<uchar> CameraServer::serializeMap(const std::unordered_map<int, ImageData>& map) {
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

bool CameraServer::send_visualize_data(int sockfd, const std::vector<cv::Mat>& images, const std::vector<std::string>& names) {
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

        // 2.1 发送图像宽度、高度和通道数
        uint32_t width = image.cols;
        uint32_t height = image.rows;
        uint32_t channels = image.channels();
        
        send(sockfd, &width, sizeof(width), 0);
        send(sockfd, &height, sizeof(height), 0);
        send(sockfd, &channels, sizeof(channels), 0);

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

bool CameraServer::encodeImage(const cv::Mat& img, std::vector<unsigned char>& buffer, const std::string& format = ".jpg") {
    std::vector<int> params;
    if (format == ".jpg") {
        params = {cv::IMWRITE_JPEG_QUALITY, 95};
    } else if (format == ".png") {
        params = {cv::IMWRITE_PNG_COMPRESSION, 9};
    } else {
        return false;
    }
    return cv::imencode(format, img, buffer, params);
}

void CameraServer::uploadImages(const std::vector<cv::Mat>& images, const std::vector<string> names, 
                                const DataLoopInfo& info) {
    if (images.empty()) {
        cout << "No images to upload" << endl;
        return;
    } else {
        cout << "Uploading Images Size: " << images.size() << endl;
    }
    
    CURL* curl = curl_easy_init();
    if (!curl) return;

    curl_mime* mime = curl_mime_init(curl);

    for (size_t i = 0; i < images.size(); ++i) {
        std::vector<unsigned char> buffer;
        if (!encodeImage(images[i], buffer, ".jpg")) {
            std::cerr << "编码图像失败: " << i << std::endl;
            continue;
        } 

        curl_mimepart* part = curl_mime_addpart(mime);
        curl_mime_name(part, "images");

        std::string filename = names[i];
        curl_mime_filename(part, filename.c_str());
        curl_mime_type(part, "image/jpeg");
        curl_mime_data(part, reinterpret_cast<const char*>(buffer.data()), buffer.size());
    }

    {
        curl_mimepart* part = curl_mime_addpart(mime);
        curl_mime_name(part, "project_id");
        curl_mime_data(part, info.project_id.c_str(), CURL_ZERO_TERMINATED); //"231"
    }

    {
        curl_mimepart* part = curl_mime_addpart(mime);
        curl_mime_name(part, "sample_id");
        curl_mime_data(part, info.sample_id.c_str(), CURL_ZERO_TERMINATED); // "20250620"
    }

    {
        curl_mimepart* part = curl_mime_addpart(mime);
        curl_mime_name(part, "operator");
        curl_mime_data(part, info.operator_name.c_str(), CURL_ZERO_TERMINATED); // "X5"
    }

    {
        curl_mimepart* part = curl_mime_addpart(mime);
        curl_mime_name(part, "type");
        curl_mime_data(part, info.type.c_str(), CURL_ZERO_TERMINATED); // "NonSequential"
    }

    {
        curl_mimepart* part = curl_mime_addpart(mime);
        curl_mime_name(part, "version");
        curl_mime_data(part, info.version.c_str(), CURL_ZERO_TERMINATED); // "1"
    }

    curl_easy_setopt(curl, CURLOPT_URL, info.url.c_str());
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "请求失败: " << curl_easy_strerror(res) << std::endl;
    }

    curl_mime_free(mime);
    curl_easy_cleanup(curl);
}

std::string CameraServer::generateTimestamp(int suffix = 6) {
    // 获取当前时间
    std::time_t now = std::time(nullptr);
    std::tm* local_time = std::localtime(&now);

    // 缓冲区用于存储格式化后的时间字符串
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", local_time);  // 格式为 YYYYMMDDHHMMSS

    // 拼接后缀
    std::string timestamp = std::string(buffer) + "_" + std::to_string(suffix);
    return timestamp;
}

void CameraServer::write_timevals_to_binary_file(const std::string& filename, 
                                                const std::vector<struct timeval>& data) {
    FILE* file = fopen(filename.c_str(), "w");
    if (!file) {
        perror("Failed to open file for writing");
        return;
    }

    for (const auto& tv : data) {
        struct tm* tm_utc = gmtime(&tv.tv_sec);  // 转换为 UTC 时间
        char datetime[64];
        strftime(datetime, sizeof(datetime), "%Y-%m-%dT%H:%M:%S", tm_utc);  // 格式化日期时间
        fprintf(file, "%s.%06ld\n", datetime, tv.tv_usec);  // 写入 ISO 8601 格式时间戳
    }

    fclose(file);
}

std::string CameraServer::generateResponse(
    const std::unordered_map<int, ImageData>& img2save, 
    const std::string& timestamp) 
{
    image_data::Response response;

    // 将 img2save 转换为 Protobuf 的 Map 类型
    std::unordered_map<int, image_data::ImageData> img2saveProtobuf = convertProtoMap(img2save);

    // 将 img2save 中的数据复制到 response 的 image_map 中
    for (const auto& [key, imageData] : img2saveProtobuf) {
    (*response.mutable_image_map())[key] = imageData;
    }

    response.set_timestamp(timestamp.c_str());

    // 序列化为二进制
    std::string serialized;
    response.SerializeToString(&serialized);
    return serialized;
}

std::unordered_map<int, image_data::ImageData> CameraServer::convertProtoMap(
    const std::unordered_map<int, ImageData>& custom_img2save) {
    
    std::unordered_map<int, image_data::ImageData> proto_img2save;
    
    for (const auto& [id, custom_data] : custom_img2save) {
        image_data::ImageData proto_data;
        // 检查图像和文件名数量是否一致
        if (custom_data.images.size() != custom_data.names.size()) {
            std::cerr << "Error: Image and name counts mismatch for ID " << id << std::endl;
            continue;
        }
        
        for (size_t i = 0; i < custom_data.images.size(); ++i) {
            // 为每个图像创建一个新的Image消息
            auto* proto_image = proto_data.add_images();
            
            // 将cv::Mat转换为字节数组
            std::vector<uchar> buffer;
            cv::imencode(".jpg", custom_data.images[i], buffer);
            
            // 设置图像数据和名称
            proto_image->set_image_data(buffer.data(), buffer.size());
            proto_image->set_name(custom_data.names[i]);
        }
        
        // 将转换后的数据添加到新的map中
        proto_img2save[id] = proto_data;
    }
    
    return proto_img2save;
}

bool CameraServer::getState() {
    return running;
}

// 信号处理
volatile sig_atomic_t keep_running = 1;

void signal_handler(int sig) {
    keep_running = 0;
    cout << "\nReceived signal to terminate. Cleaning up..." << endl;
}

int main() {
    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // 创建并启动Socket服务器
    int cam1_port = 8085;
    int cam2_port = 8086;
    int node_index = 1; // 节点索引
    CameraServer server(node_index, cam1_port, cam2_port);
    server.start();

    // 主循环等待信号
    while (keep_running) {
        this_thread::sleep_for(chrono::seconds(1));
    }

    // 停止服务器
    server.stop();
    
    cout << "Server stopped. Exiting." << endl;

    // Tracker tracker;

    // string video_path = "/home/sunrise/qimeng3/dataset/videos/output_2.avi";
    // tracker.track_video(video_path);

    return 0;
}
