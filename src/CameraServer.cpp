#include "CameraServer.hpp"

// 服务器主循环
void CameraServer::serverLoop(SocketInfo& socket_info) {
    // 4. 创建视频写入器
    std::string output_video = "output_video.avi";
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // MP4 编码
    double fps = 106.0; // 帧率
    cv::Size frame_size(1920, 1080);
    
    cv::VideoWriter video_writer;
    if (!video_writer.open(output_video, fourcc, fps, frame_size)) {
        std::cerr << "Could not open video writer for: " << output_video << std::endl;
        return;
    }

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
            std::vector<uchar> serializedData;

            if (socket_info.port == cam1_server_port && running) {
                cout << "Starting recording on Camera 1..." << endl;
                camera_manager.startRecording(camera_manager.camera1);
                cv::Mat bgr_img;
                bool over = false;
                int frame_count = 0;

                while (cam1_tracker.is_track_over() == false && !over) {
                    if (camera_manager.getFrameQueueSize(camera_manager.camera1) > 0) {
                        Frame frame = camera_manager.getFrameQueue(camera_manager.camera1);
                        cv::cvtColor(frame.image, bgr_img, cv::COLOR_YUV2BGR_NV12); // Decode NV12 to BGR
                        cam1_tracker.track(bgr_img);
                        // video_writer.write(bgr_img);
                        cv::imwrite(cv::format("%d.png", frame_count), bgr_img);
                        // cv::imshow("MIPICAM Stream", bgr_img);
                        // int key = cv::waitKey(1);
                        cout << "remain: " << camera_manager.getFrameQueueSize(camera_manager.camera1) << endl;
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

                while (camera_manager.getFrameQueueSize(camera_manager.camera1) > 0) {
                    Frame frame = camera_manager.getFrameQueue(camera_manager.camera1);
                    cv::cvtColor(frame.image, bgr_img, cv::COLOR_YUV2BGR_NV12); // Decode NV12 to BGR
                    // video_writer.write(bgr_img);
                    cv::imwrite(cv::format("%d.png", frame_count), bgr_img);
                    cam1_tracker.track(bgr_img);
                    frame_count++;
                }

                // video_writer.release();
                std::cout << "Video saved to: " << output_video << std::endl;

                cam1_tracker.post_process();
                serializedData = serializeMap(cam1_tracker.getTrackResults());
                cam1_tracker.reset(); // Reset tracker for next session

            } else if (socket_info.port == cam2_server_port && running) {
                cout << "Starting recording on Camera 2..." << endl;
                camera_manager.startRecording(camera_manager.camera2);
                cv::Mat bgr_img;
                bool over = false;
                int frame_count = 0;

                while (cam2_tracker.is_track_over() == false && !over) {
                    if (camera_manager.getFrameQueueSize(camera_manager.camera2) > 0) {
                        Frame frame = camera_manager.getFrameQueue(camera_manager.camera2);
                        cv::cvtColor(frame.image, bgr_img, cv::COLOR_YUV2BGR_NV12); // Decode NV12 to BGR
                        cam2_tracker.track(bgr_img);

                        frame_count++;
                        if (frame_count + camera_manager.getFrameQueueSize(camera_manager.camera1) > 280) {
                            cout << "over" << endl;
                            over = true;
                        }
                    } else {
                        this_thread::sleep_for(chrono::milliseconds(1));
                    }
                }

                camera_manager.camera2.stopCapture();
                cout << "Stop recording on Camera 2..." << endl;
                
                cam2_tracker.post_process();
                serializedData = serializeMap(cam2_tracker.getTrackResults());
                cam2_tracker.reset(); // Reset tracker for next session

            }

            // send track results to client
            size_t dataSize = serializedData.size();
            
            // 发送数据大小
            if (send(client_socket, &dataSize, sizeof(dataSize), 0) != sizeof(dataSize)) {
                cerr << "Failed to send data size" << endl;
                close(client_socket);
                continue;
            }
            
            // 发送数据
            const uchar* ptr = serializedData.data();
            size_t remaining = dataSize;
            while (remaining > 0) {
                ssize_t sent = send(client_socket, ptr, remaining, 0);
                if (sent <= 0) {
                    cerr << "Failed to send data" << endl;
                    break;
                }
                ptr += sent;
                remaining -= sent;
            }
        }
        close(client_socket);
        cout << "close client socket" << endl;
    }

    cout << "Stop Running" << endl;
    
    close(socket_info.fd);
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
    cam1_server_thread = thread(&CameraServer::serverLoop, this, ref(cam1_socket_info));
    cam2_server_thread = thread(&CameraServer::serverLoop, this, ref(cam2_socket_info));

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

    // 使用shutdown而非close，确保accept()立即返回
    if (cam1_socket_info.fd >= 0) {
        shutdown(cam1_socket_info.fd, SHUT_RDWR);  // 中断所有I/O操作
        close(cam1_socket_info.fd);
        cam1_socket_info.fd = -1;
    }

    if (cam2_socket_info.fd >= 0) {
        shutdown(cam2_socket_info.fd, SHUT_RDWR);  // 中断所有I/O操作
        close(cam2_socket_info.fd);
        cam2_socket_info.fd = -1;
    }
    
    if (cam1_server_thread.joinable()) {
        cam1_server_thread.join();
    }

    if (cam2_server_thread.joinable()) {
        cam2_server_thread.join();
    }
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
    // // 注册信号处理
    // signal(SIGINT, signal_handler);
    // signal(SIGTERM, signal_handler);
    
    // // 创建并启动Socket服务器
    // int cam1_port = 8085;
    // int cam2_port = 8086;
    // CameraServer server(cam1_port, cam2_port);
    // server.start();

    // // 主循环等待信号
    // while (keep_running) {
    //     this_thread::sleep_for(chrono::seconds(1));
    // }

    // // 停止服务器
    // server.stop();
    
    // cout << "Server stopped. Exiting." << endl;

    Tracker tracker;

    string video_path = "/home/sunrise/qimeng3/dataset/videos/output_2.avi";
    tracker.track_video(video_path);

    return 0;
}
