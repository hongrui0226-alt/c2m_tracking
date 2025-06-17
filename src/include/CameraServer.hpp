#include "CameraManager.hpp"
#include "tracker.hpp"

struct SocketInfo {
    int fd;  // Socket文件描述符
    int port;        // Socket端口号
    sockaddr_in address; // Socket地址
    int addrlen; // Socket地址长度
};

// Socket Server类
class CameraServer {

private:
    int cam1_server_fd, cam2_server_fd;
    int cam1_server_port, cam2_server_port;
    SocketInfo cam1_socket_info, cam2_socket_info;

    atomic<bool> running;
    thread cam1_server_thread, cam2_server_thread;
    CameraManager camera_manager;
    Tracker cam1_tracker, cam2_tracker;

    mutex mtx;
    
    // 服务器主循环
    void serverLoop(SocketInfo& socket_info);
    int connectToClient(const string& clientIp, int clientPort);
    void initServerSocket(SocketInfo& socket_info);
    
public:
    CameraServer(int cam1_server_port, int cam2_server_port) : 
        cam1_server_port(cam1_server_port), 
        cam2_server_port(cam2_server_port), 
        running(false) {
            cam1_socket_info.port = cam1_server_port;
            cam2_socket_info.port = cam2_server_port;

            // Init server sockets
            initServerSocket(cam1_socket_info);
            initServerSocket(cam2_socket_info);

            // Init camera_manager
            if (!camera_manager.isInitialized()) {
                if (!camera_manager.initialize()) {
                    std::cerr << "Camera initialization failed, exiting server loop" << std::endl;
                    return;
                }
            }
        }
    
    ~CameraServer() {
        stop();
    }

    void start();
    void stop();

    vector<uchar> serializeMat(const cv::Mat& mat);
    vector<uchar> serializeImageData(const ImageData& data);
    vector<uchar> serializeMap(const unordered_map<int, ImageData>& map);
    bool getState();

};
