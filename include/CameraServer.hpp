#include "CameraManager.hpp"
#include "tracker.hpp"
#include <curl/curl.h>

struct SocketInfo {
    int fd;  // Socket文件描述符
    int port;        // Socket端口号
    sockaddr_in address; // Socket地址
    int addrlen; // Socket地址长度
};

struct DataLoopInfo {
    string project_id;
    string sample_id;
    string operator_name;
    string type;
    string version;
    string url = "http://172.25.12.10/dataserver/api/samples/upload_inference";
};

// Socket Server类
class CameraServer {

private:
    int cam1_server_fd, cam2_server_fd;
    int cam1_server_port, cam2_server_port;
    int node_index;
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
    CameraServer(int node_index, int cam1_server_port, int cam2_server_port) : 
        cam1_server_port(cam1_server_port), 
        cam2_server_port(cam2_server_port), 
        node_index(node_index),
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
    bool sendDataToClient(int client_socket, const std::vector<uchar>& serializedData, int timeoutSeconds);
    bool send_visualize_data(int sockfd, const std::vector<cv::Mat>& images, const std::vector<std::string>& names);
    bool encodeImage(const cv::Mat& img, std::vector<unsigned char>& buffer, const std::string& format);
    void uploadImages(const std::vector<cv::Mat>& images, const std::vector<string> names, const DataLoopInfo& info);
    std::string generateTimestamp(int suffix);
    void write_timevals_to_binary_file(const std::string& filename, const std::vector<struct timeval>& data);
};
