#include "CameraManager.hpp"
#include "tracker.hpp"
#include <curl/curl.h>
#include "image_data.pb.h"
#include "httplib.h"
#include <functional>
#include <gflags/gflags.h>

using HandlerFunction = std::function<void(httplib::Response&)>;

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
    string url = "http://10.1.7.250/dataserver/api/samples/upload_inference";
};

class ThreadGuard {
public:
    explicit ThreadGuard(std::thread t) : thread_(std::move(t)) {}
    ~ThreadGuard() {
        if (thread_.joinable()) {
            thread_.join();  // 或 thread_.join();
            cout << "---------------- thread join ----------------------" << endl;
        }
    }
private:
    std::thread thread_;
};

// Socket Server类
class CameraServer {

private:
    int cam1_server_fd, cam2_server_fd;
    int cam1_server_port, cam2_server_port;
    int node_index;
    httplib::Server svr1, svr2;

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
    CameraServer(int node_index, int cam1_server_port, int cam2_server_port, string model_path) : 
        cam1_server_port(cam1_server_port), 
        cam2_server_port(cam2_server_port), 
        node_index(node_index),
        cam1_tracker(model_path), cam2_tracker(model_path),
        running(false) {

            svr1.Get("/get_data", [this, cam1_server_port](const httplib::Request&, httplib::Response& res) {
                auto t = std::thread([this, cam1_server_port, &res]() {
                    this->create_serverloop(cam1_server_port, res);
                });
                ThreadGuard guard(std::move(t));  // 自动管理线程释放
            });

            svr2.Get("/get_data", [this, cam2_server_port](const httplib::Request&, httplib::Response& res) {
                auto t = std::thread([this, cam2_server_port, &res]() {
                    this->create_serverloop(cam2_server_port, res);
                });
                ThreadGuard guard(std::move(t));  // 自动管理线程释放
            });

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
    bool sendDataToClient(int client_socket, const std::vector<uchar>& serializedData, const string& timestamp, int timeoutSeconds);
    bool send_visualize_data(int sockfd, const std::vector<cv::Mat>& images, const std::vector<std::string>& names);
    bool encodeImage(const cv::Mat& img, std::vector<unsigned char>& buffer, const std::string& format);
    void uploadImages(const std::vector<cv::Mat>& images, const std::vector<string> names, const DataLoopInfo& info);
    std::string generateTimestamp(int suffix);
    void write_timevals_to_binary_file(const std::string& filename, const std::vector<struct timeval>& data);
    std::string generateResponse(const std::unordered_map<int, ImageData>& img2save, const std::string& timestamp);
    void create_serverloop(int port, httplib::Response& res);
    std::unordered_map<int, image_data::ImageData> convertProtoMap(
        const std::unordered_map<int, ImageData>& custom_img2save);
};
