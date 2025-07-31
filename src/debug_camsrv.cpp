#include "CameraServer.hpp"

// 信号处理
volatile sig_atomic_t keep_running = 1;

void signal_handler(int sig) {
    keep_running = 0;
    cout << "\nReceived signal to terminate. Cleaning up..." << endl;
}

int main(int argc, char** argv) {
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

    return 0;
}
