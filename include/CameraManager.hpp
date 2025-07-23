#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "mipicam.hpp"

using namespace std;

// Init params of video writer
const int FRAME_WIDTH = 1920;
const int FRAME_HEIGHT = 1080;
const double FPS = 120;
const int TARGET_FRAMES = 350;

// Define Class 
class CameraManager {
private:
    mutex mtx;
    
    DeviceConfig config;
    bool is_initialized;
    atomic<bool> is_recording;
    
public:
    MIPICAM camera1, camera2;

    CameraManager() : is_initialized(false), is_recording(false) {
        config.sensor_index = 28;
	    config.exposure_seconds = 125e-6f;
    }

    bool isInitialized() const {
        return is_initialized;
    }

    bool initialize() {
        if (is_initialized) return true;
        
        if (camera1.init(config) != 0) {
            cerr << "Camera 1 initialization failed" << endl;
            return false;
        }

        if (camera2.init(config) != 0) {
            cerr << "Camera 2 initialization failed" << endl;
            return false;
        }
        
        is_initialized = true;
        return true;
    }

    bool sendImage(int socket, const cv::Mat& image) {
        // 1. 发送图像尺寸和通道数
        int width = image.cols;
        int height = image.rows;
        int channels = image.channels();
        
        if (send(socket, &width, sizeof(int), 0) != sizeof(int) ||
            send(socket, &height, sizeof(int), 0) != sizeof(int) ||
            send(socket, &channels, sizeof(int), 0) != sizeof(int)) {
            return false;
        }
        
        // 2. 发送图像数据
        size_t dataSize = image.total() * image.elemSize();
        if (send(socket, image.data, dataSize, 0) != static_cast<int>(dataSize)) {
            return false;
        }
        
        return true;
    }
    
    bool startRecording(MIPICAM& camera) {
        // if (cam_index != 1 && cam_index != 2) {
        //     cerr << "Invalid camera index" << endl;
        //     return false;
        // }
        // MIPICAM& camera = (cam_index == 1) ? camera1 : camera2;

        if (camera.startCapture() != 0 ) {
            cerr << "Failed to start capture" << endl;
            return false;
        }

        {
            lock_guard<mutex> lock(mtx);
            is_recording = true;
        }
        
        // // capture TARGET_FRAMES 
        // while ((getFrameQueueSize(camera) < TARGET_FRAMES) && is_recording) {
        //     std::this_thread::sleep_for(std::chrono::milliseconds(2));
        // }
        // camera.stopCapture();
        // cout << "Stopping capture... " << endl;
        
        // // save TARGET_FRAMES
        // while(getFrameQueueSize(camera) > 0) {
        //     Frame frame = getFrameQueue(camera);
        //     sendImage(socket, frame.image);
        // }
        
        return true;
    }
    
    void stopRecording() {
        is_recording = false;
    }

    Frame getFrameQueue(MIPICAM& camera) {
        Frame frame;
        if (!camera.frame_queue_.pop(frame)) {
            std::cerr << "Failed to pop from camera queue !!!" << std::endl;
        }
        return frame;
    }

    size_t getFrameQueueSize(MIPICAM& camera) {
        return camera.frame_queue_.size();
    }

    void clearFrameQueue(MIPICAM& camera) {
        camera.frame_queue_.clear();
    }

    void startCapturing(MIPICAM& camera) {
        if (camera.startCapture() != 0) {
            cerr << "Failed to start capture" << endl;
            return;
        }
        cout << "Camera started capturing." << endl;
        
    }
    
    ~CameraManager() {
        if (is_initialized) {
            camera1.deinit();
            camera2.deinit();
        }
    }
};