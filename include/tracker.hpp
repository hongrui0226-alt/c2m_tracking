#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <numeric>
#include <filesystem>
#include <unordered_set>
#include <algorithm>
#include <regex>
#include <ranges>
#include <optional>
#include <utility> // 结构化绑定需要
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <charconv>

#include "hb_infer.hpp"

using namespace std;

namespace fs = std::filesystem;  // 命名空间别名
using TrackXY = std::array<int, 2>;
// 定义外层键为 std::string，内层为 std::unordered_map<int, double>
using InnerMap = std::unordered_map<int, double>;
using OuterMap = std::unordered_map<std::string, InnerMap>;
struct ImageData {
    std::vector<cv::Mat> images;      // 存储图像
    std::vector<std::string> names;   // 存储文件名
};

// KMeans类的简单实现（替代scikit-learn）
class KMeans {
private:
    int n_clusters;
    int max_iter;
    unsigned int random_state;
    
public:
    KMeans(int clusters = 2, int iterations = 100, unsigned int seed = 42)
        : n_clusters(clusters), max_iter(iterations), random_state(seed) {}
    
    // 简化的KMeans拟合方法
    std::vector<cv::Point2f> fit(const std::vector<cv::Point2f>& points) {
        // 这里仅为示例，实际实现需要完整的KMeans算法
        std::vector<cv::Point2f> centers(n_clusters);
        // 简化：返回前n_clusters个点作为中心
        for (int i = 0; i < n_clusters && i < points.size(); ++i) {
            centers[i] = points[i];
        }
        return centers;
    }
};

struct FileInfo {
    string name;
    int frame_id;
    bool is_merged = false;
    bool is_separated = false;
    TrackXY xy;
    TrackXY motion;
};

bool endsWith(const std::string& str, const std::string& suffix);
FileInfo parse_file_info(const string& name);

struct Match {
    int a_idx;
    int b_idx;
    int other_idx;
    double distance;
};


struct MatchResult {
    std::vector<Match> matches;
    std::vector<size_t> unmatched_a;
    std::vector<size_t> unmatched_b;
};

struct TrackedData {
    int cx;
    int cy;
    float areas;
    cv::Mat image;  // 224x224x3的图像
    
    TrackedData() : cx(0), cy(0), areas(0.0f), image(224, 224, CV_8UC3) {}
};

struct TrackInfo
{
    TrackXY xy;
    TrackXY motion;
    int id = -1;
    int state = -1;
    float areas;
    cv::Mat current_canvas;
};

class Tracker {
private:
    // ROI相关参数
    int roi_x1, roi_y1, roi_x2, roi_y2;
    int image_w, image_h;
    float scale_x, scale_y;
    int roix1_orig, roiy1_orig, roix2_orig, roiy2_orig;
    int width_orig, length_orig;
    int right_width;
    float check_length = 80.0f;
    
    // Track 参数
    TrackXY init_motion = {-30, 0};  // 初始运动向量
    const int dis_threshold = 40;
    const int merge_dis_threshold = 80;
    const int separate_dis_threshold = 80;
    const int vaild_threshold = 500;
    const int head_threshold = 440;
    const int head_buffer_threshold = 400;
    const int tail_threshold = 80;
    const int tail_buffer_threshold = 105;
    const int vaild_area = 35;
    const int hsv_separation = 255;  // HSV色调分离阈值  TODO：待定
    const int tolerance_undetected_num = 100;
    const int binary_threshold = 15;
    const int MIN_CONTOUR_AREA = 80; // 需根据实际定义
    const int SOBEL_THRESH = 50;       // 需根据实际定义

    // Track Flag
    bool detected_flag = false;
    bool track_over_flag = false;
    bool first_frame_flag = true;
    int invaild_num = 15;
    int undetected_frame_count = 0;

    // Track 数据
    std::vector<cv::Mat> visualize_images, cv_debug_images;
    std::vector<string> visualize_names, cv_debug_names;
    std::vector<TrackInfo> last_frame_info, current_frame_info, untracked_info_dict;
    std::vector<TrackXY> last_frame_xy, current_frame_xy, untracked_info_xy;
    std::vector<std::string> frame_logs;
    std::unordered_map<int, ImageData> img2save;
    cv::Mat blank_orig;
    cv::Mat blank_resize, blank_rect;
    cv::Mat blank_frame_tail_area, blank_frame_head_area;
    int total_frame;
    int tracked_id = 0;
    int frame_count = 0;

    // 文件路径
    std::string output_image_folder;
    std::string result_dir;
    std::string visualize_dir;
    std::string cv_debug_dir;
    std::string model_path = "yolo11n-modified.bin";

    // 检测器
    HBInfer hb_infer;
    
    // // 共享内存（简化实现）
    // std::vector<TrackedData> shared_xy_areas;
    // std::shared_mutex shm_mutex;  // 用于线程安全

    // 一把大锁保平安
    mutex mtx;
    
    // 算法模型
    KMeans kmeans;
    
    // 可视化配置
    struct VisualizeConfig {
        int font = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.6;
        cv::Scalar color_tracked = cv::Scalar(0, 255, 0);
        cv::Scalar color_untracked = cv::Scalar(0, 0, 255);
        int thickness = 1;
    } visualize_config;
    int log_put_start_x = 10;  // 日志文本起始位置
    int log_put_start_y = 10;  // 日志文本起始位置
    

public:
    // 构造函数
    Tracker() : 
        roi_x1(20), roi_y1(5), roi_x2(620), roi_y2(475),
        image_w(roi_x2 - roi_x1),
        image_h(roi_y2 - roi_y1),
        scale_x(640.0f / 1920.0f),
        scale_y(480.0f / 1080.0f),
        roix1_orig(static_cast<int>((roi_x1) / scale_x)),
        roiy1_orig(static_cast<int>((roi_y1) / scale_y)),
        roix2_orig(static_cast<int>((roi_x2) / scale_x)),
        roiy2_orig(static_cast<int>((roi_y2) / scale_y)),
        width_orig(roix2_orig - roix1_orig),
        length_orig(roiy2_orig - roiy1_orig),
        // right_width(static_cast<int>((check_length) / scale_x)),
        total_frame(1),
        kmeans(2, 100, 42) {
        
        right_width = static_cast<int>(check_length / scale_x);
        // std::cout << "right_width (after cast): " << right_width << std::endl;
        
        // 初始化可视化配置
        visualize_config.color_tracked = cv::Scalar(255, 0, 0);    // 绿色
        visualize_config.color_untracked = cv::Scalar(0, 0, 255);  // 红色
        visualize_config.font = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
        visualize_config.font_scale = 0.7;
        visualize_config.thickness = 1;

        hb_infer.load_model(model_path);
        hb_infer.allocCachedMem();
        
    }
    
    // 析构函数
    ~Tracker() {
        hb_infer.releaseMem();
        hb_infer.releaseModel();
    }
    
    cv::Rect getROI() const {
        return cv::Rect(roi_x1, roi_y1, image_w, image_h);
    }
    
    // int find_vaild_frame(int dir);
    bool detect_block(const cv::Mat& frame, const cv::Mat& blank_frame);
    cv::Mat crop_and_pad_by_contour(const cv::Mat& image, const cv::Mat& blank, int target_size);
    cv::Mat createGridImage(const std::vector<cv::Mat>& images, int rows, int cols);
    vector<TrackedData> cv_process_frame(const cv::Mat& frame, bool debug);
    vector<TrackedData> detection(const cv::Mat& frame);
    void tracking_group(const cv::Mat& frame, 
                        vector<TrackedData>& tracked_datavec, 
                        bool visualize);
    bool is_all_white(const cv::Mat& img);
    MatchResult match_points(const std::vector<TrackXY>& set_A, 
                            const std::vector<TrackXY>& set_B);
    tuple<vector<string>, vector<string>, vector<string>>
        occlusion_spilt(const vector<string>& name_list, bool color_similar);
    void save_results(bool save_error);
    void track(const cv::Mat& frame, bool visualize);
    void track_FromVideo(const string& video_path);
    void track_FromVideosFolder(const string& video_folder);
    void track_FromImgs(const string& img_folder, bool cv_debug);
    void track_FromImgsFolder(const string& imgs_parent_folder);
    std::vector<cv::Mat> videoToFrames(const std::string& videoPath);
    bool is_track_over();
    std::unordered_map<int, ImageData> getTrackResults();
    void post_process();
    void reset();
    void setOutputFolder(const string& path);
    vector<cv::Mat> getVisImages();
    vector<string> getVisNames(); 
    void reset_each_frame();
    void drawFrameLogs(cv::Mat& image, const vector<std::string>& logs);
    void cv_debug_FromImgsFolder(const string& imgs_parent_folder);

};
