#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <curl/curl.h>
using namespace std;

struct DataLoopInfo {
    string project_id;
    string sample_id;
    string operator_name;
    string type;
    string version;
    string url = "http://172.25.12.10/dataserver/api/samples/upload_inference";
};

std::string generateTimestamp(int suffix = 6) {
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

// 编码图像为字节流
bool encodeImage(const cv::Mat& img, std::vector<unsigned char>& buffer, const std::string& format = ".jpg") {
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

// // 上传图像
// void uploadImages(const std::vector<cv::Mat>& images, const std::string& url) {
//     CURL* curl = curl_easy_init();
//     if (!curl) return;

//     curl_mime* mime = curl_mime_init(curl);

//     for (size_t i = 0; i < images.size(); ++i) {
//         std::vector<unsigned char> buffer;
//         if (!encodeImage(images[i], buffer, ".jpg")) {
//             std::cerr << "编码图像失败: " << i << std::endl;
//             continue;
//         }

//         curl_mimepart* part = curl_mime_addpart(mime);
//         curl_mime_name(part, "images");

//         std::string filename = "image_" + std::to_string(i) + ".jpg";
//         curl_mime_filename(part, filename.c_str());
//         curl_mime_type(part, "image/jpeg");
//         curl_mime_data(part, reinterpret_cast<const char*>(buffer.data()), buffer.size());
//     }

//     // 可选字段
//     {
//         curl_mimepart* part = curl_mime_addpart(mime);
//         curl_mime_name(part, "project_id");
//         curl_mime_data(part, "231", CURL_ZERO_TERMINATED);
//     }

//     {
//         curl_mimepart* part = curl_mime_addpart(mime);
//         curl_mime_name(part, "sample_id");
//         curl_mime_data(part, "20250620", CURL_ZERO_TERMINATED);
//     }

//     {
//         curl_mimepart* part = curl_mime_addpart(mime);
//         curl_mime_name(part, "operator");
//         curl_mime_data(part, "X51", CURL_ZERO_TERMINATED);
//     }

//     {
//         curl_mimepart* part = curl_mime_addpart(mime);
//         curl_mime_name(part, "type");
//         curl_mime_data(part, "NonSequential", CURL_ZERO_TERMINATED);
//     }

//     {
//         curl_mimepart* part = curl_mime_addpart(mime);
//         curl_mime_name(part, "version");
//         curl_mime_data(part, "1", CURL_ZERO_TERMINATED);
//     }

//     curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
//     curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);

//     CURLcode res = curl_easy_perform(curl);
//     if (res != CURLE_OK) {
//         std::cerr << "请求失败: " << curl_easy_strerror(res) << std::endl;
//     }

//     curl_mime_free(mime);
//     curl_easy_cleanup(curl);
// }

void uploadImages(const std::vector<cv::Mat>& images, const std::vector<string> names, 
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

        std::string filename = "image_" + std::to_string(i) + ".jpg";
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

int main() {
    // 示例：加载图像
    std::vector<cv::Mat> images;
    std::vector<string> names;
    cv::Mat img1 = cv::imread("../test_data/img1.jpg");
    cv::Mat img2 = cv::imread("../test_data/img2.jpg");
    images.push_back(img1);
    names.push_back("img1.jpg");
    images.push_back(img2);
    names.push_back("img2.jpg");

    if (images[0].empty() || images[1].empty()) {
        std::cerr << "无法读取图像" << std::endl;
        return 1;
    }

    for (int i=2; i<10; i++) {
        images.push_back(img1.clone());
        names.push_back(cv::format("img%d.jpg", i));
    }

    // 上传图像数据
    DataLoopInfo info;
    info.project_id = "231";
    info.sample_id = generateTimestamp(66);
    info.operator_name = "discover";
    info.type = "NonSequential";
    info.version = "1";

    std::cout << "start uploading!" << std::endl;
    uploadImages(images, names, info);

    // std::string url = "http://172.25.12.10/dataserver/api/samples/upload_inference";
    // std::cout << "start uploading!" << std::endl;
    // uploadImages(images, url);

    return 0;
}