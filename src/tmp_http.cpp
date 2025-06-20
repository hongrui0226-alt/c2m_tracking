#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

// 用于存储HTTP响应的结构体
struct UploadResult {
    std::string data;
    size_t size;
};

// libcurl写入回调函数
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    UploadResult* result = static_cast<UploadResult*>(userp);
    
    result->data.append(static_cast<char*>(contents), realsize);
    result->size += realsize;
    return realsize;
}

// 从文件读取图片数据
std::vector<unsigned char> readImageFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return {};
    }
    
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<unsigned char> buffer(fileSize);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    return buffer;
}

bool sendImageArrayToPython(const std::string& url, const std::vector<std::string>& imagePaths) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize curl" << std::endl;
        return false;
    }
    
    UploadResult result;
    std::string postFields;
    nlohmann::json jsonData;
    
    // 准备图片数据
    for (const auto& path : imagePaths) {
        auto imageData = readImageFile(path);
        if (imageData.empty()) {
            curl_easy_cleanup(curl);
            return false;
        }
        
        // 将图片数据编码为base64（实际项目中可考虑直接发送二进制数据）
        // 注：base64编码会增加约33%的数据量，大图片建议使用二进制传输
        std::string base64Data(imageData.begin(), imageData.end());
        jsonData.push_back(base64Data);
    }
    
    postFields = jsonData.dump();
    
    // 设置curl选项
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    
    // 设置请求头
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    // 设置请求体
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postFields.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, postFields.size());
    
    // 执行请求
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return false;
    }
    
    // 打印响应
    std::cout << "HTTP响应状态码: " << result.size << std::endl;
    std::cout << "响应内容: " << result.data << std::endl;
    
    // 清理资源
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return true;
}

int main() {
    // 初始化libcurl
    curl_global_init(CURL_GLOBAL_ALL);
    
    // Python服务端URL
    std::string serverUrl = "http://127.0.0.1:5000/upload/images";
    
    // 图片文件路径数组
    std::vector<std::string> imagePaths = {
        "/path/to/image1.jpg",
        "/path/to/image2.png"
    };
    
    // 发送图片数组
    bool success = sendImageArrayToPython(serverUrl, imagePaths);
    
    // 清理libcurl
    curl_global_cleanup();
    return success ? 0 : 1;
}
