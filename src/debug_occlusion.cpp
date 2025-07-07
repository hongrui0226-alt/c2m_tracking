#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <regex>
#include <ranges>
#include <fstream>

using namespace std;

struct FileInfo {
    string name;
    int frame_id;
    bool is_merged = false;
    bool is_separated = false;
};

bool endsWith(const std::string& str, const std::string& suffix) {
    const size_t str_len = str.size();
    const size_t suffix_len = suffix.size();
    
    if (str_len < suffix_len) return false;
    
    return std::memcmp(
        str.data() + str_len - suffix_len,  // 字符串末尾指针
        suffix.data(),                     // 后缀起始指针
        suffix_len                         // 比较长度
    ) == 0;
}

// 解析文件名信息
FileInfo parse_file_info(const string& name) {
    static const regex frame_pattern(R"(^(\d+)_)");
    smatch match;
    
    FileInfo info{name};
    if (regex_search(name, match, frame_pattern) && match.size() > 1) {
        info.frame_id = stoi(match[1].str());
    }
    
    info.is_merged = endsWith(name, "_m.png");
    info.is_separated = name.find("state(2)") != string::npos;
    // cout << "name: " << name << endl;
    // cout << "frame_id: " << info.frame_id << endl;
    // cout << "is_merged: " << info.is_merged << endl;
    // cout << "is_separated: " << info.is_separated << endl;
    
    return info;
}

tuple<vector<string>, vector<string>, vector<string>>
occlusion_spilt(const vector<string>& name_list, bool color_similar = true) {
    // 解析所有文件信息
    vector<FileInfo> parsed_files;
    parsed_files.reserve(name_list.size());
    
    for (const auto& name : name_list) {
        parsed_files.push_back(parse_file_info(name));
    }
    
    // 计算关键帧索引
    int last_merged=0;
    int first_separated=100000;
    int merged_index;
    int separated_index; 
    cout << "last_merged: " << last_merged << endl;
    cout << "first_separated: " << first_separated << endl;
    
    // 分类文件
    vector<string> merged_list, separated_list;
    vector<string> before_merged_list, after_separated_list, merging_list;
    vector<FileInfo> temp_list;
    
    for (const auto& file : parsed_files) {
        if (file.is_merged) {
            merged_list.push_back(file.name);
            merged_index = file.frame_id;
            last_merged = merged_index > last_merged ? merged_index : last_merged;
        } else if (file.is_separated) {
            separated_list.push_back(file.name);
            separated_index = file.frame_id;
            first_separated = separated_index < first_separated ? separated_index : first_separated;
        } else {
            temp_list.push_back(file);
        }
    }
    
    for  (const auto& file : temp_list) { 
        if (file.frame_id <= last_merged) {
            before_merged_list.push_back(file.name);
        }
        else if (file.frame_id >= first_separated) {
            after_separated_list.push_back(file.name);
        }
        else {
            merging_list.push_back(file.name);
        }
    }

    // 返回结果
    if (color_similar) {
        const size_t merged_size = merged_list.size() + before_merged_list.size();
        const size_t separated_size = separated_list.size() + after_separated_list.size();
        
        return merged_size > separated_size
            ? make_tuple(merged_list, before_merged_list, merging_list)
            : make_tuple(separated_list, after_separated_list, merging_list);
    } else {
        vector<string> combined;
        combined.reserve(merged_list.size() + before_merged_list.size() + 
                         separated_list.size() + after_separated_list.size());
        
        combined.insert(combined.end(), merged_list.begin(), merged_list.end());
        combined.insert(combined.end(), before_merged_list.begin(), before_merged_list.end());
        combined.insert(combined.end(), separated_list.begin(), separated_list.end());
        combined.insert(combined.end(), after_separated_list.begin(), after_separated_list.end());
        
        return make_tuple(combined, merging_list, vector<string>{});
    }
}

std::vector<std::string> loadFromFile(const std::string& filename) {
    std::ifstream in(filename);
    std::vector<std::string> result;

    if (!in) {
        std::cerr << "无法打开文件进行读取: " << filename << std::endl;
        return result;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            result.push_back(line);
        }
    }
    in.close();

    return result;
}

int main(int argc, char** argv) {
    vector<string> test_data = loadFromFile("../test_data.txt");
    auto [obj1_list, obj2_list, merging_list] = occlusion_spilt(test_data);
    cout << "obj1_list: " << endl;
    for (const auto& str : obj1_list) {
        cout << str << endl;
    }
    cout << "obj2_list: " << endl;
    for (const auto& str : obj2_list) {
        cout << str << endl;
    }
    cout << "merging_list: " << endl;
    for (const auto& str : merging_list) {
        cout << str << endl;
    }
}