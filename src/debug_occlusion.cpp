#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <regex>
#include <ranges>
#include <fstream>
#include <gflags/gflags.h>
#include <charconv>
#include <unordered_map>
#include <cmath>
#include <unordered_set>
#include <optional>
#include <numeric>

DEFINE_string(file_path, "../../test_data.txt", "Data to test algorithm");

using namespace std;
using TrackXY = std::array<int, 2>;

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

struct FileInfo {
    string name;
    int frame_id;
    bool is_merged = false;
    bool is_separated = false;
    TrackXY xy;
};

inline double euclidean_distance(const TrackXY& a, const TrackXY& b) {
    int dx = a[0] - b[0];
    int dy = a[1] - b[1];
    return std::sqrt(dx * dx + dy * dy); // 直接计算，避免调用
}

MatchResult match_points(const std::vector<TrackXY>& set_A, const std::vector<TrackXY>& set_B) {
    // 生成未匹配的点集
    std::vector<size_t> unmatched_a;
    std::vector<size_t> unmatched_b;

    const size_t size_A = set_A.size();
    const size_t size_B = set_B.size();
    
    if (size_A == 0 || size_B == 0) {
        return {
            {},
            size_A > 0 ? std::vector<size_t>(size_A) : std::vector<size_t>(),
            size_B > 0 ? std::vector<size_t>(size_B) : std::vector<size_t>()
        };
    }
    
    // 为每个A中的点找到B中最近的点（带索引）
    std::vector<std::optional<Match>> tentative_matches(size_A);
    
    for (int a_idx = 0; a_idx < size_A; ++a_idx) {
        double min_dist = std::numeric_limits<double>::max();
        int min_b_idx = 0;
        
        for (int b_idx = 0; b_idx < size_B; ++b_idx) {
            const double dist = euclidean_distance(set_A[a_idx], set_B[b_idx]);
            if (dist < min_dist) {
                min_dist = dist;
                min_b_idx = b_idx;
            }
        }
        
        tentative_matches[a_idx] = Match{a_idx, min_b_idx, -1, min_dist};
    }
    
    // 按距离排序，优先处理距离近的匹配
    std::vector<size_t> sorted_indices(size_A);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), 
        [&](size_t i, size_t j) {
            return tentative_matches[i]->distance < tentative_matches[j]->distance;
        });
    
    // 最终匹配结果和已匹配的B点集合
    std::vector<Match> final_matches;
    std::unordered_set<size_t> matched_b_indices;
    
    // 处理排序后的匹配
    for (size_t idx : sorted_indices) {
        const auto& match = *tentative_matches[idx];
        if (matched_b_indices.find(match.b_idx) == matched_b_indices.end()) {
            final_matches.push_back(match);
            matched_b_indices.insert(match.b_idx);
        }
        else {
            final_matches.push_back({match.a_idx, -1, match.b_idx, match.distance});
            // Add 未匹配的A点
            unmatched_a.push_back(match.a_idx);
        }
    }
    
    // 未匹配的B点
    for (int b_idx = 0; b_idx < size_B; ++b_idx) {
        if (matched_b_indices.find(b_idx) == matched_b_indices.end()) {
            unmatched_b.push_back(b_idx);
        }
    }

    sort(final_matches.begin(), final_matches.end(), 
          [](const Match& m1, const Match& m2) {
              return m1.a_idx < m2.a_idx;  // 升序排序
          });
    
    return {final_matches, unmatched_a, unmatched_b};
}

TrackXY extractCoordinates(std::string_view filename) {
    // 定位关键字符位置
    auto start = filename.find("_(");
    if (start == std::string_view::npos) 
        return {0, 0};
    start += 2;  // 跳过 "_("
    
    auto comma = filename.find(',', start);
    if (comma == std::string_view::npos || comma == start)
        return {0, 0};
    
    auto end = filename.find(')', comma);
    if (end == std::string_view::npos || end == comma + 1)
        return {0, 0};

    // 提取数字部分
    std::string_view x_str = filename.substr(start, comma - start);
    std::string_view y_str = filename.substr(comma + 1, end - comma - 1);

    // 高效转换整数
    TrackXY result;
    if (std::from_chars(x_str.data(), x_str.data() + x_str.size(), result[0]).ec != std::errc{})
        return {0, 0};
    if (std::from_chars(y_str.data(), y_str.data() + y_str.size(), result[1]).ec != std::errc{})
        return {0, 0};

    return result;
}

std::vector<int> findLongestConsSeg(const std::vector<int>& a) {
    if (a.empty()) return {};
    
    int max_start = 0;      // 最长连续段的起始位置
    int max_length = 1;     // 最长连续段的长度
    int current_start = 0;  // 当前连续段的起始位置
    
    for (int i = 1; i < a.size(); ++i) {
        // 检查是否连续
        if (a[i] != a[i-1] + 1) {
            // 发现不连续点，检查当前段是否是最长
            int current_length = i - current_start;
            if (current_length > max_length) {
                max_length = current_length;
                max_start = current_start;
            }
            current_start = i;  // 开始新的连续段
        }
    }
    
    // 检查最后一个连续段
    int final_length = static_cast<int>(a.size()) - current_start;
    if (final_length > max_length) {
        max_length = final_length;
        max_start = current_start;
    }
    
    // 创建结果向量（使用迭代器范围构造）
    return std::vector<int>(a.begin() + max_start, 
                           a.begin() + max_start + max_length);
}

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
    info.xy = extractCoordinates(name);
    // cout << "name: " << name << endl;
    // cout << "frame_id: " << info.frame_id << endl;
    // cout << "is_merged: " << info.is_merged << endl;
    // cout << "is_separated: " << info.is_separated << endl;
    // cout << "xy: " << info.xy[0] << info.xy[1] << endl;
    
    return info;
}

tuple<vector<string>, vector<string>, vector<string>>
occlusion_spilt(const vector<string>& name_list, bool color_similar = true) {
    // 解析所有文件信息
    vector<FileInfo> parsed_files;
    vector<int> ids_vec;
    unordered_map<int, size_t> ids_counts;
    unordered_map<int, vector<FileInfo>> multi_occlu_files;
    vector<int> frame_ids, cons_frame_ids;
    vector<string> a, b, c;
    int max_num = 0;
    parsed_files.reserve(name_list.size());
    
    for (const auto& name : name_list) {
        FileInfo info = parse_file_info(name);
        parsed_files.push_back(info);
        ids_vec.push_back(info.frame_id);
        ids_counts[info.frame_id]++;
        max_num = ids_counts[info.frame_id]>max_num ? ids_counts[info.frame_id] : max_num;
    }

    // 三遮挡及以上
    if (max_num > 2) {  
        vector<vector<FileInfo>> result_files(max_num);
        vector<vector<int>> results;
        vector<int> tmp_results;
        for (int m=0; m < max_num; m++) {
            tmp_results.push_back(m);
        }
        results.push_back({0,0,0,0});

        // 获取帧内目标最多的帧id
        for (const auto& [frame_id, num] : ids_counts) {
            if (num == max_num) {
                frame_ids.push_back(frame_id);
            }
        }
        // 找到最大连续帧
        sort(frame_ids.begin(), frame_ids.end());
        cons_frame_ids = findLongestConsSeg(frame_ids);

        for (const auto& info : parsed_files) {
            if (find(cons_frame_ids.begin(), cons_frame_ids.end(), info.frame_id) != cons_frame_ids.end()) {
                multi_occlu_files[info.frame_id].push_back(info);
            }
        }
        
        for (int i=0; i < (cons_frame_ids.size()-1); i++) {
            int id = cons_frame_ids[i];
            // Match
            vector<TrackXY> tmp_xy_1, xy_1, xy_2;
            for (const auto& info : multi_occlu_files[id]) {
                tmp_xy_1.push_back(info.xy);
            }
            for (const auto& index : tmp_results) {
                xy_1.push_back(tmp_xy_1[index]);
            }
            tmp_results.clear();
            for (const auto& info : multi_occlu_files[id+1]) {
                xy_2.push_back(info.xy);
            }
            auto [matches, unmatched_a, unmatched_b] = match_points(xy_1, xy_2);
            for (const auto& match : matches) {
                tmp_results.push_back(match.b_idx);
            }
            // 对未匹配到的积木二次匹配
            std::vector<TrackXY> unmatched_last_xy, unmatched_current_xy;
            std::vector<size_t> tmp_unmatched_a, tmp_unmatched_b;
            for (size_t i = 0; i < unmatched_a.size(); ++i) { 
                unmatched_last_xy.push_back(xy_1[unmatched_a[i]]);
            }
            for (size_t i = 0; i < unmatched_b.size(); ++i) { 
                unmatched_current_xy.push_back(xy_2[unmatched_b[i]]);
            }
            // Secondly Match
            auto [sec_matches, sec_unmatched_a, sec_unmatched_b] = match_points(unmatched_last_xy, unmatched_current_xy);
            for (const auto& match : sec_matches) {
                // tmp_results.push_back(match.b_idx);
                tmp_results[unmatched_a[match.a_idx]] = unmatched_b[match.b_idx];
            }

            results.push_back(tmp_results);
        }

        // 默认的：max_num 等于 result_files.size() 、cons_frame_ids.size() 等于 multi_occlu_files 的size
        size_t vaild_frame_num = cons_frame_ids.size();
        for (int i=0; i < max_num; i++) {
            for (int j=0; j < vaild_frame_num; j++) {
                cout << "results[j][i]: " << results[j][i] << endl;
                result_files[i].push_back(multi_occlu_files[cons_frame_ids[j]][results[j][i]]);
            }
        }

        for (const auto& single_group : result_files) {
            for (const auto& single_file : single_group) {
                a.push_back(single_file.name);
            }
            a.push_back("xxxxx");
        }

        for (const auto& name : a) {
            cout << "name: " << name << endl;
        }

        c.push_back("MultiOcclusionSpilt");
        
        return make_tuple(a, b, c);
    } 
    // 二遮挡
    else { 
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
        
        for (const auto& file : temp_list) { 
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
    google::ParseCommandLineFlags(&argc, &argv, true);

    vector<string> test_data = loadFromFile(FLAGS_file_path);
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