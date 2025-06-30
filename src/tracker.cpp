#include "tracker.hpp"

inline double euclidean_distance(const TrackXY& a, const TrackXY& b) {
    int dx = a[0] - b[0];
    int dy = a[1] - b[1];
    return std::sqrt(dx * dx + dy * dy); // 直接计算，避免调用
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
    
    return info;
}

// int Tracker::find_vaild_frame(int dir) {
//     int step = 3;
//     std::cout << "total_frame num: " << total_frame << std::endl;

//     if (dir == 1) {
//         // 反向遍历帧（从后往前）
//         for (int i = frames.size() - 1; i >= 0; i -= step) {
//             cv::Mat frame = frames[i](
//                 cv::Rect(roix1_orig, roiy1_orig, width_orig, length_orig)
//             );
            
//             // 截取检查区域（左侧）
//             cv::Mat check_frame = frame(
//                 cv::Rect(0, 0, static_cast<int>(80 / scale_x), length_orig)
//             );
            
//             if (detect_block(check_frame, blank_frame_tail_area)) {
//                 return total_frame - ((frames.size() - 1 - i) / step - 1) * 3;
//             }
//         }
//     } 
//     else if (dir == 2) {
//         // 正向遍历帧（从前往后）
//         for (size_t i = 0; i < frames.size(); i += step) {
//             cv::Mat frame = frames[i](
//                 cv::Rect(roix1_orig, roiy1_orig, width_orig, length_orig)
//             );
            
//             // 截取检查区域（右侧）
//             int right_width = static_cast<int>(40 / scale_x);
//             cv::Mat check_frame = frame(
//                 cv::Rect(width_orig - right_width, 0, right_width, length_orig)
//             );
            
//             if (detect_block(check_frame, blank_frame_head_area)) {
//                 return (i / step - 1) * 3;
//             }
//         }
//     }

//     return -1;
// }

bool Tracker::detect_block(const cv::Mat& frame, const cv::Mat& blank_frame) {
    // 获得原始区域
    cv::Mat rect_frame = frame(
        cv::Rect(roix1_orig, roiy1_orig, width_orig, length_orig)
    );
    cv::Mat rect_blank_frame = blank_frame(
        cv::Rect(roix1_orig, roiy1_orig, width_orig, length_orig)
    );

    // 截取检查区域（右侧）
    cv::Mat check_frame = rect_frame(
        cv::Rect(width_orig - right_width, 0, right_width, length_orig)
    );
    cv::Mat check_blank_frame = rect_blank_frame(
        cv::Rect(width_orig - right_width, 0, right_width, length_orig)
    );

    // 背景减除器+开运算
    cv::Mat gray_check, gray_ref, diff, fg_mask, thresh;
    
    // 转换为灰度图
    cv::cvtColor(check_frame, gray_check, cv::COLOR_BGR2GRAY);
    cv::cvtColor(check_blank_frame, gray_ref, cv::COLOR_BGR2GRAY);
    
    // 使用OpenCV subtract并计算绝对值
    cv::subtract(gray_check, gray_ref, diff, cv::noArray(), CV_8S); // 有符号8位
    cv::convertScaleAbs(diff, diff); // 计算绝对值并转为CV_8U
    
    // 阈值处理
    cv::threshold(diff, thresh, binary_threshold, 255, cv::THRESH_BINARY);
    
    // 形态学操作（开运算：先腐蚀后膨胀）
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(thresh, fg_mask, kernel, cv::Point(-1, -1), 2);
    cv::dilate(fg_mask, fg_mask, kernel, cv::Point(-1, -1), 1);
    
    // 计算非零像素数量
    int pixel_count = cv::countNonZero(fg_mask);
    
    // 阈值判断
    int min_pixel_threshold = 400;
    return pixel_count > min_pixel_threshold;
}

cv::Mat Tracker::crop_and_pad_by_contour(const cv::Mat& image, const cv::Mat& blank, int target_size) {
    // 输入校验
    if (image.empty() || image.channels() != 3) {
        throw std::invalid_argument("输入图像必须是一个有效的三通道矩阵");
    }
    if (target_size <= 0) {
        throw std::invalid_argument("目标尺寸必须是一个大于零的整数");
    }

    try {
        // 转为灰度并二值化
        cv::Mat gray, blank_gray, diff, thresh, thresh_clean;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(blank, blank_gray, cv::COLOR_BGR2GRAY);
        cv::absdiff(gray, blank_gray, diff);
        GaussianBlur(diff, diff, cv::Size(5, 5), 0);  // 模糊去噪，利于抠图
        cv::threshold(diff, thresh, binary_threshold, 255, cv::THRESH_BINARY);

        // 形态学开运算去噪
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(thresh, thresh_clean, cv::MORPH_OPEN, kernel);

        // 创建白色背景
        cv::Mat white_image(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));

        // 使用掩码选择性保留原图或替换为白色
        cv::Mat mask_inv, foreground, background, result;
        cv::bitwise_not(thresh_clean, mask_inv);
        cv::bitwise_and(image, image, foreground, thresh_clean);
        cv::bitwise_and(white_image, white_image, background, mask_inv);
        cv::add(foreground, background, result);

        // 获取图像尺寸
        int h = result.rows;
        int w = result.cols;

        // 计算缩放比例并缩放图像
        if (w == 0 || h == 0) {
            throw std::invalid_argument("输入图像的宽度或高度不能为零");
        }
        float scale = std::min(static_cast<float>(target_size) / w, 
                              static_cast<float>(target_size) / h);
        int new_w = static_cast<int>(w * scale);
        int new_h = static_cast<int>(h * scale);
        cv::Mat resized;
        cv::resize(result, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_NEAREST);

        // 创建白色画布并居中填充
        cv::Mat canvas(cv::Size(target_size, target_size), CV_8UC3, cv::Scalar(255, 255, 255));
        int dx = (target_size - new_w) / 2;
        int dy = (target_size - new_h) / 2;
        cv::Rect roi(dx, dy, new_w, new_h);
        resized.copyTo(canvas(roi));

        return canvas;
    } 
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("图像处理过程中发生错误: ") + e.what());
    }
}

vector<TrackedData> Tracker::cv_process_frame(const cv::Mat& frame) {
    using namespace std::chrono;
    static const int MIN_CONTOUR_AREA = 100; // 需根据实际定义
    static const int SOBEL_THRESH = 50;       // 需根据实际定义
    
    // int bianli = front_frame;
    auto start_time = high_resolution_clock::now();

    // TrackedData tracked_data;
    vector<TrackedData> res_data;

    // int current_cpu = sched_getcpu(); // 获取当前CPU核心
    // std::cout << "当前进程 (PID: " << getpid() << ") 正在 CPU " << current_cpu << " 上运行\n";

    std::vector<double> resize_time, roi_time, cvt_color_time, np_abs_time,
        cv2_threshold_time, cv2_erode_time, cv2_dilate_time,
        cv2_find_counter_time, cv2_sobel_time, effect_image_time;

    cv::Size target_size(640, 480);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    auto effect_start = high_resolution_clock::now();

    // 1. 图像缩放
    cv::Mat resized_frame;
    auto t1 = high_resolution_clock::now();
    cv::resize(frame, resized_frame, target_size, 0, 0, cv::INTER_NEAREST);
    resize_time.push_back(duration_cast<milliseconds>(high_resolution_clock::now() - t1).count());

    // 2. ROI截取
    auto t2 = high_resolution_clock::now();
    cv::Mat roi_frame = resized_frame(
        cv::Rect(roi_x1, roi_y1, roi_x2-roi_x1, roi_y2-roi_y1)
    );
    roi_time.push_back(duration_cast<milliseconds>(high_resolution_clock::now() - t2).count());

    // 3. 灰度转换
    auto t3 = high_resolution_clock::now();
    cv::Mat frame_gray;
    cv::cvtColor(roi_frame, frame_gray, cv::COLOR_BGR2GRAY);
    cvt_color_time.push_back(duration_cast<milliseconds>(high_resolution_clock::now() - t3).count());

    // 4. 差异计算
    auto t4 = high_resolution_clock::now();
    cv::Mat diff;
    cv::absdiff(blank_rect_gray, frame_gray, diff);
    np_abs_time.push_back(duration_cast<milliseconds>(high_resolution_clock::now() - t4).count());

    // 5. 阈值处理
    auto t6 = high_resolution_clock::now();
    cv::Mat thresh;
    cv::threshold(diff, thresh, binary_threshold, 255, cv::THRESH_BINARY);
    cv2_threshold_time.push_back(duration_cast<milliseconds>(high_resolution_clock::now() - t6).count());

    // 6. 形态学操作
    auto t7 = high_resolution_clock::now();
    cv::Mat fg_mask;
    GaussianBlur(thresh, thresh, cv::Size(5, 5), 0);
    cv::erode(thresh, fg_mask, kernel, cv::Point(-1, -1), 1);
    cv2_erode_time.push_back(duration_cast<milliseconds>(high_resolution_clock::now() - t7).count());

    auto t8 = high_resolution_clock::now();
    cv::dilate(fg_mask, fg_mask, kernel, cv::Point(-1, -1), 1);
    cv2_dilate_time.push_back(duration_cast<milliseconds>(high_resolution_clock::now() - t8).count());

    // // 高斯模糊减少噪声
    // cv::Mat blurred;
    // GaussianBlur(diff, blurred, cv::Size(5, 5), 0);

    // // // 使用Canny边缘检测
    // cv::Mat edges;
    // // Canny(blurred, edges, 30, 90);

    // // 创建椭圆结构元素
    // cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));

    // // 形态学闭运算
    // morphologyEx(fg_mask, edges, cv::MORPH_CLOSE, close_kernel);

    // 7. 轮廓检测
    auto t9 = high_resolution_clock::now();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fg_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv2_find_counter_time.push_back(duration_cast<milliseconds>(high_resolution_clock::now() - t9).count());
    frame_logs.push_back(cv::format("Found %zu contours", contours.size()));  // add log
    int vaild_num_contours = 0;

    int valid_count = 0;
    for (const auto& contour : contours) {
        // 边界框计算
        cv::Rect bbox = cv::boundingRect(contour);
        int x = bbox.x, y = bbox.y, w = bbox.width, h = bbox.height;
        if (w == 0 || h == 0) continue;

        // 面积过滤
        double area = cv::contourArea(contour);
        if ((area <= MIN_CONTOUR_AREA || area >= 460*450) && (w*h <= 2*area)) {
            frame_logs.push_back(
                cv::format("Contour area %f = %dx%d , skipping", area, w, h)
            );
            continue;
        }
        // // 边缘检测（Laplacian算子）
        // auto t10 = high_resolution_clock::now();
        // cv::Mat roi_gray = frame_gray(bbox);
        // cv::Mat lap, lap_mag;
        // cv::Laplacian(roi_gray, lap, CV_64F);
        // cv::convertScaleAbs(lap, lap_mag);
        // double max_edge = cv::minMaxLoc(lap_mag).maxVal;
        // cv2_sobel_time.push_back(duration_cast<milliseconds>(high_resolution_clock::now() - t10).count());

        // // 边缘强度过滤
        // if (max_edge < SOBEL_THRESH) continue;

        // 坐标映射
        int cx = x + w/2, cy = y + h/2;
        int x_orig = static_cast<int>((x + roi_x1)/scale_x) - 5;
        int y_orig = static_cast<int>((y + roi_y1)/scale_y) - 5;
        int w_orig = static_cast<int>(w/scale_x) + 10;
        int h_orig = static_cast<int>(h/scale_y) + 10;

        // 边界检查
        x_orig = std::max(0, x_orig);
        y_orig = std::max(0, y_orig);
        w_orig = std::min(w_orig, blank_orig.cols - x_orig);
        h_orig = std::min(h_orig, blank_orig.rows - y_orig);

        // 图像裁剪与填充
        cv::Mat contour_region = frame(
            cv::Rect(x_orig, y_orig, w_orig, h_orig)
        );
        cv::Mat blank_region = blank_orig(
            cv::Rect(x_orig, y_orig, w_orig, h_orig)
        );
        // cv::imwrite("contour_region.jpg", contour_region);
        // cv::imwrite("blank_region.jpg", blank_region);

        cv::Mat output = crop_and_pad_by_contour(contour_region, blank_region, 224);  // Traget Size

        // 存储到共享内存
        // if (index >= 800) break;
        // xy_areas[index].frame_id = bianli - first_valid_frame;
        // xy_areas[index].cx = cx;
        // xy_areas[index].cy = cy;
        // xy_areas[index].areas = static_cast<float>(area);
        // output.copyTo(xy_areas[index].image);
        
        TrackedData data;
        data.frame_id = 1;
        data.cx = cx;
        data.cy = cy;
        data.areas = static_cast<float>(area);
        output.copyTo(data.image);
        res_data.push_back(data);

        effect_image_time.push_back(
            duration_cast<milliseconds>(high_resolution_clock::now() - effect_start).count()
        );
        vaild_num_contours++;
    }

    frame_logs.push_back(
        cv::format("Valid contours: %d", vaild_num_contours)
    );
    // 输出性能统计
    auto print_time = [](const std::string& name, const std::vector<double>& times) {
        double total = std::accumulate(times.begin(), times.end(), 0.0);
        std::cout << name << ": " << total << "ms\n";
    };

    // print_time("resize_time", resize_time);
    // print_time("roi_time", roi_time);
    // print_time("cvt_color_time", cvt_color_time);
    // print_time("np_abs_time", np_abs_time);
    // print_time("cv2_threshold_time", cv2_threshold_time);
    // print_time("cv2_erode_time", cv2_erode_time);
    // print_time("cv2_dilate_time", cv2_dilate_time);
    // print_time("cv2_find_counter_time", cv2_find_counter_time);
    // print_time("cv2_sobel_time", cv2_sobel_time);
    // std::cout << "cv_time: " << duration_cast<milliseconds>(high_resolution_clock::now() - start_time).count() << "ms\n";

    return res_data;
}

bool Tracker::is_all_white(const cv::Mat& img) {
    if (img.empty()) return false;
    
    cv::Mat temp;
    cv::bitwise_not(img, temp); // 全白→全黑（像素值0），非白→非零值
    int non_zero = cv::countNonZero(temp); // 统计非零像素数
    return non_zero == 0;
}

MatchResult Tracker::match_points(const std::vector<TrackXY>& set_A, const std::vector<TrackXY>& set_B) {
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
        }
    }
    
    // 生成未匹配的点集
    std::vector<size_t> unmatched_a;
    std::vector<size_t> unmatched_b;
    
    // 未匹配的A点
    std::unordered_set<size_t> matched_a_indices;
    for (const auto& match : final_matches) {
        matched_a_indices.insert(match.a_idx);
    }
    for (int a_idx = 0; a_idx < size_A; ++a_idx) {
        if (matched_a_indices.find(a_idx) == matched_a_indices.end()) {
            unmatched_a.push_back(a_idx);
        }
    }
    
    // 未匹配的B点
    for (int b_idx = 0; b_idx < size_B; ++b_idx) {
        if (matched_b_indices.find(b_idx) == matched_b_indices.end()) {
            unmatched_b.push_back(b_idx);
        }
    }
    
    return {final_matches, unmatched_a, unmatched_b};
}

tuple<vector<string>, vector<string>, vector<string>>
Tracker::occlusion_spilt(const vector<string>& name_list, bool color_similar = true) {
    // 解析所有文件信息
    vector<FileInfo> parsed_files;
    parsed_files.reserve(name_list.size());
    
    for (const auto& name : name_list) {
        parsed_files.push_back(parse_file_info(name));
    }
    
    // 计算关键帧索引
    int last_merged;
    int first_separated;
    int merged_index;
    int separated_index; 
    
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

void Tracker::tracking_group(const cv::Mat& frame, 
                            vector<TrackedData>& tracked_datavec, 
                            bool visualize=true) {
    frame_count++;
    current_frame_info.clear();
    current_frame_xy.clear();

    // 当前帧没有数据
    if (tracked_datavec.empty()) {
        return;
    }
    //  && !is_all_white(tracked_data.image)
    for (auto& tracked_data : tracked_datavec) {
        if (tracked_data.cx < vaild_threshold && 
            tracked_data.cx > tail_threshold) {
            TrackInfo track_info;
            track_info.current_canvas = tracked_data.image;
            track_info.xy = {tracked_data.cx, tracked_data.cy};
            track_info.areas = tracked_data.areas;
            current_frame_info.push_back(track_info);
            current_frame_xy.push_back(track_info.xy);
        }
    }

    if (current_frame_info.empty()) {
        return;
    }

    if (last_frame_info.empty()) {
        for (auto& info : current_frame_info) {
            if (info.xy[0] >= head_threshold) {  // 靠近入口
                info.motion = init_motion;
                info.id = tracked_id++;
                info.state = 0;

                img2save[info.id].images.push_back(std::move(info.current_canvas));
                img2save[info.id].names.push_back(
                    cv::format("%d_(%d,%d)_area(%.1f)_state(%d).png", 
                            frame_count+1, info.xy[0], info.xy[1], info.areas, info.state)
                );

                frame_logs.push_back(
                    cv::format("LastEmpty: New %d -> (%d, %d)",
                                info.id, info.xy[0], info.xy[1])
                );

            } else {
                cerr << "发现新积木，但不在入口区域！" << endl;
            }

            // 更新上一帧数据（使用算法优化）
            last_frame_xy.clear();
            last_frame_info.clear();
            for (const auto& info : current_frame_info) {
                if (info.state == -1) continue;  // 过滤未初始化的运动向量
                TrackXY pred_xy = {
                    info.xy[0] + info.motion[0],
                    info.xy[1] + info.motion[1]
                };
                if (pred_xy[0] <= tail_threshold) continue;
                last_frame_xy.push_back(pred_xy);
                last_frame_info.push_back(info);
            }

            // 可视化部分（使用RAII和范围循环）
            if (visualize) {
                cv::Mat visualized_frame = frame.clone();
                resize(visualized_frame, visualized_frame, cv::Size(640, 480));
                
                for (const auto& info : current_frame_info) {
                    cv::Point origin_xy(info.xy[0] + roi_x1 - 10, 
                                    info.xy[1] + roi_y1 + 5);
                    
                    if (info.id == 0) {  // 未追踪到
                        putText(visualized_frame, "x", origin_xy, 
                                visualize_config.font, visualize_config.font_scale,
                                visualize_config.color_untracked, visualize_config.thickness);
                        putText(visualized_frame, "x" + to_string(info.id), origin_xy, 
                                visualize_config.font, visualize_config.font_scale,
                                visualize_config.color_tracked, visualize_config.thickness);
                    } else {  // 已追踪到
                        putText(visualized_frame, to_string(info.id), origin_xy, 
                                visualize_config.font, visualize_config.font_scale,
                                visualize_config.color_tracked, visualize_config.thickness);
                    }
                }

                // Put log text on the frame
                drawFrameLogs(visualized_frame, frame_logs);

                {
                    lock_guard<mutex> lock(mtx);
                    visualize_images.push_back(visualized_frame);
                    visualize_names.push_back(
                        cv::format("frame_%d.jpg", frame_count+1)
                    );
                }
                
            }
        }
        return;
    }

    auto [matches, unmatched_a, unmatched_b] = match_points(last_frame_xy, current_frame_xy);
    for (size_t item_index = 0; item_index < matches.size(); ++item_index) {
        const auto& [a_idx, b_idx, other_idx, distance] = matches[item_index];  // 直接解包
        if (b_idx == -1) {
            // 未匹配到
            continue;
        }
        if (distance > dis_threshold) {
            // 距离过大，认为是新的积木
            unmatched_a.push_back(a_idx);
            unmatched_b.push_back(b_idx);
            continue;
        }
        // current frame info
        auto& xy = current_frame_info[b_idx].xy;
        auto& areas = current_frame_info[b_idx].areas;
        auto& current_canvas = current_frame_info[b_idx].current_canvas;
        
        // last frame info
        auto& last_xy = last_frame_info[a_idx].xy;
        TrackXY motion = {xy[0] - last_xy[0]
                        , xy[1] - last_xy[1]};
        auto& id = last_frame_info[a_idx].id;
        auto& state = last_frame_info[a_idx].state;
        
        // update current frame info
        current_frame_info[b_idx].motion = motion;
        current_frame_info[b_idx].id = id;
        current_frame_info[b_idx].state = state;

        // update imgs to save
        img2save[id].images.push_back(current_canvas);
        img2save[id].names.push_back(
            cv::format("%d_(%d,%d)_area(%.1f)_state(%d).png", 
                    frame_count+1, xy[0], xy[1], areas, state)
        );
    }

    // 对未匹配到的积木二次匹配
    std::vector<TrackXY> unmatched_last_xy, unmatched_current_xy;
    std::vector<size_t> tmp_unmatched_a, tmp_unmatched_b;
    for (size_t i = 0; i < unmatched_a.size(); ++i) { 
        unmatched_last_xy.push_back(last_frame_info[unmatched_a[i]].xy);
    }
    for (size_t i = 0; i < unmatched_b.size(); ++i) { 
        unmatched_current_xy.push_back(current_frame_info[unmatched_b[i]].xy);
    }
    
    if ((unmatched_a.size() > 0) && (unmatched_b.size() > 0)) {
        cout << "sec match!" << endl;
        auto [sec_matches, sec_unmatched_a, sec_unmatched_b] = match_points(unmatched_last_xy, unmatched_current_xy);

        // 二次匹配后正常追踪到的积木
        for (size_t item_index = 0; item_index < sec_matches.size(); ++item_index) {
            const auto& [sec_a_idx, sec_b_idx, sec_other_idx, distance] = sec_matches[item_index];  // 直接解包
            const auto& a_idx = unmatched_a[sec_a_idx];
            const auto& b_idx = unmatched_b[sec_b_idx];
            
            if (b_idx == -1) {
                // 未匹配到
                continue;
            }
            if (distance > dis_threshold) {
                // 距离过大，认为是新的积木
                sec_unmatched_a.push_back(sec_a_idx);
                sec_unmatched_b.push_back(sec_b_idx);
                continue;
            }
            // current frame info
            auto& xy = current_frame_info[b_idx].xy;
            auto& areas = current_frame_info[b_idx].areas;
            auto& current_canvas = current_frame_info[b_idx].current_canvas;
            
            // last frame info
            auto& last_xy = last_frame_info[a_idx].xy;
            TrackXY motion = {xy[0] - last_xy[0]
                            , xy[1] - last_xy[1]};
            auto& id = last_frame_info[a_idx].id;
            auto& state = last_frame_info[a_idx].state;
            
            // update current frame info
            current_frame_info[b_idx].motion = motion;
            current_frame_info[b_idx].id = id;
            current_frame_info[b_idx].state = state;
    
            // update imgs to save
            img2save[id].images.push_back(current_canvas);
            img2save[id].names.push_back(
                cv::format("%d_(%d,%d)_area(%.1f)_state(%d).png", 
                        frame_count+1, xy[0], xy[1], areas, state)
            );
        }  // sec_unmatched_a  和  unmatched_a 命名不一致的问题

        for (size_t i = 0; i < sec_unmatched_a.size(); ++i) { 
            sec_unmatched_a[i] = unmatched_a[sec_unmatched_a[i]];
        } 
        for (size_t i = 0; i < sec_unmatched_b.size(); ++i) { 
            sec_unmatched_b[i] = unmatched_b[sec_unmatched_b[i]];
        } 

        unmatched_a = sec_unmatched_a;
        unmatched_b = sec_unmatched_b;

    }

    // 处理异常追踪 ：last未追踪到的 -> 多合一遮挡
    for (auto& idx : unmatched_a) {
        
        auto& last_id = last_frame_info[idx].id;
        auto& min_index = matches[idx].other_idx;
        auto& min_dist = matches[idx].distance;
        auto& id = current_frame_info[min_index].id;
        auto& state = current_frame_info[min_index].state;

        if (state == -1) {  // 如果追踪到的也是未追踪的积木，则跳过
            frame_logs.push_back(
                cv::format("UntrackLast: %d tracked untracked blocks, skip", last_id)
            );
            continue;
        }
        if (id == last_id) {  // 追踪到和自己id相同的积木，则跳过
            frame_logs.push_back(
                cv::format("UntrackLast: %d tracked same id blocks, skip", last_id)
            );
            cout << "Repeated occlusion ! " << endl;
            continue;
        }

        // 判断是否处于可以合并的距离之内
        if (min_dist < merge_dis_threshold) {
            // Modyfy names of the merged images
            auto name_list = img2save[last_id].names;
            img2save[last_id].names.clear();
            for (const auto& name : name_list) {
                std::string new_name = name;
                size_t pos = new_name.find(".png");
                if (pos != std::string::npos) {
                    new_name.replace(pos, 4, "_m.png");
                }
                img2save[last_id].names.push_back(new_name);
            }
            // Merge the two img2save
            img2save[id].images.insert(
                img2save[id].images.end(),
                img2save[last_id].images.begin(),
                img2save[last_id].images.end()
            );
            img2save[id].names.insert(
                img2save[id].names.end(),
                img2save[last_id].names.begin(),
                img2save[last_id].names.end()
            );
            // Delete the last_id
            img2save.erase(last_id);

            frame_logs.push_back(
                cv::format("UntrackLast : Merged %d with %d", last_id, id)
            );
        }
        else {
            cout << last_id << "is lost with min_dist " << min_dist << endl;
            frame_logs.push_back(
                cv::format("UntrackLast : Lost %d with min_dist %f", last_id, min_dist)
            );
        }
    }

    // 处理异常追踪 ：current未追踪到的（取决于xy位置）-> 新积木 or 遮挡后一分多
    for (auto& idx : unmatched_b) { 
        auto& xy = current_frame_info[idx].xy;

        //  靠近入口 -> 新积木
        if (xy[0] >= head_threshold) {    
            current_frame_info[idx].id = tracked_id++;
            current_frame_info[idx].motion = init_motion;
            current_frame_info[idx].state = 0;
            
            frame_logs.push_back(
                cv::format("New block %d -> (%d, %d)", current_frame_info[idx].id, xy[0], xy[1])
            );
        }
        //  遮挡后一分多
        else {
            // 计算该积木与当前帧所有其他的积木的距离，并找到最近的
            if (current_frame_info.size() <= 1) {
                frame_logs.push_back(
                    cv::format("UntrackCur: New %d but no other blocks", 
                        current_frame_info[idx].id)
                );
                cout << "No other blocks in current frame! 新积木但不在入口处" << endl;
                continue;
            }
            double min_dist = std::numeric_limits<double>::max();
            int min_idx = 0;
            for (size_t i = 0; i < current_frame_info.size(); ++i) {
                if (i == idx)
                    continue;
                const double dist = euclidean_distance(xy, current_frame_info[i].xy);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = i;
                }
            }
            // 匹配到一个未被匹配的积木
            if (current_frame_info[min_idx].state == -1) {
                cout << "Matched with another not matched block !" << endl;
                frame_logs.push_back(
                    cv::format("UntrackCur: Matched %d with %d but %d is not matched", 
                        current_frame_info[idx].id, 
                        current_frame_info[min_idx].id, 
                        current_frame_info[min_idx].id)
                );
                continue;
            }
            // 距离过大，认为是异常追踪
            if (min_dist >= separate_dis_threshold) { 
                cout << "Separate too far !" << min_dist << " > " << separate_dis_threshold << endl;
                frame_logs.push_back(
                    cv::format("UntrackCur: Matched %d with %d but distance %f > %d",
                        current_frame_info[idx].id,
                        current_frame_info[min_idx].id,
                        min_dist, separate_dis_threshold)
                );
                continue;
            }

            TrackXY motion = {static_cast<int>((vaild_threshold - xy[0]) * 0.1), 0};
            current_frame_info[idx].motion = motion;
            current_frame_info[idx].id = current_frame_info[min_idx].id;
            current_frame_info[idx].state = 2;

            frame_logs.push_back(
                cv::format("UntrackCur: Matched %d with %d -> (%d, %d) state(2) with dist %f",
                    current_frame_info[idx].id,
                    current_frame_info[min_idx].id,
                    xy[0], xy[1], min_dist)
            );

        }

        img2save[current_frame_info[idx].id].images.push_back(current_frame_info[idx].current_canvas);
        img2save[current_frame_info[idx].id].names.push_back(
            cv::format("%d_(%d,%d)_area(%.1f)_state(%d).png", 
                    frame_count+1, xy[0], xy[1], 
                    current_frame_info[idx].areas, 
                    current_frame_info[idx].state)
        );
    }

    // Update last_frames_pred_xy and last_frames_info
    last_frame_xy.clear();
    last_frame_info.clear();
    for (const auto& info : current_frame_info) {
        if (info.state == -1) continue;  // 过滤未初始化的运动向量
        TrackXY pred_xy = {
            info.xy[0] + info.motion[0],
            info.xy[1] + info.motion[1]
        };
        if (pred_xy[0] <= tail_threshold) continue;
        last_frame_xy.push_back(pred_xy);
        last_frame_info.push_back(info);
    }

    // Visualize the tracking process
    if (visualize) {
        cv::Mat visualized_frame = frame.clone();
        resize(visualized_frame, visualized_frame, cv::Size(640, 480));
        
        for (const auto& info : current_frame_info) {
            cv::Point origin_xy(info.xy[0] + roi_x1 - 10, 
                            info.xy[1] + roi_y1 + 5);
            
            if (info.state == -1) {  // 未追踪到
                putText(visualized_frame, "x", origin_xy, 
                        visualize_config.font, visualize_config.font_scale,
                        visualize_config.color_untracked, visualize_config.thickness);
            } else {  // 已追踪到
                putText(visualized_frame, to_string(info.id), origin_xy, 
                        visualize_config.font, visualize_config.font_scale,
                        visualize_config.color_tracked, visualize_config.thickness);
            }
        }

        // Put log text on the frame
        drawFrameLogs(visualized_frame, frame_logs);

        {
            lock_guard<mutex> lock(mtx);
            visualize_images.push_back(visualized_frame);
            visualize_names.push_back(
                cv::format("frame_%d.jpg", frame_count+1)
            );
        }
            
    }

}

void Tracker::save_results(bool save_error=true) {
    
    for (const auto& [id, data] : img2save) {
        size_t real_length = data.images.size();
        size_t index_begin, index_end;
        size_t pos = data.names[0].find("_");
        if (pos != std::string::npos) {
            index_begin = std::stoi(data.names[0].substr(0, pos));
        }
        pos = data.names.back().find("_");
        if (pos != std::string::npos) {
            index_end = std::stoi(data.names.back().substr(0, pos));
        }
        size_t target_length = index_end - index_begin + 1;

        // 判断这组积木存在遮挡
        if (target_length < real_length) {
            // 创建文件夹
            fs::create_directories(fs::path(result_dir) / ("c_" + std::to_string(id) + "_0"));
            fs::create_directories(fs::path(result_dir) / ("c_" + std::to_string(id) + "_1"));

            vector<double> hsv_means;
            for (const auto& img : data.images) {
                // 1. 创建掩码：像素值 < 250 的区域
                vector<cv::Mat> channels;
                cv::Mat mask_b, mask_g, mask_r, mask;
                cv::split(img, channels);
                cv::threshold(channels[0], mask_b, 250, 255, cv::THRESH_BINARY_INV);
                cv::threshold(channels[1], mask_g, 250, 255, cv::THRESH_BINARY_INV);
                cv::threshold(channels[2], mask_r, 250, 255, cv::THRESH_BINARY_INV);
                cv::bitwise_and(mask_b, mask_g, mask);
                cv::bitwise_and(mask, mask_r, mask);

                // 2. 转换为HSV色彩空间
                cv::Mat hsv_img;
                cv::cvtColor(img, hsv_img, cv::COLOR_BGR2HSV);
                
                // 3. 分离HSV通道
                std::vector<cv::Mat> hsv_channels;
                cv::split(hsv_img, hsv_channels);
                cv::Mat h_channel = hsv_channels[0];  // H通道
                
                // 4. 计算掩码区域内的H通道均值
                cv::Scalar mean_value = cv::mean(h_channel, mask);
                
                // 5. 存储结果
                hsv_means.push_back(mean_value[0]);
            }

            // 如果色调差异小于阈值，则进行轨迹切割，反之则进行颜色聚类
            auto [min, max] = minmax_element(hsv_means.begin(), hsv_means.end());
            if ((*max - *min) < hsv_separation) {
                auto [obj1_list, obj2_list, merging_list] = occlusion_spilt(data.names, true);
                for (const auto& name : obj1_list) {
                    auto pos = std::find(data.names.begin(), data.names.end(), name);
                    cv::imwrite((fs::path(result_dir) / ("c_" + std::to_string(id) + "_0") / name).string(), 
                                data.images[pos - data.names.begin()]);
                }
                for (const auto& name : obj2_list) {
                    auto pos = std::find(data.names.begin(), data.names.end(), name);
                    cv::imwrite((fs::path(result_dir) / ("c_" + std::to_string(id) + "_1") / name).string(), 
                                data.images[pos - data.names.begin()]);
                }

                if (save_error) {
                    fs::create_directories(fs::path(result_dir) / ("e_" + std::to_string(id)));
                    for (const auto& name : merging_list) {
                        auto pos = std::find(data.names.begin(), data.names.end(), name);
                        cv::imwrite((fs::path(result_dir) / ("e_" + std::to_string(id)) / name).string(), 
                                    data.images[pos - data.names.begin()]);
                    }
                }
            }
            else {
                auto [combinded, merging_list, other] = occlusion_spilt(data.names, false);

                vector<double> hsv_unmerged;
                vector<string> obj1_list, obj2_list;
                for (const auto& name : combinded) {
                    auto pos = std::find(data.names.begin(), data.names.end(), name);
                    hsv_unmerged.push_back(hsv_means[pos - data.names.begin()]);
                }
                std::sort(hsv_unmerged.begin(), hsv_unmerged.end());
                for (size_t i=0; i<hsv_unmerged.size(); i++) {
                    double hsv_dist_head = hsv_unmerged[i] - hsv_unmerged[0];
                    double hsv_dist_tail = hsv_unmerged.back() - hsv_unmerged[i];
                    if (hsv_dist_head > hsv_dist_tail) {
                        for (size_t j=0; j<i; j++)
                            obj1_list.push_back(combinded[j]);
                        for (size_t j=i; j<combinded.size(); j++)
                            obj2_list.push_back(combinded[j]);
                    }
                }
                for (const auto& name : obj1_list) {
                    auto pos = std::find(data.names.begin(), data.names.end(), name);
                    cv::imwrite((fs::path(result_dir) / ("c_" + std::to_string(id) + "_0") / name).string(), 
                                data.images[pos - data.names.begin()]);
                }
                for (const auto& name : obj2_list) {
                    auto pos = std::find(data.names.begin(), data.names.end(), name);
                    cv::imwrite((fs::path(result_dir) / ("c_" + std::to_string(id) + "_1") / name).string(), 
                                data.images[pos - data.names.begin()]);
                }

                if (save_error) {
                    fs::create_directories(fs::path(result_dir) / ("e_" + std::to_string(id)));
                    for (const auto& name : merging_list) {
                        auto pos = std::find(data.names.begin(), data.names.end(), name);
                        cv::imwrite((fs::path(result_dir) / ("e_" + std::to_string(id)) / name).string(), 
                                    data.images[pos - data.names.begin()]);
                    }
                }
            }
        }
        // 判断这组积木不存在遮挡
        else {
            fs::create_directories(fs::path(result_dir) / std::to_string(id));
            for (size_t i=0; i<data.names.size(); i++) {
                cv::imwrite((fs::path(result_dir) / std::to_string(id) / data.names[i]).string(), 
                            data.images[i]);
            }
        }
    }

    // Save visualize images
    if (!visualize_images.empty()) {
        for (int i=0; i<visualize_images.size(); i++) {
            cv::imwrite((fs::path(visualize_dir) / visualize_names[i]).string(), visualize_images[i]);
        }
    }
}

void Tracker::post_process() {
    vector<int> nonused_ids;

    for (const auto& [id, data] : img2save) {
        if (data.images.empty()) {
            std::cerr << "Error: img2save-id"<< id <<" is empty or invalid!" << std::endl;
            continue;
        }
        size_t real_length = data.images.size();
        size_t index_begin, index_end;
        size_t pos = data.names[0].find("_");
        if (pos != std::string::npos) {
            index_begin = std::stoi(data.names[0].substr(0, pos));
        }
        pos = data.names.back().find("_");
        if (pos != std::string::npos) {
            index_end = std::stoi(data.names.back().substr(0, pos));
        }
        size_t target_length = index_end - index_begin + 1;

        // 判断这组积木存在遮挡
        if (target_length < real_length) {
            int id_1 = tracked_id++;
            int id_2 = tracked_id++;

            vector<double> hsv_means;
            for (const auto& img : data.images) {
                // 1. 创建掩码：像素值 < 250 的区域
                vector<cv::Mat> channels;
                cv::Mat mask_b, mask_g, mask_r, mask;
                cv::split(img, channels);
                cv::threshold(channels[0], mask_b, 250, 255, cv::THRESH_BINARY_INV);
                cv::threshold(channels[1], mask_g, 250, 255, cv::THRESH_BINARY_INV);
                cv::threshold(channels[2], mask_r, 250, 255, cv::THRESH_BINARY_INV);
                cv::bitwise_and(mask_b, mask_g, mask);
                cv::bitwise_and(mask, mask_r, mask);

                // 2. 转换为HSV色彩空间
                cv::Mat hsv_img;
                cv::cvtColor(img, hsv_img, cv::COLOR_BGR2HSV);
                
                // 3. 分离HSV通道
                std::vector<cv::Mat> hsv_channels;
                cv::split(hsv_img, hsv_channels);
                cv::Mat h_channel = hsv_channels[0];  // H通道
                
                // 4. 计算掩码区域内的H通道均值
                cv::Scalar mean_value = cv::mean(h_channel, mask);
                
                // 5. 存储结果
                hsv_means.push_back(mean_value[0]);
            }

            // 如果色调差异小于阈值，则进行轨迹切割，反之则进行颜色聚类
            auto [min, max] = minmax_element(hsv_means.begin(), hsv_means.end());
            if ((*max - *min) < hsv_separation) {
                auto [obj1_list, obj2_list, merging_list] = occlusion_spilt(data.names, true);
                cout << "id: " << id << endl;
                cout << "obj1_list size: " << obj1_list.size() << endl;
                cout << "obj2_list size: " << obj2_list.size() << endl;
                cout << "merging_list size: " << merging_list.size() << endl;
                for (const auto& name : obj1_list) {
                    auto pos = std::find(data.names.begin(), data.names.end(), name);
                    img2save[id_1].images.push_back(data.images[pos - data.names.begin()].clone());
                    img2save[id_1].names.push_back(name);
                }
                for (const auto& name : obj2_list) {
                    auto pos = std::find(data.names.begin(), data.names.end(), name);
                    img2save[id_2].images.push_back(data.images[pos - data.names.begin()].clone());
                    img2save[id_2].names.push_back(name);
                }

            }
            else {
                auto [combinded, merging_list, other] = occlusion_spilt(data.names, false);
                cout << "id: " << id << endl;
                cout << "combinded size: " << combinded.size() << endl;
                cout << "merging_list size: " << merging_list.size() << endl;
                vector<double> hsv_unmerged;
                vector<string> obj1_list, obj2_list;
                for (const auto& name : combinded) {
                    auto pos = std::find(data.names.begin(), data.names.end(), name);
                    hsv_unmerged.push_back(hsv_means[pos - data.names.begin()]);
                }
                std::sort(hsv_unmerged.begin(), hsv_unmerged.end());
                for (size_t i=0; i<hsv_unmerged.size(); i++) {
                    double hsv_dist_head = hsv_unmerged[i] - hsv_unmerged[0];
                    double hsv_dist_tail = hsv_unmerged.back() - hsv_unmerged[i];
                    if (hsv_dist_head > hsv_dist_tail) {
                        for (size_t j=0; j<i; j++)
                            obj1_list.push_back(combinded[j]);
                        for (size_t j=i; j<combinded.size(); j++)
                            obj2_list.push_back(combinded[j]);
                    }
                }
                for (const auto& name : obj1_list) {
                    auto pos = std::find(data.names.begin(), data.names.end(), name);
                    img2save[id_1].images.push_back(data.images[pos - data.names.begin()].clone());
                    img2save[id_1].names.push_back(name);
                }
                for (const auto& name : obj2_list) {
                    auto pos = std::find(data.names.begin(), data.names.end(), name);
                    img2save[id_2].images.push_back(data.images[pos - data.names.begin()].clone());
                    img2save[id_2].names.push_back(name);
                }
            }

            // 删除这组遮挡的id内容
            // img2save.erase(id);
            nonused_ids.push_back(id);
        }
    }
    // 安全删除方法
    auto it = img2save.begin();
    while (it != img2save.end()) {
        if (std::find(nonused_ids.begin(), nonused_ids.end(), it->first) != nonused_ids.end()) {
            // erase返回下一个有效迭代器，避免迭代器失效
            it = img2save.erase(it);
        } else {
            ++it;
        }
    }
}

void Tracker::setOutputFolder(const string& path) {
    output_image_folder = path;
    result_dir = output_image_folder + "/result";
    visualize_dir = output_image_folder + "/visualize";

    vector<std::string> dirs_to_create = {
        result_dir, visualize_dir
    };
    
    for (const auto& dir : dirs_to_create) {
        fs::create_directories(dir);
    }
}

void Tracker::reset() {
    // 重置追踪器状态
    lock_guard<mutex> lock(mtx);
    detected_flag = false;
    track_over_flag = false;
    first_frame_flag = true;
    undetected_frame_count = 0;
    invaild_num = 5;
    
    img2save.clear();
    last_frame_xy.clear();
    last_frame_info.clear();
    current_frame_info.clear();
    current_frame_xy.clear();
    visualize_images.clear();
    visualize_names.clear();
    
    tracked_id = 0;
    frame_count = 0;
}

void Tracker::reset_each_frame() {
    frame_logs.clear();
}

vector<cv::Mat> Tracker::getVisImages() {
    lock_guard<mutex> lock(mtx);
    return visualize_images;
}

vector<string> Tracker::getVisNames() {
    lock_guard<mutex> lock(mtx);
    return visualize_names;
}

std::unordered_map<int, ImageData> Tracker::getTrackResults() {
    lock_guard<mutex> lock(mtx);
    return img2save;
}

bool Tracker::is_track_over() {
    lock_guard<mutex> lock(mtx);
    return track_over_flag;
}

void Tracker::drawFrameLogs(cv::Mat& image, const vector<std::string>& logs) {
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.4;
    int thickness = 1;
    cv::Scalar color(255, 255, 255);
    int log_put_pos_x = log_put_start_x;
    int log_put_pos_y = log_put_start_y;

    for (const auto& line : logs) {
        cv::putText(image, line, cv::Point(log_put_pos_x, log_put_pos_y), fontFace, fontScale, color, thickness);
        log_put_pos_y += 10; // 每行间隔
        if (log_put_pos_y >= image.rows) {
            log_put_pos_y = log_put_start_y;
            log_put_pos_x += 260;
        }
    }
}

// 整体流程：
    // 循环等待有效帧
    // 初始化数据  ： 获得空白帧、以及空白帧灰度图
    // --循环....
    //    对每一帧做cv_process
    //    做完cv处理给到Tracking_group
    // --循环结束....
    // save results 处理结果
    // detected_flag、track_over_flag、tracked_id 重置
    // 传回有效数据，把数据交给server send处理，这里可能要注意数据结构以及处理过程中的数据保存

void Tracker::track(const cv::Mat& frame, bool visualize) {
    if (detected_flag) {  // 检测到有效帧后
        reset_each_frame();
        vector<TrackedData> cv_res = cv_process_frame(frame);
        if (cv_res.size() > 0) {
            // 检测到目标后，再执行tracking group
            tracking_group(frame, cv_res, visualize);
            undetected_frame_count = 0;
            // cout << "出现目标" << cv_res.size() << endl;
        }
        else {
            // 连续5帧未检测到目标，则认为下落结束
            undetected_frame_count++;
            if (undetected_frame_count == tolerance_undetected_num) {
                lock_guard<mutex> lock(mtx);
                track_over_flag = true;
                cout << "下落结束, 共 " << frame_count << " 帧" << endl;
            }
        }
    }
    else {  // 未检测到有效帧
        if (invaild_num > 0) {  // 跳过前几个无效帧
            blank_orig = frame;
            // first_frame_flag = false;
            invaild_num--;
            return;
        }

        detected_flag = detect_block(frame, blank_orig);
        
        if (detected_flag) {
            cv::cvtColor(blank_orig, blank_orig_gray, cv::COLOR_BGR2GRAY);
            cv::resize(blank_orig_gray, blank_resize_gray, cv::Size(640, 480), cv::INTER_NEAREST);
            blank_rect_gray = blank_resize_gray(
                cv::Rect(roi_x1, roi_y1, roi_x2-roi_x1, roi_y2-roi_y1)
            );

            vector<TrackedData> cv_res = cv_process_frame(frame);
            tracking_group(frame, cv_res, visualize);
            cout << "检测到有效帧" << endl;
        }
        else {
            blank_orig = frame;
            // cout << "未检测到有效帧" << endl;
        }
            
    }
    
}

vector<cv::Mat> Tracker::videoToFrames(const string& videoPath) {
    vector<cv::Mat> frames;
    
    // 打开视频文件
    cv::VideoCapture cap(videoPath);
    
    // 检查视频是否成功打开
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << videoPath << std::endl;
        return frames;
    }
    
    // 获取视频帧率和总帧数（可选）
    double fps = cap.get(cv::CAP_PROP_FPS);
    int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Video FPS: " << fps << ", Total frames: " << frameCount << std::endl;
    
    cv::Mat frame;
    // 循环读取每一帧
    while (cap.read(frame)) {
        // 复制当前帧到 vector
        frames.push_back(frame.clone());
        
        // 可选：显示当前帧
        // cv::imshow("Frame", frame);
        // if (cv::waitKey(25) >= 0) break;
    }
    
    // 释放资源
    cap.release();

    return frames;
}

void Tracker::track_video(const string& video_path) {
    // 读取视频，然后存成图片数组
    // 初始化数据：初始化保存路径
    
    // 创建输出目录
    string video_name = fs::path(video_path).stem().string();
    output_image_folder = fs::path(video_path).parent_path().parent_path() / "tracking_images" / video_name;
    result_dir = output_image_folder + "/result";
    visualize_dir = output_image_folder + "/visualize";

    vector<std::string> dirs_to_create = {
        result_dir, visualize_dir
    };
    
    for (const auto& dir : dirs_to_create) {
        fs::create_directories(dir);
    }

    vector<cv::Mat> frames = videoToFrames(video_path);

    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& frame : frames) {
        track(frame, true);
        if (track_over_flag) break;
    }
    save_results();
    reset();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Track 执行时间: " << duration << " 微秒" << std::endl;

}

// 提取文件名中的数字
int extractNumber(const std::string& filename) {
    std::string name = fs::path(filename).stem().string(); // 去掉扩展名
    return std::stoi(name);
}

// 自定义比较函数：按数字排序
bool naturalSort(const fs::path& a, const fs::path& b) {
    return extractNumber(a.string()) < extractNumber(b.string());
}
void Tracker::track_imgs(const string& img_folder) {
    std::vector<fs::path> imageFiles;

    // 创建输出目录
    string video_name = fs::path(img_folder).stem().string();
    output_image_folder = fs::path(img_folder).parent_path() / "tracking_images" / video_name;
    result_dir = output_image_folder + "/result";
    visualize_dir = output_image_folder + "/visualize";

    vector<std::string> dirs_to_create = {
        result_dir, visualize_dir
    };
    
    for (const auto& dir : dirs_to_create) {
        fs::create_directories(dir);
    }

    cout << "Processing images in " << img_folder << endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // 读取图片文件夹下的所有图片
    for (const auto& entry : fs::directory_iterator(img_folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            imageFiles.push_back(entry.path());
        }
    }
    // 按自然顺序排序
    std::sort(imageFiles.begin(), imageFiles.end(), naturalSort);
    
    for (const auto& entry : imageFiles) {
        string img_path = entry.string();
        cout << "Processing image: " << img_path << endl;
        cv::Mat frame = cv::imread(img_path);
        if (frame.empty()) {
            cout << "Error: Could not read image: " << img_path << endl;
            continue;
        }
        track(frame, true);
        if (track_over_flag) break;
    }
    save_results();
    reset();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Track 执行时间: " << duration << " 微秒" << std::endl;
}

void Tracker::track_video_folder(const string& video_folder) {
    // 读取视频文件夹下的所有视频
    for (const auto& entry : fs::directory_iterator(video_folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".avi") {
            string video_path = entry.path().string();
            std::cout << "Processing video: " << video_path << std::endl;
            track_video(video_path);
        }
    }
}

void Tracker::track_imgs_folder(const string& imgs_parent_folder) { 
    for (const auto& entry : fs::directory_iterator(imgs_parent_folder)) {
        if (entry.is_directory()) {
            string img_folder = entry.path().string();
            track_imgs(img_folder);
        }
    }

}

// int main() {
//     Tracker tracker;

//     string video_path = "/home/sunrise/qimeng3/dataset/videos/20250611222932_4.avi";
//     tracker.track_video(video_path);

//     return -1;
// }