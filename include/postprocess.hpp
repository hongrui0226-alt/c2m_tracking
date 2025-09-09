#ifndef POSTPROCESS_HPP
#define POSTPROCESS_HPP

// 函数声明或定义

#endif // POSTPROCESS_HPP

// C/C++ Standard Librarys
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

// Thrid Party Librarys
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "dnn/hb_dnn.h"

// 模型的类别数量, 默认80
// Number of classes in the model, default is 80
#define CLASSES_NUM 1
// NMS的阈值, 默认0.45
// Non-Maximum Suppression (NMS) threshold, default is 0.45
#define NMS_THRESHOLD 0.2
// 分数阈值, 默认0.25
// Score threshold, default is 0.25
#define SCORE_THRESHOLD 0.1
#define NMS_TOP_K 300
// 控制回归部分离散化程度的超参数, 默认16
// A hyperparameter that controls the discretization level of the regression part, default is 16
#define REG 16
// 绘制标签的字体尺寸, 默认1.0
// Font size for drawing labels, default is 1.0.
#define FONT_SIZE 1.0
// 绘制标签的字体粗细, 默认 1.0
// Font thickness for drawing labels, default is 1.0.
#define FONT_THICKNESS 1.0
// 绘制矩形框的线宽, 默认2.0
// Line width for drawing bounding boxes, default is 2.0.
#define LINE_SIZE 2.0

using namespace std;

struct DetectionResult {
    cv::Rect2d rect;
    int id;
    float score;
};

// 7. YOLO11-Detect 后处理
// 7. Postprocess
std::vector<DetectionResult> hb_postprocess(hbDNNTensor* output, const cv::Mat& img, int32_t input_H, int32_t input_W, int* order);