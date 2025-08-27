#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP

// 函数声明或定义

#endif // PREPROCESS_HPP

// C/C++ Standard Librarys
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

// Thrid Party Librarys
#include <opencv2/opencv.hpp>

// 前处理方式选择, 0:Resize, 1:LetterBox
// Preprocessing method selection, 0: Resize, 1: LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

using namespace std;

cv::Mat letterbox(const cv::Mat& img, int32_t input_H, int32_t input_W);
cv::Mat hb_preprocess(const cv::Mat& img, int32_t input_H, int32_t input_W);
