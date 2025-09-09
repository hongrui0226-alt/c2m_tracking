#include "preprocess.hpp"
cv::Mat letterbox(const cv::Mat& img, int32_t input_H, int32_t input_W) {
    // 3.2 前处理
    // 3.2 Preprocess
    float y_scale = 1.0;
    float x_scale = 1.0;
    int x_shift = 0;
    int y_shift = 0;
    cv::Mat resize_img;
    if (PREPROCESS_TYPE == LETTERBOX_TYPE) // letter box
    {
        auto begin_time = std::chrono::system_clock::now();
        x_scale = std::min(1.0 * input_H / img.rows, 1.0 * input_W / img.cols);
        y_scale = x_scale;
        if (x_scale <= 0 || y_scale <= 0)
        {
            throw std::runtime_error("Invalid scale factor.");
        }

        int new_w = img.cols * x_scale;
        x_shift = (input_W - new_w) / 2;
        int x_other = input_W - new_w - x_shift;

        int new_h = img.rows * y_scale;
        y_shift = (input_H - new_h) / 2;
        int y_other = input_H - new_h - y_shift;

        cv::Size targetSize(new_w, new_h);
        cv::resize(img, resize_img, targetSize);
        cv::copyMakeBorder(resize_img, resize_img, y_shift, y_other, x_shift, x_other, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        // cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);

        // std::cout << "\033[31m pre process (LetterBox) time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;
    }
    else if (PREPROCESS_TYPE == RESIZE_TYPE) // resize
    {
        auto begin_time = std::chrono::system_clock::now();

        cv::Size targetSize(input_W, input_H);
        cv::resize(img, resize_img, targetSize);
        // cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);

        y_scale = 1.0 * input_H / img.rows;
        x_scale = 1.0 * input_W / img.cols;
        y_shift = 0;
        x_shift = 0;

        // std::cout << "\033[31m pre process (Resize) time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;
    }
    // std::cout << "y_scale = " << y_scale << ", ";
    // std::cout << "x_scale = " << x_scale << std::endl;
    // std::cout << "y_shift = " << y_shift << ", ";
    // std::cout << "x_shift = " << x_shift << std::endl;

    return resize_img;
}

cv::Mat hb_preprocess(const cv::Mat& img, int32_t input_H, int32_t input_W) {
    cv::Mat img_tensor = letterbox(img, input_H, input_W);
    img_tensor.convertTo(img_tensor, CV_32FC3, 1.0 / 255);
    return img_tensor;
}
