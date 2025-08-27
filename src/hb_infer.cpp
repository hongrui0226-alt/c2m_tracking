// Custom Librarys
#include "hb_infer.hpp"

// Thrid Party Librarys
#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>

// C/C++ Standard Librarys
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;

DEFINE_string(image_path, "/home/sunrise/test.png", "Path of image that you want to infer");

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    string model_path = "yolo11n-modified.bin";
    cv::Mat img_data = cv::imread(FLAGS_image_path);

    HBInfer yolo11n_infer;
    
    // Load model as initialization
    yolo11n_infer.load_model(model_path);
    // Allocate memory for input and output tensors
    yolo11n_infer.allocCachedMem();

    for (int i = 0; i < 10; i++)
    {
        cout << "Infer time: " << i << ": " << endl;
        
        cv::Mat img_copy;
        img_data.copyTo(img_copy);
        // cout << "copy addr:" << img_copy.data << endl;

        // Infer that includes preprocess
        yolo11n_infer.infer(img_copy);
        // Postprocess
        yolo11n_infer.postprocess(img_copy);
        // Release task handle each infer time
        yolo11n_infer.releaseTask();    
    }
    
    // Release model handle after all infer time
    yolo11n_infer.releaseMem();
    yolo11n_infer.releaseModel();
}