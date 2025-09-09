// RDK BPU libDNN API
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"

// C/C++ Standard Librarys
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

// Custom Pre/Post Process
#include "preprocess.hpp"
#include "postprocess.hpp"

// Third Party Libraries
#include "opencv2/opencv.hpp"

using namespace std;

#define RDK_CHECK_SUCCESS(value, errmsg)                                         \
    do                                                                           \
    {                                                                            \
        auto ret_code = value;                                                   \
        if (ret_code != 0)                                                       \
        {                                                                        \
            std::cout << "[ERROR] " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cout << errmsg << ", error code:" << ret_code << std::endl;     \
        }                                                                        \
    } while (0);


class HBInfer {
private:
    hbPackedDNNHandle_t packed_dnn_handle_;
    hbDNNHandle_t dnn_handle_;
    int32_t input_count_, output_count_;
    int32_t input_H_, input_W_;
    hbDNNTensorProperties input_properties_;
    hbDNNTaskHandle_t task_handle_ = nullptr;
    hbDNNTensor input_;
    hbDNNTensor *output_ = new hbDNNTensor[output_count_];
    int order[6] = {0, 1, 2, 3, 4, 5}; // 非通用变量

public:
    HBInfer() {

    }

    void load_model(const std::string &model_path) {
        // 0. 加载bin模型
        auto begin_time = std::chrono::system_clock::now();

        const char *model_file_name = model_path.c_str();
        RDK_CHECK_SUCCESS(
            hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_file_name, 1),
            "hbDNNInitializeFromFiles failed");

        std::cout << "\033[31m Load D-Robotics Quantize model time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

        // 1. 打印相关版本信息
        // std::cout << "OpenCV build details: " << cv::getBuildInformation() << std::endl;
        std::cout << "[INFO] MODEL_PATH: " << model_path << std::endl;
        std::cout << "[INFO] CLASSES_NUM: " << CLASSES_NUM << std::endl;
        std::cout << "[INFO] NMS_THRESHOLD: " << NMS_THRESHOLD << std::endl;
        std::cout << "[INFO] SCORE_THRESHOLD: " << SCORE_THRESHOLD << std::endl;

        // 2.1 模型名称
        // 2.1 Model names
        const char **model_name_list;
        int model_count = 0;
        RDK_CHECK_SUCCESS(
            hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle_),
            "hbDNNGetModelNameList failed");

        // 如果这个bin模型有多个打包，则只使用第一个，一般只有一个
        // If this bin model has multiple packages, only the first one is used, usually there is only one.
        if (model_count > 1)
        {
            std::cout << "This model file have more than 1 model, only use model 0.";
        }
        const char *model_name = model_name_list[0];
        std::cout << "[model name]: " << model_name << std::endl;

        // 2.2 获得Packed模型的第一个模型的handle
        // 2.2 Get the handle of the first model in the packed model
        RDK_CHECK_SUCCESS(
            hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name),
            "hbDNNGetModelHandle failed");

        // 2.3 模型输入检查
        // 2.3 Model input check
        RDK_CHECK_SUCCESS(
            hbDNNGetInputCount(&input_count_, dnn_handle_),
            "hbDNNGetInputCount failed");

        
        RDK_CHECK_SUCCESS(
            hbDNNGetInputTensorProperties(&input_properties_, dnn_handle_, 0),
            "hbDNNGetInputTensorProperties failed");

        // 2.3.1 D-Robotics YOLO11 *.bin 模型应该为单输入
        // 2.3.1 D-Robotics YOLO11 *.bin model should have only one input
        if (input_count_ > 1)
        {
            std::cout << "Your Model have more than 1 input, please check!" << std::endl;
            return;
        }

        // // 2.3.2 D-Robotics YOLO11 *.bin 模型输入Tensor类型应为RGB
        // // tensor type: HB_DNN_IMG_TYPE_RGB
        // if (input_properties_.tensorType == HB_DNN_IMG_TYPE_RGB)
        // {
        //     std::cout << "input tensor type: HB_DNN_IMG_TYPE_RGB" << std::endl;
        // }
        // else
        // {
        //     std::cout << "input tensor type is not HB_DNN_IMG_TYPE_RGB, please check!" << std::endl;
        //     return;
        // }

        // // 2.3.3 D-Robotics YOLO11 *.bin 模型输入Tensor数据排布应为NCHW
        // // tensor layout: HB_DNN_LAYOUT_NCHW
        // std::cout << "input_properties.tensorType: " << input_properties_.tensorType << std::endl;
        // if (input_properties_.tensorLayout == HB_DNN_LAYOUT_NCHW)
        // {
        //     std::cout << "input tensor layout: HB_DNN_LAYOUT_NCHW" << std::endl;
        // }
        // else
        // {
        //     std::cout << "input tensor layout is not HB_DNN_LAYOUT_NCHW, please check!" << std::endl;
        //     return;
        // }

        // 2.3.4 D-Robotics YOLO11 *.bin 模型输入Tensor数据的valid shape应为(1,3,H,W)
        // valid shape: (1,3,640,640)
        if (input_properties_.validShape.numDimensions == 4)
        {
            input_H_ = input_properties_.validShape.dimensionSize[1];
            input_W_ = input_properties_.validShape.dimensionSize[2];
            std::cout << "input tensor valid shape: (" << input_properties_.validShape.dimensionSize[0];
            std::cout << ", " << input_properties_.validShape.dimensionSize[3];
            std::cout << ", " << input_H_;
            std::cout << ", " << input_W_ << ")" << std::endl;
        }
        else
        {
            std::cout << "input tensor validShape.numDimensions is not 4 such as (1,3,640,640), please check!" << std::endl;
            return;
        }

        // 2.4 模型输出检查
        // 2.4 Model output check
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputCount(&output_count_, dnn_handle_),
            "hbDNNGetOutputCount failed");

        // 2.4.1 D-Robotics YOLO11 *.bin 模型应该有6个输出
        // 2.4.1 D-Robotics YOLO11 *.bin model should have 6 outputs
        if (output_count_ == 6)
        {
            for (int i = 0; i < 6; i++)
            {
                hbDNNTensorProperties output_properties;
                RDK_CHECK_SUCCESS(
                    hbDNNGetOutputTensorProperties(&output_properties, dnn_handle_, i),
                    "hbDNNGetOutputTensorProperties failed");
                std::cout << "output[" << i << "] ";
                std::cout << "valid shape: (" << output_properties.validShape.dimensionSize[0];
                std::cout << ", " << output_properties.validShape.dimensionSize[1];
                std::cout << ", " << output_properties.validShape.dimensionSize[2];
                std::cout << ", " << output_properties.validShape.dimensionSize[3] << "), ";
                if (output_properties.quantiType == SHIFT)
                    std::cout << "QuantiType: SHIFT" << std::endl;
                if (output_properties.quantiType == SCALE)
                    std::cout << "QuantiType: SCALE" << std::endl;
                if (output_properties.quantiType == NONE)
                    std::cout << "QuantiType: NONE" << std::endl;
            }
        }
        else
        {
            std::cout << "Your Model's outputs num is not 6, please check!" << std::endl;
            return;
        }

        // 2.4.2 可选：调整输出头顺序的映射
        // 2.4.2 Adjust the mapping of output order

        int32_t H_8 = input_H_ / 8;
        int32_t H_16 = input_H_ / 16;
        int32_t H_32 = input_H_ / 32;
        int32_t W_8 = input_W_ / 8;
        int32_t W_16 = input_W_ / 16;
        int32_t W_32 = input_W_ / 32;
        int32_t order_we_want[6][3] = {
            {H_8, W_8, CLASSES_NUM},   // output[order[3]]: (1, H // 8,  W // 8,  CLASSES_NUM)
            {H_8, W_8, 64},            // output[order[0]]: (1, H // 8,  W // 8,  64)
            {H_16, W_16, CLASSES_NUM}, // output[order[4]]: (1, H // 16, W // 16, CLASSES_NUM)
            {H_16, W_16, 64},          // output[order[1]]: (1, H // 16, W // 16, 64)
            {H_32, W_32, CLASSES_NUM}, // output[order[5]]: (1, H // 32, W // 32, CLASSES_NUM)
            {H_32, W_32, 64},          // output[order[2]]: (1, H // 32, W // 32, 64)
        };
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                hbDNNTensorProperties output_properties;
                RDK_CHECK_SUCCESS(
                    hbDNNGetOutputTensorProperties(&output_properties, dnn_handle_, j),
                    "hbDNNGetOutputTensorProperties failed");
                int32_t h = output_properties.validShape.dimensionSize[1];
                int32_t w = output_properties.validShape.dimensionSize[2];
                int32_t c = output_properties.validShape.dimensionSize[3];
                if (h == order_we_want[i][0] && w == order_we_want[i][1] && c == order_we_want[i][2])
                {
                    order[i] = j;
                    break;
                }
            }
        }

        // 2.4.3 可选：打印并检查调整后的输出头顺序的映射
        // 2.4.3 Print and check the mapping of output order
        if (order[0] + order[1] + order[2] + order[3] + order[4] + order[5] == 0 + 1 + 2 + 3 + 4 + 5)
        {
            std::cout << "Outputs order check SUCCESS, continue." << std::endl;
            std::cout << "order = {";
            for (int i = 0; i < 6; i++)
            {
                std::cout << order[i] << ", ";
            }
            std::cout << "}" << std::endl;
        }
        else
        {
            std::cout << "Outputs order check FAILED, use default" << std::endl;
            for (int i = 0; i < 6; i++)
                order[i] = i;
        }

    }

    void prepare_input(const cv::Mat& img_data) {
        // 3.4 cv::Mat的NHWC-BGR888格式转为NCHW-RGB888格式
        // 3.4 Convert NHWC-BGR888 to NCHW-RGB888
        auto begin_time = std::chrono::system_clock::now();
        cv::Mat img_tensor = hb_preprocess(img_data, input_H_, input_W_);
        
        if (!img_tensor.isContinuous()) {
            cout << "img_tensor is not continuous" << endl;
            img_tensor = img_tensor.clone(); // 确保数据连续
        }
        float* img_ptr = reinterpret_cast<float *>(img_tensor.data);
        float* input_ptr = reinterpret_cast<float *>(input_.sysMem[0].virAddr);
        int32_t data_size = input_H_ * input_W_ * 3 * 4;
        memcpy(input_ptr, img_ptr, data_size);

        // std::cout << "\033[31m (u8-128)->s8 time = " << 
        //     std::fixed << std::setprecision(2) << 
        //     std::chrono::duration_cast<std::chrono::microseconds>
        //     (std::chrono::system_clock::now() - begin_time).count() / 1000.0 
        //     << " ms\033[0m" << std::endl;

        hbSysFlushMem(&input_.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    }

    void allocCachedMem() { 

        // 准备模型输入数据的空间
        // Prepare the space for model input data
        input_.properties = input_properties_;
        hbSysAllocCachedMem(&input_.sysMem[0], int(4 * 3 * input_H_ * input_W_));

        // 准备模型输出数据的空间
        // Prepare the space for model output data
        for (int i = 0; i < 6; i++)
        {
            hbDNNTensorProperties &output_properties = output_[i].properties;
            hbDNNGetOutputTensorProperties(&output_properties, dnn_handle_, i);
            int out_aligned_size = output_properties.alignedByteSize;
            hbSysMem &mem = output_[i].sysMem[0];
            hbSysAllocCachedMem(&mem, out_aligned_size);
        }
    }

    void infer(const cv::Mat& img_data) {
        // Prepare input data
        prepare_input(img_data);

        auto begin_time = std::chrono::system_clock::now();
        // 5. 推理模型
        // 5. Inference
        hbDNNInferCtrlParam infer_ctrl_param;
        HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
        hbDNNInfer(&task_handle_, &output_, &input_, dnn_handle_, &infer_ctrl_param);
        
        // 6. 等待任务结束
        // 6. Wait for task to finish
        hbDNNWaitTaskDone(task_handle_, 0);

        // 7.1.2 对缓存的BPU内存进行刷新
        // 7.1.2 Flush the cached BPU memory
        hbSysFlushMem(&(output_[order[0]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output_[order[1]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output_[order[2]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output_[order[3]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output_[order[4]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output_[order[5]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

        // std::cout << "\033[31m forward time = " << 
        // std::fixed << std::setprecision(2) << 
        // std::chrono::duration_cast<std::chrono::microseconds>
        // (std::chrono::system_clock::now() - begin_time).count() / 1000.0 
        // << " ms\033[0m" << std::endl;
    }

    vector<DetectionResult> postprocess(const cv::Mat& img) { 
        cv::Mat resize_img = letterbox(img, input_H_, input_W_);
        return hb_postprocess(output_, resize_img, input_H_, input_W_, order);
    }

    void releaseTask() {
        hbDNNReleaseTask(task_handle_);
        task_handle_ = nullptr;
    }

    void releaseMem() {
        hbSysFreeMem(&(input_.sysMem[0]));
        hbSysFreeMem(&(output_->sysMem[0]));
    }

    void releaseModel() {
        hbDNNRelease(packed_dnn_handle_);
    }

};