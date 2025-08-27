#include "postprocess.hpp"

// 7. YOLO11-Detect 后处理
// 7. Postprocess
std::vector<DetectionResult> hb_postprocess(hbDNNTensor* output, const cv::Mat& img, int32_t input_H, int32_t input_W, int* order) {  
    std::vector<DetectionResult> results;

    float CONF_THRES_RAW = -log(1 / SCORE_THRESHOLD - 1);     // 利用反函数作用阈值，利用单调性筛选
    std::vector<std::vector<cv::Rect2d>> bboxes(CLASSES_NUM); // 每个id的xyhw 信息使用一个std::vector<cv::Rect2d>存储
    std::vector<std::vector<float>> scores(CLASSES_NUM);      // 每个id的score信息使用一个std::vector<float>存储
    std::vector<cv::Rect2d> all_boxes;
    std::vector<float> all_scores;
    std::vector<int> all_clsIds;
    static int32_t H_8 = input_H / 8;
    static int32_t H_16 = input_H / 16;
    static int32_t H_32 = input_H / 32;
    static int32_t W_8 = input_W / 8;
    static int32_t W_16 = input_W / 16;
    static int32_t W_32 = input_W / 32;

    auto begin_time = std::chrono::system_clock::now();

    // 7.1 小目标特征图
    // 7.1 Small Object Feature Map
    // output[order[0]]: (1, H // 8,  W // 8,  CLASSES_NUM)
    // output[order[1]]: (1, H // 8,  W // 8,  4 * REG)

    // 7.1.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    // 7.1.1 Check if the dequantization type complies with the bin model specification exported in the RDK Model Zoo README.
    if (output[order[0]].properties.quantiType != NONE)
    {
        std::cout << "output[order[0]] QuantiType is not NONE, please check!" << std::endl;
        return {};
    }
    if (output[order[1]].properties.quantiType != SCALE)
    {
        std::cout << "output[order[1]] QuantiType is not SCALE, please check!" << std::endl;
        return {};
    }

    // 7.1.3 将BPU推理完的内存地址转换为对应类型的指针
    // 7.1.3 Convert the memory address of BPU inference to a pointer of the corresponding type
    auto *s_cls_raw = reinterpret_cast<float *>(output[order[0]].sysMem[0].virAddr);
    auto *s_bbox_raw = reinterpret_cast<int32_t *>(output[order[1]].sysMem[0].virAddr);
    auto *s_bbox_scale = reinterpret_cast<float *>(output[order[1]].properties.scale.scaleData);
    for (int h = 0; h < H_8; h++)
    {
        for (int w = 0; w < W_8; w++)
        {
            // 7.1.4 取对应H和W位置的C通道, 记为数组的形式
            // cls对应CLASSES_NUM个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
            // bbox对应4个坐标乘以REG的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
            // 7.1.4 Get the C channel at the corresponding H and W positions, represented as an array.
            // cls corresponds to CLASSES_NUM raw score values, which are the values before Sigmoid calculation. Here, we use the monotonicity of the function to filter first, then calculate.
            // bbox corresponds to the raw values of 4 coordinates multiplied by REG, which are the values before DFL calculation. This part of the calculation is only performed if the score is qualified.
            float *cur_s_cls_raw = s_cls_raw;
            int32_t *cur_s_bbox_raw = s_bbox_raw;

            // 7.1.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
            // 7.1.5 Find the index of the maximum score value and discard if the maximum value is less than the threshold
            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++)
            {
                if (cur_s_cls_raw[i] > cur_s_cls_raw[cls_id])
                {
                    cls_id = i;
                }
            }

            // 7.1.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
            // 7.1.6 If not qualified, skip to avoid unnecessary dequantization, DFL and dist2bbox calculation
            if (cur_s_cls_raw[cls_id] < CONF_THRES_RAW)
            {
                s_cls_raw += CLASSES_NUM;
                s_bbox_raw += REG * 4;
                continue;
            }

            // 7.1.7 计算这个目标的分数
            // 7.1.7 Calculate the score of the target
            float score = 1 / (1 + std::exp(-cur_s_cls_raw[cls_id]));

            // 7.1.8 对bbox_raw信息进行反量化, DFL计算
            // 7.1.8 Dequantize bbox_raw information, DFL calculation
            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++)
            {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < REG; j++)
                {
                    int index_id = REG * i + j;
                    dfl = std::exp(float(cur_s_bbox_raw[index_id]) * s_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            // 7.1.9 剔除不合格的框   if(x1 >= x2 || y1 >=y2) continue;
            // 7.1.9 Remove unqualified boxes
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
            {
                s_cls_raw += CLASSES_NUM;
                s_bbox_raw += REG * 4;
                continue;
            }

            // 7.1.10 dist 2 bbox (ltrb 2 xyxy)
            float x1 = (w + 0.5 - ltrb[0]) * 8.0;
            float y1 = (h + 0.5 - ltrb[1]) * 8.0;
            float x2 = (w + 0.5 + ltrb[2]) * 8.0;
            float y2 = (h + 0.5 + ltrb[3]) * 8.0;

            // 7.1.11 对应类别加入到对应的std::vector中
            // 7.1.11 Add the corresponding class to the corresponding std::vector.
            // bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            // scores[cls_id].push_back(score);
            all_boxes.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            all_scores.push_back(score);
            all_clsIds.push_back(cls_id);

            s_cls_raw += CLASSES_NUM;
            s_bbox_raw += REG * 4;
        }
    }

    // 7.2 中目标特征图
    // 7.2 Media Object Feature Map
    // output[order[2]]: (1, H // 16,  W // 16,  CLASSES_NUM)
    // output[order[3]]: (1, H // 16,  W // 16,  4 * REG)

    // 7.2.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    // 7.2.1 Check if the dequantization type complies with the bin model specification exported in the RDK Model Zoo README.
    if (output[order[2]].properties.quantiType != NONE)
    {
        std::cout << "output[order[2]] QuantiType is not NONE, please check!" << std::endl;
        return {};
    }
    if (output[order[3]].properties.quantiType != SCALE)
    {
        std::cout << "output[order[3]] QuantiType is not SCALE, please check!" << std::endl;
        return {};
    }

    // 7.2.3 将BPU推理完的内存地址转换为对应类型的指针
    // 7.2.3 Convert the memory address of BPU inference to a pointer of the corresponding type
    auto *m_cls_raw = reinterpret_cast<float *>(output[order[2]].sysMem[0].virAddr);
    auto *m_bbox_raw = reinterpret_cast<int32_t *>(output[order[3]].sysMem[0].virAddr);
    auto *m_bbox_scale = reinterpret_cast<float *>(output[order[3]].properties.scale.scaleData);
    for (int h = 0; h < H_16; h++)
    {
        for (int w = 0; w < W_16; w++)
        {
            // 7.2.4 取对应H和W位置的C通道, 记为数组的形式
            // cls对应CLASSES_NUM个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
            // bbox对应4个坐标乘以REG的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
            // 7.2.4 Get the C channel at the corresponding H and W positions, represented as an array.
            // cls corresponds to CLASSES_NUM raw score values, which are the values before Sigmoid calculation. Here, we use the monotonicity of the function to filter first, then calculate.
            // bbox corresponds to the raw values of 4 coordinates multiplied by REG, which are the values before DFL calculation. This part of the calculation is only performed if the score is qualified.
            float *cur_m_cls_raw = m_cls_raw;
            int32_t *cur_m_bbox_raw = m_bbox_raw;

            // 7.2.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
            // 7.2.5 Find the index of the maximum score value and discard if the maximum value is less than the threshold
            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++)
            {
                if (cur_m_cls_raw[i] > cur_m_cls_raw[cls_id])
                {
                    cls_id = i;
                }
            }

            // 7.2.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
            // 7.2.6 If not qualified, skip to avoid unnecessary dequantization, DFL and dist2bbox calculation
            if (cur_m_cls_raw[cls_id] < CONF_THRES_RAW)
            {
                m_cls_raw += CLASSES_NUM;
                m_bbox_raw += REG * 4;
                continue;
            }

            // 7.2.7 计算这个目标的分数
            // 7.2.7 Calculate the score of the target
            float score = 1 / (1 + std::exp(-cur_m_cls_raw[cls_id]));

            // 7.2.8 对bbox_raw信息进行反量化, DFL计算
            // 7.2.8 Dequantize bbox_raw information, DFL calculation
            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++)
            {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < REG; j++)
                {
                    int index_id = REG * i + j;
                    dfl = std::exp(float(cur_m_bbox_raw[index_id]) * m_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            // 7.2.9 剔除不合格的框   if(x1 >= x2 || y1 >=y2) continue;
            // 7.2.9 Remove unqualified boxes
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
            {
                m_cls_raw += CLASSES_NUM;
                m_bbox_raw += REG * 4;
                continue;
            }

            // 7.2.10 dist 2 bbox (ltrb 2 xyxy)
            float x1 = (w + 0.5 - ltrb[0]) * 16.0;
            float y1 = (h + 0.5 - ltrb[1]) * 16.0;
            float x2 = (w + 0.5 + ltrb[2]) * 16.0;
            float y2 = (h + 0.5 + ltrb[3]) * 16.0;

            // 7.2.11 对应类别加入到对应的std::vector中
            // 7.2.11 Add the corresponding class to the corresponding std::vector.
            // bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            // scores[cls_id].push_back(score);
            all_boxes.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            all_scores.push_back(score);
            all_clsIds.push_back(cls_id);

            m_cls_raw += CLASSES_NUM;
            m_bbox_raw += REG * 4;
        }
    }

    // 7.3 大目标特征图
    // 7.3 Big Object Feature Map
    // output[order[4]]: (1, H // 32,  W // 32,  CLASSES_NUM)
    // output[order[5]]: (1, H // 32,  W // 32,  4 * REG)

    // 7.3.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    // 7.3.1 Check if the dequantization type complies with the bin model specification exported in the RDK Model Zoo README.
    if (output[order[4]].properties.quantiType != NONE)
    {
        std::cout << "output[order[4]] QuantiType is not NONE, please check!" << std::endl;
        return {};
    }
    if (output[order[5]].properties.quantiType != SCALE)
    {
        std::cout << "output[order[5]] QuantiType is not SCALE, please check!" << std::endl;
        return {};
    }

    // 7.3.3 将BPU推理完的内存地址转换为对应类型的指针
    // 7.3.3 Convert the memory address of BPU inference to a pointer of the corresponding type
    auto *l_cls_raw = reinterpret_cast<float *>(output[order[4]].sysMem[0].virAddr);
    auto *l_bbox_raw = reinterpret_cast<int32_t *>(output[order[5]].sysMem[0].virAddr);
    auto *l_bbox_scale = reinterpret_cast<float *>(output[order[5]].properties.scale.scaleData);
    for (int h = 0; h < H_32; h++)
    {
        for (int w = 0; w < W_32; w++)
        {
            // 7.3.4 取对应H和W位置的C通道, 记为数组的形式
            // cls对应CLASSES_NUM个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
            // bbox对应4个坐标乘以REG的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
            // 7.3.4 Get the C channel at the corresponding H and W positions, represented as an array.
            // cls corresponds to CLASSES_NUM raw score values, which are the values before Sigmoid calculation. Here, we use the monotonicity of the function to filter first, then calculate.
            // bbox corresponds to the raw values of 4 coordinates multiplied by REG, which are the values before DFL calculation. This part of the calculation is only performed if the score is qualified.
            float *cur_l_cls_raw = l_cls_raw;
            int32_t *cur_l_bbox_raw = l_bbox_raw;

            // 7.3.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
            // 7.3.5 Find the index of the maximum score value and discard if the maximum value is less than the threshold
            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++)
            {
                if (cur_l_cls_raw[i] > cur_l_cls_raw[cls_id])
                {
                    cls_id = i;
                }
            }

            // 7.3.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
            // 7.3.6 If not qualified, skip to avoid unnecessary dequantization, DFL and dist2bbox calculation
            if (cur_l_cls_raw[cls_id] < CONF_THRES_RAW)
            {
                l_cls_raw += CLASSES_NUM;
                l_bbox_raw += REG * 4;
                continue;
            }

            // 7.3.7 计算这个目标的分数
            // 7.3.7 Calculate the score of the target
            float score = 1 / (1 + std::exp(-cur_l_cls_raw[cls_id]));

            // 7.3.8 对bbox_raw信息进行反量化, DFL计算
            // 7.3.8 Dequantize bbox_raw information, DFL calculation
            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++)
            {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < REG; j++)
                {
                    int index_id = REG * i + j;
                    dfl = std::exp(float(cur_l_bbox_raw[index_id]) * l_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            // 7.3.9 剔除不合格的框   if(x1 >= x2 || y1 >=y2) continue;
            // 7.3.9 Remove unqualified boxes
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
            {
                l_cls_raw += CLASSES_NUM;
                l_bbox_raw += REG * 4;
                continue;
            }

            // 7.3.10 dist 2 bbox (ltrb 2 xyxy)
            float x1 = (w + 0.5 - ltrb[0]) * 32.0;
            float y1 = (h + 0.5 - ltrb[1]) * 32.0;
            float x2 = (w + 0.5 + ltrb[2]) * 32.0;
            float y2 = (h + 0.5 + ltrb[3]) * 32.0;

            // 7.3.11 对应类别加入到对应的std::vector中
            // 7.3.11 Add the corresponding class to the corresponding std::vector.
            // bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            // scores[cls_id].push_back(score);
            all_boxes.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            all_scores.push_back(score);
            all_clsIds.push_back(cls_id);

            l_cls_raw += CLASSES_NUM;
            l_bbox_raw += REG * 4;
        }
    }

    // 7.4 对每一个类别进行NMS
    // 7.4 NMS
    std::vector<int> all_indices;
    // std::vector<std::vector<int>> indices(CLASSES_NUM);
    // for (int i = 0; i < CLASSES_NUM; i++)
    // {
    //     cv::dnn::NMSBoxes(bboxes[i], scores[i], SCORE_THRESHOLD, NMS_THRESHOLD, indices[i], 1.f, NMS_TOP_K);
    // }
    cv::dnn::NMSBoxes(all_boxes, all_scores, SCORE_THRESHOLD, NMS_THRESHOLD, all_indices, 1.f, NMS_TOP_K);

    std::cout << "\033[31m Post Process time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

    for (std::vector<int>::iterator it = all_indices.begin(); it != all_indices.end(); ++it) {
        bboxes[all_clsIds[*it]].push_back(all_boxes[*it]);
        scores[all_clsIds[*it]].push_back(all_scores[*it]);

        DetectionResult result;
        result.id = all_clsIds[*it];
        result.rect = all_boxes[*it];
        result.score = all_scores[*it];
        results.push_back(result);
    }

    // 8. 渲染
    // 8. Render
    begin_time = std::chrono::system_clock::now();
    for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++)
    {
        // 8.1 每一个类别分别渲染
        // 8.1 Render for each class
        size_t box_num = bboxes[cls_id].size();
        for (int i = 0; i < box_num; i++)
        {
            // 8.2 获取基本的 bbox 信息
            // 8.2 Get basic bbox information
            float x1 = bboxes[cls_id][i].x;
            float y1 = bboxes[cls_id][i].y;
            float x2 = x1 + bboxes[cls_id][i].width;
            float y2 = y1 + bboxes[cls_id][i].height;
            float score = scores[cls_id][i];
            // std::string name = object_names[cls_id % CLASSES_NUM];

            // 8.3 绘制矩形
            // 8.3 Draw rect
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), LINE_SIZE);

            // 8.4 绘制字体
            // 8.4 Draw text
            std::string text = to_string(cls_id) + ": " + std::to_string(static_cast<int>(score * 100)) + "%";
            cv::putText(img, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, cv::Scalar(0, 0, 255), FONT_THICKNESS, cv::LINE_AA);

            // 8.5 打印检测信息
            // 8.5 Print detection information
            std::cout << "(" << x1 << " " << y1 << " " << x2 << " " << y2 << "): \t" << text << std::endl;
        }
    }
    std::cout << "\033[31m Draw Result time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;
    
    // 保存结果
    cv::imwrite("result.jpg", img);
    cout << "Result saved to result.jpg" << endl;

    // for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++) {

    //     for (std::vector<int>::iterator it = indices[cls_id].begin(); it!=indices[cls_id].end(); ++it) {
    //         DetectionResult result;
    //         result.id = cls_id;
    //         result.rect = bboxes[cls_id][*it];
    //         result.score = scores[cls_id][*it];
    //         results.push_back(result);
    //     }

    // }

    return results;
}