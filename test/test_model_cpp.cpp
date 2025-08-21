/**
 * @file        test_model_cpp.cpp
 * @brief       C++版本的触摸模型预测器测试脚本
 * @description 用于验证模型预测结果并与Python版本进行比较
 */

#include "TouchModelPredictor.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cstring>

// 创建测试用的触摸帧
TouchFrame createTestFrame(uint32_t tx_count = 18, uint32_t rx_count = 36, uint32_t value = 100) {
    TouchFrame frame;
    frame.txn = tx_count;
    frame.rxn = rx_count;
    frame.size = tx_count * rx_count;
    frame.timestamp = 0;
    frame.frame_id = 1;
    
    // 分配内存
    frame.data = new int32_t[frame.size];
    
    // 填充测试数据
    for (uint32_t i = 0; i < tx_count; ++i) {
        for (uint32_t j = 0; j < rx_count; ++j) {
            uint32_t idx = i * rx_count + j;
            if ((i + j) % 3 == 0) {
                frame.data[idx] = static_cast<int32_t>(value * 1.5);
            } else if ((i + j) % 5 == 0) {
                frame.data[idx] = static_cast<int32_t>(value * 0.5);
            } else {
                frame.data[idx] = static_cast<int32_t>(value);
            }
        }
    }
    
    return frame;
}

// 打印帧数据样本
void printFrameSample(const TouchFrame& frame, const std::string& label) {
    std::cout << label << " (形状: " << frame.txn << "x" << frame.rxn << "):\n";
    
    // 查找最小值和最大值
    uint32_t min_val = UINT32_MAX;
    uint32_t max_val = 0;
    for (uint32_t i = 0; i < frame.size; ++i) {
        if (frame.data[i] < min_val) min_val = frame.data[i];
        if (frame.data[i] > max_val) max_val = frame.data[i];
    }
    std::cout << "数据范围: " << min_val << " - " << max_val << std::endl;
    
    // 打印前3行3列的样本
    std::cout << "数据样本:\n";
    for (uint32_t i = 0; i < std::min(frame.txn, 3u); ++i) {
        for (uint32_t j = 0; j < std::min(frame.rxn, 3u); ++j) {
            std::cout << std::setw(6) << frame.data[i * frame.rxn + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// 比较C++和Python结果
void compareToPython(const TouchFrame& predicted_frame) {
    // 尝试读取Python生成的测试输出文件
    std::cout << "尝试加载Python测试结果进行比较..." << std::endl;
    
    // 这里我们加载同目录下的 `test_input.npy` 和 `test_output.npy`
    std::string input_file = "test_input.npy";
    std::string output_file = "test_output.npy";
    if (!std::ifstream(input_file).good() || !std::ifstream(output_file).good()) {
        std::cerr << "错误: 无法找到Python测试输入或输出文件，请确保它们存在于当前目录" << std::endl;
        return;
    }
    // 读取Python生成的输入数据
    std::ifstream input_stream(input_file, std::ios::binary);
    std::vector<int32_t> python_input_data(predicted_frame.size);
    input_stream.read(reinterpret_cast<char*>(python_input_data.data()), predicted_frame.size * sizeof(int32_t));
    input_stream.close();
    // 读取Python生成的输出数据
    std::ifstream output_stream(output_file, std::ios::binary);
    std::vector<int32_t> python_output_data(predicted_frame.size);
    output_stream.read(reinterpret_cast<char*>(python_output_data.data()), predicted_frame.size * sizeof(int32_t));
    output_stream.close();

    // 进行比较
    bool match = true;
    for (uint32_t i = 0; i < predicted_frame.size; ++i) {
        if (predicted_frame.data[i] != python_output_data[i]) {
            match = false;
            std::cout << "不匹配: C++预测[" << i << "] = " << predicted_frame.data[i]
                      << ", Python输出[" << i << "] = " << python_output_data[i] << std::endl;
        }
    }
    if (match) {
        std::cout << "C++预测结果与Python输出完全匹配！" << std::endl;
    } else {
        std::cout << "C++预测结果与Python输出不匹配，请检查数据处理逻辑。" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "请手动与Python脚本输出的结果进行比较" << std::endl;
}

// 调试帧到张量的转换
void debugFrameToTensor(const TouchModelPredictor& predictor, const TouchFrame& frame) {
    std::cout << "DEBUG: 检查frameToTensor内部实现\n";
    
    // 打印输入帧的一些详细信息
    std::cout << "输入帧尺寸: " << frame.txn << "x" << frame.rxn << std::endl;
    std::cout << "输入帧前几个值: ";
    for (uint32_t i = 0; i < std::min(frame.size, 5u); ++i) {
        std::cout << frame.data[i] << " ";
    }
    std::cout << std::endl;
    
    // 模拟frameToTensor的实现
    std::vector<float> buffer(frame.size);
    for (uint32_t i = 0; i < frame.size; ++i) {
        buffer[i] = static_cast<float>(frame.data[i]) / 1000.0f;
    }
    
    // 打印归一化后的值
    std::cout << "归一化后的前几个值: ";
    for (uint32_t i = 0; i < std::min(frame.size, 5u); ++i) {
        std::cout << buffer[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "========== C++触摸模型预测器测试 ==========" << std::endl;
    
    // 确定模型路径
    std::string model_path;
    if (argc > 1) {
        model_path = argv[1];
    } else {
        // 尝试查找build/models目录下的模型
        model_path = "build/models/latest_model.pt";
        if (!std::ifstream(model_path).good()) {
            model_path = "build/models/touch_filter_model.pt";
            if (!std::ifstream(model_path).good()) {
                std::cerr << "错误: 未找到默认模型文件，请指定模型路径作为参数" << std::endl;
                return 1;
            }
        }
    }
    
    std::cout << "使用模型: " << model_path << std::endl;
    
    // 创建预测器
    auto predictor = createTouchModelPredictor(model_path);
    if (!predictor->isModelLoaded()) {
        std::cerr << "错误: 模型加载失败" << std::endl;
        return 1;
    }
    
    std::cout << "模型已成功加载" << std::endl;
    
    // 创建测试帧 - 与Python测试脚本相同的参数
    TouchFrame test_frame = createTestFrame();
    std::cout << "已创建测试帧 (" << test_frame.txn << "x" << test_frame.rxn << ")" << std::endl;
    
    // 打印输入帧样本
    printFrameSample(test_frame, "输入数据");
    
    // 调试帧到张量的转换
    debugFrameToTensor(*predictor, test_frame);
    
    // 执行预测
    std::cout << "执行预测..." << std::endl;
    try {
        TouchFrame predicted_frame = predictor->predict(test_frame);
        
        // 打印预测结果
        printFrameSample(predicted_frame, "预测结果");
        
        // 比较与Python结果
        compareToPython(predicted_frame);
        
        // 释放内存
        delete[] predicted_frame.data;
    } catch (const std::exception& e) {
        std::cerr << "预测失败: " << e.what() << std::endl;
    }
    
    // 释放测试帧内存
    delete[] test_frame.data;
    
    std::cout << "========== 测试完成 ==========" << std::endl;
    return 0;
}