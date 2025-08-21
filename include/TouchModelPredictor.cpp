/**
 * @file        TouchModelPredictor.cpp
 * @brief       触摸模型预测器实现
 * @version     0.1
 * @date        2025-07-15
 */

#include "TouchModelPredictor.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>

// 检查CUDA是否可用的辅助函数
bool isCudaAvailable() {
#ifdef USE_CUDA
    return torch::cuda::is_available();
#else
    return false;
#endif
}

// 构造函数 - 初始化预测器并加载模型
TouchModelPredictor::TouchModelPredictor(const std::string& model_path)
    : model_path_(model_path), model_loaded_(false), use_cuda_(false) {
    // 尝试加载模型
    model_loaded_ = loadModel(model_path);
}

// 加载模型
bool TouchModelPredictor::loadModel(const std::string& model_path) {
    try {
        // 始终在CPU上加载模型，避免CUDA兼容性问题
        torch::Device device(torch::kCPU);
        
        // 使用map_location确保模型在CPU上加载
        model_ = torch::jit::load(model_path, device);
        
        // 设置为评估模式
        model_.eval();
        
        std::cout << "模型已加载到CPU设备" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "加载模型失败: " << e.what() << std::endl;
        return false;
    }
}

// 预测单帧触摸数据
TouchFrame TouchModelPredictor::predict(const TouchFrame& input_frame) {
    if (!model_loaded_) {
        throw std::runtime_error("模型未成功加载，无法执行预测");
    }
    
    // 将输入帧转换为Tensor
    torch::Tensor input_tensor = frameToTensor(input_frame);
    
    // 执行预测
    torch::NoGradGuard no_grad; // 禁用梯度计算
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    
    // 执行前向传播
    torch::Tensor output_tensor;
    try {
        output_tensor = model_.forward(inputs).toTensor();
    } catch (const std::exception& e) {
        std::cerr << "预测失败: " << e.what() << std::endl;
        throw;
    }
    
    // 将Tensor转换回TouchFrame
    return tensorToFrame(output_tensor, input_frame);
}

// 将TouchFrame转换为Tensor
torch::Tensor TouchModelPredictor::frameToTensor(const TouchFrame& frame) const {
    // 检查数据有效性
    if (!frame.data || frame.size == 0 || frame.txn == 0 || frame.rxn == 0) {
        throw std::invalid_argument("无效的触摸帧数据");
    }
    
    // 创建选项，始终使用CPU
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCPU);
    
    // 分配临时缓冲区并复制数据
    std::vector<float> buffer(frame.size);
    
    // 将uint32_t数据转换为归一化的float数据 (除以1000.0)
    for (uint32_t i = 0; i < frame.size; ++i) {
        buffer[i] = static_cast<float>(frame.data[i]) / 1000.0f;
    }
    
    // 确保与 Python 下一致
    torch::Tensor tensor = torch::from_blob(buffer.data(), { 1, 1, frame.txn, frame.rxn }, options);
    
    // 由于from_blob不复制数据，创建一个新的张量进行复制
    return tensor.clone();
}

// 将Tensor转换为TouchFrame (如果使用int32_t)
TouchFrame TouchModelPredictor::tensorToFrame(const torch::Tensor& tensor, const TouchFrame& original_frame) const {
    // 创建新的TouchFrame，复制原始帧的元数据
    TouchFrame output_frame;
    output_frame.txn = original_frame.txn;
    output_frame.rxn = original_frame.rxn;
    output_frame.size = original_frame.size;
    output_frame.timestamp = original_frame.timestamp;
    output_frame.frame_id = original_frame.frame_id;
    
    // 分配内存存储结果数据
    output_frame.data = new int32_t[output_frame.size];
    
    // 确保张量在CPU上并将其转换为连续的布局
    auto cpu_tensor = tensor.to(torch::kCPU).contiguous();
    
    // 获取张量的裸指针
    float* tensor_data = cpu_tensor.data_ptr<float>();
    
    // 将float数据转换为int32_t数据 (乘以1000.0)
    for (uint32_t i = 0; i < output_frame.size; ++i) {
        float value = tensor_data[i] * 1000.0f;
        output_frame.data[i] = static_cast<int32_t>(value);
    }
    
    return output_frame;
}

// 检查模型是否成功加载
bool TouchModelPredictor::isModelLoaded() const {
    return model_loaded_;
}

// 设置是否使用CUDA加速 - 目前强制使用CPU
void TouchModelPredictor::setUseCuda(bool use_cuda) {
    // 在此版本中强制使用CPU，忽略use_cuda参数
    use_cuda_ = false;
}

// 获取模型路径
std::string TouchModelPredictor::getModelPath() const {
    return model_path_;
}

// 创建并初始化触摸模型预测器
std::unique_ptr<TouchModelPredictor> createTouchModelPredictor(const std::string& model_path) {
    return std::make_unique<TouchModelPredictor>(model_path);
}