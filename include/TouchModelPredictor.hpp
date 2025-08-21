/**
 * @file        TouchModelPredictor.hpp
 * @brief       触摸模型预测器
 * @version     0.1
 * @date        2025-07-15
 * 
 * @description
 *      调用模型传入单帧触摸数据，返回预测结果(单帧)
 * 
 */

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <torch/script.h>
#include "TouchData.hpp"

/**
 * @class TouchModelPredictor
 * @brief 触摸模型预测器类，用于加载模型并进行单帧预测
 */
class TouchModelPredictor {
public:
    /**
     * @brief 构造函数
     * @param model_path 模型文件路径
     */
    explicit TouchModelPredictor(const std::string& model_path);

    /**
     * @brief 析构函数
     */
    ~TouchModelPredictor() = default;

    /**
     * @brief 使用模型预测单帧触摸数据
     * @param input_frame 输入的触摸数据帧
     * @return 处理后的触摸数据帧
     */
    TouchFrame predict(const TouchFrame& input_frame);

    /**
     * @brief 检查模型是否成功加载
     * @return 模型加载状态
     */
    bool isModelLoaded() const;

    /**
     * @brief 设置是否使用CUDA加速
     * @param use_cuda 是否使用CUDA
     */
    void setUseCuda(bool use_cuda);

    /**
     * @brief 获取模型路径
     * @return 当前加载的模型路径
     */
    std::string getModelPath() const;

private:
    // LibTorch 模型
    torch::jit::script::Module model_;
    
    // 模型路径
    std::string model_path_;
    
    // 模型加载状态
    bool model_loaded_;
    
    // 是否使用CUDA加速
    bool use_cuda_;
    
    /**
     * @brief 加载模型
     * @param model_path 模型文件路径
     * @return 加载是否成功
     */
    bool loadModel(const std::string& model_path);
    
    /**
     * @brief 将TouchFrame转换为Tensor
     * @param frame 输入的触摸数据帧
     * @return 转换后的Tensor
     */
    torch::Tensor frameToTensor(const TouchFrame& frame) const;
    
    /**
     * @brief 将Tensor转换为TouchFrame
     * @param tensor 预测结果Tensor
     * @param original_frame 原始帧(用于复制元数据)
     * @return 处理后的触摸数据帧
     */
    TouchFrame tensorToFrame(const torch::Tensor& tensor, const TouchFrame& original_frame) const;
};

/**
 * @brief 创建并初始化触摸模型预测器
 * @param model_path 模型文件路径
 * @return 预测器指针
 */
std::unique_ptr<TouchModelPredictor> createTouchModelPredictor(const std::string& model_path);
