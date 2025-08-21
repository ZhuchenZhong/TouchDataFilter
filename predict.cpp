/**
 * @file    predict.cpp
 * @author  
 * @brief   触控数据滤波
 * @version 0.1
 * @date    2025-07-15
 * 
 * @attention
 *      不使用cuda，仅使用CPU进行预测(目标平台没有cuda)
 * 
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <ctime>
#include "include/TouchData.hpp"
#include "include/TouchModelPredictor.hpp"

// 命令行参数结构
struct CmdLineArgs {
    std::string model_path;    // 模型文件路径
    std::string input_file;    // 输入文件路径
    std::string output_file;   // 输出文件路径
    bool verbose = false;      // 是否详细输出
    bool use_cuda = false;     // 是否使用CUDA (默认false)
};

// 触摸数据解析器 - 简化版，仅解析基本结构
class TouchDataParser {
public:
    TouchDataParser() = default;

    // 解析文件并返回触摸帧链表
    TouchFrames parseFile(const std::string& filePath) {
        TouchFrames frames;
        frames.head = nullptr;
        frames.tail = nullptr;
        frames.count = 0;

        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filePath << std::endl;
            return frames;
        }

        std::string line;
        std::vector<std::string> headerLines;
        uint32_t frameId = 0;
        TouchFrameNode* currentNode = nullptr;

        // 读取头部信息
        while (std::getline(file, line) && !line.empty()) {
            if (line.find("MC Diff: Frame") != std::string::npos) {
                // 开始新的帧
                frameId++;
                
                // 创建新节点
                TouchFrameNode* newNode = new TouchFrameNode();
                newNode->frame.frame_id = frameId;
                newNode->frame.timestamp = static_cast<uint32_t>(std::time(nullptr)); // 使用当前时间
                newNode->frame.txn = 18;  // 默认值，实际应从文件中解析
                newNode->frame.rxn = 36;  // 默认值，实际应从文件中解析
                newNode->frame.size = newNode->frame.txn * newNode->frame.rxn;
                newNode->frame.data = new int32_t[newNode->frame.size]();
                newNode->prev = currentNode;
                newNode->next = nullptr;

                if (currentNode) {
                    currentNode->next = newNode;
                } else {
                    frames.head = newNode;
                }
                
                frames.tail = newNode;
                currentNode = newNode;
                frames.count++;

                // 解析帧数据
                uint32_t rowIndex = 0;
                while (std::getline(file, line) && !line.empty() && line.find("MC_TX") != std::string::npos) {
                    if (rowIndex >= newNode->frame.txn) break;
                    
                    // 解析一行触摸数据
                    std::istringstream iss(line);
                    std::string token;
                    iss >> token; // 跳过"MC_TX##"标识符
                    
                    for (uint32_t colIndex = 0; colIndex < newNode->frame.rxn && iss >> token; colIndex++) {
                        // 移除逗号（如果存在）
                        if (!token.empty() && token.back() == ',') {
                            token.pop_back();
                        }
                        
                        // 转换为整数
                        try {
                            newNode->frame.data[rowIndex * newNode->frame.rxn + colIndex] = static_cast<int32_t>(std::stoi(token));
                        } catch (const std::exception& e) {
                            newNode->frame.data[rowIndex * newNode->frame.rxn + colIndex] = 0;
                        }
                    }
                    rowIndex++;
                }
            } else {
                headerLines.push_back(line);
            }
        }

        file.close();
        return frames;
    }

    // 将触摸帧保存到文件
    bool saveToFile(const TouchFrames& frames, const std::string& filePath, const std::vector<std::string>& headerLines) {
        std::ofstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "无法创建文件: " << filePath << std::endl;
            return false;
        }

        // 写入头部信息
        for (const auto& line : headerLines) {
            file << line << std::endl;
        }

        // 写入每一帧数据
        TouchFrameNode* currentNode = frames.head;
        while (currentNode) {
            file << std::endl << "MC Diff: Frame " << currentNode->frame.frame_id << std::endl;
            
            // 写入行数据
            for (uint32_t tx = 0; tx < currentNode->frame.txn; tx++) {
                file << "\tMC_TX" << std::setw(2) << std::setfill('0') << tx << ",";
                
                // 写入列数据
                for (uint32_t rx = 0; rx < currentNode->frame.rxn; rx++) {
                    file << std::setw(6) << std::right 
                         << currentNode->frame.data[tx * currentNode->frame.rxn + rx];
                    if (rx < currentNode->frame.rxn - 1) {
                        file << ",";
                    }
                }
                file << std::endl;
            }

            currentNode = currentNode->next;
        }

        file.close();
        return true;
    }

private:
    std::vector<std::string> headerLines;
};

// 解析命令行参数
CmdLineArgs parseCommandLine(int argc, char* argv[]) {
    CmdLineArgs args;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--model" || arg == "-m") {
            if (i + 1 < argc) {
                args.model_path = argv[++i];
            }
        } else if (arg == "--input" || arg == "-i") {
            if (i + 1 < argc) {
                args.input_file = argv[++i];
            }
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                args.output_file = argv[++i];
            }
        } else if (arg == "--verbose" || arg == "-v") {
            args.verbose = true;
        } else if (arg == "--cuda") {
            args.use_cuda = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "用法: " << argv[0] << " [选项]" << std::endl;
            std::cout << "选项:" << std::endl;
            std::cout << "  --model, -m    指定模型文件路径" << std::endl;
            std::cout << "  --input, -i    指定输入数据文件路径" << std::endl;
            std::cout << "  --output, -o   指定输出文件路径 (默认: 输入文件名加前缀'滤波_')" << std::endl;
            std::cout << "  --verbose, -v  详细输出模式" << std::endl;
            std::cout << "  --cuda         使用CUDA加速 (如果可用)" << std::endl;
            std::cout << "  --help, -h     显示此帮助信息" << std::endl;
            exit(0);
        }
    }
    
    return args;
}

// 查找最新的模型文件
std::string findLatestModel() {
    std::string modelPath;
    
    // 首先检查 build/models/latest_model.pt
    if (std::filesystem::exists("build/models/latest_model.pt")) {
        modelPath = "build/models/latest_model.pt";
    } else {
        // 查找最新时间戳的模型
        std::filesystem::path latestModel;
        std::time_t latestTime = 0;

        for (const auto& entry : std::filesystem::directory_iterator("build/models")) {
            if (entry.path().extension() == ".pt") {
                auto modTime = std::filesystem::last_write_time(entry.path());
                auto modTimeT = std::chrono::system_clock::to_time_t(
                    std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                        modTime - std::filesystem::file_time_type::clock::now() + 
                        std::chrono::system_clock::now()));
                
                if (modTimeT > latestTime) {
                    latestTime = modTimeT;
                    latestModel = entry.path();
                }
            }
        }
        
        if (!latestModel.empty()) {
            modelPath = latestModel.string();
        }
    }
    
    return modelPath;
}

// 释放触摸帧链表内存
void freeFrames(TouchFrames& frames) {
    TouchFrameNode* current = frames.head;
    while (current) {
        TouchFrameNode* next = current->next;
        delete[] current->frame.data;
        delete current;
        current = next;
    }
    
    frames.head = nullptr;
    frames.tail = nullptr;
    frames.count = 0;
}

// 主函数
int main(int argc, char* argv[]) {
    // 解析命令行参数
    CmdLineArgs args = parseCommandLine(argc, argv);
    
    // 如果未指定模型路径，尝试找到最新的模型
    if (args.model_path.empty()) {
        args.model_path = findLatestModel();
        
        if (args.model_path.empty()) {
            std::cerr << "错误: 未指定模型路径，且无法找到默认模型" << std::endl;
            std::cerr << "请使用 --model 参数指定模型文件路径" << std::endl;
            return 1;
        }
    }
    
    if (args.verbose) {
        std::cout << "使用模型: " << args.model_path << std::endl;
    }
    
    // 如果未指定输入文件，显示错误
    if (args.input_file.empty()) {
        std::cerr << "错误: 未指定输入文件" << std::endl;
        std::cerr << "请使用 --input 参数指定输入数据文件路径" << std::endl;
        return 1;
    }
    
    // 如果未指定输出文件，生成默认输出文件名
    if (args.output_file.empty()) {
        std::filesystem::path inputPath(args.input_file);
        std::string timestamp = std::to_string(std::time(nullptr));
        args.output_file = inputPath.parent_path().string() + "/滤波_" + 
                           timestamp + "_" + inputPath.filename().string();
    }
    
    // 创建模型预测器
    auto predictor = createTouchModelPredictor(args.model_path);
    
    // 设置CUDA选项
    predictor->setUseCuda(args.use_cuda);
    
    // 检查模型是否成功加载
    if (!predictor->isModelLoaded()) {
        std::cerr << "错误: 无法加载模型: " << args.model_path << std::endl;
        return 1;
    }
    
    if (args.verbose) {
        std::cout << "模型已成功加载" << std::endl;
        std::cout << "开始处理文件: " << args.input_file << std::endl;
    }
    
    // 解析触摸数据文件
    TouchDataParser parser;
    TouchFrames inputFrames = parser.parseFile(args.input_file);
    
    if (inputFrames.count == 0) {
        std::cerr << "错误: 未能解析任何有效的触摸数据帧" << std::endl;
        return 1;
    }
    
    if (args.verbose) {
        std::cout << "成功解析 " << inputFrames.count << " 帧数据" << std::endl;
        std::cout << "开始处理数据..." << std::endl;
    }
    
    // 处理每一帧数据
    TouchFrames outputFrames;
    outputFrames.head = nullptr;
    outputFrames.tail = nullptr;
    outputFrames.count = 0;
    
    TouchFrameNode* currentInputNode = inputFrames.head;
    TouchFrameNode* lastOutputNode = nullptr;
    
    int processedFrames = 0;
    while (currentInputNode) {
        if (args.verbose && processedFrames % 10 == 0) {
            std::cout << "处理第 " << processedFrames << " 帧..." << std::endl;
        }
        
        // 预测处理当前帧
        TouchFrame processedFrame;
        try {
            processedFrame = predictor->predict(currentInputNode->frame);
            
            // 创建新输出节点
            TouchFrameNode* newOutputNode = new TouchFrameNode();
            newOutputNode->frame = processedFrame;
            newOutputNode->prev = lastOutputNode;
            newOutputNode->next = nullptr;
            
            if (lastOutputNode) {
                lastOutputNode->next = newOutputNode;
            } else {
                outputFrames.head = newOutputNode;
            }
            
            outputFrames.tail = newOutputNode;
            lastOutputNode = newOutputNode;
            outputFrames.count++;
            
        } catch (const std::exception& e) {
            std::cerr << "处理帧 " << currentInputNode->frame.frame_id << " 失败: " << e.what() << std::endl;
        }
        
        currentInputNode = currentInputNode->next;
        processedFrames++;
    }
    
    if (args.verbose) {
        std::cout << "处理完成，成功处理 " << outputFrames.count << " 帧数据" << std::endl;
        std::cout << "保存处理结果到: " << args.output_file << std::endl;
    }
    
    // 保存处理结果
    std::vector<std::string> headerLines; // 这里应该从原始文件中提取头部信息
    bool saveSuccess = parser.saveToFile(outputFrames, args.output_file, headerLines);
    
    if (!saveSuccess) {
        std::cerr << "保存处理结果失败" << std::endl;
        // 释放内存
        freeFrames(inputFrames);
        freeFrames(outputFrames);
        return 1;
    }
    
    if (args.verbose) {
        std::cout << "处理完成！" << std::endl;
    }
    
    // 释放内存
    freeFrames(inputFrames);
    freeFrames(outputFrames);
    
    return 0;
}



