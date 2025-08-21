#include "include/predict_wrapper.h"
#include <stdio.h>

int main() {
    // 初始化Python解释器
    if (!initialize_predictor()) {
        fprintf(stderr, "Failed to initialize Python\n");
        return 1;
    }
    
    // 创建预测器
    if (!create_predictor(NULL)) {
        fprintf(stderr, "Failed to create predictor\n");
        finalize_predictor();
        return 1;
    }
    
    // 预测数据
    if (!predict_touch_data("/path/to/input.txt", "/path/to/output.txt")) {
        fprintf(stderr, "Prediction failed\n");
    } else {
        printf("Prediction successful\n");
    }
    
    // 清理资源
    finalize_predictor();
    return 0;
}