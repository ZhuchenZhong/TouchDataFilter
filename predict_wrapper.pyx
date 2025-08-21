"""
将预测功能包装为C可调用的库
"""
import os
import sys
import numpy as np
import torch
from pathlib import Path

# 导入预测相关模块
from core.dataProcessor import TouchDataParser, TouchData
from train import TouchFilterNet, MODEL_PATH

# 定义模型和设备
cdef public class TouchPredictor [object PyTouchPredictor, type PyTouchPredictorType]:
    cdef object model
    cdef object device
    
    def __cinit__(self, model_path=None):
        """初始化预测器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 如果未提供模型路径，使用默认路径
        if model_path is None:
            try:
                __all = [os.path.join(Path(MODEL_PATH).parent, f) 
                         for f in sorted(os.listdir(Path(MODEL_PATH).parent))]
                _root = __all[-1]
                default_model = f'{_root}/touch_filter_model.pth'
                best_model = f'{_root}/checkpoints/best_model.pth'
                
                if os.path.exists(default_model):
                    model_path = default_model
                elif os.path.exists(best_model):
                    model_path = best_model
                else:
                    raise FileNotFoundError("未找到默认模型文件")
            except Exception as e:
                raise RuntimeError(f"加载默认模型失败: {str(e)}")
        
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path):
        """加载预训练模型"""
        try:
            # 检查是否为整数模型（通过文件名判断）
            is_int_model = 'int' in model_path.lower()
            
            # 加载模型文件
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 根据模型类型选择合适的模型类
            if is_int_model:
                # 导入整数模型类
                from core.TouchFilterNet_int import TouchFilterNet_int
                model = TouchFilterNet_int(in_channels=1).to(self.device)
            else:
                model = TouchFilterNet(in_channels=1).to(self.device)
            
            # 检查加载的内容是否为嵌套字典（包含model_state_dict键）
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # 直接加载状态字典
                model.load_state_dict(checkpoint)
            
            model.eval()  # 设置为评估模式
            return model
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")
    
    def predict_file(self, file_path, output_path=None):
        """处理文件并返回处理后的数据路径"""
        try:
            # 解析数据文件
            parser = TouchDataParser(file_path)
            header_info, input_frames = parser.parse()
            
            # 处理每一帧
            predicted_frames = []
            for frame in input_frames:
                if frame.data_matrix is not None:
                    # 归一化数据
                    input_matrix = frame.data_matrix.astype(np.float32) / 1000.0
                    input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                    
                    # 预测
                    with torch.no_grad():
                        predicted = self.model(input_tensor)
                    
                    # 转换回numpy数组
                    predicted_matrix = predicted.squeeze().cpu().numpy() * 1000.0
                    
                    # 创建预测帧
                    predicted_frame = TouchData(
                        frame_number=frame.frame_number,
                        timestamp=frame.timestamp,
                        tx_count=frame.tx_count,
                        rx_count=frame.rx_count,
                        data_matrix=predicted_matrix.astype(np.int16)
                    )
                else:
                    # 如果没有有效数据，添加空帧
                    predicted_frame = TouchData(
                        frame_number=frame.frame_number if hasattr(frame, 'frame_number') else 0
                    )
                
                predicted_frames.append(predicted_frame)
            
            # 如果指定了输出路径，保存结果
            if output_path:
                save_parser = TouchDataParser("")
                save_parser.header_info = header_info
                save_parser.touch_data = predicted_frames
                save_parser.save_to_file(output_path)
                return output_path
            
            return predicted_frames
        except Exception as e:
            raise RuntimeError(f"预测失败: {str(e)}")

# 导出C接口函数
cdef extern from "Python.h":
    void Py_Initialize()
    void Py_Finalize()
    
cdef public int initialize_predictor():
    """初始化Python解释器"""
    Py_Initialize()
    return 1

cdef public int finalize_predictor():
    """关闭Python解释器"""
    Py_Finalize()
    return 1

# 创建一个全局预测器实例
cdef TouchPredictor global_predictor = None

cdef public int create_predictor(const char* model_path):
    """创建预测器实例"""
    global global_predictor
    try:
        if model_path == NULL:
            global_predictor = TouchPredictor()
        else:
            global_predictor = TouchPredictor(model_path.decode('utf-8'))
        return 1
    except Exception:
        return 0

cdef public int predict_touch_data(const char* input_file, const char* output_file):
    """预测触摸数据"""
    global global_predictor
    try:
        if global_predictor is None:
            return 0
        global_predictor.predict_file(
            input_file.decode('utf-8'), 
            output_file.decode('utf-8')
        )
        return 1
    except Exception:
        return 0