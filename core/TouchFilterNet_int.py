"""
触摸数据滤波整数模型
实现基于整数计算的触摸数据滤波网络，用于资源受限设备
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math

from core.TouchFilterNet_fp import TouchFilterNet as TouchFilterNet_FP


class IntConv2d(nn.Module):
    """整数域卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bits=8):
        super(IntConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = padding
        self.stride = stride
        self.bits = bits
        
        # 量化范围
        self.qmin = -(1 << (bits - 1))
        self.qmax = (1 << (bits - 1)) - 1
        
        # 权重和偏置使用浮点数进行训练
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # 重置参数
        self.reset_parameters() 
        
        # 量化参数 (用于保存转换信息)
        self.input_scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.output_scale = nn.Parameter(torch.ones(1), requires_grad=False)
    
    def reset_parameters(self):
        """重置参数，使用PyTorch默认的卷积层初始化方法"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def quantize(self, x):
        """量化张量到指定位宽的整数，但保持浮点类型以支持梯度计算"""
        return torch.clamp(x.round(), float(self.qmin), float(self.qmax))
    
    def forward(self, x):
        # 量化权重和输入 (仅用于前向传播)
        quantized_weight = self.quantize(self.weight)
        quantized_bias = self.quantize(self.bias) if self.bias is not None else None
        
        # 前向传播 (使用量化后的权重但保持计算在浮点域中)
        out = F.conv2d(
            x.float(),
            quantized_weight,
            bias=quantized_bias,
            stride=self.stride,
            padding=self.padding
        )
        
        # 量化输出
        out = self.quantize(out)
        
        return out


class IntReLU(nn.Module):
    """整数域ReLU激活函数"""
    
    def __init__(self, inplace=False, bits=8):
        super(IntReLU, self).__init__()
        self.inplace = inplace
        self.bits = bits
        self.qmin = 0
        self.qmax = (1 << (bits - 1)) - 1
        
    def forward(self, x):
        # 整数域的ReLU只需要保留非负值并确保不超过最大值
        return torch.clamp(x, self.qmin, self.qmax)


class TouchFilterNet_int(nn.Module):
    """触摸数据滤波网络 - 整数版本"""

    def __init__(self, in_channels=1, bits=8):
        super(TouchFilterNet_int, self).__init__()
        self.bits = bits
        
        # 整数量化范围
        self.qmin = -(1 << (bits - 1))
        self.qmax = (1 << (bits - 1)) - 1
        
        # 特征提取
        self.encoder = nn.Sequential(
            IntConv2d(in_channels, 16, kernel_size=3, padding=1, bits=bits),
            IntReLU(inplace=True, bits=bits),
            IntConv2d(16, 32, kernel_size=3, padding=1, bits=bits),
            IntReLU(inplace=True, bits=bits),
            IntConv2d(32, 64, kernel_size=3, padding=1, bits=bits),
            IntReLU(inplace=True, bits=bits),
        )

        # 注意力机制 - 关注高差异区域
        self.attention = nn.Sequential(
            IntConv2d(64, 32, kernel_size=1, bits=bits),
            IntReLU(inplace=True, bits=bits),
            IntConv2d(32, 1, kernel_size=1, bits=bits),
            # 注: Sigmoid在整数域中由int_sigmoid函数实现
        )

        # 解码器（恢复原始分辨率）
        self.decoder = nn.Sequential(
            IntConv2d(64, 32, kernel_size=3, padding=1, bits=bits),
            IntReLU(inplace=True, bits=bits),
            IntConv2d(32, 16, kernel_size=3, padding=1, bits=bits),
            IntReLU(inplace=True, bits=bits),
            IntConv2d(16, in_channels, kernel_size=3, padding=1, bits=bits),
        )

        # 残差连接 - 使用1x1卷积学习残差映射
        self.residual = IntConv2d(in_channels, in_channels, kernel_size=1, bits=bits)

        # 差异增强层 - 学习如何根据输入差异调整输出
        self.diff_enhancer = nn.Sequential(
            IntConv2d(in_channels * 2, 8, kernel_size=1, bits=bits),
            IntReLU(inplace=True, bits=bits),
            IntConv2d(8, in_channels, kernel_size=1, bits=bits),
            # 注: Tanh在整数域中由int_tanh函数实现
        )
        
        # 初始化参数
        self._initialize_params()

    def _initialize_params(self):
        """初始化模型参数"""
        for name, module in self.named_modules():
            if isinstance(module, IntConv2d):
                # 使用均匀分布初始化权重
                nn.init.kaiming_uniform_(module.weight.data, a=np.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight.data)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(module.bias.data, -bound, bound)
                
                # 仍然使用浮点类型，但将值范围限制在整数范围内
                module.weight.data = torch.clamp(module.weight.data, float(self.qmin), float(self.qmax))
                if module.bias is not None:
                    module.bias.data = torch.clamp(module.bias.data, float(self.qmin), float(self.qmax))

    def int_sigmoid(self, x):
        """整数域的sigmoid近似函数"""
        # 使用查找表或分段线性近似
        # 简单实现：将输入规范化到[0, 2^bits-1]范围内
        scale = (1 << (self.bits - 1)) - 1  # 最大整数值
        return torch.clamp((x + scale) / 2, 0, scale)  # 简单的线性映射
        
    def int_tanh(self, x):
        """整数域的tanh近似函数"""
        # 使用查找表或分段线性近似
        # 简单实现：将输入规范化到[-scale, scale]范围内
        scale = (1 << (self.bits - 1)) - 1  # 最大整数值
        return torch.clamp(x, -scale, scale)  # 简单的裁剪
    
    def forward(self, x):
        # 保存原始输入用于残差连接和差异增强
        original = x
        residual = self.residual(x)

        # 编码
        encoded = self.encoder(x)

        # 注意力机制
        attention_raw = self.attention(encoded)
        attention_map = self.int_sigmoid(attention_raw)
        
        # 整数域的乘法需要考虑量化效应
        # 简单实现：按比例缩放attention_map后相乘
        scale_factor = 16  # 2^4，使用移位操作可以更高效
        attended = (encoded * attention_map) / scale_factor

        # 解码
        decoded = self.decoder(attended)

        # 基础输出（带残差连接）
        base_output = decoded + residual

        # 差异增强: 将原始输入与基础输出拼接
        concat = torch.cat([original, base_output], dim=1)
        diff_scale_raw = self.diff_enhancer(concat)
        diff_scale = self.int_tanh(diff_scale_raw)

        # 最终输出：基于差异调整
        # 在整数域中，我们使用缩放和移位来模拟乘法
        scale_factor = 16  # 2^4
        output = base_output + (base_output * diff_scale) / scale_factor
        
        # 确保输出在有效整数范围内
        return torch.clamp(output, self.qmin, self.qmax)


def convert_fp_to_int_model(fp_model_path, output_path=None, bits=8):
    """将浮点模型转换为整数模型
    
    Args:
        fp_model_path: 浮点模型文件路径
        output_path: 整数模型保存路径，若为None则不保存
        bits: 整数位宽，默认为8位
        
    Returns:
        转换后的整数模型实例
    """
    # 加载浮点模型
    fp_model = TouchFilterNet_FP(in_channels=1)
    
    try:
        # 加载模型权重
        loaded_model = torch.load(fp_model_path, map_location='cpu')
        if isinstance(loaded_model, dict) and 'model_state_dict' in loaded_model:
            fp_model.load_state_dict(loaded_model['model_state_dict'])
        else:
            fp_model.load_state_dict(loaded_model)
        fp_model.eval()  # 设置为评估模式
    except Exception as e:
        print(f"加载浮点模型失败: {str(e)}")
        return None
    
    # 创建整数模型
    int_model = TouchFilterNet_int(in_channels=1, bits=bits)
    
    # 计算量化范围
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    
    # 从浮点模型到整数模型的映射
    fp_to_int_mapping = {
        'encoder.0': 'encoder.0',  # Conv2d -> IntConv2d
        'encoder.2': 'encoder.2',  # Conv2d -> IntConv2d
        'encoder.4': 'encoder.4',  # Conv2d -> IntConv2d
        'attention.0': 'attention.0',  # Conv2d -> IntConv2d
        'attention.2': 'attention.2',  # Conv2d -> IntConv2d
        'decoder.0': 'decoder.0',  # Conv2d -> IntConv2d
        'decoder.2': 'decoder.2',  # Conv2d -> IntConv2d
        'decoder.4': 'decoder.4',  # Conv2d -> IntConv2d
        'residual': 'residual',  # Conv2d -> IntConv2d
        'diff_enhancer.0': 'diff_enhancer.0',  # Conv2d -> IntConv2d
        'diff_enhancer.2': 'diff_enhancer.2',  # Conv2d -> IntConv2d
    }
    
    # 转换参数
    scale_factor = (1 << (bits - 1)) - 1  # 最大正整数值
    
    for fp_name, int_name in fp_to_int_mapping.items():
        # 处理权重
        fp_weight_name = f"{fp_name}.weight"
        int_weight_name = f"{int_name}.weight"
        
        if hasattr(fp_model, fp_name) and fp_weight_name in fp_model.state_dict():
            fp_weight = fp_model.state_dict()[fp_weight_name].cpu()
            
            # 获取整数模型中对应的模块
            int_module = None
            for name, module in int_model.named_modules():
                if name == int_name:
                    int_module = module
                    break

            if int_module is not None:
                # 归一化权重到[-1, 1]范围
                weight_abs_max = torch.abs(fp_weight).max().item()
                if weight_abs_max > 0:
                    normalized_weight = fp_weight / weight_abs_max
                    # 量化到整数范围
                    quantized_weight = torch.round(normalized_weight * scale_factor)
                    quantized_weight = torch.clamp(quantized_weight, qmin, qmax).to(torch.int32)
                    int_module.weight.data = quantized_weight
                    # 保存缩放因子
                    if hasattr(int_module, 'weight_scale'):
                        int_module.weight_scale.data = torch.tensor(weight_abs_max / scale_factor)
        
        # 处理偏置
        fp_bias_name = f"{fp_name}.bias"
        int_bias_name = f"{int_name}.bias"
        
        if hasattr(fp_model, fp_name) and fp_bias_name in fp_model.state_dict():
            fp_bias = fp_model.state_dict()[fp_bias_name].cpu()
            
            # 获取整数模型中对应的模块
            int_module = None
            for name, module in int_model.named_modules():
                if name == int_name:
                    int_module = module
                    break
            
            if int_module is not None:
                # 处理偏置项，类似于权重
                bias_abs_max = torch.abs(fp_bias).max().item() if fp_bias.numel() > 0 else 0
                if bias_abs_max > 0:
                    normalized_bias = fp_bias / bias_abs_max
                    quantized_bias = torch.round(normalized_bias * scale_factor)
                    quantized_bias = torch.clamp(quantized_bias, qmin, qmax).to(torch.int32)
                    int_module.bias.data = quantized_bias
    
    # 保存整数模型
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({
            'model_state_dict': int_model.state_dict(),
            'bits': bits
        }, output_path)
        print(f"整数模型已保存至: {output_path}")
    
    return int_model
