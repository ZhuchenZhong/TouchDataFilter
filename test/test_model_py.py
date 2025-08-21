"""
Python版本的触摸模型预测器测试脚本
用于验证模型预测结果并与C++版本进行比较
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

sys.path.append(Path(__file__).parent.parent.as_posix())  # 添加项目根目录到路径

# 导入自定义模块
from train import TouchFilterNet
from core.dataProcessor import TouchData

console = Console()

def create_test_frame(tx_count=18, rx_count=36, value=100):
    """创建测试用的触摸帧"""
    # 创建一个简单的测试数据矩阵
    data_matrix = np.ones((tx_count, rx_count), dtype=np.int16) * value
    
    # 添加一些变化以便于识别
    for i in range(tx_count):
        for j in range(rx_count):
            if (i + j) % 3 == 0:
                data_matrix[i, j] = int(value * 1.5)
            elif (i + j) % 5 == 0:
                data_matrix[i, j] = int(value * 0.5)
    
    # 创建TouchData对象
    frame = TouchData(
        frame_number=1,
        timestamp=0,
        tx_count=tx_count,
        rx_count=rx_count,
        data_matrix=data_matrix
    )
    
    return frame

def load_model(model_path):
    """加载预训练模型"""
    console.print(f"[yellow]加载模型: {model_path}[/yellow]")
    
    try:
        device = torch.device("cpu")
        model = TouchFilterNet(in_channels=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        console.print("[green]模型加载成功[/green]")
        return model, device
    except Exception as e:
        console.print(f"[bold red]加载模型失败: {str(e)}[/bold red]")
        return None, None

def predict_frame(model, frame, device):
    """使用模型预测单帧数据"""
    # 打印输入数据的详细信息
    console.print(f"[blue]输入数据形状: {frame.data_matrix.shape}[/blue]")
    console.print(f"[blue]输入数据范围: {frame.data_matrix.min()} - {frame.data_matrix.max()}[/blue]")
    console.print(f"[blue]输入数据样本:\n{frame.data_matrix[:3, :3]}[/blue]")
    
    # 归一化数据
    input_matrix = frame.data_matrix.astype(np.float32) / 1000.0
    input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    console.print(f"[blue]输入张量形状: {input_tensor.shape}[/blue]")
    console.print(f"[blue]输入张量样本:\n{input_tensor[0, 0, :3, :3]}[/blue]")
    
    # 执行预测
    with torch.no_grad():
        predicted = model(input_tensor)
    
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
    
    # 打印输出数据的详细信息
    console.print(f"[green]输出数据形状: {predicted_frame.data_matrix.shape}[/green]")
    console.print(f"[green]输出数据范围: {predicted_frame.data_matrix.min()} - {predicted_frame.data_matrix.max()}[/green]")
    console.print(f"[green]输出数据样本:\n{predicted_frame.data_matrix[:3, :3]}[/green]")
    
    # 保存输入和输出数据供C++比较
    np.save("test_input.npy", frame.data_matrix)
    np.save("test_output.npy", predicted_frame.data_matrix)
    
    return predicted_frame

def main():
    console.print(Panel.fit("[bold]测试[/bold]"))
    
    # 查找最新模型
    try:
        model_dirs = sorted([d for d in os.listdir("../models") if os.path.isdir(os.path.join("../models", d))])
        latest_model_dir = model_dirs[-1]
        best_model_path = os.path.join("../models", latest_model_dir, "checkpoints", "best_model.pth")

        if not os.path.exists(best_model_path):
            best_model_path = os.path.join("../models", latest_model_dir, "touch_filter_model.pth")
    except:
        console.print("[red]未找到模型文件，请指定模型路径[/red]")
        return
    
    # 加载模型
    model, device = load_model(best_model_path)
    if model is None:
        return
    
    # 创建测试帧
    test_frame = create_test_frame()
    console.print(f"已创建测试帧 ({test_frame.tx_count}x{test_frame.rx_count})")
    
    # 预测
    predicted_frame = predict_frame(model, test_frame, device)
    
    # 打印预测结果
    console.print(Panel.fit("[bold green]预测完成[/bold green]"))
    console.print("保存的测试数据文件可用于C++测试比较：test_input.npy, test_output.npy")

if __name__ == "__main__":
    main()