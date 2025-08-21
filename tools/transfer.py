"""
    将训练好的PyTorch模型转换为LibTorch可用的格式
    并保存到build/models/目录
"""
import os
import torch
import time
import argparse
from pathlib import Path

from core.TouchFilterNet_fp import TouchFilterNet
from rich.console import Console
from rich.panel import Panel

# 初始化rich控制台
console = Console()

def main():
    parser = argparse.ArgumentParser(description="将PyTorch模型转换为LibTorch格式")
    parser.add_argument('--model', type=str, help="PyTorch模型文件的路径")
    args = parser.parse_args()

    console.print(Panel.fit("[bold green]触摸屏数据滤波模型转换工具[/bold green]", 
                           subtitle="PyTorch => LibTorch"))
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"使用设备: [bold]{device}[/bold]")
    
    # 确定模型路径
    if args.model:
        model_path = args.model
    else:
        # 查找最新的模型目录
        try:
            model_dirs = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
            latest_model_dir = sorted(model_dirs)[-1]  # 取最新的一个
            
            # 先尝试最佳模型路径
            best_model_path = os.path.join("models", latest_model_dir, "checkpoints", "best_model.pth")
            default_model_path = os.path.join("models", latest_model_dir, "touch_filter_model.pth")
            
            if os.path.exists(best_model_path):
                model_path = best_model_path
                console.print(f"使用最佳模型: [blue]{best_model_path}[/blue]")
            elif os.path.exists(default_model_path):
                model_path = default_model_path
                console.print(f"使用默认模型: [blue]{default_model_path}[/blue]")
            else:
                console.print("[bold red]错误: 未找到有效的模型文件![/bold red]")
                return
        except (IndexError, FileNotFoundError):
            console.print("[bold red]错误: 未找到模型目录或文件![/bold red]")
            return
    
    # 加载模型
    console.print(f"加载模型: [yellow]{model_path}[/yellow]")
    model = TouchFilterNet(in_channels=1)
    
    try:
        # 检查加载的是检查点还是直接的状态字典
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 这是一个检查点文件
            model.load_state_dict(checkpoint['model_state_dict'])
            console.print(f"从检查点加载模型: epoch [green]{checkpoint.get('epoch', 'unknown')}[/green]")
        else:
            # 这是直接的状态字典
            model.load_state_dict(checkpoint)
            console.print("从状态字典加载模型")
        
        # 设置为评估模式
        model.eval()
        # 确保模型在正确的设备上
        model = model.to(device)
    except Exception as e:
        console.print(f"[bold red]加载模型失败: {str(e)}[/bold red]")
        return
    
    # 创建示例输入以进行追踪
    example_input = torch.randn(1, 1, 16, 16, device=device)
    
    # 转换为TorchScript格式
    console.print("转换为TorchScript格式...")
    try:
        # 使用trace方法转换模型
        traced_script_module = torch.jit.trace(model, example_input)
    except Exception as e:
        console.print(f"[bold red]转换模型失败: {str(e)}[/bold red]")
        return
    
    # 确保输出目录存在
    output_dir = Path("build/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"touch_filter_model_{timestamp}.pt"
    
    # 保存转换后的模型
    console.print(f"保存LibTorch模型到: [green]{output_file}[/green]")
    try:
        traced_script_module.save(str(output_file))
        console.print("[bold green]模型转换成功！[/bold green]")
    except Exception as e:
        console.print(f"[bold red]保存模型失败: {str(e)}[/bold red]")
        return
    
    # 创建符号链接到最新的模型
    latest_link = output_dir / "latest_model.pt"
    try:
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(output_file.name, latest_link)
        console.print(f"创建符号链接: [blue]{latest_link}[/blue] -> [blue]{output_file.name}[/blue]")
    except Exception as e:
        console.print(f"[yellow]创建符号链接失败: {str(e)}[/yellow]")
    
    console.print(Panel.fit("[bold green]转换完成![/bold green]", subtitle=f"输出: {output_file}"))

if __name__ == "__main__":
    main()