'''
    @brief
        命令行批量预测
        
    @description
        tk 获取路径
            模型文件
            预测数据文件夹
        保存路径为预测数据文件夹下的 `predictions/{time}` 文件夹
        保存文件名为 '{原文件名}-predicted.txt'
'''
import os
import tkinter as tk
from tkinter import filedialog
import torch
import numpy as np
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from datetime import datetime
import logging
from pathlib import Path
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from rich.logging import RichHandler

from core.dataProcessor import TouchDataParser, TouchData
from core.TouchFilterNet_fp import TouchFilterNet
from train import CHECKPOINT_PATH   # 导入模型路径常量
from tools.findTFN_fp_DPL_alpha import BATCH_SIZE

logging.basicConfig(
    level="NOTSET",
    handlers=[RichHandler()]
)
logger = logging.getLogger("predict-cli")

console = Console()

def select_model_file():
    """
    打开文件对话框选择模型文件
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    model_file = filedialog.askopenfilename(
        title="选择模型文件",
        initialdir=CHECKPOINT_PATH,
        filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
    )
    
    return model_file if model_file else None

def select_data_folder():
    """
    打开文件夹对话框选择数据文件夹
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    data_folder = filedialog.askdirectory(
        title="选择预测数据文件夹",
        initialdir=os.path.join(os.path.dirname(__file__), "data")
    )
    
    return data_folder if data_folder else None

def load_model(model_path):
    """
    加载预训练模型
    """
    try:
        # 检查设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        
        # 创建模型实例
        model = TouchFilterNet(in_channels=1)
        
        # 加载模型参数
        checkpoint = torch.load(model_path, map_location=device)
        
        # 检查加载的是检查点还是直接的状态字典
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 这是一个检查点文件
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"从检查点加载模型: epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # 这是直接的状态字典
            model.load_state_dict(checkpoint)
            logger.info("从状态字典加载模型")
        
        model.to(device)
        model.eval()  # 设置为评估模式
        
        logger.info(f"成功加载模型: {model_path}")
        return model, device
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return None, None

def process_frame_batch(frames, model, device, batch_size=32):
    """
    批量处理帧数据以提高效率
    """
    predicted_frames = []
    
    # 分批处理
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        batch_tensors = []
        valid_indices = []
        
        # 准备批次数据
        for j, frame in enumerate(batch_frames):
            if frame.data_matrix is not None:
                input_matrix = frame.data_matrix.astype(np.float32) / 1000.0
                input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0)
                batch_tensors.append(input_tensor)
                valid_indices.append(j)
        
        if batch_tensors:
            # 批量预测
            batch_input = torch.cat(batch_tensors, dim=0).unsqueeze(1).to(device)
            
            with torch.no_grad():
                batch_predictions = model(batch_input)
            
            # 处理预测结果
            prediction_idx = 0
            for j, frame in enumerate(batch_frames):
                if j in valid_indices:
                    predicted_matrix = batch_predictions[prediction_idx].squeeze().cpu().numpy() * 1000.0
                    predicted_frame = TouchData(
                        frame_number=frame.frame_number,
                        timestamp=frame.timestamp,
                        tx_count=frame.tx_count,
                        rx_count=frame.rx_count,
                        data_matrix=predicted_matrix.astype(np.int16)
                    )
                    predicted_frames.append(predicted_frame)
                    prediction_idx += 1
                else:
                    # 空帧
                    predicted_frame = TouchData(
                        frame_number=frame.frame_number if hasattr(frame, 'frame_number') else 0
                    )
                    predicted_frames.append(predicted_frame)
        else:
            # 所有帧都是空的
            for frame in batch_frames:
                predicted_frame = TouchData(
                    frame_number=frame.frame_number if hasattr(frame, 'frame_number') else 0
                )
                predicted_frames.append(predicted_frame)
    
    return predicted_frames

def process_file(file_path, model, device, use_batch=True, batch_size=32):
    """
    处理单个文件并进行预测
    """
    try:
        # 解析触摸数据
        parser = TouchDataParser(file_path)
        header_info, input_frames = parser.parse()
        
        if not input_frames:
            logger.error(f"无法解析文件或文件为空: {file_path}")
            return None
        
        # 选择处理方式
        if use_batch and len(input_frames) > batch_size:
            # 批量处理
            predicted_frames = process_frame_batch(input_frames, model, device, batch_size)
        else:
            # 逐帧处理
            predicted_frames = []
            for frame in input_frames:
                if frame.data_matrix is not None:
                    # 归一化数据
                    input_matrix = frame.data_matrix.astype(np.float32) / 1000.0
                    input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    
                    # 预测
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
                    
                    predicted_frames.append(predicted_frame)
                else:
                    # 如果没有有效数据，添加空帧
                    predicted_frame = TouchData(
                        frame_number=frame.frame_number if hasattr(frame, 'frame_number') else 0
                    )
                    predicted_frames.append(predicted_frame)
        
        return predicted_frames, header_info
        
    except Exception as e:
        logger.error(f"处理文件时出错 {file_path}: {e}")
        return None

def save_prediction(header_info, predicted_frames, file_path, output_dir):
    """
    保存预测结果到文件
    """
    try:        
        # 获取原文件名并创建输出文件名
        base_name = Path(file_path).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{base_name}-predicted-{timestamp}.txt"

        # 使用解析器保存结果
        parser = TouchDataParser("")
        parser.header_info = header_info
        parser.touch_data = predicted_frames
        parser.save_to_file(output_file)
        
        logger.info(f"已保存预测结果到: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"保存预测结果时出错: {e}")
        return None

def process_single_file_worker(args):
    """
    单文件处理工作函数，用于进程池
    """
    file_path, model_path, batch_size, output_dir = args
    
    try:
        # 在工作进程中加载模型
        model, device = load_model(model_path)
        if model is None:
            return None, os.path.basename(file_path)
        
        # 处理文件
        result = process_file(file_path, model, device, use_batch=True, batch_size=batch_size)
        if result:
            predicted_frames, header_info = result
            # 保存预测结果
            output_file = save_prediction(header_info, predicted_frames, file_path, output_dir)
            if output_file:
                return True, os.path.basename(file_path)
            else:
                return False, os.path.basename(file_path)
        else:
            return False, os.path.basename(file_path)
            
    except Exception as e:
        logger.error(f"处理文件时出错 {file_path}: {e}")
        return False, os.path.basename(file_path)

def main():
    console.print(Panel.fit("[bold green]触摸屏数据滤波预测工具[/bold green]", 
                           subtitle="批量预测模式"))
    
    # 选择模型文件
    console.print("[yellow]请选择模型文件...[/yellow]")
    model_path = select_model_file()
    if not model_path:
        console.print("[red]未选择模型文件，程序退出[/red]")
        return
    
    # 选择数据文件夹
    console.print("[yellow]请选择数据文件夹...[/yellow]")
    data_folder = select_data_folder()
    if not data_folder:
        console.print("[red]未选择数据文件夹，程序退出[/red]")
        return
    
    # 获取所有txt文件
    txt_files = []
    for file in os.listdir(data_folder):
        if file.endswith('.txt'):
            file_path = os.path.join(data_folder, file)
            if os.path.isfile(file_path):
                txt_files.append(file_path)
    
    if not txt_files:
        console.print(f"[red]在 {data_folder} 中没有找到txt文件[/red]")
        return
    
    # 询问并行处理选项
    console.print(f"[green]找到 {len(txt_files)} 个文件[/green]")
    
    # 确定处理方式
    if torch.cuda.is_available():
        console.print("[yellow]检测到CUDA设备，使用GPU单进程批量处理模式[/yellow]")
        use_parallel = False
    else:
        console.print("[yellow]使用CPU，是否启用多进程并行处理？[/yellow]")
        use_parallel_input = input("输入 'y' 启用多进程处理，或按回车使用单进程: ").strip().lower()
        use_parallel = use_parallel_input == 'y'
    
    # 设置批量大小
    batch_size = BATCH_SIZE
    if not use_parallel:
        batch_input = input(f"输入批量大小 (默认 {batch_size}): ").strip()
        if batch_input.isdigit():
            batch_size = int(batch_input)
    
    success_count = 0
    failed_files = []
    
    # 创建输出目录 - 使用数据文件夹来创建
    output_dir = Path(data_folder) / "predictions" / time.asctime()[4:].replace(" ", "-")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if use_parallel:
        # 多进程并行处理
        num_processes = min(mp.cpu_count(), len(txt_files))
        console.print(f"[green]使用 {num_processes} 个进程并行处理...[/green]")
        
        # 准备参数，包括输出目录
        process_args = [(file_path, model_path, batch_size, output_dir) for file_path in txt_files]
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                
                task = progress.add_task("[cyan]处理文件...", total=len(txt_files))
                
                # 提交所有任务
                future_to_file = {executor.submit(process_single_file_worker, args): args[0] 
                                  for args in process_args}
                
                # 处理完成的任务
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        success, file_name = future.result()
                        if success:
                            success_count += 1
                        else:
                            failed_files.append(file_name)
                    except Exception as e:
                        failed_files.append(os.path.basename(file_path))
                        logger.error(f"处理文件失败 {file_path}: {e}")
                    
                    progress.update(task, advance=1)
    else:
        # 单进程处理（支持批量）
        console.print("[green]使用单进程批量处理模式...[/green]")
        
        # 加载模型
        console.print("[yellow]加载模型中...[/yellow]")
        model, device = load_model(model_path)
        if model is None:
            console.print("[red]模型加载失败，程序退出[/red]")
            return
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            
            task = progress.add_task("[cyan]处理文件...", total=len(txt_files))
            
            for file_path in txt_files:
                file_name = os.path.basename(file_path)
                progress.update(task, description=f"处理 {file_name}")
                
                # 处理文件
                result = process_file(file_path, model, device, use_batch=True, batch_size=batch_size)
                if result:
                    predicted_frames, header_info = result
                    # 保存预测结果，传入output_dir参数
                    output_file = save_prediction(header_info, predicted_frames, file_path, output_dir)
                    if output_file:
                        success_count += 1
                    else:
                        failed_files.append(file_name)
                else:
                    failed_files.append(file_name)
                
                progress.update(task, advance=1)
    
    # 显示处理结果
    console.print(f"[bold green]处理完成![/bold green]")
    console.print(f"[green]成功处理: {success_count} 个文件[/green]")
    
    if failed_files:
        console.print(f"[red]失败文件: {len(failed_files)} 个[/red]")
        for failed_file in failed_files:
            console.print(f"  - {failed_file}")
    
    console.print(f"[blue]预测结果已保存到各文件夹的 'predicted' 子目录[/blue]")

if __name__ == "__main__":
    main()



