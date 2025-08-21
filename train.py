"""
触摸数据滤波模型训练脚本
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from typing import List

from core.TouchFilterNet_fp import TouchFilterNet, DiffPreservingLoss
from core.dataProcessor import TouchDataParser

# 初始化rich控制台
console = Console()
EPOCH = 50
MODEL_PATH = f"models/{time.asctime()[4:].replace(' ','-')}/"
CHECKPOINT_PATH = f"{MODEL_PATH}checkpoints/"

class TouchDataset(Dataset):
    """触摸数据集"""
    def __init__(self, input_files: List[str], target_files: List[str]):
        """
        初始化数据集
        
        Args:
            input_files: 输入文件路径列表（有噪声的数据）
            target_files: 目标文件路径列表（无噪声的数据）
        """
        self.input_data = []
        self.target_data = []
        
        # 加载数据
        with Progress(
            TextColumn("[bold blue]加载数据...", justify="right"),
            BarColumn(),
            TextColumn("[bold green]{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]加载数据文件...", total=len(input_files))
            
            for input_file, target_file in zip(input_files, target_files):
                # 加载输入数据（有噪声）
                _, input_frames = TouchDataParser(input_file).parse()
                # 加载目标数据（无噪声）
                _, target_frames = TouchDataParser(target_file).parse()
                
                # 确保帧数匹配
                min_frames = min(len(input_frames), len(target_frames))
                
                for i in range(min_frames):
                    if (input_frames[i].data_matrix is not None and 
                        target_frames[i].data_matrix is not None):
                        # 归一化数据 - 保留差异性的归一化方式
                        input_matrix = input_frames[i].data_matrix.astype(np.float32) / 1000.0
                        target_matrix = target_frames[i].data_matrix.astype(np.float32) / 1000.0
                        
                        # 添加到数据集
                        self.input_data.append(input_matrix)
                        self.target_data.append(target_matrix)
                
                progress.update(task, advance=1)

        # 转换为张量 - 确保输入和目标有相同的维度
        self.input_data = [torch.tensor(data, dtype=torch.float32).unsqueeze(0) for data in self.input_data]
        self.target_data = [torch.tensor(data, dtype=torch.float32).unsqueeze(0) for data in self.target_data]  # 添加 .unsqueeze(0)
        
        console.print(f"[green]已加载 {len(self.input_data)} 个训练样本")
        
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda'):
    """训练模型"""
    start_time = time.time()
    best_loss = float('inf')

    # 创建保存目录
    os.makedirs(f'{MODEL_PATH}checkpoints', exist_ok=True)

    # 使用进度条显示
    with Progress(
        TextColumn("[cyan]轮次 {task.completed}/{task.total}"),
        BarColumn(),
        TextColumn("[bold green]{task.percentage:.0f}%"),
        TextColumn("[bold]{task.fields[train_loss]}{task.fields[val_loss]}"),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        # 创建轮次进度条任务
        epoch_task = progress.add_task("", total=num_epochs, train_loss="", val_loss="")

        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0

            # 处理每个批次
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # 零初始化梯度
                optimizer.zero_grad()

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 计算平均训练损失
            train_loss /= len(train_loader)

            # 验证阶段
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                val_loss /= len(val_loader)

            # 更新进度条信息，包含loss
            progress.update(epoch_task, 
                           advance=1, 
                           train_loss=f" 训练损失: {train_loss:.6f}", 
                           val_loss=f" 验证损失: {val_loss:.6f}")

            # 保存最佳模型 (不输出提示)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f'{CHECKPOINT_PATH}best_model.pth')

            # 保存检查点 (不输出提示)
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, f'{CHECKPOINT_PATH}model_epoch_{epoch+1}.pth')

    # 保存最终模型
    torch.save(model.state_dict(), f'{MODEL_PATH}touch_filter_model.pth')

    # 计算训练时间
    time_elapsed = time.time() - start_time
    console.print(f'[bold green]训练完成，用时 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒[/bold green]')
    console.print(f'[bold green]最佳验证损失: {best_loss:.6f}[/bold green]')

    return model

def display_model_info(model):
    """显示模型信息"""
    # 创建模型结构表格
    table = Table(title="模型结构")
    table.add_column("层", justify="left", style="cyan")
    table.add_column("参数数量", justify="right", style="green")
    
    # 计算总参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 为每个模块添加行
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        table.add_row(f"{name}", f"{params:,}")
    
    # 添加总参数行
    table.add_row("总参数", f"{total_params:,}")
    table.add_row("可训练参数", f"{trainable_params:,}")
    
    # 显示表格
    console.print(table)
    
    # 显示模型架构树
    tree = Tree("[bold]TouchFilterNet 架构", style="bold blue")
    
    def add_module_to_tree(tree, module, prefix=''):
        if isinstance(module, nn.Sequential):
            branch = tree.add(f"{prefix}Sequential")
            for i, layer in enumerate(module):
                add_module_to_tree(branch, layer, f"{i}: ")
        elif isinstance(module, nn.Conv2d):
            tree.add(f"{prefix}Conv2d(in={module.in_channels}, out={module.out_channels}, kernel={module.kernel_size})")
        elif isinstance(module, nn.ReLU):
            tree.add(f"{prefix}ReLU(inplace={module.inplace})")
        else:
            tree.add(f"{prefix}{module.__class__.__name__}")
    
    for name, module in model.named_children():
        branch = tree.add(f"{name}")
        add_module_to_tree(branch, module)
    
    console.print(tree)


def get_train_val_files():
    """获取训练和验证文件列表"""
    data_dir = "data"
    input_files = []
    target_files = []

    # 找到所有"亮屏"文件（不含"充电器"的）
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt") and "充电器" not in file:
                # 构建对应的"灭屏"文件名
                target_file = file.replace("亮屏", "灭屏")

                if os.path.exists(os.path.join(root, target_file)):
                    input_files.append(os.path.join(root, file))
                    target_files.append(os.path.join(root, target_file))
    # for file in os.listdir(data_dir):
    #     if file.startswith("亮屏") and "充电器" not in file:
    #         # 构建对应的"灭屏"文件名
    #         target_file = file.replace("亮屏", "灭屏")

    #         if os.path.exists(os.path.join(data_dir, target_file)):
    #             input_files.append(os.path.join(data_dir, file))
    #             target_files.append(os.path.join(data_dir, target_file))

    # 80%用于训练，20%用于验证
    split = int(0.8 * len(input_files))

    train_input = input_files[:split]
    train_target = target_files[:split]

    val_input = input_files[split:]
    val_target = target_files[split:]

    return train_input, train_target, val_input, val_target


def main():
    console.print(Panel.fit("触摸数据滤波模型训练", title="开始训练", border_style="green"))
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"使用设备: [bold]{device}[/bold]")
    
    # 获取训练和验证文件
    train_input, train_target, val_input, val_target = get_train_val_files()
    
    console.print(f"训练文件: [bold]{len(train_input)}[/bold] 对")
    console.print(f"验证文件: [bold]{len(val_input)}[/bold] 对")
    
    # 创建数据集
    train_dataset = TouchDataset(train_input, train_target)
    val_dataset = TouchDataset(val_input, val_target)
    
    # 创建数据加载器
    # 确定最佳批处理大小
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        # 根据GPU内存动态调整批处理大小
        batch_size = min(128, max(32, int(gpu_mem * 4)))
    else:
        batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 样本维度检查
    try:
        sample_input, _ = next(iter(train_loader))
        console.print(f"输入形状: {sample_input.shape}")
    except StopIteration:
        console.print("[bold red]警告: 数据集为空！[/bold red]")
        return
    
    # 创建模型
    model = TouchFilterNet(in_channels=1).to(device)
    
    # 定义差异保留损失函数和优化器
    criterion = DiffPreservingLoss(alpha=1.8, beta=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 显示模型信息
    # console.print(Panel.fit("模型信息", title="模型架构", border_style="blue"))
    # display_model_info(model)
    
    # 训练模型
    console.print(Panel.fit("开始训练过程", title="训练", border_style="yellow"))
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=EPOCH,
        device=device
    )
    
    # 显示最终模型信息
    console.print(Panel.fit("训练完成", title="结果", border_style="green"))
    console.print(f"最终模型已保存到: [bold]{MODEL_PATH}touch_filter_model.pth[/bold]")
    console.print(f"最佳模型已保存到: [bold]{CHECKPOINT_PATH}best_model.pth[/bold]")


if __name__ == "__main__":
    main()
