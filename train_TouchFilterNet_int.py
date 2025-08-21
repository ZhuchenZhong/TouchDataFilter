"""
触摸数据滤波整数模型训练与转换脚本
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from typing import List

# 导入浮点模型和数据处理器
from train import get_train_val_files
from core.TouchFilterNet_fp import TouchFilterNet as TouchFilterNet_FP
from core.dataProcessor import TouchDataParser

from core.TouchFilterNet_int import TouchFilterNet_int, convert_fp_to_int_model

# 初始化rich控制台
console = Console()
EPOCH = 50
MODEL_PATH = f"models/{time.asctime()[4:].replace(' ','-')}-int/"
CHECKPOINT_PATH = f"{MODEL_PATH}checkpoints/"
Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

class IntDataset(Dataset):
    """整数类型的触摸数据集"""

    def __init__(self, input_files: List[str], target_files: List[str], bits: int = 8):
        """
        初始化数据集

        Args:
            input_files: 输入文件路径列表（有噪声的数据）
            target_files: 目标文件路径列表（无噪声的数据）
            bits: 整数量化位数
        """
        self.input_data = []
        self.target_data = []
        self.bits = bits

        # 计算整数量化范围
        self.qmin = -(1 << (bits - 1))
        self.qmax = (1 << (bits - 1)) - 1

        # 添加数据缓存机制
        cache_file = f"{MODEL_PATH}data_cache_{hash(tuple(sorted(input_files + target_files)))}.pt"
        if os.path.exists(cache_file):
            console.print(f"[green]从缓存加载数据: {cache_file}")
            cache_data = torch.load(cache_file)
            self.input_data = cache_data['input']
            self.target_data = cache_data['target']
            return

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
                    if (
                        input_frames[i].data_matrix is not None
                        and target_frames[i].data_matrix is not None
                    ):
                        # 直接使用整数数据，不需要归一化
                        input_matrix = input_frames[i].data_matrix.astype(np.int32)
                        target_matrix = target_frames[i].data_matrix.astype(np.int32)

                        # 确保数据在整数范围内
                        input_matrix = np.clip(input_matrix, self.qmin, self.qmax)
                        target_matrix = np.clip(target_matrix, self.qmin, self.qmax)

                        # 添加到数据集
                        self.input_data.append(input_matrix)
                        self.target_data.append(target_matrix)

                progress.update(task, advance=1)

        # 转换为张量 - 保持整数类型
        self.input_data = [
            torch.tensor(data, dtype=torch.int32).unsqueeze(0)
            for data in self.input_data
        ]
        self.target_data = [
            torch.tensor(data, dtype=torch.int32).unsqueeze(0)
            for data in self.target_data
        ]

        # 保存到缓存
        torch.save({
            'input': self.input_data,
            'target': self.target_data
        }, cache_file)
        console.print(f"[green]数据已缓存到: {cache_file}")

        console.print(f"[green]已加载 {len(self.input_data)} 个训练样本")

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]


class IntMSELoss(nn.Module):
    """整数模型的MSE损失函数"""

    def __init__(self):
        super(IntMSELoss, self).__init__()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 计算差的平方
        diff = outputs - targets
        squared_diff = diff * diff
        # 计算均值
        return torch.mean(squared_diff.float())  # 转换为float以便梯度计算


class IntDiffPreservingLoss(nn.Module):
    """整数版本的差异保留损失函数"""

    def __init__(self, alpha=2.0, beta=0.5):
        super(IntDiffPreservingLoss, self).__init__()
        self.alpha = alpha  # 控制高差异区域的权重系数
        self.beta = beta  # MSE损失和差异保留损失的平衡系数

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 基本MSE损失
        diff = outputs - targets
        mse_loss = (diff * diff).float()  # 转换为float以便梯度计算

        # 计算差异权重 - 绝对值越大的区域权重越高
        diff_weights = torch.abs(targets.float()).pow(self.alpha) + 0.5

        # 加权MSE损失
        weighted_loss = (mse_loss * diff_weights).mean()

        # 同时考虑普通MSE和加权MSE
        total_loss = self.beta * mse_loss.mean() + (1 - self.beta) * weighted_loss

        return total_loss


def train_integer_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=25,
    device="cuda",
    bits=8,
):
    """训练整数模型"""
    start_time = time.time()
    best_loss = float("inf")

    # 创建保存目录
    os.makedirs(f"{MODEL_PATH}checkpoints", exist_ok=True)

    # 添加混合精度训练支持 - 修复弃用警告
    use_amp = device != "cpu" and hasattr(torch, 'amp')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

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

        # 优化验证循环
        def evaluate_model():
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    # 验证时也使用混合精度 - 修复弃用警告
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    val_loss += loss.item()
            return val_loss / len(val_loader)

        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0

            # 处理每个批次
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)  # 更高效的梯度重置

                # 使用混合精度训练
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                    # 缩放损失并执行反向传播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 原始训练流程
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            # 计算平均训练损失
            train_loss /= len(train_loader)

            # 验证阶段
            val_loss = evaluate_model()

            # 更新学习率
            scheduler.step(val_loss)

            # 显示当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            progress.update(
                epoch_task,
                advance=1,
                train_loss=f" 训练损失: {train_loss:.6f}",
                val_loss=f" 验证损失: {val_loss:.6f} LR: {current_lr:.6f}",
            )

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f"{CHECKPOINT_PATH}best_model_int.pth")

            # 保存检查点
            if (epoch + 1) % 5 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "bits": bits,
                    },
                    f"{CHECKPOINT_PATH}model_int_epoch_{epoch+1}.pth",
                )

    # 保存最终模型
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "bits": bits,
        },
        f"{MODEL_PATH}touch_filter_model_int.pth",
    )

    # 计算训练时间
    time_elapsed = time.time() - start_time
    console.print(
        f"[bold green]训练完成，用时 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒[/bold green]"
    )
    console.print(f"[bold green]最佳验证损失: {best_loss:.6f}[/bold green]")

    return model


# def get_train_val_files():
#     """获取训练和验证文件列表"""
#     data_dir = "data"
#     input_files = []
#     target_files = []

#     # 找到所有"亮屏"文件（不含"充电器"的）
#     for root, _, files in os.walk(data_dir):
#         for file in files:
#             if file.endswith(".txt") and "充电器" not in file and "亮屏" in file:
#                 # 构建对应的"灭屏"文件名
#                 target_file = file.replace("亮屏", "灭屏")

#                 target_path = os.path.join(root, target_file)
#                 if os.path.exists(target_path):
#                     input_files.append(os.path.join(root, file))
#                     target_files.append(target_path)

#     # 80%用于训练，20%用于验证
#     split = int(0.8 * len(input_files))

#     train_input = input_files[:split]
#     train_target = target_files[:split]

#     val_input = input_files[split:]
#     val_target = target_files[split:]

#     return train_input, train_target, val_input, val_target


def create_integer_model(in_channels=1, bits=8):
    """创建一个新的整数模型"""
    model = TouchFilterNet_int(in_channels=in_channels, bits=bits)
    return model


def convert_model_from_fp(fp_model_path, bits=8, save_path=None):
    """从浮点模型转换为整数模型"""
    console.print(
        Panel.fit(
            f"从浮点模型转换为{bits}位整数模型", title="转换模型", border_style="blue"
        )
    )

    # 加载浮点模型
    fp_model = TouchFilterNet_FP(in_channels=1)
    try:
        fp_model.load_state_dict(torch.load(fp_model_path, map_location="cpu"))
        fp_model.eval()
        console.print(f"[green]已成功加载浮点模型: {fp_model_path}[/green]")
    except Exception as e:
        console.print(f"[bold red]加载浮点模型失败: {str(e)}[/bold red]")
        return None

    # 转换模型
    int_model = convert_fp_to_int_model(fp_model_path, output_path=save_path, bits=bits)

    if save_path:
        console.print(f"[green]已保存整数模型到: {save_path}[/green]")

    return int_model


def display_model_info(model, bits=8):
    """显示整数模型信息"""
    # 创建模型信息表格
    table = Table(title="整数模型信息")
    table.add_column("参数", justify="left", style="cyan")
    table.add_column("值", justify="left", style="green")

    table.add_row("量化位数", f"{bits}位")
    table.add_row("整数范围", f"{-(1 << (bits - 1))} 到 {(1 << (bits - 1)) - 1}")

    # 显示表格
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="触摸数据滤波整数模型训练与转换")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "convert"],
        help="操作模式: train=从头训练整数模型, convert=转换现有浮点模型",
    )
    parser.add_argument("--bits", type=int, default=8, help="整数量化位数(默认: 8)")
    parser.add_argument(
        "--model", type=str, default=None, help="用于转换的浮点模型路径"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCH, help=f"训练轮次(默认: {EPOCH})"
    )

    args = parser.parse_args()

    # 全局位数设置
    bits = args.bits

    if args.mode == "convert":
        if not args.model:
            # 尝试查找最近的模型
            model_dirs = sorted(
                [
                    d
                    for d in os.listdir("models")
                    if os.path.isdir(os.path.join("models", d))
                ]
            )
            if not model_dirs:
                console.print("[bold red]错误: 未找到任何模型目录![/bold red]")
                return

            latest_model_dir = os.path.join("models", model_dirs[-1])
            default_model = os.path.join(latest_model_dir, "touch_filter_model_int.pth")
            best_model = os.path.join(latest_model_dir, "checkpoints", "best_model_int.pth")

            if os.path.exists(default_model):
                fp_model_path = default_model
            elif os.path.exists(best_model):
                fp_model_path = best_model
            else:
                console.print("[bold red]错误: 未找到默认模型或最佳模型![/bold red]")
                return
        else:
            fp_model_path = args.model

        # 创建保存路径
        save_path = f"{MODEL_PATH}touch_filter_model_int_{bits}bit.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 转换模型
        convert_model_from_fp(fp_model_path, bits=bits, save_path=save_path)
        console.print(f"[bold green]整数模型已保存到: {save_path}[/bold green]")

    elif args.mode == "train":
        console.print(
            Panel.fit(
                f"训练{bits}位整数触摸数据滤波模型",
                title="开始训练",
                border_style="green",
            )
        )

        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"使用设备: [bold]{device}[/bold]")

        # 获取训练和验证文件
        train_input, train_target, val_input, val_target = get_train_val_files()

        console.print(f"训练文件: [bold]{len(train_input)}[/bold] 对")
        console.print(f"验证文件: [bold]{len(val_input)}[/bold] 对")

        # 显示整数模型信息
        display_model_info(None, bits=bits)

        # 创建数据集
        train_dataset = IntDataset(train_input, train_target, bits=bits)
        val_dataset = IntDataset(val_input, val_target, bits=bits)

        # 确定最佳批处理大小
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            # 根据GPU内存动态调整批处理大小
            batch_size = min(128, max(32, int(gpu_mem * 4)))
        else:
            batch_size = 32
        
        console.print(f"使用批处理大小: [bold]{batch_size}[/bold]")
        
        # 创建数据加载器，启用pin_memory和持久性工作进程
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=min(8, os.cpu_count()),
            pin_memory=device != "cpu",
            persistent_workers=True if min(4, os.cpu_count()) > 0 else False,
            prefetch_factor=2 if device != "cpu" else None
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        # 样本维度检查
        try:
            sample_input, _ = next(iter(train_loader))
            console.print(f"输入形状: {sample_input.shape}")
        except StopIteration:
            console.print("[bold red]警告: 数据集为空！[/bold red]")
            return

        # 创建整数模型
        model = create_integer_model(in_channels=1, bits=bits).to(device)

        # 定义损失函数和优化器
        criterion = IntDiffPreservingLoss(alpha=2.0, beta=0.5)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        console.print(Panel.fit("开始训练过程", title="训练", border_style="yellow"))
        model = train_integer_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=args.epochs,
            device=device,
            bits=bits,
        )

        # 显示最终模型信息
        console.print(Panel.fit("训练完成", title="结果", border_style="green"))
        console.print(
            f"最终整数模型已保存到: [bold]{MODEL_PATH}touch_filter_model_int.pth[/bold]"
        )
        console.print(
            f"最佳整数模型已保存到: [bold]{CHECKPOINT_PATH}best_model_int.pth[/bold]"
        )


if __name__ == "__main__":
    main()
