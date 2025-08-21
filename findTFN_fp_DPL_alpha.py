"""
触摸数据滤波模型 - DiffPreservingLoss alpha参数灵敏度分析
此脚本用于探索不同alpha值对模型性能的影响
"""
import os
import time
from pathlib import Path
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table

from core.TouchFilterNet_fp import TouchFilterNet, DiffPreservingLoss
from train import TouchDataset, get_train_val_files

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 初始化rich控制台
console = Console()

# 设置实验参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 确定最佳批处理大小
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    # 根据GPU内存动态调整批处理大小
    BATCH_SIZE = min(128, max(32, int(gpu_mem * 4)))
else:
    BATCH_SIZE = 32
EPOCHS = 50
ALPHA_VALUES = [1.8, 2.0]
BETA = 0.5  # 固定beta值
RESULTS_DIR = f"results/alpha_sensitivity_{time.strftime('%Y%m%d_%H%M%S')}"


def train_and_evaluate(alpha, train_loader, val_loader, epochs=100):
    """使用指定的alpha值训练和评估模型"""
    model = TouchFilterNet(in_channels=1).to(DEVICE)
    criterion = DiffPreservingLoss(alpha=alpha, beta=BETA)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []
    
    # 使用进度条显示
    with Progress(
        TextColumn(f"[cyan]Alpha={alpha:.1f} 轮次 {{task.completed}}/{{task.total}}"),
        BarColumn(),
        TextColumn("[bold green]{task.percentage:.0f}%"),
        TextColumn("[bold]{task.fields[train_loss]}{task.fields[val_loss]}"),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        # 创建轮次进度条任务
        epoch_task = progress.add_task("", total=epochs, train_loss="", val_loss="")

        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0

            # 处理每个批次
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 计算平均训练损失
            train_loss /= len(train_loader)
            training_losses.append(train_loss)

            # 验证阶段
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                val_loss /= len(val_loader)
                validation_losses.append(val_loss)

            # 更新进度条信息
            progress.update(epoch_task, 
                           advance=1, 
                           train_loss=f" 训练损失: {train_loss:.6f}", 
                           val_loss=f" 验证损失: {val_loss:.6f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 确保目录存在
                os.makedirs(f'{RESULTS_DIR}/models', exist_ok=True)
                torch.save({
                    'alpha': alpha,
                    'beta': BETA,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, f'{RESULTS_DIR}/models/best_model_alpha_{alpha:.1f}.pth')
    
    return {
        'alpha': alpha,
        'final_train_loss': training_losses[-1],
        'final_val_loss': validation_losses[-1],
        'best_val_loss': best_val_loss,
        'training_losses': training_losses,
        'validation_losses': validation_losses
    }


def plot_results(all_results):
    """绘制不同alpha值的结果比较图"""
    plt.figure(figsize=(15, 10))
    
    # 1. 最终验证损失 vs alpha值
    plt.subplot(2, 2, 1)
    alphas = [result['alpha'] for result in all_results]
    final_val_losses = [result['final_val_loss'] for result in all_results]
    best_val_losses = [result['best_val_loss'] for result in all_results]
    
    plt.plot(alphas, final_val_losses, 'o-', label='最终验证损失')
    plt.plot(alphas, best_val_losses, 's-', label='最佳验证损失')
    plt.xlabel('Alpha值')
    plt.ylabel('验证损失')
    # plt.title('不同Alpha值的验证损失比较')
    plt.title('Different Alpha Values vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 2. 每个alpha值的训练过程
    plt.subplot(2, 2, 2)
    for result in all_results:
        plt.plot(range(1, len(result['validation_losses'])+1), 
                 result['validation_losses'], 
                 label=f"Alpha={result['alpha']:.1f}")
    
    plt.xlabel('轮次')
    plt.ylabel('验证损失')
    # plt.title('不同Alpha值的验证损失随轮次变化')
    plt.title('Validation Loss vs Epochs for Different Alpha Values')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 3. 训练损失 vs 验证损失 (最佳alpha值)
    best_result = min(all_results, key=lambda x: x['best_val_loss'])
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(best_result['training_losses'])+1), best_result['training_losses'], 'b-', label='训练损失')
    plt.plot(range(1, len(best_result['validation_losses'])+1), best_result['validation_losses'], 'r-', label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    # plt.title(f"最佳Alpha={best_result['alpha']:.1f}的训练过程")
    plt.title(f"Training and Validation Losses for Best Alpha={best_result['alpha']:.1f}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 4. 所有alpha值的最终训练损失和验证损失对比
    plt.subplot(2, 2, 4)
    final_train_losses = [result['final_train_loss'] for result in all_results]
    
    width = 0.35
    x = np.arange(len(alphas))
    plt.bar(x - width/2, final_train_losses, width, label='训练损失')
    plt.bar(x + width/2, final_val_losses, width, label='验证损失')
    plt.xlabel('Alpha值')
    plt.ylabel('损失')
    # plt.title('最终训练损失和验证损失对比')
    plt.title('Final Training and Validation Losses Comparison')
    plt.xticks(x, [f"{alpha:.1f}" for alpha in alphas])
    plt.legend()
    plt.show()
    
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(f'{RESULTS_DIR}/alpha_sensitivity_results.png')
    plt.close()


def save_results_table(all_results):
    """保存结果表格到文件"""
    # 创建结果表格
    table = Table(title="Alpha参数灵敏度分析结果")
    table.add_column("Alpha", justify="center", style="cyan")
    table.add_column("最终训练损失", justify="right", style="green")
    table.add_column("最终验证损失", justify="right", style="yellow")
    table.add_column("最佳验证损失", justify="right", style="red")
    
    # 排序结果（按最佳验证损失）
    sorted_results = sorted(all_results, key=lambda x: x['best_val_loss'])
    
    # 填充表格
    for result in sorted_results:
        table.add_row(
            f"{result['alpha']:.1f}",
            f"{result['final_train_loss']:.6f}",
            f"{result['final_val_loss']:.6f}",
            f"{result['best_val_loss']:.6f}"
        )
    
    # 显示表格
    console.print(table)
    
    # 保存结果到CSV文件
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f'{RESULTS_DIR}/alpha_sensitivity_results.csv', 'w') as f:
        f.write("Alpha,最终训练损失,最终验证损失,最佳验证损失\n")
        for result in sorted_results:
            f.write(f"{result['alpha']:.1f},{result['final_train_loss']:.6f},{result['final_val_loss']:.6f},{result['best_val_loss']:.6f}\n")


def main():
    console.print(Panel.fit("触摸数据滤波模型 - DiffPreservingLoss Alpha参数灵敏度分析", 
                           title="开始分析", border_style="green"))
    
    # 获取训练和验证文件
    train_input, train_target, val_input, val_target = get_train_val_files()
    
    console.print(f"训练文件: [bold]{len(train_input)}[/bold] 对")
    console.print(f"验证文件: [bold]{len(val_input)}[/bold] 对")
    
    # 创建数据集
    train_dataset = TouchDataset(train_input, train_target)
    val_dataset = TouchDataset(val_input, val_target)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 创建结果目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 对不同的alpha值进行训练和评估
    all_results = []
    
    start_time = time.time()
    
    for alpha in ALPHA_VALUES:
        console.print(f"[bold yellow]开始测试 Alpha = {alpha}[/bold yellow]")
        result = train_and_evaluate(alpha, train_loader, val_loader, epochs=EPOCHS)
        all_results.append(result)
        console.print(f"[bold green]Alpha = {alpha} 测试完成: 最佳验证损失 = {result['best_val_loss']:.6f}[/bold green]")
    
    # 分析和可视化结果
    console.print(Panel.fit("生成结果分析", title="分析阶段", border_style="magenta"))
    
    # 保存结果表格
    save_results_table(all_results)
    
    # 绘制结果图表
    plot_results(all_results)
    
    # 找出最佳alpha值
    best_result = min(all_results, key=lambda x: x['best_val_loss'])
    best_alpha = best_result['alpha']
    
    # 计算总时间
    time_elapsed = time.time() - start_time
    
    # 显示最终结果
    console.print(Panel.fit(
        f"分析完成！最佳Alpha值为 [bold]{best_alpha:.1f}[/bold]\n"
        f"最佳验证损失: [bold]{best_result['best_val_loss']:.6f}[/bold]\n"
        f"结果已保存到 [bold]{RESULTS_DIR}[/bold]\n"
        f"总耗时: [bold]{time_elapsed//60:.0f}分{time_elapsed%60:.0f}秒[/bold]",
        title="分析结果", border_style="green"))


if __name__ == "__main__":
    main()