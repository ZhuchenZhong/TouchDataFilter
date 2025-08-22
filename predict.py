"""
触摸数据滤波预测脚本
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from datetime import datetime
from pathlib import Path

# 导入自定义模块
from core.dataProcessor import TouchDataParser, TouchData
from train import TouchFilterNet               # 复用训练脚本中的模型定义
from train import MODEL_PATH                   # 导入模型路径常量

# plt.rcParams['font.sans-serif'] = ['SimHei']   # 设置中文字体
# plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# 初始化rich控制台
console = Console()

class TouchDataPredictor:
    def __init__(self, model_path):
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"使用设备: [bold]{self.device}[/bold]")
        
        # 加载模型
        self.model = self.load_model(model_path)
        
        # 初始化GUI
        self.root = tk.Tk()
        self.root.title("触摸数据滤波预测器")
        self.root.geometry("1200x800")
        self.setup_gui(str(Path(model_path).parent))  # 传入模型路径的父目录

    def load_model(self, model_path):
        """加载预训练模型，支持浮点模型和整数模型"""
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
                console.print("[cyan]检测到整数模型，使用TouchFilterNet_int架构[/cyan]")
            else:
                model = TouchFilterNet(in_channels=1).to(self.device)
                console.print("[cyan]使用标准浮点模型架构[/cyan]")
            
            # 检查加载的内容是否为嵌套字典（包含model_state_dict键）
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                console.print("[cyan]检测到嵌套模型状态字典[/cyan]")
                model.load_state_dict(checkpoint["model_state_dict"])
                
                # 如果有bits信息，显示出来
                if "bits" in checkpoint:
                    console.print(f"[green]加载的整数模型位宽: {checkpoint['bits']}位[/green]")
            else:
                # 直接加载状态字典
                model.load_state_dict(checkpoint)
            
            model.eval()  # 设置为评估模式
            console.print(f"[green]已成功加载模型: {model_path}[/green]")
            return model
        except Exception as e:
            console.print(f"[bold red]加载模型失败: {str(e)}[/bold red]")
            
            # 如果常规加载失败且不是整数模型尝试，尝试以整数模型格式加载
            if not is_int_model: # type: ignore
                console.print("尝试使用整形格式加载。")
                try:
                    from core.TouchFilterNet_int import TouchFilterNet_int
                    model = TouchFilterNet_int(in_channels=1).to(self.device)
                    
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        model.load_state_dict(checkpoint)
                        
                    model.eval()
                    console.print(f"[green]已成功以整数模型格式加载: {model_path}[/green]")
                    return model
                except Exception as e2:
                    console.print(f"[bold red]加载整形模型失败: {str(e2)}[/bold red]")
            
            return None

    def setup_gui(self, model_path):
        """设置GUI界面"""
        # 创建顶部控制区域
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 文件选择按钮
        select_btn = tk.Button(control_frame, text="选择触摸数据文件", command=self.select_file, padx=10, pady=5)
        select_btn.pack(side=tk.LEFT, padx=5)
        
        # 模型选择
        model_label = tk.Label(control_frame, text="选择模型:")
        model_label.pack(side=tk.LEFT, padx=(20, 5))

        self.model_var = tk.StringVar(value=f"{model_path}/touch_filter_model.pth")
        model_options = [f"{model_path}/touch_filter_model.pth", f"{model_path}/checkpoints/best_model.pth"]

        # 查找所有检查点
        # checkpoint_dir = Path(model_path) / "checkpoints"
        # if checkpoint_dir.exists():
        #     for cp_file in checkpoint_dir.glob("model_epoch_*.pth"):
        #         model_options.append(str(cp_file))
        
        model_menu = tk.OptionMenu(control_frame, self.model_var, *model_options)
        model_menu.pack(side=tk.LEFT, padx=5)
        model_menu.config(width=20)
        
        # 加载模型按钮
        load_model_btn = tk.Button(control_frame, text="加载模型", command=self.reload_model, padx=10, pady=5)
        load_model_btn.pack(side=tk.LEFT, padx=5)
        
        # 预测并保存按钮
        self.predict_btn = tk.Button(control_frame, text="预测并保存", command=self.predict_and_save, padx=10, pady=5, state=tk.DISABLED)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # 创建显示区域
        display_frame = tk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 分割显示区域为两部分
        left_frame = tk.Frame(display_frame, width=580, height=600)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = tk.Frame(display_frame, width=580, height=600)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # 左侧显示原始数据
        left_label = tk.Label(left_frame, text="原始触摸数据")
        left_label.pack()
        
        self.fig1, self.ax1 = plt.subplots(figsize=(6, 6))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=left_frame)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 右侧显示预测数据
        right_label = tk.Label(right_frame, text="滤波后触摸数据")
        right_label.pack()
        
        self.fig2, self.ax2 = plt.subplots(figsize=(6, 6))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=right_frame)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪。请选择一个触摸数据文件...")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 添加帧选择滑块
        slider_frame = tk.Frame(self.root)
        slider_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        slider_label = tk.Label(slider_frame, text="帧选择:")
        slider_label.pack(side=tk.LEFT, padx=5)
        
        self.frame_slider = tk.Scale(slider_frame, from_=0, to=0, orient=tk.HORIZONTAL, 
                                     command=self.update_display, length=800)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 初始化数据存储
        self.selected_file = None
        self.input_frames = []
        self.predicted_frames = []

    def select_file(self):
        """选择数据文件"""
        file_path = filedialog.askopenfilename(
            initialdir="data",
            title="选择触摸数据文件",
            filetypes=(("文本文件", "*.txt"), ("所有文件", "*.*"))
        )
        
        if file_path:
            self.selected_file = file_path
            self.status_var.set(f"已选择文件: {os.path.basename(file_path)}")
            self.load_data(file_path)
            self.predict_btn.config(state=tk.NORMAL)

    def reload_model(self):
        """重新加载所选模型"""
        model_path = self.model_var.get()
        
        # 检查所选模型是否存在
        if not os.path.exists(model_path):
            messagebox.showerror("错误", f"找不到模型文件: {model_path}")
            return
        
        self.model = self.load_model(model_path)
        
        if self.model is not None:
            self.status_var.set(f"已加载模型: {os.path.basename(model_path)}")
            
            # 如果已有数据，重新预测
            if hasattr(self, 'input_frames') and len(self.input_frames) > 0:
                self.process_data()
        else:
            self.status_var.set("加载模型失败")

    def load_data(self, file_path):
        """加载触摸数据"""
        try:
            # 显示加载进度
            with Progress(
                TextColumn("[bold blue]加载数据...", justify="right"),
                BarColumn(),
                TextColumn("[bold green]{task.percentage:.0f}%"),
            ) as progress:
                task = progress.add_task("[cyan]加载文件...", total=1)
                
                # 解析数据文件
                self.header_info, self.input_frames = TouchDataParser(file_path).parse()
                
                progress.update(task, completed=1)
            
            # 更新滑块范围
            max_frames = len(self.input_frames) - 1
            self.frame_slider.config(to=max_frames)
            self.frame_slider.set(0)
            
            console.print(f"[green]已加载 {len(self.input_frames)} 帧数据[/green]")
            self.status_var.set(f"已加载 {len(self.input_frames)} 帧数据")
            
            # 处理数据（进行预测）
            self.process_data()
            
        except Exception as e:
            console.print(f"[bold red]加载数据失败: {str(e)}[/bold red]")
            messagebox.showerror("错误", f"加载数据失败: {str(e)}")

    def process_data(self):
        """处理数据并预测"""
        if self.model is None:
            messagebox.showwarning("警告", "请先加载有效的模型")
            return
        
        self.predicted_frames = []
        
        with Progress(
            TextColumn("[bold blue]预测中...", justify="right"),
            BarColumn(),
            TextColumn("[bold green]{task.percentage:.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]处理帧...", total=len(self.input_frames))
            
            for frame in self.input_frames:
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
                    
                    self.predicted_frames.append(predicted_frame)
                else:
                    # 如果没有有效数据，添加空帧，但仍需提供frame_number
                    predicted_frame = TouchData(
                        frame_number=frame.frame_number if hasattr(frame, 'frame_number') else 0
                    )
                    self.predicted_frames.append(predicted_frame)
                    
                progress.update(task, advance=1)
                
        console.print(f"[green]预测完成: {len(self.predicted_frames)} 帧[/green]")
        self.status_var.set(f"预测完成: {len(self.predicted_frames)} 帧")
        
        # 显示第一帧
        self.update_display(0)

    def update_display(self, frame_idx):
        """更新显示指定帧的图像"""
        frame_idx = int(frame_idx)
        
        # 清除现有图像
        self.ax1.clear()
        self.ax2.clear()
        
        if frame_idx < len(self.input_frames) and self.input_frames[frame_idx].data_matrix is not None:
            # 绘制原始数据
            self.ax1.imshow(
                self.input_frames[frame_idx].data_matrix, # type: ignore
                cmap='viridis',
                interpolation='none',
                vmin=0,
                vmax=1000
            )
            self.ax1.set_title(f"RawData (Frame {frame_idx})")
            
            # 绘制预测数据
            if frame_idx < len(self.predicted_frames) and self.predicted_frames[frame_idx].data_matrix is not None:
                self.ax2.imshow(
                    self.predicted_frames[frame_idx].data_matrix,
                    cmap='viridis',
                    interpolation='none',
                    vmin=0,
                    vmax=1000
                )
                self.ax2.set_title(f"ProcessedData (Frame {frame_idx})")
        
        self.canvas1.draw()
        self.canvas2.draw()

    def predict_and_save(self):
        """预测并保存结果"""
        if not self.selected_file or not self.predicted_frames:
            messagebox.showwarning("警告", "没有可保存的数据")
            return
        
        # 创建保存文件名
        base_name = os.path.basename(self.selected_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"滤波_{timestamp}_{base_name}"
        
        save_path = filedialog.asksaveasfilename(
            initialdir="data",
            initialfile=save_name,
            title="保存滤波数据",
            filetypes=(("文本文件", "*.txt"), ("所有文件", "*.*"))
        )
        
        if save_path:
            try:
                # 使用原始解析器的格式保存预测结果
                parser = TouchDataParser("")
                parser.header_info = self.header_info
                parser.touch_data = self.predicted_frames
                parser.save_to_file(save_path)
                
                console.print(f"[green]已保存滤波数据到: {save_path}[/green]")
                messagebox.showinfo("成功", f"已保存滤波数据到:\n{save_path}")
                self.status_var.set(f"已保存滤波数据到: {os.path.basename(save_path)}")
            except Exception as e:
                console.print(f"[bold red]保存数据失败: {str(e)}[/bold red]")
                messagebox.showerror("错误", f"保存数据失败: {str(e)}")

    def run(self):
        """运行应用"""
        self.root.mainloop()


def main():
    console.print(Panel.fit("触摸数据滤波预测工具", title="开始", border_style="green"))

    if len(sys.argv) > 1:
        # 如果有命令行参数，使用第一个参数作为模型目录
        _root = sys.argv[1]
        if not os.path.exists(_root):
            console.print(f"[bold red]错误: 指定的模型目录不存在: {_root}[/bold red]")
            messagebox.showerror("错误", f"指定的模型目录不存在: {_root}")
            return
    else:
        # 如果没有命令行参数，查找默认模型目录
        try:
            __all = [os.path.join(Path(MODEL_PATH).parent, f) for f in sorted(os.listdir(Path(MODEL_PATH).parent))]
            console.print(f"[blue]找到模型目录: {__all}[/blue]")
            _root = __all[-1]
        except IndexError:
            console.print("[bold red]错误: 未找到模型目录![/bold red]")
            messagebox.showerror("错误", "未找到模型目录!\n请先训练模型或将模型文件放置在models目录下。")
            return

    # 检查默认模型是否存在
    default_model = f'{_root}/touch_filter_model.pth'
    best_model = f'{_root}/checkpoints/best_model.pth'

    model_path = None
    if os.path.exists(default_model):
        model_path = default_model
        console.print(f"[blue]使用默认模型: {default_model}[/blue]")
    elif os.path.exists(best_model):
        model_path = best_model
        console.print(f"[blue]使用最佳模型: {best_model}[/blue]")
    else:
        # 查找任何可用的模型
        for root, _, files in os.walk("models"):
            for file in files:
                if file.endswith(".pth"):
                    model_path = os.path.join(root, file)
                    if not os.path.exists(model_path):
                        continue
                    console.print(f"[yellow]找到可用模型: {model_path}[/yellow]")
                    break
            if model_path:
                break

    if not model_path:
        console.print("[bold red]错误: 未找到任何可用的模型文件![/bold red]")
        messagebox.showerror("错误", "未找到任何可用的模型文件!\n请先训练模型或将模型文件放置在models目录下。")
        return

    # 启动预测器
    predictor = TouchDataPredictor(model_path)
    predictor.run()


if __name__ == "__main__":
    main()
