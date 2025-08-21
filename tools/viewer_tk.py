"""
    @brief         触摸数据查看器 (Tkinter版)

    @date          2024-11-07

    @version       1.0.0

    @description   基于 TouchDataParser 的触摸数据可视化工具，使用Tkinter实现
"""

import os
import time
import threading
import re
import numpy as np
from configparser import ConfigParser
from typing import List, Optional, Tuple, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from core.dataProcessor import TouchDataParser, TouchData, HeaderInfo

# 配置文件处理
configPath: str = "config.ini"
configReset: bool = False
config: ConfigParser = ConfigParser()

if not os.path.exists(configPath):
    configReset = True
else:
    try:
        config.read(configPath)
    except:
        messagebox.showerror("配置错误", "配置文件已重置")
        configReset = True

if configReset:
    config["DEFAULT"] = {
        "ColorHigh": "139,0,0",
        "ColorMedium": "255,0,0",
        "ColorLow": "255,182,193",
        "ColorHighNeg": "0,0,139",  # 深蓝色
        "ColorMediumNeg": "0,0,255",  # 蓝色
        "ColorLowNeg": "173,216,230",  # 浅蓝色
        "ThresholdHigh": "300",
        "ThresholdMedium": "150",
        "ThresholdLow": "50",
        "WinStartSizeRow": "800",
        "WinStartSizeCol": "1000",
    }
    with open(configPath, "w") as configfile:
        config.write(configfile)

# 全局配置
WINSTARTSIZE_ROW = int(config["DEFAULT"]["WinStartSizeRow"])
WINSTARTSIZE_COL = int(config["DEFAULT"]["WinStartSizeCol"])

DIFF_THRESHOLD_HIGH: int = int(config["DEFAULT"]["ThresholdHigh"])
DIFF_THRESHOLD_MEDIUM: int = int(config["DEFAULT"]["ThresholdMedium"])
DIFF_THRESHOLD_LOW: int = int(config["DEFAULT"]["ThresholdLow"])

DIFF_THRESHOLD_HIGH_COLOR: List[int] = [
    int(rgb) for rgb in config["DEFAULT"]["ColorHigh"].split(",")
]
DIFF_THRESHOLD_MEDIUM_COLOR: List[int] = [
    int(rgb) for rgb in config["DEFAULT"]["ColorMedium"].split(",")
]
DIFF_THRESHOLD_LOW_COLOR: List[int] = [
    int(rgb) for rgb in config["DEFAULT"]["ColorLow"].split(",")
]

# 添加负值颜色配置
DIFF_THRESHOLD_HIGH_COLOR_NEG: List[int] = [
    int(rgb) for rgb in config["DEFAULT"].get("ColorHighNeg", "0,0,139").split(",")
]
DIFF_THRESHOLD_MEDIUM_COLOR_NEG: List[int] = [
    int(rgb) for rgb in config["DEFAULT"].get("ColorMediumNeg", "0,0,255").split(",")
]
DIFF_THRESHOLD_LOW_COLOR_NEG: List[int] = [
    int(rgb) for rgb in config["DEFAULT"].get("ColorLowNeg", "173,216,230").split(",")
]


class ColorManager:
    """管理不同数值对应的颜色"""

    @staticmethod
    def getColorForValue(value: int) -> str:
        """获取对应数值的颜色(返回十六进制颜色代码)"""
        absValue: int = abs(value)

        # 根据值的正负选择不同的色系
        if value > 0:  # 正值使用暖色系（红色系）
            if absValue > DIFF_THRESHOLD_HIGH:
                return f"#{DIFF_THRESHOLD_HIGH_COLOR[0]:02x}{DIFF_THRESHOLD_HIGH_COLOR[1]:02x}{DIFF_THRESHOLD_HIGH_COLOR[2]:02x}"
            elif DIFF_THRESHOLD_MEDIUM < absValue <= DIFF_THRESHOLD_HIGH:
                return f"#{DIFF_THRESHOLD_MEDIUM_COLOR[0]:02x}{DIFF_THRESHOLD_MEDIUM_COLOR[1]:02x}{DIFF_THRESHOLD_MEDIUM_COLOR[2]:02x}"
            elif DIFF_THRESHOLD_LOW < absValue <= DIFF_THRESHOLD_MEDIUM:
                return f"#{DIFF_THRESHOLD_LOW_COLOR[0]:02x}{DIFF_THRESHOLD_LOW_COLOR[1]:02x}{DIFF_THRESHOLD_LOW_COLOR[2]:02x}"
            else:
                return "#FFFFFF"  # 白色
        else:  # 负值使用冷色系（蓝色系）
            if absValue > DIFF_THRESHOLD_HIGH:
                return f"#{DIFF_THRESHOLD_HIGH_COLOR_NEG[0]:02x}{DIFF_THRESHOLD_HIGH_COLOR_NEG[1]:02x}{DIFF_THRESHOLD_HIGH_COLOR_NEG[2]:02x}"
            elif DIFF_THRESHOLD_MEDIUM < absValue <= DIFF_THRESHOLD_HIGH:
                return f"#{DIFF_THRESHOLD_MEDIUM_COLOR_NEG[0]:02x}{DIFF_THRESHOLD_MEDIUM_COLOR_NEG[1]:02x}{DIFF_THRESHOLD_MEDIUM_COLOR_NEG[2]:02x}"
            elif DIFF_THRESHOLD_LOW < absValue <= DIFF_THRESHOLD_MEDIUM:
                return f"#{DIFF_THRESHOLD_LOW_COLOR_NEG[0]:02x}{DIFF_THRESHOLD_LOW_COLOR_NEG[1]:02x}{DIFF_THRESHOLD_LOW_COLOR_NEG[2]:02x}"
            else:
                return "#FFFFFF"  # 白色


class TouchDataFileProcessor:
    """处理触摸数据文件"""
    
    @staticmethod
    def loadTouchData(file_path: str) -> tuple[Optional[HeaderInfo], List[TouchData]]:
        """从文件加载触摸数据"""
        parser = TouchDataParser(file_path)
        header_info, touch_data = parser.parse()
        return header_info, touch_data

class DiffCalculator:
    """计算触摸数据差异"""
    
    @staticmethod
    def calculateDiff(
        touch_data: List[TouchData],
        method: str = "base",
        baseFrameIndex: int = 0,
        startFrame: int = None,
        endFrame: int = None,
        interval: int = 1,
    ) -> List[TouchData]:
        """计算帧间差异"""
        if not touch_data:
            return []
            
        if method == "base":
            return DiffCalculator._calculateBaseDiff(touch_data, baseFrameIndex)
        elif method == "average":
            return DiffCalculator._calculateAverageDiff(touch_data, startFrame, endFrame)
        elif method == "interval":
            return DiffCalculator._calculateIntervalDiff(touch_data, interval)
        else:
            raise ValueError("未知的计算方法")
    
    @staticmethod
    def _calculateBaseDiff(touch_data: List[TouchData], baseFrameIndex: int) -> List[TouchData]:
        """基于基准帧计算差异"""
        if baseFrameIndex >= len(touch_data):
            baseFrameIndex = 0
            
        base_frame = touch_data[baseFrameIndex]
        diff_data = []
        
        for i, frame in enumerate(touch_data[1:], 1):
            if frame.data_matrix is not None and base_frame.data_matrix is not None:
                diff_matrix = frame.data_matrix - base_frame.data_matrix
                diff_frame = TouchData(
                    frame_number=frame.frame_number,
                    timestamp=frame.timestamp,
                    tx_count=frame.tx_count,
                    rx_count=frame.rx_count,
                    data_matrix=diff_matrix
                )
                diff_data.append(diff_frame)
        
        return diff_data
    
    @staticmethod
    def _calculateAverageDiff(touch_data: List[TouchData], startFrame: int, endFrame: int) -> List[TouchData]:
        """基于区间平均值计算差异"""
        if startFrame is None or endFrame is None:
            startFrame, endFrame = 0, min(5, len(touch_data) - 1)
        
        startFrame = max(0, startFrame)
        endFrame = min(len(touch_data) - 1, endFrame)
        
        # 计算平均矩阵
        avg_matrix = None
        count = 0
        for i in range(startFrame, endFrame + 1):
            if i < len(touch_data) and touch_data[i].data_matrix is not None:
                if avg_matrix is None:
                    avg_matrix = touch_data[i].data_matrix.copy()
                else:
                    avg_matrix += touch_data[i].data_matrix
                count += 1
        
        if count > 0:
            avg_matrix = avg_matrix / count
        
        # 计算差异
        diff_data = []
        for i, frame in enumerate(touch_data[1:], 1):
            if frame.data_matrix is not None and avg_matrix is not None:
                diff_matrix = frame.data_matrix - avg_matrix
                diff_frame = TouchData(
                    frame_number=frame.frame_number,
                    timestamp=frame.timestamp,
                    tx_count=frame.tx_count,
                    rx_count=frame.rx_count,
                    data_matrix=diff_matrix.astype(int)
                )
                diff_data.append(diff_frame)
        
        return diff_data
    
    @staticmethod
    def _calculateIntervalDiff(touch_data: List[TouchData], interval: int) -> List[TouchData]:
        """基于帧间隔计算差异"""
        diff_data = []
        
        # 前几帧用零填充
        for i in range(interval):
            if i < len(touch_data):
                frame = touch_data[i]
                if frame.data_matrix is not None:
                    zero_matrix = frame.data_matrix * 0
                    diff_frame = TouchData(
                        frame_number=frame.frame_number,
                        timestamp=frame.timestamp,
                        tx_count=frame.tx_count,
                        rx_count=frame.rx_count,
                        data_matrix=zero_matrix
                    )
                    diff_data.append(diff_frame)
        
        # 计算间隔差异
        for i in range(interval, len(touch_data)):
            current_frame = touch_data[i]
            prev_frame = touch_data[i - interval]
            
            if (current_frame.data_matrix is not None and 
                prev_frame.data_matrix is not None):
                diff_matrix = current_frame.data_matrix - prev_frame.data_matrix
                diff_frame = TouchData(
                    frame_number=current_frame.frame_number,
                    timestamp=current_frame.timestamp,
                    tx_count=current_frame.tx_count,
                    rx_count=frame.rx_count,
                    data_matrix=diff_matrix
                )
                diff_data.append(diff_frame)
        
        return diff_data

class DataGrid(tk.Frame):
    """自定义数据表格组件"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.cells = {}  # 保存所有单元格
        self.tx_labels = []  # 保存TX标签
        self.rx_labels = []  # 保存RX标签
        self.tx_size = 0
        self.rx_size = 0
        
        self.canvas = tk.Canvas(self)
        self.vscroll = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hscroll = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=self.vscroll.set, xscrollcommand=self.hscroll.set)
        
        self.frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        
        self.vscroll.pack(side="right", fill="y")
        self.hscroll.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.frame.bind("<Configure>", self._on_frame_configure)
        
    def _on_frame_configure(self, event=None):
        """更新滚动区域"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def updateGrid(self, touch_data: TouchData) -> None:
        """更新表格显示"""
        if touch_data.data_matrix is None:
            return
        
        tx_size = touch_data.tx_count
        rx_size = touch_data.rx_count
        
        if tx_size != self.tx_size or rx_size != self.rx_size:
            self._adjustGridSize(tx_size, rx_size)
        
        self._populateGrid(touch_data.data_matrix, tx_size, rx_size)
    
    def _adjustGridSize(self, tx_size: int, rx_size: int) -> None:
        """调整表格大小"""
        # 清除现有单元格
        for widget in self.frame.winfo_children():
            widget.destroy()
        
        self.cells = {}
        self.tx_labels = []
        self.rx_labels = []
        
        # 创建表头
        empty_corner = tk.Label(self.frame, text="", width=4, height=2, 
                               relief="ridge", bg="#f0f0f0")
        empty_corner.grid(row=0, column=0, sticky="nsew")
        
        # 创建列标签 (RX)
        for rx in range(rx_size):
            label = tk.Label(self.frame, text=f"RX{rx+1}", width=4, height=1, 
                            relief="ridge", bg="#f0f0f0")
            label.grid(row=0, column=rx+1, sticky="nsew")
            self.rx_labels.append(label)
        
        # 创建行标签 (TX)
        for tx in range(tx_size):
            label = tk.Label(self.frame, text=f"TX{tx+1}", width=4, height=1, 
                            relief="ridge", bg="#f0f0f0")
            label.grid(row=tx+1, column=0, sticky="nsew")
            self.tx_labels.append(label)
        
        # 创建单元格
        for tx in range(tx_size):
            row_cells = {}
            for rx in range(rx_size):
                cell = tk.Label(self.frame, text="0", width=4, height=1, 
                               relief="ridge", bg="white")
                cell.grid(row=tx+1, column=rx+1, sticky="nsew")
                row_cells[rx] = cell
            self.cells[tx] = row_cells
        
        self.tx_size = tx_size
        self.rx_size = rx_size
        
        # 更新画布滚动区域
        self.frame.update_idletasks()
        self._on_frame_configure()
    
    def _populateGrid(self, data_matrix, tx_size: int, rx_size: int) -> None:
        """填充表格数据"""
        for tx in range(tx_size):
            for rx in range(rx_size):
                value = int(data_matrix[tx, rx])
                cell = self.cells[tx][rx]
                cell.config(
                    text=str(value), 
                    bg=ColorManager.getColorForValue(value),
                    fg="white" if ColorManager.getColorForValue(value) != "#FFFFFF" else "black"
                )

class TouchDataViewerApp:
    """触摸数据查看器应用"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("触摸数据查看器 (Tkinter版)")
        self.root.geometry(f"{WINSTARTSIZE_ROW}x{WINSTARTSIZE_COL}")
        
        self.header_info: Optional[HeaderInfo] = None
        self.touch_data: List[TouchData] = []
        self.diff_data: List[TouchData] = []
        self.current_frame_index: int = 0
        self.total_frames: int = 0
        self.auto_play_active: bool = False
        self.diff_method: str = "base"
        self.base_frame_index: int = 0
        self.start_frame: int = 0
        self.end_frame: int = 0
        self.interval: int = 1
        self.first_load: bool = True
        
        self._initialize_ui()
        
    def _initialize_ui(self) -> None:
        """初始化用户界面"""
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)
        
        # 文件选择
        self._create_file_selection_section()
        
        # 模式选择
        self._create_mode_selection_section()
        
        # 差异计算选项
        self._create_diff_options_section()
        
        # 数据表格
        self.data_grid = DataGrid(self.main_frame)
        self.data_grid.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 导航控件
        self._create_navigation_section()
        
        # 播放间隔设置
        self._create_interval_time_section()
        
        # 跳转控件
        self._create_jump_section()
        
        # 刷新按钮
        self._create_refresh_button()
        
        # 信息显示
        self._create_info_section()
    
    def _create_file_selection_section(self) -> None:
        """创建文件选择区域"""
        file_frame = tk.Frame(self.main_frame)
        file_frame.pack(fill="x", padx=5, pady=5)
        
        self.file_path_var = tk.StringVar()
        self.file_entry = tk.Entry(file_frame, textvariable=self.file_path_var)
        self.file_entry.pack(side="left", fill="x", expand=True)
        
        browse_btn = tk.Button(file_frame, text="浏览...", command=self._browse_file)
        browse_btn.pack(side="left", padx=5)
        
        load_btn = tk.Button(file_frame, text="加载文件", command=self.on_load_file)
        load_btn.pack(side="left")
    
    def _browse_file(self) -> None:
        """浏览文件"""
        file_path = filedialog.askopenfilename(
            title="选择触摸数据文件",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
    
    def _create_mode_selection_section(self) -> None:
        """创建模式选择区域"""
        mode_frame = tk.Frame(self.main_frame)
        mode_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(mode_frame, text="显示模式:").pack(side="left", padx=5)
        
        self.display_mode = tk.StringVar(value="raw")
        tk.Radiobutton(mode_frame, text="原始数据", variable=self.display_mode, 
                      value="raw", command=self._display_frame).pack(side="left")
        tk.Radiobutton(mode_frame, text="差异数据", variable=self.display_mode, 
                      value="diff", command=self._display_frame).pack(side="left")
    
    def _create_diff_options_section(self) -> None:
        """创建差异计算选项区域"""
        diff_frame = tk.LabelFrame(self.main_frame, text="差异计算选项")
        diff_frame.pack(fill="x", padx=5, pady=5)
        
        # 差异计算方法选择
        self.diff_method_var = tk.StringVar(value="base")
        
        # 基准帧模式
        base_frame = tk.Frame(diff_frame)
        base_frame.pack(fill="x", padx=5, pady=2)
        
        tk.Radiobutton(base_frame, text="基准帧模式", variable=self.diff_method_var,
                      value="base", command=self.on_diff_method_change).pack(side="left")
        tk.Label(base_frame, text="基准帧号:").pack(side="left", padx=(20,5))
        
        self.base_frame_var = tk.StringVar(value="0")
        tk.Entry(base_frame, textvariable=self.base_frame_var, width=5).pack(side="left")
        
        # 区间平均模式
        avg_frame = tk.Frame(diff_frame)
        avg_frame.pack(fill="x", padx=5, pady=2)
        
        tk.Radiobutton(avg_frame, text="区间平均模式", variable=self.diff_method_var,
                      value="average", command=self.on_diff_method_change).pack(side="left")
        tk.Label(avg_frame, text="区间:").pack(side="left", padx=(20,5))
        
        self.start_frame_var = tk.StringVar(value="0")
        tk.Entry(avg_frame, textvariable=self.start_frame_var, width=5).pack(side="left")
        
        tk.Label(avg_frame, text="到").pack(side="left", padx=5)
        
        self.end_frame_var = tk.StringVar(value="5")
        tk.Entry(avg_frame, textvariable=self.end_frame_var, width=5).pack(side="left")
        
        # 间隔帧模式
        interval_frame = tk.Frame(diff_frame)
        interval_frame.pack(fill="x", padx=5, pady=2)
        
        tk.Radiobutton(interval_frame, text="间隔帧模式", variable=self.diff_method_var,
                      value="interval", command=self.on_diff_method_change).pack(side="left")
        tk.Label(interval_frame, text="间隔:").pack(side="left", padx=(20,5))
        
        self.interval_var = tk.StringVar(value="1")
        tk.Entry(interval_frame, textvariable=self.interval_var, width=5).pack(side="left")
    
    def _create_navigation_section(self) -> None:
        """创建导航区域"""
        nav_frame = tk.Frame(self.main_frame)
        nav_frame.pack(fill="x", padx=5, pady=5)
        
        prev_btn = tk.Button(nav_frame, text="上一帧", command=self.on_previous_frame)
        prev_btn.pack(side="left", padx=5)
        
        next_btn = tk.Button(nav_frame, text="下一帧", command=self.on_next_frame)
        next_btn.pack(side="left", padx=5)
        
        self.play_btn = tk.Button(nav_frame, text="自动播放", command=self.on_toggle_auto_play)
        self.play_btn.pack(side="left", padx=5)
    
    def _create_interval_time_section(self) -> None:
        """创建播放间隔设置区域"""
        interval_frame = tk.Frame(self.main_frame)
        interval_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(interval_frame, text="播放间隔(秒):").pack(side="left", padx=5)
        
        self.interval_time_var = tk.StringVar(value="0.5")
        tk.Entry(interval_frame, textvariable=self.interval_time_var, width=5).pack(side="left")
    
    def _create_jump_section(self) -> None:
        """创建跳转区域"""
        jump_frame = tk.Frame(self.main_frame)
        jump_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(jump_frame, text="跳转到帧:").pack(side="left", padx=5)
        
        self.jump_frame_var = tk.StringVar()
        tk.Entry(jump_frame, textvariable=self.jump_frame_var, width=5).pack(side="left")
        
        jump_btn = tk.Button(jump_frame, text="跳转", command=self.on_jump_to_frame)
        jump_btn.pack(side="left", padx=5)
    
    def _create_refresh_button(self) -> None:
        """创建刷新按钮"""
        refresh_btn = tk.Button(self.main_frame, text="刷新", command=self.on_refresh)
        refresh_btn.pack(padx=5, pady=5)
    
    def _create_info_section(self) -> None:
        """创建信息显示区域"""
        info_frame = tk.Frame(self.main_frame)
        info_frame.pack(fill="x", padx=5, pady=5)
        
        # 帧信息
        self.frame_info_var = tk.StringVar(value="当前帧: 0 / 0")
        tk.Label(info_frame, textvariable=self.frame_info_var).pack(pady=2)
        
        # 头部信息
        self.header_info_var = tk.StringVar(value="头部信息: 未加载")
        tk.Label(info_frame, textvariable=self.header_info_var).pack(pady=2)
    
    def on_load_file(self) -> None:
        """加载文件事件"""
        file_path = self.file_path_var.get()
        if file_path:
            self.header_info, self.touch_data = TouchDataFileProcessor.loadTouchData(file_path)
            self.total_frames = len(self.touch_data)
            self.current_frame_index = 0
            
            # 更新头部信息显示
            if self.header_info:
                header_text = f"头部信息: {self.header_info.date} {self.header_info.time}, TX={self.header_info.tx}, RX={self.header_info.rx}"
                self.header_info_var.set(header_text)
            
            self.on_refresh()
    
    def on_diff_method_change(self) -> None:
        """差异计算方法改变事件"""
        self.diff_method = self.diff_method_var.get()
        
        if self.diff_method == "base":
            try:
                self.base_frame_index = int(self.base_frame_var.get())
            except ValueError:
                self.base_frame_index = 0
                self.base_frame_var.set("0")
        
        elif self.diff_method == "average":
            try:
                self.start_frame = int(self.start_frame_var.get())
                self.end_frame = int(self.end_frame_var.get())
            except ValueError:
                self.start_frame = 0
                self.end_frame = 5
                self.start_frame_var.set("0")
                self.end_frame_var.set("5")
        
        elif self.diff_method == "interval":
            try:
                self.interval = int(self.interval_var.get())
            except ValueError:
                self.interval = 1
                self.interval_var.set("1")
    
    def on_refresh(self) -> None:
        """刷新事件"""
        if not self.touch_data:
            return
        
        if not self.first_load:
            self.on_diff_method_change()
        else:
            self.first_load = False
        
        self.diff_data = DiffCalculator.calculateDiff(
            self.touch_data,
            self.diff_method,
            self.base_frame_index,
            self.start_frame,
            self.end_frame,
            self.interval,
        )
        
        self._update_frame_info()
        self._display_frame()
    
    def _display_frame(self) -> None:
        """显示当前帧"""
        if not self.touch_data:
            return
        
        if self.display_mode.get() == "raw":
            if self.current_frame_index < len(self.touch_data):
                frame_data = self.touch_data[self.current_frame_index]
                self.data_grid.updateGrid(frame_data)
        else:  # diff mode
            if self.current_frame_index > 0 and self.current_frame_index - 1 < len(self.diff_data):
                frame_data = self.diff_data[self.current_frame_index - 1]
                self.data_grid.updateGrid(frame_data)
    
    def _update_frame_info(self) -> None:
        """更新帧信息显示"""
        self.frame_info_var.set(f"当前帧: {self.current_frame_index + 1} / {self.total_frames}")
    
    def on_previous_frame(self) -> None:
        """上一帧事件"""
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self._update_frame_info()
            self._display_frame()
    
    def on_next_frame(self) -> None:
        """下一帧事件"""
        if self.current_frame_index < self.total_frames - 1:
            self.current_frame_index += 1
            self._update_frame_info()
            self._display_frame()
    
    def on_jump_to_frame(self) -> None:
        """跳转到指定帧事件"""
        try:
            frame_number = int(self.jump_frame_var.get()) - 1
            if 0 <= frame_number < self.total_frames:
                self.current_frame_index = frame_number
                self._update_frame_info()
                self._display_frame()
            else:
                messagebox.showerror("错误", "无效的帧号")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的帧号")
    
    def on_toggle_auto_play(self) -> None:
        """切换自动播放状态"""
        if not self.auto_play_active:
            self.auto_play_active = True
            self.play_btn.config(text="停止播放")
            self._start_auto_play()
        else:
            self.auto_play_active = False
            self.play_btn.config(text="自动播放")
    
    def _start_auto_play(self) -> None:
        """开始自动播放"""
        def auto_play():
            try:
                interval = float(self.interval_time_var.get())
            except ValueError:
                messagebox.showerror("错误", "请输入有效的间隔时间")
                self.root.after(0, self._stop_auto_play)
                return
            
            while self.auto_play_active and self.current_frame_index < self.total_frames - 1:
                self.root.after(0, self.on_next_frame)
                time.sleep(interval)
            
            self.root.after(0, self._stop_auto_play)
        
        threading.Thread(target=auto_play, daemon=True).start()
    
    def _stop_auto_play(self) -> None:
        """停止自动播放"""
        self.auto_play_active = False
        self.play_btn.config(text="自动播放")

if __name__ == "__main__":
    root = tk.Tk()
    app = TouchDataViewerApp(root)
    root.mainloop()
