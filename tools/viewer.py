"""
    @brief         触摸数据查看器

    @date          2024-11-07

    @version       1.0.0

    @description   基于 TouchDataParser 的触摸数据可视化工具
"""

import os
import time
import threading
from configparser import ConfigParser
from typing import List, Optional
import wx
import wx.grid as gridlib
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
        wx.MessageBox("配置文件已重置", "配置错误", wx.OK | wx.ICON_ERROR)
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
                    rx_count=current_frame.rx_count,
                    data_matrix=diff_matrix
                )
                diff_data.append(diff_frame)
        
        return diff_data

class GridManager:
    """管理表格显示"""
    
    def __init__(self, grid: gridlib.Grid):
        self.grid: gridlib.Grid = grid
        # 设置表格基本属性
        self.grid.EnableEditing(False)  # 禁用编辑
        self.grid.SetDefaultCellAlignment(wx.ALIGN_CENTER, wx.ALIGN_CENTER)  # 居中对齐
    
    def updateGrid(self, touch_data: TouchData) -> None:
        """更新表格显示"""
        if touch_data.data_matrix is None:
            return
            
        tx_size = touch_data.tx_count
        rx_size = touch_data.rx_count
        
        self._adjustGridSize(tx_size, rx_size)
        self._populateGrid(touch_data.data_matrix, tx_size, rx_size)
        self._adjustCellSize()  # 调整单元格大小为正方形
    
    def _adjustGridSize(self, tx_size: int, rx_size: int) -> None:
        """调整表格大小"""
        # 调整行数
        if self.grid.GetNumberRows() != tx_size:
            if tx_size < self.grid.GetNumberRows():
                self.grid.DeleteRows(pos=0, numRows=self.grid.GetNumberRows() - tx_size)
            else:
                self.grid.AppendRows(tx_size - self.grid.GetNumberRows())
        
        # 调整列数
        if self.grid.GetNumberCols() != rx_size:
            if rx_size < self.grid.GetNumberCols():
                self.grid.DeleteCols(pos=0, numCols=self.grid.GetNumberCols() - rx_size)
            else:
                self.grid.AppendCols(rx_size - self.grid.GetNumberCols())
        
        # 设置标签
        for tx in range(tx_size):
            self.grid.SetRowLabelValue(tx, f"TX{tx + 1}")
        for rx in range(rx_size):
            self.grid.SetColLabelValue(rx, f"RX{rx + 1}")
    
    def _adjustCellSize(self) -> None:
        """调整单元格大小为正方形"""
        # 设置单元格大小为正方形
        cell_size = 25  # 正方形边长
        
        # 设置所有行高
        for row in range(self.grid.GetNumberRows()):
            self.grid.SetRowSize(row, cell_size)
        
        # 设置所有列宽
        for col in range(self.grid.GetNumberCols()):
            self.grid.SetColSize(col, cell_size)
        
        # 设置行标签和列标签宽度
        self.grid.SetRowLabelSize(50)  # 行标签宽度
        self.grid.SetColLabelSize(30)  # 列标签高度
        
        # 设置字体大小
        font = wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.grid.SetDefaultCellFont(font)
        
        # 设置标签字体
        label_font = wx.Font(7, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.grid.SetLabelFont(label_font)
    
    def _populateGrid(self, data_matrix, tx_size: int, rx_size: int) -> None:
        """填充表格数据"""
        for tx in range(tx_size):
            for rx in range(rx_size):
                value = int(data_matrix[tx, rx])
                self.grid.SetCellValue(tx, rx, str(value))
                self.grid.SetCellBackgroundColour(
                    tx, rx, ColorManager.getColorForValue(value)
                )
                
                # 设置单元格文本颜色（根据背景色调整）
                bg_color = ColorManager.getColorForValue(value)
                if bg_color != wx.WHITE:
                    # 如果背景色不是白色，使用白色文字
                    self.grid.SetCellTextColour(tx, rx, wx.WHITE)
                else:
                    # 白色背景使用黑色文字
                    self.grid.SetCellTextColour(tx, rx, wx.BLACK)
        
        self.grid.ForceRefresh()

class TouchDataViewerFrame(wx.Frame):
    """触摸数据查看器主窗口"""
    
    def __init__(self, *args, **kw):
        super(TouchDataViewerFrame, self).__init__(*args, **kw)
        
        self.mainPanel: wx.Panel = wx.Panel(self)
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
        self._bind_events()
    
    def _initialize_ui(self) -> None:
        """初始化用户界面"""
        main_layout = wx.BoxSizer(wx.VERTICAL)
        
        # 文件选择
        main_layout.Add(self._create_file_selection_section(), flag=wx.EXPAND)
        
        # 模式选择
        main_layout.Add(self._create_mode_selection_section(), flag=wx.EXPAND)
        
        # 差异计算选项
        self.diff_options_panel = self._create_diff_options_section()
        main_layout.Add(self.diff_options_panel, flag=wx.EXPAND)
        
        # 数据表格
        self.data_grid = gridlib.Grid(self.mainPanel)
        self.grid_manager = GridManager(self.data_grid)
        self.data_grid.CreateGrid(0, 0)
        main_layout.Add(self.data_grid, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        
        # 导航控件
        main_layout.Add(self._create_navigation_section(), flag=wx.ALIGN_CENTER)
        
        # 播放间隔设置
        main_layout.Add(self._create_interval_time_section(), flag=wx.ALIGN_CENTER)
        
        # 跳转控件
        main_layout.Add(self._create_jump_section(), flag=wx.ALIGN_CENTER)
        
        # 刷新按钮
        main_layout.Add(self._create_refresh_button(), flag=wx.ALIGN_CENTER)
        
        # 信息显示
        self._create_info_section(main_layout)
        
        self.mainPanel.SetSizer(main_layout)
    
    def _create_file_selection_section(self) -> wx.BoxSizer:
        """创建文件选择区域"""
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.file_picker = wx.FilePickerCtrl(
            self.mainPanel, message="选择触摸数据文件"
        )
        hbox.Add(self.file_picker, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        self.load_button = wx.Button(self.mainPanel, label="加载文件")
        hbox.Add(self.load_button, flag=wx.ALL, border=5)
        return hbox
    
    def _create_mode_selection_section(self) -> wx.BoxSizer:
        """创建模式选择区域"""
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.raw_mode_radio = wx.RadioButton(
            self.mainPanel, label="原始数据", style=wx.RB_GROUP
        )
        self.diff_mode_radio = wx.RadioButton(
            self.mainPanel, label="差异数据"
        )
        hbox.Add(
            wx.StaticText(self.mainPanel, label="显示模式:"),
            flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            border=5,
        )
        hbox.Add(self.raw_mode_radio, flag=wx.ALL, border=5)
        hbox.Add(self.diff_mode_radio, flag=wx.ALL, border=5)
        return hbox
    
    def _create_diff_options_section(self) -> wx.Panel:
        """创建差异计算选项区域"""
        panel = wx.Panel(self.mainPanel)
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        # 基准帧模式
        self.base_frame_radio = wx.RadioButton(
            panel, label="基准帧模式", style=wx.RB_GROUP
        )
        hbox_base = wx.BoxSizer(wx.HORIZONTAL)
        hbox_base.Add(
            wx.StaticText(panel, label="基准帧号:"),
            flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            border=5,
        )
        self.base_frame_entry = wx.TextCtrl(panel, value="0")
        hbox_base.Add(self.base_frame_entry, flag=wx.ALL, border=5)
        vbox.Add(self.base_frame_radio, flag=wx.ALL, border=5)
        vbox.Add(hbox_base, flag=wx.EXPAND)
        
        # 区间平均模式
        self.average_frame_radio = wx.RadioButton(
            panel, label="区间平均模式"
        )
        hbox_avg = wx.BoxSizer(wx.HORIZONTAL)
        hbox_avg.Add(self.average_frame_radio, flag=wx.ALL, border=5)
        hbox_avg.Add(
            wx.StaticText(panel, label="区间:"),
            flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            border=5,
        )
        self.start_frame_entry = wx.TextCtrl(panel, value="0", size=(50, -1))
        hbox_avg.Add(self.start_frame_entry, flag=wx.ALL, border=5)
        hbox_avg.Add(
            wx.StaticText(panel, label="到"),
            flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            border=5,
        )
        self.end_frame_entry = wx.TextCtrl(panel, value="5", size=(50, -1))
        hbox_avg.Add(self.end_frame_entry, flag=wx.ALL, border=5)
        vbox.Add(hbox_avg, flag=wx.EXPAND)
        
        # 间隔帧模式
        self.interval_frame_radio = wx.RadioButton(panel, label="间隔帧模式")
        hbox_interval = wx.BoxSizer(wx.HORIZONTAL)
        hbox_interval.Add(self.interval_frame_radio, flag=wx.ALL, border=5)
        hbox_interval.Add(
            wx.StaticText(panel, label="间隔:"),
            flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            border=5,
        )
        self.interval_entry = wx.TextCtrl(panel, value="1")
        hbox_interval.Add(self.interval_entry, flag=wx.ALL, border=5)
        vbox.Add(hbox_interval, flag=wx.EXPAND)
        
        panel.SetSizer(vbox)
        return panel
    
    def _create_navigation_section(self) -> wx.BoxSizer:
        """创建导航区域"""
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.previous_button = wx.Button(self.mainPanel, label="上一帧")
        self.next_button = wx.Button(self.mainPanel, label="下一帧")
        self.auto_play_button = wx.Button(self.mainPanel, label="自动播放")
        hbox.Add(self.previous_button, flag=wx.ALL, border=5)
        hbox.Add(self.next_button, flag=wx.ALL, border=5)
        hbox.Add(self.auto_play_button, flag=wx.ALL, border=5)
        return hbox
    
    def _create_interval_time_section(self) -> wx.BoxSizer:
        """创建播放间隔设置区域"""
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(
            wx.StaticText(self.mainPanel, label="播放间隔(秒):"),
            flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            border=5,
        )
        self.interval_time_entry = wx.TextCtrl(
            self.mainPanel, value="0.5", size=(50, -1)
        )
        hbox.Add(self.interval_time_entry, flag=wx.ALL, border=5)
        return hbox
    
    def _create_jump_section(self) -> wx.BoxSizer:
        """创建跳转区域"""
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(
            wx.StaticText(self.mainPanel, label="跳转到帧:"),
            flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            border=5,
        )
        self.jump_frame_entry = wx.TextCtrl(self.mainPanel, size=(50, -1))
        hbox.Add(self.jump_frame_entry, flag=wx.ALL, border=5)
        self.jump_button = wx.Button(self.mainPanel, label="跳转")
        hbox.Add(self.jump_button, flag=wx.ALL, border=5)
        return hbox
    
    def _create_refresh_button(self) -> wx.Button:
        """创建刷新按钮"""
        self.refresh_button = wx.Button(self.mainPanel, label="刷新")
        return self.refresh_button
    
    def _create_info_section(self, main_layout: wx.BoxSizer) -> None:
        """创建信息显示区域"""
        # 帧信息
        self.frame_info_text = wx.StaticText(
            self.mainPanel, label="当前帧: 0 / 0"
        )
        main_layout.Add(self.frame_info_text, flag=wx.ALL | wx.ALIGN_CENTER, border=5)
        
        # 头部信息
        self.header_info_text = wx.StaticText(
            self.mainPanel, label="头部信息: 未加载"
        )
        main_layout.Add(self.header_info_text, flag=wx.ALL | wx.ALIGN_CENTER, border=5)
    
    def _bind_events(self) -> None:
        """绑定事件"""
        self.Bind(wx.EVT_BUTTON, self.on_load_file, self.load_button)
        self.Bind(wx.EVT_BUTTON, self.on_previous_frame, self.previous_button)
        self.Bind(wx.EVT_BUTTON, self.on_next_frame, self.next_button)
        self.Bind(wx.EVT_BUTTON, self.on_jump_to_frame, self.jump_button)
        self.Bind(wx.EVT_BUTTON, self.on_toggle_auto_play, self.auto_play_button)
        self.Bind(wx.EVT_BUTTON, self.on_refresh, self.refresh_button)
        self.Bind(wx.EVT_RADIOBUTTON, self.on_diff_method_change, self.base_frame_radio)
        self.Bind(wx.EVT_RADIOBUTTON, self.on_diff_method_change, self.average_frame_radio)
        self.Bind(wx.EVT_RADIOBUTTON, self.on_diff_method_change, self.interval_frame_radio)
    
    def on_load_file(self, event: wx.CommandEvent) -> None:
        """加载文件事件"""
        file_path = self.file_picker.GetPath()
        if file_path:
            self.header_info, self.touch_data = TouchDataFileProcessor.loadTouchData(file_path)
            self.total_frames = len(self.touch_data)
            self.current_frame_index = 0
            
            # 更新头部信息显示
            if self.header_info:
                header_text = f"头部信息: {self.header_info.date} {self.header_info.time}, TX={self.header_info.tx}, RX={self.header_info.rx}"
                self.header_info_text.SetLabel(header_text)
            
            self.on_refresh(None)
    
    def on_diff_method_change(self, event: wx.CommandEvent) -> None:
        """差异计算方法改变事件"""
        if self.base_frame_radio.GetValue():
            self.diff_method = "base"
            self.base_frame_index = (
                int(self.base_frame_entry.GetValue())
                if self.base_frame_entry.GetValue().isdigit()
                else 0
            )
        elif self.average_frame_radio.GetValue():
            self.diff_method = "average"
            self.start_frame = (
                int(self.start_frame_entry.GetValue())
                if self.start_frame_entry.GetValue().isdigit()
                else 0
            )
            self.end_frame = (
                int(self.end_frame_entry.GetValue())
                if self.end_frame_entry.GetValue().isdigit()
                else 5
            )
        elif self.interval_frame_radio.GetValue():
            self.diff_method = "interval"
            self.interval = (
                int(self.interval_entry.GetValue())
                if self.interval_entry.GetValue().isdigit()
                else 1
            )
    
    def on_refresh(self, event: wx.CommandEvent) -> None:
        """刷新事件"""
        if not self.touch_data:
            return
        
        if not self.first_load:
            self.on_diff_method_change(event)
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
        
        if self.raw_mode_radio.GetValue():
            if self.current_frame_index < len(self.touch_data):
                frame_data = self.touch_data[self.current_frame_index]
                self.grid_manager.updateGrid(frame_data)
        else:
            if self.current_frame_index > 0 and self.current_frame_index - 1 < len(self.diff_data):
                frame_data = self.diff_data[self.current_frame_index - 1]
                self.grid_manager.updateGrid(frame_data)
    
    def _update_frame_info(self) -> None:
        """更新帧信息显示"""
        self.frame_info_text.SetLabel(
            f"当前帧: {self.current_frame_index + 1} / {self.total_frames}"
        )
    
    def on_previous_frame(self, event: wx.CommandEvent) -> None:
        """上一帧事件"""
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self._update_frame_info()
            self._display_frame()
    
    def on_next_frame(self, event: wx.CommandEvent) -> None:
        """下一帧事件"""
        if self.current_frame_index < self.total_frames - 1:
            self.current_frame_index += 1
            self._update_frame_info()
            self._display_frame()
    
    def on_jump_to_frame(self, event: wx.CommandEvent) -> None:
        """跳转到指定帧事件"""
        try:
            frame_number = int(self.jump_frame_entry.GetValue()) - 1
            if 0 <= frame_number < self.total_frames:
                self.current_frame_index = frame_number
                self._update_frame_info()
                self._display_frame()
            else:
                wx.MessageBox("无效的帧号", "错误", wx.OK | wx.ICON_ERROR)
        except ValueError:
            wx.MessageBox("请输入有效的帧号", "错误", wx.OK | wx.ICON_ERROR)
    
    def on_toggle_auto_play(self, event: wx.CommandEvent) -> None:
        """切换自动播放状态"""
        if not self.auto_play_active:
            self.auto_play_active = True
            self.auto_play_button.SetLabel("停止播放")
            self._start_auto_play()
        else:
            self.auto_play_active = False
            self.auto_play_button.SetLabel("自动播放")
    
    def _start_auto_play(self) -> None:
        """开始自动播放"""
        def auto_play():
            try:
                interval = float(self.interval_time_entry.GetValue())
            except ValueError:
                wx.MessageBox("请输入有效的间隔时间", "错误", wx.OK | wx.ICON_ERROR)
                return
            
            while self.auto_play_active and self.current_frame_index < self.total_frames - 1:
                wx.CallAfter(self.on_next_frame, None)
                time.sleep(interval)
            
            wx.CallAfter(self._stop_auto_play)
        
        threading.Thread(target=auto_play, daemon=True).start()
    
    def _stop_auto_play(self) -> None:
        """停止自动播放"""
        self.auto_play_active = False
        self.auto_play_button.SetLabel("自动播放")

class TouchDataViewerApp(wx.App):
    """触摸数据查看器应用"""
    
    def OnInit(self) -> bool:
        self.frame = TouchDataViewerFrame(
            None, title="触摸数据查看器", size=(WINSTARTSIZE_ROW, WINSTARTSIZE_COL)
        )
        self.frame.Show()
        return True

if __name__ == "__main__":
    app = TouchDataViewerApp()
    app.MainLoop()
