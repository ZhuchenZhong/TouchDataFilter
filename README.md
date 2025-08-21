# TouchDataFilter

一个基于深度学习的触摸数据滤波系统，用于处理触摸屏传感器数据中的噪声，提升数据质量和触摸体验。

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [模型原理](#模型原理)
- [环境要求](#环境要求)
- [安装说明](#安装说明)
- [使用方法](#使用方法)
- [数据格式](#数据格式)
- [训练说明](#训练说明)
- [预测与推理](#预测与推理)
- [模型详细信息](#模型详细信息)
- [工具说明](#工具说明)
- [配置文件](#配置文件)
- [许可证](#许可证)

## 项目概述

TouchDataFilter 是一个专门为触摸屏数据滤波设计的深度学习系统。该系统能够有效去除触摸传感器数据中的噪声，同时保留重要的触摸特征和细节信息。项目提供了两种模型实现：

- **浮点模型 (TouchFilterNet_fp)**: 高精度的浮点数模型，适用于服务器端和高性能设备
- **整数模型 (TouchFilterNet_int)**: 量化的整数模型，适用于资源受限的嵌入式设备

## 系统架构

系统采用模块化设计，主要包含以下组件：

```
TouchDataFilter/
├── core/                     # 核心模块
│   ├── dataProcessor.py      # 数据处理器
│   ├── TouchFilterNet_fp.py  # 浮点模型
│   └── TouchFilterNet_int.py # 整数模型
├── data/                     # 数据集
├── models/                   # 训练好的模型
├── tools/                    # 辅助工具
├── docs/                     # 文档
└── resource/                 # 资源文件
```

## 模型原理

### 网络架构

TouchFilterNet 采用**编码器-解码器架构**，结合了以下关键技术：

#### 1. 编码器-解码器结构

- **编码器**: 使用多层卷积网络提取触摸数据的层次特征
- **解码器**: 逐步恢复原始分辨率，重建滤波后的数据

#### 2. 注意力机制

- 自适应地关注高差异区域（重要的触摸特征）
- 使用Sigmoid激活函数生成注意力权重图
- 帮助模型专注于需要重点处理的区域

#### 3. 残差连接

- 使用1×1卷积学习残差映射
- 保证网络的可训练性和梯度流动
- 防止信息丢失和梯度消失

#### 4. 差异增强机制

- 根据输入数据的差异特性动态调整输出
- 对高差异区域进行增强处理
- 对低差异区域保持原有特征

### 损失函数

#### 差异保留损失 (DiffPreservingLoss)

$$
总损失 = \beta \times MSE损失 + (1 - \beta) \times 加权MSE损失
$$

- **$\alpha$ 参数**: 控制高差异区域的权重系数 (默认1.8)
- **$\beta$ 参数**: 平衡MSE损失和差异保留损失 (默认0.5)
- **差异权重**: $|targets|^{\alpha} + 0.5$ ，为高差异区域分配更大权重

### 量化策略（整数模型）

整数模型采用量化技术将浮点运算转换为整数运算：

- **位宽**: 支持8位量化 (可配置)
- **量化范围**: $[-2^{7}, 2^{7}-1] = [-128, 127]$ (8位)
- **激活函数**: 使用查找表或分段线性近似
- **整数卷积**: 自定义IntConv2d层实现整数域卷积

## 环境要求

### 基本要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (GPU训练可选)

### 依赖包

> PyTorch 注意需要选择对应的 cuda / cpu 版本

```bash
torch>=1.12.0
torchvision
numpy
matplotlib
rich
tkinter
pathlib
dataclasses
```

## 安装说明

1. **克隆项目**

```bash
git clone https://github.com/ZhuchenZhong/TouchDataFilter.git
cd TouchDataFilter
```

2. **创建虚拟环境**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

3. **安装依赖**

```bash
pip install torch torchvision numpy matplotlib rich
```

4. **配置项目**

```bash
python init.py  # 自动检测GPU并生成配置
```

## 使用方法

### 快速开始

1. **训练浮点模型**

```bash
python train.py
```

2. **训练整数模型**

```bash
python train_TouchFilterNet_int.py
```

3. **运行预测**

```bash
python predict.py
```

4. **命令行预测**

```bash
python predict-cli.py --model_path models/your_model.pth --input_file data/test_data.txt
```

### GUI预测工具

运行 `predict.py` 将启动图形化界面：

- 选择预训练模型
- 加载触摸数据文件
- 实时查看滤波效果
- 保存处理结果

## 数据格式

### 输入数据格式

触摸数据文件采用特定的文本格式：

```
DATE,2024-01-01
TIME,12:00:00
FW Ver: 1.0.0
TX,32
RX,18
TX_VOLTAGE,5000
Cf,100

Frame[1]:
<触摸数据矩阵>
...
```

### 数据结构

- **HeaderInfo**: 包含日期、时间、固件版本、TX/RX通道数等元信息
- **TouchData**: 包含帧号、时间戳、数据矩阵等触摸信息

## 训练说明

### 训练参数

```python
EPOCH = 50                          # 训练轮数
BATCH_SIZE = 32-128                 # 自动根据GPU内存调整
LEARNING_RATE = 1e-3                # 学习率 η = 10^{-3}
OPTIMIZER = Adam                    # 优化器
SCHEDULER = ReduceLROnPlateau       # 学习率调度器
```

### 训练流程

1. **数据预处理**: 自动解析触摸数据文件，处理噪声和正常数据对
2. **模型初始化**: 创建网络结构并初始化参数
3. **训练循环**:
   - 前向传播计算损失
   - 反向传播更新参数
   - 验证集评估性能
4. **模型保存**: 定期保存检查点和最佳模型

### 混合精度训练

支持自动混合精度(AMP)训练以提高训练效率：

- 自动检测GPU支持
- 使用GradScaler进行梯度缩放
- 减少内存使用，提高训练速度

## 预测与推理

### 模型加载

```python
from core.TouchFilterNet_fp import TouchFilterNet
from core.TouchFilterNet_int import TouchFilterNet_int

# 加载浮点模型
model_fp = TouchFilterNet()
model_fp.load_state_dict(torch.load('model_fp.pth'))

# 加载整数模型
model_int = TouchFilterNet_int(bits=8)
model_int.load_state_dict(torch.load('model_int.pth'))
```

### 数据预处理

```python
from core.dataProcessor import TouchDataParser

# 解析触摸数据
parser = TouchDataParser('data_file.txt')
header, touch_data = parser.parse()

# 转换为模型输入格式
input_tensor = torch.tensor(touch_data[0].data_matrix).unsqueeze(0).unsqueeze(0)
```

## 模型详细信息

### 浮点模型 (TouchFilterNet_fp)

#### 网络结构

- **输入**: 单通道触摸数据 $(1 \times H \times W)$
- **编码器**: 3层卷积 $(1 \rightarrow 16 \rightarrow 32 \rightarrow 64$ 通道$)$
- **注意力**: $64 \rightarrow 32 \rightarrow 1$ 通道，Sigmoid激活
- **解码器**: 3层卷积 $(64 \rightarrow 32 \rightarrow 16 \rightarrow 1$ 通道$)$
- **残差**: $1 \times 1$ 卷积学习残差映射
- **差异增强**: $2$ 通道输入 $\rightarrow 8 \rightarrow 1$ 通道输出

#### 参数量

- 总参数: $\sim 50K$
- 可训练参数: $\sim 50K$
- 内存占用: $\sim 200MB$ (训练时)

### 整数模型 (TouchFilterNet_int)

#### 量化特性

- **量化位宽**: 8位整数
- **权重量化**: 对称量化，范围 $[-128, 127]$
- **激活量化**: ReLU使用 $[0, 127]$ 范围
- **特殊函数**: Sigmoid/Tanh使用查找表近似

#### 优化特性

- **计算效率**: 整数运算替代浮点运算
- **内存效率**: 8位存储替代32位浮点
- **硬件友好**: 适合FPGA/ASIC部署

## 工具说明

### 数据可视化工具

1. **viewer.py**: 基于matplotlib的数据查看器
2. **viewer_tk.py**: 基于tkinter的交互式查看器
3. **findTFN_fp_DPL_alpha.py**: 差异保留损失参数寻找工具

### 使用示例

```bash
# 可视化触摸数据
python tools/viewer.py --file data/test_data.txt

# 交互式数据查看
python tools/viewer_tk.py

# 参数优化
python tools/findTFN_fp_DPL_alpha.py --data_dir data/
```

## 配置文件

### settings.ini

```ini
[PATHS]
project_root = /path/to/TouchDataFilter
data_root = /path/to/data
model_root = /path/to/models

[GPU]
gpu_support = True
batch_size = 47
```

### 自动配置

运行 `init.py` 将自动：

- 检测GPU支持情况
- 计算最优批次大小
- 生成配置文件
- 创建必要目录

## 性能指标

### 模型性能

- **浮点模型精度**: $MSE < 0.001$ (验证集)
- **整数模型精度**: $MSE < 0.005$ (验证集)
- **推理速度**:
  - 浮点模型: $\sim 10ms$ (GPU), $\sim 50ms$ (CPU)
  - 整数模型: $\sim 5ms$ (GPU), $\sim 20ms$ (CPU)

### 资源消耗

- **浮点模型**: $200MB$ 内存, $50K$ 参数
- **整数模型**: $50MB$ 内存, $50K$ 参数 (8位量化)

## 贡献指南

欢迎提交Issue和Pull Request来改进项目：

1. Fork项目仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 许可证

本项目采用 Mozilla Public License 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 更新日志

### v1.0.0 (2025-07-15)

- 初始版本发布
- 实现浮点和整数两种模型
- 添加GUI预测工具
- 完善文档和示例

---

**项目维护者**: ZhuchenZhong
**联系方式**: [GitHub](https://github.com/ZhuchenZhong)
**项目地址**: https://github.com/ZhuchenZhong/TouchDataFilter
