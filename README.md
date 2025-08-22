# TouchDataFilter

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python&logoColor=white)![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+cu124-red?style=flat-square&logo=pytorch&logoColor=white)![License](https://img.shields.io/badge/License-MPL--2.0-green?style=flat-square)![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat-square)![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)[![GitHub](https://img.shields.io/badge/GitHub-ZhuchenZhong-pink?style=flat-square&logo=github)](https://github.com/ZhuchenZhong)[![Project](https://img.shields.io/badge/Project-TouchDataFilter-black?style=flat-square&logo=github)](https://github.com/ZhuchenZhong/TouchDataFilter)

**基于深度学习的触摸数据滤波系统**

*v0.1.0 @ 2025/08/22*

---

#### 快速开始

##### 数据集结构

📂`./data/ 是放置训练数据 ，对于训练和验证数据的匹配一般来说我们遵循

- .txt (PlainText) 文件

- 文件名称中仅有亮屏/灭屏之分

  这里定义亮屏文件所存储的触摸数据是滤波前的数据，灭屏数据则是期望数据

  这里有正则表达式定义两者文件之分 -> `^(?=.*(亮|灭))屏.*\.txt$`

此文件夹的所有文件将会由训练脚本进行搜索，因此📂 `./data` 下的文件存放结构不做定义

> 同一对训练数据需要存放在同一路径下



##### 环境配置

项目使用 `uv` 作为虚拟环境管理器，当然可以使用其他如 `pdm` 等，这里使用 `uv`

```bash
uv venv
uv pip install -r requirements.txt
```

> 注意 PyTorch 的安装不在此处，需要自行去其 [PyTouch官网](https://pytorch.org/get-started/locally/) 选择对应平台的版本



##### 训练

脚本 `train.py` 会自动根据📂 `./data` 下的数据文件训练，同时会将其保存在 📂 `./models` 下。

默认情况下，脚本将遵循

- 使用📂`./data`下所有可用的训练数据
- 训练集和验证集的划分为 $80\%$ 和 $20\%$
- 训练轮数(epoch) 为 $50$
- Batch Size 为 32 (会根据当前环境动态调整，上限为128)

> 当前版本需要在 `train.py` 自行修改，后续将会移至 📂 `./config`下

运行 `uv run train.py` 后会自动运行



##### 预测 & 查看数据

运行 `uv run predict.py` 后，脚本会自动调用 📂`./models` 下按照字典顺序最后的模型文件(后续会将其更改成默认最新训练的模型且支持切换模型目录)。如果需要指定某个模型文件夹则需要指定路径 `uv run predict.py /path/to/your/model`

![image-20250822102332562](resource/public/assert/image-20250822102332562.png)

如果需要查看具体的数据则需要 📂`./tools`  下的 `viewer_tk.py` 。

相同的，运行 `uv run viewer_tk.py` 则能查看具体的预测/原始数据



#### 计划

##### 短期计划

- [ ] 将模型和数据加载器抽象成Python包
- [ ] 模型加载遵循时间顺序
- [ ] 统一项目结构和代码风格
- [ ] 配置文件单独控制
- [ ] 完善当前文档

##### 长期计划

- [ ] 支持 $int 2^n$ 量化模型
- [ ] 支持 C++/LibTorch 调用
  - [ ] 支持 fp32 模型导出 LibTorch 格式(JIT)
- [ ] 支持 C 语言调用
- [ ] 支持环境受限的平台









---
