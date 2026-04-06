# KCF Tracker

一个基于 KCF (Kernelized Correlation Filter) 的目标跟踪项目，包含单视频演示、OTB100 批量运行与结果评估脚本。

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 运行演示

```bash
python demo.py
```

### 3) 批量运行（OTB100）

```bash
python run_tracker.py
```

### 4) 评估结果

```bash
python evaluate.py
```

## 数据与目录说明

- OTB100 数据默认放在项目根目录下的 `otb100/`。
- 每个序列通常包含：`img/` 与 `groundtruth_rect.txt`。
- 批量运行结果默认写入：`results/KCF/`。
- 评估图示例：`precision_plot.png`、`success_plot.png`。

## 项目文件（核心）

- `kcf.py`：KCF 跟踪器实现。
- `fhog.py`：特征相关实现。
- `demo.py`：单视频/可视化演示入口。
- `run_tracker.py`：数据集批量跟踪入口。
- `evaluate.py`：精度与成功率评估。

## 备注

- 若路径不一致，请在脚本中将数据根目录改为你的本地路径。
- 先运行 `run_tracker.py`，再运行 `evaluate.py`。
