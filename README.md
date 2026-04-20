# KCF Tracker

一个基于 KCF (Kernelized Correlation Filter) 的目标跟踪项目，支持以下流程：

- 单视频可视化演示
- OTB100 批量跟踪与评估
- VOT2018 跟踪与评估

## 1. 环境与依赖

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
pip install vot-toolkit
```

## 2. 当前项目结构

```text
KCF_tracker/
	core/                     # 全局核心代码
		kcf.py
		fhog.py
	demos/                    # 演示脚本
		demo.py
	otb/                      # OTB100 运行与评估脚本
		run_tracker.py
		evaluate.py
	vot2018/                  # VOT2018 运行与评估脚本
		run_vot2018.py
		evaluate_vot2018.py
		vot_wrapper.py
		vot_local.py
	vot2018_workspace/        # VOT toolkit 工作区
		trackers.ini
		sequences/
		results/
		logs/
	otb100/                   # OTB100 数据集
	results/
		KCF/                    # OTB 跟踪结果
```

## 3. 运行方法

以下命令默认在项目根目录执行。

### 3.1 单视频演示

```bash
python demos/demo.py
```

### 3.2 OTB100 批量跟踪

```bash
python otb/run_tracker.py
```

### 3.3 OTB100 评估

```bash
python otb/evaluate.py
```

### 3.4 VOT2018 快速验证（单序列）

```bash
python vot2018/run_vot2018.py --quick
```

### 3.5 VOT2018 完整运行

```bash
python vot2018/run_vot2018.py
```

### 3.6 VOT2018 评估

```bash
python vot2018/evaluate_vot2018.py
```

如果当前目录在 `vot2018/`，也可直接执行：

```bash
python run_vot2018.py
python evaluate_vot2018.py
```

## 4. 结果输出位置

- OTB 跟踪结果：`results/KCF/*.txt`
- OTB 评估图：`success_plot.png`、`precision_plot.png`
- VOT 跟踪结果：`vot2018_workspace/results/KCF_Tracker/baseline/**/.bin`
- VOT 评估结果：`vot2018_workspace/evaluation_results.txt`
- VOT 准确率图：`vot2018_accuracy.png`

## 5. VOT 配置注意事项

`vot2018_workspace/trackers.ini` 中 `command` 当前使用绝对路径指向 `vot2018/vot_wrapper.py`。

如果你移动了项目目录，需要同步更新该路径，否则可能出现：

```text
Unable to connect to tracker
```

排查日志目录：`vot2018_workspace/logs/`

## 6. 常见问题

### 6.1 提示找不到 vot 命令

安装 vot-toolkit 并确认当前环境生效：

```bash
pip install vot-toolkit
```

### 6.2 VOT 运行中断但返回码异常

`vot2018/run_vot2018.py` 已增加结果文件二次校验。若未生成 `.bin`，脚本会直接判定失败并提示查看日志。
