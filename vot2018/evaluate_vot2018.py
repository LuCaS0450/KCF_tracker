"""
评估 VOT2018 测试结果
计算 EAO (Expected Average Overlap)、准确率 (Accuracy) 和鲁棒性 (Robustness)
参数对齐 VOT2018 supervised 官方设置：burnin=10, skip_initialize=5, EAO=[100, 356]
"""
import sys
import io

# 设置标准输出编码为 UTF-8,解决 Windows 控制台中文乱码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# VOT2018 short-term 官方评估参数
BURNIN_FRAMES = 10
SKIP_INITIALIZE_FRAMES = 5
EAO_WINDOW_LOW = 100
EAO_WINDOW_HIGH = 356

def load_groundtruth(gt_file):
    """加载真实标注文件"""
    try:
        # VOT 格式可能是 4 列 (x,y,w,h) 或 8 列 (四边形)
        data = np.loadtxt(gt_file, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # 如果是 8 列（四边形），转换为矩形框
        if data.shape[1] == 8:
            x_min = data[:, 0::2].min(axis=1)
            y_min = data[:, 1::2].min(axis=1)
            x_max = data[:, 0::2].max(axis=1)
            y_max = data[:, 1::2].max(axis=1)
            data = np.column_stack([x_min, y_min, x_max - x_min, y_max - y_min])
        
        return data
    except Exception as e:
        print(f"加载 {gt_file} 失败: {e}")
        return None

def load_results(res_file):
    """加载跟踪结果文件"""
    try:
        data = np.loadtxt(res_file, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    except Exception as e:
        print(f"加载 {res_file} 失败: {e}")
        return None

def compute_overlap(rect1, rect2):
    """计算两个矩形框的 IoU (Intersection over Union)"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    # 计算交集
    xx1 = max(x1, x2)
    yy1 = max(y1, y2)
    xx2 = min(x1 + w1, x2 + w2)
    yy2 = min(y1 + h1, y2 + h2)
    
    intersection = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    
    # 计算并集
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union < 1e-6:
        return 0.0
    
    return intersection / union

def build_overlap_segments(frame_overlaps):
    """将按帧重叠率序列切分为连续有效片段（用 NaN 作为分隔）。"""
    segments = []
    current = []

    for value in frame_overlaps:
        if np.isnan(value):
            if current:
                segments.append(np.array(current, dtype=np.float32))
                current = []
        else:
            current.append(float(value))

    if current:
        segments.append(np.array(current, dtype=np.float32))

    return segments

def compute_expected_overlap_curve(segments):
    """根据片段计算 expected overlap 曲线。"""
    if not segments:
        return np.array([], dtype=np.float32)

    max_len = max(len(seg) for seg in segments)
    curve = np.full(max_len, np.nan, dtype=np.float32)

    for t in range(1, max_len + 1):
        prefix_means = [np.mean(seg[:t]) for seg in segments if len(seg) >= t]
        if prefix_means:
            curve[t - 1] = float(np.mean(prefix_means))

    return curve

def compute_sequence_eao(segments, low=EAO_WINDOW_LOW, high=EAO_WINDOW_HIGH):
    """计算单序列 EAO（Expected Average Overlap）。"""
    curve = compute_expected_overlap_curve(segments)
    if len(curve) == 0:
        return 0.0

    start = max(1, int(low))
    end = min(int(high), len(curve))

    if end < start:
        valid = curve[np.isfinite(curve)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    window = curve[start - 1:end]
    window = window[np.isfinite(window)]
    return float(np.mean(window)) if len(window) > 0 else 0.0

def find_bin_run_files(results_dir, seq_name):
    """查找某序列所有 baseline 重复运行的 bin 结果文件。"""
    results_dir = Path(results_dir)
    run_files = []

    for tracker_dir in results_dir.iterdir():
        if not tracker_dir.is_dir():
            continue

        baseline_seq_dir = tracker_dir / 'baseline' / seq_name
        if baseline_seq_dir.exists():
            run_files = sorted(baseline_seq_dir.glob(f'{seq_name}_*.bin'))
            if run_files:
                return run_files

    return run_files

def evaluate_single_run(gt, run_data, use_special_frames):
    """评估单次运行结果，支持 VOT bin 特殊帧语义和 burn-in 规则。"""
    min_len = min(len(gt), len(run_data))
    gt = gt[:min_len]

    overlaps = []
    frame_overlaps = []
    failures = 0
    tracked_frames = 0
    burnin_left = BURNIN_FRAMES

    if use_special_frames:
        for i in range(min_len):
            frame_info = run_data[i]
            code = frame_info.get('code')

            # 初始化帧后启用 burn-in（与 VOT2018 分析参数保持一致）
            if code == 1:
                burnin_left = BURNIN_FRAMES

            if frame_info.get('failure', False):
                failures += 1

            if frame_info.get('valid', False):
                iou = compute_overlap(frame_info['bbox'], gt[i])
                if burnin_left > 0:
                    burnin_left -= 1
                    frame_overlaps.append(np.nan)
                else:
                    overlaps.append(iou)
                    frame_overlaps.append(iou)
                    tracked_frames += 1
            else:
                frame_overlaps.append(np.nan)
    else:
        # 对普通 txt 结果做近似处理：序列起始 burn-in，失效定义为 IoU<=0
        for i in range(min_len):
            iou = compute_overlap(run_data[i], gt[i])

            if burnin_left > 0:
                burnin_left -= 1
                frame_overlaps.append(np.nan)
            else:
                overlaps.append(iou)
                frame_overlaps.append(iou)
                tracked_frames += 1

            if iou <= 0:
                failures += 1

    overlaps = np.array(overlaps, dtype=np.float32)
    segments = build_overlap_segments(np.array(frame_overlaps, dtype=np.float32))

    return {
        'accuracy': float(np.mean(overlaps)) if len(overlaps) > 0 else 0.0,
        # VOT AR 的 R 倾向以失败次数表征，数值越低越好
        'robustness': float(failures),
        'failures': int(failures),
        'tracked_frames': int(tracked_frames),
        'length': int(min_len),
        'overlaps': overlaps,
        'segments': segments
    }

def load_results_bin(res_file):
    """加载 VOT Toolkit 格式的 bin 文件"""
    import sys
    import os
    import numpy as np
    
    # 临时修改 sys.path 以导入真正的 vot-toolkit 库，避免与当前目录下的 vot.py 冲突
    old_path = sys.path[:]
    try:
        sys.path = [p for p in sys.path if not os.path.exists(os.path.join(p, 'vot.py'))]
        import vot.region.io as vrio
        traj = vrio.read_trajectory(res_file)

        # 保留 VOT 特殊帧语义（初始化/失效/跳过），避免误计入 IoU 和失效统计
        frames = []
        for r in traj:
            r_type = type(r).__name__
            if r_type == 'Rectangle':
                frames.append({
                    'bbox': [r.x, r.y, r.width, r.height],
                    'valid': True,
                    'failure': False,
                    'code': None
                })
            elif r_type == 'Polygon':
                x_min = min(p[0] for p in r.points)
                y_min = min(p[1] for p in r.points)
                x_max = max(p[0] for p in r.points)
                y_max = max(p[1] for p in r.points)
                frames.append({
                    'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                    'valid': True,
                    'failure': False,
                    'code': None
                })
            elif r_type == 'Special':
                code = int(getattr(r, 'code', 0))
                frames.append({
                    'bbox': None,
                    'valid': False,
                    'failure': (code == 2),
                    'code': code
                })
            else:
                frames.append({
                    'bbox': None,
                    'valid': False,
                    'failure': False,
                    'code': None
                })

        return frames
    except Exception as e:
        print(f"加载 bin 文件失败 {res_file}: {e}")
        return None
    finally:
        sys.path = old_path

def evaluate_sequence(seq_name, results_dir, sequences_dir):
    """评估单个序列"""
    # 加载真实标注
    gt_file = os.path.join(sequences_dir, seq_name, 'groundtruth.txt')
    if not os.path.exists(gt_file):
        print(f"警告: 找不到 {gt_file}")
        return None
    
    gt = load_groundtruth(gt_file)
    if gt is None:
        return None

    # 优先使用 baseline 重复运行的 bin 结果（VOT2018 官方设置）
    run_bin_files = find_bin_run_files(results_dir, seq_name)

    if run_bin_files:
        run_metrics = []
        all_segments = []
        all_overlaps = []
        total_failures = 0
        total_tracked_frames = 0

        for bin_file in run_bin_files:
            res = load_results_bin(str(bin_file))
            if res is None:
                continue

            run_result = evaluate_single_run(gt, res, use_special_frames=True)
            run_metrics.append(run_result)
            all_segments.extend(run_result['segments'])
            all_overlaps.append(run_result['overlaps'])
            total_failures += run_result['failures']
            total_tracked_frames += run_result['tracked_frames']

        if not run_metrics:
            print(f"警告: {seq_name} 的重复运行结果读取失败")
            return None

        accuracy = float(np.mean([r['accuracy'] for r in run_metrics]))
        robustness = float(np.mean([r['robustness'] for r in run_metrics]))
        eao = compute_sequence_eao(all_segments)
        min_len = int(np.min([r['length'] for r in run_metrics]))
        overlaps = np.concatenate(all_overlaps) if all_overlaps else np.array([], dtype=np.float32)

        return {
            'sequence': seq_name,
            'accuracy': accuracy,
            'robustness': robustness,
            'eao': eao,
            'failures': total_failures,
            'tracked_frames': total_tracked_frames,
            'length': min_len,
            'runs': len(run_metrics),
            'segments': all_segments,
            'overlaps': overlaps
        }

    # 回退：兼容单次 txt 输出
    res_file_txt = os.path.join(results_dir, f'{seq_name}.txt')
    if not os.path.exists(res_file_txt):
        print(f"警告: 找不到 {seq_name} 的 bin 重复结果与 txt 结果")
        return None

    res = load_results(res_file_txt)
    if res is None:
        return None

    run_result = evaluate_single_run(gt, res, use_special_frames=False)
    eao = compute_sequence_eao(run_result['segments'])

    return {
        'sequence': seq_name,
        'accuracy': run_result['accuracy'],
        'robustness': run_result['robustness'],
        'eao': eao,
        'failures': run_result['failures'],
        'tracked_frames': run_result['tracked_frames'],
        'length': run_result['length'],
        'runs': 1,
        'segments': run_result['segments'],
        'overlaps': run_result['overlaps']
    }

def evaluate_all(workspace_dir=None):
    """评估所有序列"""
    if workspace_dir is None:
        workspace_dir = PROJECT_ROOT / 'vot2018_workspace'
    else:
        workspace_dir = Path(workspace_dir)

    results_dir = workspace_dir / 'results'
    sequences_dir = workspace_dir / 'sequences'
    
    if not os.path.exists(results_dir):
        print(f"错误: 结果目录不存在: {results_dir}")
        return
    
    if not os.path.exists(sequences_dir):
        print(f"错误: 序列目录不存在: {sequences_dir}")
        return
    
    # 获取所有序列
    sequences = [d.name for d in sequences_dir.iterdir() if d.is_dir()]
    
    print("=" * 70)
    print("VOT2018 评估结果")
    print("=" * 70)
    print(f"{'序列名称':<20} {'A(Accuracy)':<12} {'R(Failures)':<12} {'EAO':<10} {'Runs':<8} {'总帧数':<10}")
    print("-" * 70)
    
    all_results = []
    total_accuracy = 0.0
    total_robustness = 0.0
    total_eao = 0.0
    total_failures = 0
    total_tracked_frames = 0
    total_frames = 0
    global_eao_segments = []
    
    for seq_name in sorted(sequences):
        result = evaluate_sequence(seq_name, results_dir, sequences_dir)
        if result is not None:
            all_results.append(result)
            total_accuracy += result['accuracy']
            total_robustness += result['robustness']
            total_eao += result['eao']
            total_failures += result['failures']
            total_tracked_frames += result.get('tracked_frames', result['length'])
            total_frames += result['length'] * result.get('runs', 1)
            global_eao_segments.extend(result.get('segments', []))
            
            print(
                f"{seq_name:<20} {result['accuracy']:<12.4f} "
                f"{result['robustness']:<12.4f} {result['eao']:<10.4f} "
                f"{result.get('runs', 1):<8} {result['length']:<10}"
            )
    
    print("-" * 70)
    
    # 计算总体指标
    num_sequences = len(all_results)
    if num_sequences > 0:
        avg_accuracy = total_accuracy / num_sequences
        avg_robustness = total_robustness / num_sequences
        avg_eao = total_eao / num_sequences
        dataset_eao = compute_sequence_eao(global_eao_segments)
        failure_rate = total_failures / total_frames if total_frames > 0 else 0.0
        
        print(f"\n总体统计:")
        print(f"  平均准确率 A (Accuracy):       {avg_accuracy:.4f}")
        print(f"  平均鲁棒性 R (Failures):       {avg_robustness:.4f}")
        print(f"  平均序列 EAO:                  {avg_eao:.4f}")
        print(f"  数据集 EAO (官方口径近似):     {dataset_eao:.4f}")
        print(f"  总失效次数:              {total_failures}")
        print(f"  失效率 (Failure Rate):   {failure_rate:.4f}")
        print(f"  评估序列数:              {num_sequences}")
        print(f"  有效预测帧数:            {total_tracked_frames}")
        print(f"  总帧数(含重复运行):      {total_frames}")
        print(f"  EAO 时间窗:              [{EAO_WINDOW_LOW}, {EAO_WINDOW_HIGH}]")
        print(f"  Burn-in 帧数:            {BURNIN_FRAMES}")
        print(f"  Re-init 跳帧(实验参数):  {SKIP_INITIALIZE_FRAMES}")
        
        # 为 A/R/EAO 生成序列级可视化
        plot_metrics_per_sequence(all_results)
        
        # 保存结果到文件
        save_results(all_results, workspace_dir)
    
    print("\n" + "=" * 70)

def plot_metric_per_sequence(results, metric_key, title, ylabel, output_filename, color):
    """绘制单个指标的序列级分面柱状图。"""
    values = [float(r[metric_key]) for r in results]
    seq_names = [r['sequence'] for r in results]

    if not values:
        return

    chunk_size = 20
    num_panels = int(np.ceil(len(values) / chunk_size))
    fig_seq, axes = plt.subplots(num_panels, 1, figsize=(12, 3.4 * num_panels), sharey=True)
    if num_panels == 1:
        axes = [axes]

    if metric_key in ('accuracy', 'eao'):
        y_upper = 1.0
    else:
        y_upper = max(0.05, float(np.max(values)) * 1.15)

    for panel_idx, ax in enumerate(axes):
        start = panel_idx * chunk_size
        end = min((panel_idx + 1) * chunk_size, len(values))

        if start >= len(values):
            ax.axis('off')
            continue

        x = np.arange(start, end)
        panel_values = values[start:end]
        panel_names = [name[:12] for name in seq_names[start:end]]

        ax.bar(x, panel_values, color=color, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(panel_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, y_upper)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'Sequences {start + 1}-{end}', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    axes[-1].set_xlabel('Sequence', fontsize=11)
    fig_seq.suptitle(title, fontsize=14)
    fig_seq.tight_layout(rect=[0, 0.02, 1, 0.96])

    output_seq_plot = PROJECT_ROOT / output_filename
    fig_seq.savefig(output_seq_plot, dpi=300, bbox_inches='tight')
    print(f"  指标图已保存: {output_seq_plot}")
    plt.close(fig_seq)

def plot_metrics_per_sequence(results):
    """为 A/R/EAO 三个指标生成序列级图像。"""
    plot_metric_per_sequence(
        results=results,
        metric_key='accuracy',
        title='Tracking Accuracy per Sequence (A)',
        ylabel='A (Avg IoU)',
        output_filename='vot2018_accuracy_per_sequence.png',
        color='steelblue'
    )
    plot_metric_per_sequence(
        results=results,
        metric_key='robustness',
        title='Tracking Robustness per Sequence (R)',
        ylabel='R (Failures)',
        output_filename='vot2018_robustness_per_sequence.png',
        color='darkorange'
    )
    plot_metric_per_sequence(
        results=results,
        metric_key='eao',
        title='Expected Average Overlap per Sequence (EAO)',
        ylabel='EAO',
        output_filename='vot2018_eao_per_sequence.png',
        color='seagreen'
    )

def save_results(results, workspace_dir):
    """保存评估结果到文件"""
    output_file = Path(workspace_dir) / 'evaluation_results.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("VOT2018 Evaluation Results\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(
            f"{'Sequence':<20} {'Accuracy(A)':<12} {'Robustness(R)':<14} "
            f"{'EAO':<10} {'Runs':<8} {'Failures':<10} {'Tracked':<10} {'Frames':<10}\n"
        )
        f.write("-" * 70 + "\n")
        
        for r in results:
            f.write(
                f"{r['sequence']:<20} {r['accuracy']:<12.4f} {r['robustness']:<14.4f} "
                f"{r['eao']:<10.4f} {r.get('runs', 1):<8} {r['failures']:<10} "
                f"{r.get('tracked_frames', r['length']):<10} {r['length']:<10}\n"
            )
        
        f.write("-" * 70 + "\n")
        
        total_accuracy = sum(r['accuracy'] for r in results)
        total_robustness = sum(r['robustness'] for r in results)
        total_eao = sum(r['eao'] for r in results)
        total_failures = sum(r['failures'] for r in results)
        total_tracked = sum(r.get('tracked_frames', r['length']) for r in results)
        total_frames = sum(r['length'] * r.get('runs', 1) for r in results)
        all_segments = []
        for r in results:
            all_segments.extend(r.get('segments', []))
        num_sequences = len(results)
        
        f.write(f"\nOverall Statistics:\n")
        f.write(f"  Average Accuracy (A):     {total_accuracy / num_sequences:.4f}\n")
        f.write(f"  Average Robustness (R):   {total_robustness / num_sequences:.4f}\n")
        f.write(f"  Average Sequence EAO:     {total_eao / num_sequences:.4f}\n")
        f.write(f"  Dataset EAO:              {compute_sequence_eao(all_segments):.4f}\n")
        f.write(f"  Total Failures:           {total_failures}\n")
        f.write(f"  Failure Rate:             {total_failures / total_frames:.4f}\n")
        f.write(f"  EAO Window:               [{EAO_WINDOW_LOW}, {EAO_WINDOW_HIGH}]\n")
        f.write(f"  Burn-in:                  {BURNIN_FRAMES}\n")
        f.write(f"  Skip Initialize:          {SKIP_INITIALIZE_FRAMES}\n")
        f.write(f"  Num Sequences:            {num_sequences}\n")
        f.write(f"  Tracked Frames:           {total_tracked}\n")
        f.write(f"  Total Frames:             {total_frames}\n")
    
    print(f"  评估结果已保存: {output_file}")

if __name__ == '__main__':
    evaluate_all()
