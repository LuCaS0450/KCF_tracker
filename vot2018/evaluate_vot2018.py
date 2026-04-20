"""
评估 VOT2018 测试结果
计算 EAO (Expected Average Overlap)、准确率 (Accuracy) 和鲁棒性 (Robustness)
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

    # 加载跟踪结果 (优先读取 txt; 如果没有，尝试寻找 bin)
    res_file_txt = os.path.join(results_dir, f'{seq_name}.txt')
    res_file_bin = os.path.join(results_dir, 'KCF_Tracker', 'baseline', seq_name, f'{seq_name}_001.bin')
    
    use_bin_results = False
    if os.path.exists(res_file_txt):
        res = load_results(res_file_txt)
    elif os.path.exists(res_file_bin):
        res = load_results_bin(res_file_bin)
        use_bin_results = True
    else:
        print(f"警告: 找不到 {res_file_txt} 以及 {res_file_bin}")
        return None

    if res is None:
        return None

    # 确保长度一致
    min_len = min(len(gt), len(res))
    gt = gt[:min_len]
    res = res[:min_len]
    
    overlaps = []
    failures = 0
    tracked_frames = 0

    if use_bin_results:
        # VOT bin 结果中包含 Special 帧：
        # code=1 初始化, code=2 失效事件, code=0 跳过/未知
        # 准确率仅在有效预测框帧上统计；失效次数仅统计 code=2。
        for i in range(min_len):
            frame_info = res[i]
            if frame_info.get('valid', False):
                iou = compute_overlap(frame_info['bbox'], gt[i])
                overlaps.append(iou)
                tracked_frames += 1

            if frame_info.get('failure', False):
                failures += 1
    else:
        # 兼容普通 txt 结果：维持原始按帧 IoU 统计方式
        for i in range(min_len):
            iou = compute_overlap(res[i], gt[i])
            overlaps.append(iou)
        tracked_frames = len(overlaps)
        failures = int(np.sum(np.array(overlaps) == 0))

    overlaps = np.array(overlaps, dtype=np.float32)

    # 计算准确率
    accuracy = np.mean(overlaps) if len(overlaps) > 0 else 0.0
    
    return {
        'sequence': seq_name,
        'accuracy': accuracy,
        'failures': failures,
        'tracked_frames': tracked_frames,
        'length': min_len,
        'overlaps': overlaps
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
    print(f"{'序列名称':<20} {'准确率':<10} {'失效次数':<10} {'有效帧数':<10} {'总帧数':<10}")
    print("-" * 70)
    
    all_results = []
    total_accuracy = 0.0
    total_failures = 0
    total_tracked_frames = 0
    total_frames = 0
    
    for seq_name in sorted(sequences):
        result = evaluate_sequence(seq_name, results_dir, sequences_dir)
        if result is not None:
            all_results.append(result)
            total_accuracy += result['accuracy']
            total_failures += result['failures']
            total_tracked_frames += result.get('tracked_frames', result['length'])
            total_frames += result['length']
            
            print(f"{seq_name:<20} {result['accuracy']:<10.4f} "
                f"{result['failures']:<10} {result.get('tracked_frames', result['length']):<10} {result['length']:<10}")
    
    print("-" * 70)
    
    # 计算总体指标
    num_sequences = len(all_results)
    if num_sequences > 0:
        avg_accuracy = total_accuracy / num_sequences
        failure_rate = total_failures / total_frames if total_frames > 0 else 0.0
        
        print(f"\n总体统计:")
        print(f"  平均准确率 (Accuracy):  {avg_accuracy:.4f}")
        print(f"  总失效次数:              {total_failures}")
        print(f"  失效率 (Failure Rate):   {failure_rate:.4f}")
        print(f"  评估序列数:              {num_sequences}")
        print(f"  有效预测帧数:            {total_tracked_frames}")
        print(f"  总帧数:                  {total_frames}")
        
        # 绘制准确率分布图
        plot_accuracy_distribution(all_results)
        
        # 保存结果到文件
        save_results(all_results, workspace_dir)
    
    print("\n" + "=" * 70)

def plot_accuracy_distribution(results):
    """绘制准确率分布图"""
    accuracies = [r['accuracy'] for r in results]
    seq_names = [r['sequence'] for r in results]
    
    plt.figure(figsize=(14, 6))
    
    # 柱状图
    plt.subplot(1, 2, 1)
    x = range(len(accuracies))
    plt.bar(x, accuracies, color='steelblue', alpha=0.7)
    plt.xlabel('Sequence Index', fontsize=12)
    plt.ylabel('Accuracy (Average IoU)', fontsize=12)
    plt.title('Tracking Accuracy per Sequence', fontsize=14)
    plt.xticks(x, [name[:10] for name in seq_names], rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    
    # 直方图
    plt.subplot(1, 2, 2)
    plt.hist(accuracies, bins=15, color='coral', alpha=0.7, edgecolor='black')
    plt.xlabel('Accuracy (Average IoU)', fontsize=12)
    plt.ylabel('Number of Sequences', fontsize=12)
    plt.title('Accuracy Distribution', fontsize=14)
    plt.axvline(np.mean(accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(accuracies):.3f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_plot = PROJECT_ROOT / 'vot2018_accuracy.png'
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"  准确率分布图已保存: {output_plot}")
    plt.close()

def save_results(results, workspace_dir):
    """保存评估结果到文件"""
    output_file = Path(workspace_dir) / 'evaluation_results.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("VOT2018 Evaluation Results\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Sequence':<20} {'Accuracy':<12} {'Failures':<10} {'Tracked':<10} {'Frames':<10}\n")
        f.write("-" * 70 + "\n")
        
        for r in results:
            f.write(f"{r['sequence']:<20} {r['accuracy']:<12.4f} "
                   f"{r['failures']:<10} {r.get('tracked_frames', r['length']):<10} {r['length']:<10}\n")
        
        f.write("-" * 70 + "\n")
        
        total_accuracy = sum(r['accuracy'] for r in results)
        total_failures = sum(r['failures'] for r in results)
        total_tracked = sum(r.get('tracked_frames', r['length']) for r in results)
        total_frames = sum(r['length'] for r in results)
        num_sequences = len(results)
        
        f.write(f"\nOverall Statistics:\n")
        f.write(f"  Average Accuracy:  {total_accuracy / num_sequences:.4f}\n")
        f.write(f"  Total Failures:    {total_failures}\n")
        f.write(f"  Failure Rate:      {total_failures / total_frames:.4f}\n")
        f.write(f"  Num Sequences:     {num_sequences}\n")
        f.write(f"  Tracked Frames:    {total_tracked}\n")
        f.write(f"  Total Frames:      {total_frames}\n")
    
    print(f"  评估结果已保存: {output_file}")

if __name__ == '__main__':
    evaluate_all()
