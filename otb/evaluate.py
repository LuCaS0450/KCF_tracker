import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOT_FIGSIZE = (10, 7)
PLOT_DPI = 300
PLOT_FONT_SIZE = 14
PLOT_TICK_SIZE = 12

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Assumes boxes are numpy arrays with format [x, y, w, h].
    """
    x1, y1, w1, h1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    x2, y2, w2, h2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    area1 = w1 * h1
    area2 = w2 * h2

    # Calculate intersection coordinates
    ix1 = np.maximum(x1, x2)
    iy1 = np.maximum(y1, y2)
    ix2 = np.minimum(x1 + w1, x2 + w2)
    iy2 = np.minimum(y1 + h1, y2 + h2)

    iw = np.maximum(0, ix2 - ix1)
    ih = np.maximum(0, iy2 - iy1)

    intersection = iw * ih
    union = area1 + area2 - intersection

    # Avoid division by zero
    iou = intersection / (union + 1e-16)
    return iou

def compute_cle(box1, box2):
    """
    Compute Center Location Error (CLE) between two bounding boxes.
    Assumes boxes are numpy arrays with format [x, y, w, h].
    """
    cx1 = box1[:, 0] + box1[:, 2] / 2
    cy1 = box1[:, 1] + box1[:, 3] / 2

    cx2 = box2[:, 0] + box2[:, 2] / 2
    cy2 = box2[:, 1] + box2[:, 3] / 2

    cle = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return cle

def load_boxes(path):
    boxes = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            delimiter = ',' if ',' in line else None
            vals = line.split(delimiter)
            if len(vals) < 4:
                vals = line.split()
            boxes.append([float(v) for v in vals[:4]])
    return np.array(boxes)

def evaluate_otb():
    otb_dir = PROJECT_ROOT / 'otb100'
    res_dir = PROJECT_ROOT / 'results' / 'KCF'

    if not res_dir.exists():
        print("Predictions not found. Please run otb/run_tracker.py first.")
        return

    if not otb_dir.exists():
        print(f"OTB dataset directory not found: {otb_dir}")
        return

    sequences = sorted(os.listdir(otb_dir))

    # 用于存放每个视频的独立评估曲线 (Macro-average 准备)
    all_seq_success = []
    all_seq_precision = []

    thresholds_iou = np.linspace(0, 1, 100)
    thresholds_cle = np.arange(0, 51) # 0 to 50 pixels

    print("Evaluating sequences...")
    for seq in sequences:
        seq_dir = otb_dir / seq
        gt_path = seq_dir / 'groundtruth_rect.txt'
        res_path = res_dir / f"{seq}.txt"

        if not gt_path.exists() or not res_path.exists():
            continue

        gt_boxes = load_boxes(gt_path)
        res_boxes = load_boxes(res_path)

        # ==========================================
        # 修改点 3: 完美处理 David 等特殊视频的帧错位问题
        # ==========================================
        # 在 OTB 中，缺失 GT 的帧总是发生在视频的开头（例如 David 从 300 帧开始，但图片有 770 张）。
        # GT 的结尾总是与视频的最后一帧对齐。
        # 因此，如果预测框数量大于 GT 数量，说明跟踪器从第 1 帧开始盲跑了。
        # 我们只需要截取预测框的"尾部"与 GT 对齐即可：
        if len(res_boxes) > len(gt_boxes):
            res_boxes = res_boxes[-len(gt_boxes):]
        elif len(res_boxes) < len(gt_boxes):
            # 防止跟踪器中途崩溃或提前退出，截断 GT 以匹配预测长度
            gt_boxes = gt_boxes[:len(res_boxes)]

        if len(gt_boxes) == 0:
            continue

        ious = compute_iou(res_boxes, gt_boxes)
        cles = compute_cle(res_boxes, gt_boxes)

        # ==========================================
        # 修改点 1 & 2: OTB 官方标准 (Macro-average) 且 <= 阈值
        # ==========================================
        # 分别计算当前这【一个视频】的曲线
        seq_success_curve = [np.mean(ious > th) for th in thresholds_iou]
        seq_precision_curve = [np.mean(cles <= th) for th in thresholds_cle]  # 改为 <=

        # 将该视频的曲线存入总池子中
        all_seq_success.append(seq_success_curve)
        all_seq_precision.append(seq_precision_curve)

    if not all_seq_success:
        print("No valid sequences evaluated.")
        return

    # 对所有视频的曲线求均值，保证每个视频权重为 1:1（长短视频一视同仁）
    mean_success_curve = np.mean(all_seq_success, axis=0)
    mean_precision_curve = np.mean(all_seq_precision, axis=0)

    # 最终指标计算
    auc = np.mean(mean_success_curve)
    prec_20 = mean_precision_curve[20]  # thresholds_cle[20] 就是 20px

    plt.rcParams.update({
        'font.size': PLOT_FONT_SIZE,
        'axes.titlesize': PLOT_FONT_SIZE + 2,
        'axes.labelsize': PLOT_FONT_SIZE,
        'xtick.labelsize': PLOT_TICK_SIZE,
        'ytick.labelsize': PLOT_TICK_SIZE,
        'legend.fontsize': PLOT_TICK_SIZE,
    })

    # 绘制成功率曲线 (Success Plot)
    plt.figure(figsize=PLOT_FIGSIZE)
    plt.plot(thresholds_iou, mean_success_curve, label=f'KCF [AUC: {auc:.3f}]', color='red', linewidth=2.5)
    plt.title('Success Plot of OPE')
    plt.xlabel('Overlap threshold')
    plt.ylabel('Success rate')
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'success_plot.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    # 绘制精确度曲线 (Precision Plot)
    plt.figure(figsize=PLOT_FIGSIZE)
    plt.plot(thresholds_cle, mean_precision_curve, label=f'KCF [DP@20: {prec_20:.3f}]', color='blue', linewidth=2.5)
    plt.title('Precision Plot of OPE')
    plt.xlabel('Location error threshold')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'precision_plot.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    print(f"Evaluation complete.")
    print(f"Success AUC: {auc:.3f}")
    print(f"Precision @ 20px: {prec_20:.3f}")

if __name__ == '__main__':
    evaluate_otb()