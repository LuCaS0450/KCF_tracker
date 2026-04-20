import os
import cv2
import glob
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.kcf import KCFTracker

def get_sequence_info(seq_dir):
    """
    Parses the groundtruth and image paths for a sequence.
    """
    seq_dir = Path(seq_dir)
    img_dir = seq_dir / 'img'
    gt_path = seq_dir / 'groundtruth_rect.txt'

    if not gt_path.exists() or not img_dir.exists():
        return None, None

    with open(gt_path, 'r') as f:
        lines = f.readlines()

    gt = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        delimiter = ',' if ',' in line else None
        vals = line.split(delimiter)
        if len(vals) < 4:
            vals = line.split()
        gt.append([float(v) for v in vals[:4]])

    img_files = sorted(glob.glob(str(img_dir / '*.jpg')) + glob.glob(str(img_dir / '*.png')))

    return img_files, gt

def run_otb():
    otb_dir = PROJECT_ROOT / 'otb100'
    res_dir = PROJECT_ROOT / 'results' / 'KCF'
    res_dir.mkdir(parents=True, exist_ok=True)

    if not otb_dir.exists():
        raise FileNotFoundError(f"OTB dataset directory not found: {otb_dir}")

    sequences = sorted(os.listdir(otb_dir))

    for seq in tqdm(sequences, desc="Evaluating OTB100"):
        seq_dir = otb_dir / seq
        if not seq_dir.is_dir():
            continue

        img_files, gt = get_sequence_info(seq_dir)
        if img_files is None or len(img_files) == 0 or len(gt) == 0:
            continue

        # =========== OTB 数据集帧对齐逻辑 ===========
        # 处理 GT 不是从第 1 帧开始的情况（如 David 序列）
        if len(img_files) > len(gt):
            # 截取图片列表的尾部，使其与 GT 的长度和对应关系完美匹配
            img_files = img_files[-len(gt):]
        elif len(img_files) < len(gt):
            # 防止跟踪器中途崩溃或提前退出，截断 GT 以匹配预测长度
            gt = gt[:len(img_files)]
        # ============================================

        tracker = KCFTracker()

        # Read the first frame
        frame = cv2.imread(img_files[0])
        init_bbox = gt[0]
        tracker.init(frame, init_bbox)

        res = [init_bbox]

        # Track through sequence
        for i in range(1, len(img_files)):
            frame = cv2.imread(img_files[i])
            if frame is None:
                break
            bbox = tracker.update(frame)
            res.append(bbox)

        # Save tracking results
        res_path = res_dir / f"{seq}.txt"
        with open(res_path, 'w', encoding='utf-8') as f:
            for bbox in res:
                f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")

if __name__ == '__main__':
    run_otb()
