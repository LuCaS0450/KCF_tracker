import os
import cv2
import glob
from tqdm import tqdm
from kcf import KCFTracker

def get_sequence_info(seq_dir):
    """
    Parses the groundtruth and image paths for a sequence.
    """
    img_dir = os.path.join(seq_dir, 'img')
    gt_path = os.path.join(seq_dir, 'groundtruth_rect.txt')

    if not os.path.exists(gt_path) or not os.path.exists(img_dir):
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

    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))

    return img_files, gt

def run_otb():
    otb_dir = 'otb100'
    res_dir = os.path.join('results', 'KCF')
    os.makedirs(res_dir, exist_ok=True)

    sequences = sorted(os.listdir(otb_dir))

    for seq in tqdm(sequences, desc="Evaluating OTB100"):
        seq_dir = os.path.join(otb_dir, seq)
        if not os.path.isdir(seq_dir):
            continue

        img_files, gt = get_sequence_info(seq_dir)
        if img_files is None or len(img_files) == 0:
            continue

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
        res_path = os.path.join(res_dir, f"{seq}.txt")
        with open(res_path, 'w') as f:
            for bbox in res:
                f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")

if __name__ == '__main__':
    run_otb()
