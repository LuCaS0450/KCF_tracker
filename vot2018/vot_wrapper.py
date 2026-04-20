import sys
import cv2
from pathlib import Path
import math
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vot2018 import vot_local as vot
from core.kcf import KCFTracker


def main():
    # 1. 启动 TraX 协议与 VOT 评测工具进行握手
    # 参数 "rectangle" 告诉官方工具：我的 KCF 只能处理矩形框，请把多边形自动帮我转换好
    handle = vot.VOT("rectangle")

    # 获取初始框和第一帧图片路径
    selection = handle.region()
    imagefile = handle.frame()

    if not imagefile:
        sys.exit(0)

    # 2. 将 VOT 传过来的对象转换为 KCF 认识的格式 [x, y, w, h]
    init_bbox = [selection.x, selection.y, selection.width, selection.height]

    # 3. 初始化 KCF 跟踪器
    image = cv2.imread(imagefile)
    if image is None:
        with open("error_log.txt", "a") as f:
            f.write(f"Init Image is None for imagefile: {imagefile}\n")
            sys.exit(0)
            
    tracker = KCFTracker()
    try:
        tracker.init(image, init_bbox)
    except Exception as e:
        with open("error_log.txt", "a") as f:
            f.write(f"Exception during init: {e}\n{traceback.format_exc()}\n")
        sys.exit(0)

    # 重要：立即报告第一帧的结果（使用初始框）
    handle.report(vot.Rectangle(float(init_bbox[0]), float(init_bbox[1]), float(init_bbox[2]), float(init_bbox[3])))

    # 4. 进入实时跟踪交互循环
    while True:
        # 向评测工具索要下一帧的图片
        imagefile = handle.frame()
        if not imagefile:
            break  # 视频结束

        image = cv2.imread(imagefile)
        
        # 检查图像是否成功加载
        if image is None:
            with open("error_log.txt", "a") as f:
                f.write(f"Image is None for imagefile: {imagefile}\n")
            handle.report([])
            continue

        # VOT 在失效后会发送 initialize 请求，这里需要重置跟踪器状态
        reinit_region = handle.new_region()
        if reinit_region is not None:
            init_bbox = [reinit_region.x, reinit_region.y, reinit_region.width, reinit_region.height]
            tracker = KCFTracker()
            try:
                tracker.init(image, init_bbox)
                handle.report(vot.Rectangle(float(init_bbox[0]), float(init_bbox[1]), float(init_bbox[2]), float(init_bbox[3])))
            except Exception as e:
                with open("error_log.txt", "a") as f:
                    f.write(f"Exception during re-init: {e}\n{traceback.format_exc()}\n")
                handle.report([])
            continue

        # 你的 KCF 进行预测
        try:
            bbox = tracker.update(image)
        except Exception as e:
            with open("error_log.txt", "a") as f:
                f.write(f"Exception during update: {e}\n{traceback.format_exc()}\n")
            handle.report([])
            continue
        
        # 检查 bbox 是否有效
        if bbox is None or len(bbox) != 4:
            # 如果跟踪失败，报告空列表（表示失效）
            handle.report([])
        else:
            x, y, w, h = bbox
            # 检查是否有无效值
            if math.isnan(x) or math.isnan(y) or math.isnan(w) or math.isnan(h):
                handle.report([])
            elif math.isinf(x) or math.isinf(y) or math.isinf(w) or math.isinf(h):
                handle.report([])
            elif w <= 0 or h <= 0:
                handle.report([])
            else:
                # 将预测结果打包成 VOT 官方格式，并通过 TraX 报告给评测工具
                handle.report(vot.Rectangle(float(x), float(y), float(w), float(h)))


if __name__ == '__main__':
    main()