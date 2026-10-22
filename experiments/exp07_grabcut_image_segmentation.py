import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def improved_graphcut_segmentation(img_path, iter_base=6, iter_refine=4):
    # 读取图像
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"图片 {img_path} 读取失败！")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    # ==========预处理：高斯滤波降噪，改善分割边界==========
    img_smooth = cv2.GaussianBlur(img_bgr, (5,5), 1.2)

    # 初始化mask
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 改进ROI：框稍微向内收缩，框外全部标记为确定背景
    margin_x = int(w * 0.12)
    margin_y = int(h * 0.12)
    rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)

    # ============ 计时：仅图割核心图搜索部分 ============
    start_time = time.perf_counter()

    # 第一轮：矩形初始化GraphCut
    cv2.grabCut(img_smooth, mask, rect, bgdModel, fgdModel, iter_base, cv2.GC_INIT_WITH_RECT)
    # 第二轮：用上一轮得到的mask继续迭代精修（GC_EVAL 继续求解最小割）
    cv2.grabCut(img_smooth, mask, None, bgdModel, fgdModel, iter_refine, cv2.GC_EVAL)

    end_time = time.perf_counter()
    seg_time = end_time - start_time

    # 生成二值掩码：GC_FGD(3) GC_PR_FGD(1) 作为前景；0,2为背景
    mask_bin = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # =========后处理形态学优化：去除孤立噪点，填充孔洞，平滑轮廓========
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel_small) # 闭运算：填充小孔
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel_small)  # 开运算：去除小噪点
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel_big)

    # 生成分割结果
    seg_result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_bin)
    return img_rgb, mask_bin, seg_result, seg_time


if __name__ == "__main__":
    # 替换成你的3张图片路径
    image_paths = [
        "data/test.jpg",
        "data/test1.jpg",
        "data/test2.jpg"
    ]

    plt.figure(figsize=(18, 12))
    for idx, path in enumerate(image_paths):
        try:
            origin, mask, seg_img, t = improved_graphcut_segmentation(path)
            print(f"【图片{idx+1}】{path} | 图割分割耗时：{t:.4f} s")

            # 子图：原图 / 掩码 / 分割结果
            plt.subplot(3,3, idx*3 + 1)
            plt.title(f"原图{idx+1}")
            plt.imshow(origin)
            plt.axis("off")

            plt.subplot(3,3, idx*3 + 2)
            plt.title(f"分割掩码")
            plt.imshow(mask, cmap="gray")
            plt.axis("off")

            plt.subplot(3,3, idx*3 + 3)
            plt.title(f"分割结果 | time={t:.4f}s")
            plt.imshow(seg_img)
            plt.axis("off")
        except Exception as e:
            print(f"处理 {path} 异常：{e}")

    plt.tight_layout()
    plt.show()
