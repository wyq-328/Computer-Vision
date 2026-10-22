import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
import os

# 解决matplotlib显示中文的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def histogram_contrast_saliency(img_bgr):
    """
    1. 基于直方图对比度的显著性检测 (Histogram Contrast - HC)
    """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    bins = 12
    # 计算量化值
    quant = (img_lab // (256 // bins)).astype(np.int32)
    # 【关键修复1】裁剪量化值，确保不会越界（最大值限制为11）
    quant = np.clip(quant, 0, bins - 1)

    # 将量化后的3D坐标映射到1D索引
    index = quant[:, :, 0] * bins * bins + quant[:, :, 1] * bins + quant[:, :, 2]

    # 计算每个bin的频率
    hist = np.bincount(index.flatten(), minlength=bins ** 3).astype(np.float32)
    # 【关键修复2】强制截断长度，确保严格等于 bins**3 (1728)
    hist = hist[:bins ** 3]
    hist /= hist.sum()

    # 获取每个bin的中心颜色
    colors = []
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                colors.append([i * (256 // bins) + (256 // bins // 2),
                               j * (256 // bins) + (256 // bins // 2),
                               k * (256 // bins) + (256 // bins // 2)])
    colors = np.array(colors, dtype=np.float32)

    # 计算显著性
    saliency_bins = np.zeros(bins ** 3)
    for i in range(bins ** 3):
        if hist[i] == 0:
            continue
        dist = np.linalg.norm(colors - colors[i], axis=1)
        saliency_bins[i] = np.sum(dist * hist)

    saliency_map = saliency_bins[index]
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    saliency_map = cv2.GaussianBlur(saliency_map, (21, 21), 10)

    return saliency_map


def region_contrast_saliency(img_bgr):

    # 转换到Lab颜色空间
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    height, width = img_bgr.shape[:2]

    # 使用SLIC超像素分割将图像分成若干区域
    # n_segments可根据图像大小调整，这里设为200
    segments = slic(img_bgr, n_segments=200, compactness=10, sigma=1, start_label=0)

    # 获取分割区域的数量
    num_regions = np.max(segments) + 1

    # 计算每个区域的平均颜色、空间中心位置和像素数量
    region_colors = np.zeros((num_regions, 3), dtype=np.float32)
    region_centers = np.zeros((num_regions, 2), dtype=np.float32)
    region_sizes = np.zeros(num_regions, dtype=np.float32)

    for i in range(num_regions):
        mask = (segments == i)
        if np.sum(mask) == 0:
            continue
        region_sizes[i] = np.sum(mask)
        # 平均颜色
        region_colors[i] = np.mean(img_lab[mask], axis=0)
        # 空间中心
        y_coords, x_coords = np.where(mask)
        region_centers[i] = [np.mean(y_coords), np.mean(x_coords)]

    # 计算区域间的显著性
    saliency_regions = np.zeros(num_regions, dtype=np.float32)

    for i in range(num_regions):
        if region_sizes[i] == 0:
            continue
        # 与所有其他区域计算对比度
        color_dist = np.linalg.norm(region_colors - region_colors[i], axis=1)
        spatial_dist = np.linalg.norm(region_centers - region_centers[i], axis=1)

        # 空间距离加权 (距离越远，影响越小，使用高斯函数)
        spatial_weight = np.exp(-spatial_dist / (0.2 * max(height, width)))

        # 显著性 = 颜色距离 * 区域大小 * 空间权重 的累加
        saliency_regions[i] = np.sum(color_dist * region_sizes * spatial_weight)

    # 映射回像素级
    saliency_map = saliency_regions[segments]

    # 归一化到 0-255
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return saliency_map


def process_images(image_paths):

    for idx, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f"警告：找不到图片 {img_path}，跳过。")
            continue

        print(f"正在处理第 {idx + 1} 张图片: {img_path}")
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            print(f"读取失败: {img_path}")
            continue

        # 限制图片大小，防止计算过慢 (例如限制最长边为500像素)
        max_dim = 500
        h, w = img_bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

        # 计算 HC 和 RC 显著图
        hc_map = histogram_contrast_saliency(img_bgr)
        rc_map = region_contrast_saliency(img_bgr)

        # 转换颜色空间用于Matplotlib显示
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 绘图展示
        plt.figure(figsize=(15, 4))
        plt.suptitle(f"图片 {idx + 1}: {os.path.basename(img_path)}", fontsize=16)

        plt.subplot(1, 3, 1)
        plt.imshow(img_rgb)
        plt.title("原始图像")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(hc_map, cmap='gray')
        plt.title("1. 基于直方图对比度 (HC)")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(rc_map, cmap='gray')
        plt.title("2. 基于区域对比度 (RC)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    image_list = [
        "data/test.jpg",
        "data/test1.jpg",
        "data/test2.jpg"
    ]

    process_images(image_list)
