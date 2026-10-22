import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# 解决matplotlib显示中文的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ====================== 【自定义函数区】======================
def contrast_distribution_saliency(img, dir_num=8):
    """
    自定义函数：基于对比度分布的显著性检测
    :param img: 输入原图（RGB）
    :param dir_num: 方向数量，8 / 16
    :return: saliency_map 显著性图(0~255)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    saliency = np.zeros_like(gray, dtype=np.float32)

    # 生成dir_num个方向角
    angles = np.linspace(0, 2 * np.pi, dir_num, endpoint=False)
    kernel_size = 3

    for theta in angles:
        # 构造该方向的Sobel梯度核
        dx = np.cos(theta)
        dy = np.sin(theta)
        # 求该方向图像梯度
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        grad_dir = sobel_x * dx + sobel_y * dy
        grad_mag = np.abs(grad_dir)
        saliency += grad_mag

    # 归一化到0~255
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)
    saliency = saliency.astype(np.uint8)
    return saliency


def min_direction_contrast_saliency(img):
    """
    自定义函数：基于最小方向对比度显著性检测
    对每个像素，取所有方向梯度的最小值作为显著值
    :param img: 输入RGB图像
    :return: saliency_map 显著性图
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    # 使用16个方向
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    grad_list = []

    for theta in angles:
        dx = np.cos(theta)
        dy = np.sin(theta)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F,1,0,ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F,0,1,ksize=3)
        grad_dir = sobel_x * dx + sobel_y * dy
        grad_mag = np.abs(grad_dir)
        grad_list.append(grad_mag)

    grad_stack = np.stack(grad_list, axis=-1)
    # 最小方向对比度：取每个像素所有方向梯度里最小值
    saliency_min = np.min(grad_stack, axis=-1)
    saliency_min = cv2.normalize(saliency_min, None,0,255,cv2.NORM_MINMAX)
    saliency_min = saliency_min.astype(np.uint8)
    return saliency_min


# ====================== 【主程序入口，固定执行代码】=====================
if __name__ == "__main__":
    # 【修改这里：放你的3张图片路径】
    img_paths = ["data/test1.jpg", "data/test2.jpg", "data/test4.jpg"]

    for idx, path in enumerate(img_paths):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # -------- 1.对比度分布 8方向 --------
        t0 = time.time()
        sal8 = contrast_distribution_saliency(img, dir_num=8)
        t8 = time.time() - t0

        # -------- 2.对比度分布 16方向 --------
        t0 = time.time()
        sal16 = contrast_distribution_saliency(img, dir_num=16)
        t16 = time.time() - t0

        # --------3.最小方向对比度 --------
        t0 = time.time()
        sal_min = min_direction_contrast_saliency(img)
        t_min = time.time() - t0

        print(f"\n=====图片{idx+1} 耗时=====")
        print(f"对比度分布 8方向：{t8:.4f} s")
        print(f"对比度分布16方向：{t16:.4f} s")
        print(f"最小方向对比度：{t_min:.4f} s")

        # 绘图展示结果
        plt.figure(figsize=(14, 3.5))
        plt.subplot(1,4,1)
        plt.imshow(img)
        plt.title("原图")
        plt.axis('off')

        plt.subplot(1,4,2)
        plt.imshow(sal8, cmap='gray')
        plt.title("对比度分布 8方向")
        plt.axis('off')

        plt.subplot(1,4,3)
        plt.imshow(sal16, cmap='gray')
        plt.title("对比度分布 16方向")
        plt.axis('off')

        plt.subplot(1,4,4)
        plt.imshow(sal_min, cmap='gray')
        plt.title("最小方向对比度")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
