import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------- 1.SUSAN算子实现（37点圆形模板） ----------------------
def susan_edge_detect(gray_img, t=25, g=18):
    h, w = gray_img.shape
    out = np.zeros_like(gray_img, dtype=np.float32)
    # 37点SUSAN圆形模板坐标（半径3.4）
    mask_offsets = [
        (-3, -1), (-3,0), (-3,1),
        (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2),
        (-1,-3), (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2), (-1,3),
        (0,-3),  (0,-2),  (0,-1),          (0,1),  (0,2),  (0,3),
        (1,-3),  (1,-2),  (1,-1),  (1,0), (1,1),  (1,2),  (1,3),
        (2,-2),  (2,-1),  (2,0),   (2,1), (2,2),
        (3,-1),  (3,0),   (3,1)
    ]
    for y in range(3, h-3):
        for x in range(3, w-3):
            nucleus = gray_img[y, x]
            n = 0
            for dy, dx in mask_offsets:
                ny = y + dy
                nx = x + dx
                if abs(int(gray_img[ny, nx]) - int(nucleus)) <= t:
                    n +=1
            r = g - n
            if r > 0:
                out[y, x] = r
    out = cv2.normalize(out, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)
    _, edge_susan = cv2.threshold(out, 30, 255, cv2.THRESH_BINARY)
    return edge_susan

# ---------------------- 2.Sobel算子 ----------------------
def sobel_edge_detect(gray_img):
    H, W = gray_img.shape
    Gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    result = np.zeros((H, W), dtype=np.float32)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            p = gray_img[i - 1:i + 2, j - 1:j + 2]
            gx = np.sum(Gx_kernel * p)
            gy = np.sum(Gy_kernel * p)
            grad = np.sqrt(gx ** 2 + gy ** 2)
            result[i, j] = grad
    return result

# ---------------------- 3.添加高斯噪声 ----------------------
def add_gauss_noise(img, sigma=15):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy,0,255).astype(np.uint8)

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    img = cv2.imread("data/test1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_noisy = add_gauss_noise(gray, sigma=18)

    # 干净图像检测
    susan_clean = susan_edge_detect(gray, t=25, g=18)
    sobel_clean = sobel_edge_detect(gray)

    # 噪声图像检测
    susan_noisy = susan_edge_detect(gray_noisy, t=25, g=18)
    sobel_noisy = sobel_edge_detect(gray_noisy)

    # 绘图对比
    plt.figure(figsize=(12, 8))
    plt.subplot(2,3,1);plt.title("原图灰度");plt.imshow(gray,cmap="gray");plt.axis('off')
    plt.subplot(2,3,2);plt.title("SUSAN(干净图)");plt.imshow(susan_clean,cmap="gray");plt.axis('off')
    plt.subplot(2,3,3);plt.title("Sobel(干净图)");plt.imshow(sobel_clean,cmap="gray");plt.axis('off')

    plt.subplot(2,3,1+3);plt.title("加噪声图像");plt.imshow(gray_noisy,cmap="gray");plt.axis('off')
    plt.subplot(2,3,2+3);plt.title("SUSAN(噪声图)");plt.imshow(susan_noisy,cmap="gray");plt.axis('off')
    plt.subplot(2,3,3+3);plt.title("Sobel(噪声图)");plt.imshow(sobel_noisy,cmap="gray");plt.axis('off')

    plt.tight_layout()
    plt.show()