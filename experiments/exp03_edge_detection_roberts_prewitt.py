import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

img = cv2.imread("data/test1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# roberts算子
def roberts(src):
    H, W = src.shape
    Gx_kernel = np.array([[1, 0], [0, -1]])
    Gy_kernel = np.array([[0, 1], [-1, 0]])
    result = np.zeros((H, W), dtype=np.float32)

    for i in range(H - 2):
        for j in range(W - 2):
            p = src[i:i + 2, j:j + 2]
            gx = np.sum(Gx_kernel * p)
            gy = np.sum(Gy_kernel * p)
            grad = np.sqrt(gx ** 2 + gy ** 2)
            result[i, j] = grad
    return result
#prewitt算子
def prewitt(src):
    H, W = src.shape
    Gx_kernel = np.array([[-1, 0,1], [-1, 0,1],[-1, 0,1]])
    Gy_kernel = np.array([[1,1, 1], [0, 0,0],[-1,-1,-1]])
    result = np.zeros((H, W), dtype=np.float32)
    for i in range(1,H - 1):
        for j in range(1,W - 1):
            p = src[i-1:i + 2, j-1:j + 2]
            gx = np.sum(Gx_kernel * p)
            gy = np.sum(Gy_kernel * p)
            grad = np.sqrt(gx ** 2 + gy ** 2)
            result[i, j] = grad
    return result
#sobel算子
def sobel(src):
    H, W = src.shape
    Gx_kernel = np.array([[-1, 0,1], [-2, 0,2],[-1, 0,1]])
    Gy_kernel = np.array([[1,2, 1], [0, 0,0],[-1,-2,-1]])
    result = np.zeros((H, W), dtype=np.float32)
    for i in range(1,H - 1):
        for j in range(1,W - 1):
            p = src[i-1:i + 2, j-1:j + 2]
            gx = np.sum(Gx_kernel * p)
            gy = np.sum(Gy_kernel * p)
            grad = np.sqrt(gx ** 2 + gy ** 2)
            result[i, j] = grad
    return result

#laplacian
def laplacian(src):
    H, W = src.shape
    out = np.zeros((H, W), dtype=np.float32)


    src_blur = cv2.GaussianBlur(src, (3, 3), sigmaX=1)

    lap_kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            patch = src_blur[i-1:i+2, j-1:j+2]
            val = np.sum(patch * lap_kernel)
            out[i, j] = val

    out = np.abs(out)
    out = np.clip(out, 0, 60)

    out = (out / 60) * 255
    out = out.astype(np.uint8)

    return out

img_roberts = roberts(gray)
img_prewitt = prewitt(gray)
img_sobel = sobel(gray)
img_lap = laplacian(gray)


plt.figure(figsize=[10, 10])
plt.subplot(2, 3, 1)
plt.title("原图")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("roberts")
plt.imshow(img_roberts, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("prewitt")
plt.imshow(img_prewitt, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("sobel")
plt.imshow(img_sobel, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("laplacian")
plt.imshow(img_lap, cmap="gray")
plt.axis("off")
plt.show()


