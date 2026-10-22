import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

plt.rcParams["font.sans-serif"] = ["SimHei"]

img_origin = cv2.imread("data/test.jpg")
img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
img2=img_origin.copy()
h, w = img_origin.shape[:2]
H,W,C = img_origin.shape

# 下采样、上采样
img_down = cv2.resize(img_origin, (w//2, h//2), interpolation=cv2.INTER_AREA)
img_up = cv2.resize(img_down, (w, h), interpolation=cv2.INTER_CUBIC)
#加减乘除
img_add = cv2.add(img_origin,40)
img_sub = cv2.subtract(img_origin,40)
img_mul = cv2.multiply(img_origin,1.3)
img_div = cv2.divide(img_origin,1.3)
#平移
def my_translate(src, tx, ty):
    H, W, C = src.shape
    dst = np.zeros_like(src)   # 创建全黑输出图
    # 遍历输出图每个像素 (y,x)
    for y_dst in range(H):
        for x_dst in range(W):
            # 反向映射：输出坐标 → 原图坐标
            x_src = x_dst - tx
            y_src = y_dst - ty
            # 判断是否在原图范围内
            if 0 <= x_src < W and 0 <= y_src < H:
                dst[y_dst, x_dst] = src[y_src, x_src]
            # 超出范围保持黑色0
    return dst

img_my_trans = my_translate(img_origin, tx=100, ty=50)
#旋转
def my_rotate(src, angle_deg):
    H, W, C = src.shape
    dst = np.zeros_like(src)
    cx = W / 2
    cy = H / 2
    theta = np.radians(angle_deg)
    cos_ = np.cos(theta)
    sin_ = np.sin(theta)

    # 遍历输出图像每一个像素
    for y_dst in range(H):
        for x_dst in range(W):
            # 逆变换，输出坐标映射回原图坐标
            xd = x_dst - cx
            yd = y_dst - cy
            x_src = xd * cos_ + yd * sin_ + cx
            y_src = -xd * sin_ + yd * cos_ + cy

            # 最近邻插值：取四舍五入整数像素
            x_nn = round(x_src)
            y_nn = round(y_src)

            # 判断坐标是否合法
            if 0 <= x_nn < W and 0 <= y_nn < H:
                dst[y_dst, x_dst] = src[y_nn, x_nn]
    return dst


img_my_rot = my_rotate(img_origin, 45)  #逆时针45度

fig = plt.figure(figsize=(7, 10))
# 3行1列，高度比例4 : 2 :1，实现图片一大一中一小
gs = GridSpec(3, 1, height_ratios=[4, 2, 1])

ax1 = fig.add_subplot(gs[0])
ax1.imshow(img_up)
ax1.set_title("up")
ax1.axis("off")

ax2 = fig.add_subplot(gs[1])
ax2.imshow(img_origin)
ax2.set_title("original")
ax2.axis("off")

ax3 = fig.add_subplot(gs[2])
ax3.imshow(img_down)
ax3.set_title("down")
ax3.axis("off")

plt.figure(figsize=(7, 10))

plt.subplot(2,2,1)
plt.title("加")
plt.imshow(img_add)
plt.axis("off")

plt.subplot(2,2,2)
plt.title("减")
plt.imshow(img_sub)
plt.axis("off")

plt.subplot(2,2,3)
plt.title("乘")
plt.imshow(img_mul)
plt.axis("off")

plt.subplot(2,2,4)
plt.title("除")
plt.imshow(img_div)
plt.axis("off")

plt.figure(figsize=(7, 10))

plt.subplot(1,2,1)
plt.title("平移")
plt.imshow(img_my_trans)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("旋转")
plt.imshow(img_my_rot)
plt.axis("off")
plt.tight_layout()
plt.show()