import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

#读取图片
img = cv2.imread("data/test.jpg")
if img is None:
    raise FileNotFoundError("图片读取失败，请检查图片路径！")
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#灰度变换
#线性变换
alpha = 1.3
beta = 10
gray_linear = cv2.convertScaleAbs(gray,alpha=alpha,beta=beta)
#对数变换
c = 40
gray_log = c*np.log1p(gray.astype(np.float32))
gray_log = cv2.normalize(gray_log,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
#伽马变换
gamma = 0.6
inv_gamma = 1.0/gamma
table = np.array([((i/255.0)** inv_gamma)*255 for i in np.arange(0,256)]).astype("uint8")
gray_gamma = cv2.LUT(gray,table)
#二值化处理
ret1,binary_fixed = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
ret2,binary_otsu=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
binary_adaptive=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize=11,C=2)
#局部像素
x1,y1=60,60
x2,y2=180,180
cv2.rectangle(img_rgb,(x1,y1),(x2,y2),color=(255,0,0),thickness=2)
roi_color=img[y1:y2,x1:x2]
print(f"彩色ROI shape:{roi_color.shape}")
print(roi_color)

#图像可视化
plt.figure(figsize=(14,10))

plt.subplot(2,4,1)
plt.title("原图RGB")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(2,4,2)
plt.title("灰度图")
plt.imshow(gray,cmap="gray")
plt.axis("off")

plt.subplot(2,4,3)
plt.title("线性变换 a=1.3")
plt.imshow(gray_linear,cmap="gray")
plt.axis("off")

plt.subplot(2,4,4)
plt.title("对数变换")
plt.imshow(gray_log,cmap="gray")
plt.axis("off")

plt.subplot(2,4,5)
plt.title("伽马变换 γ=0.6")
plt.imshow(gray_gamma,cmap="gray")
plt.axis("off")

plt.subplot(2,4,6)
plt.title("固定阈值二值化 T=127")
plt.imshow(binary_fixed,cmap="gray")
plt.axis("off")

plt.subplot(2,4,7)
plt.title("OTSU大津二值化")
plt.imshow(binary_otsu,cmap="gray")
plt.axis("off")

plt.subplot(2,4,8)
plt.title("自适应二值化")
plt.imshow(binary_adaptive,cmap="gray")
plt.axis("off")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("原图（红色框选ROI）")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("裁剪出来彩色ROI")
plt.imshow(cv2.cvtColor(roi_color,cv2.COLOR_BGR2RGB))
plt.axis("off")


plt.tight_layout()
plt.show()
