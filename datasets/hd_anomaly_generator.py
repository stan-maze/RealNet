import os
import random
from typing import Tuple

import cv2
import numpy as np
from scipy.ndimage import median_filter
from vcam import meshGen, vcam


def line_mask_of_(
    img: np.ndarray,
    thin_line_threshold: int = 50,
    thick_line_threshold: int = 130,
    denoising_rounds: int = 5,
    denoising_kernel_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成图像对应粗细线条区域的掩码。

    Args:
        img (np.ndarray): 要根据亮度识别粗细线条区域的图片（灰度图）
        thin_line_threshold (int, optional): （谨慎调整）细线条识别阈值（亮度值）
        thick_line_threshold (int, optional): （谨慎调整）粗线条识别阈值（亮度值）
        denoising_rounds (int, optional): （谨慎调整）线条识别时的中值滤波去噪轮数
        denoising_kernel_size (int, optional): （谨慎调整）线条识别去噪时的中值滤波核大小

    Returns:
        Tuple:
            - np.ndarray: 指示细线条区域的掩码
            - np.ndarray: 指示粗线条区域的掩码
    """
    # 获取图片的宽高
    img_h, img_w = img.shape
    # 创建对应大小的双层布尔类型全零矩阵作为掩码，分别记录两种线条区域
    mask = np.zeros((img_h, img_w, 2), dtype=bool)
    # 如果图片像素亮度大于粗线条阈值，则将对应层掩码的对应位置设为真值
    mask[:, :, 0] = (img > thin_line_threshold) & (img <= thick_line_threshold)
    # 如果图片像素亮度大于细线条阈值且小于粗线条阈值，则将对应层掩码的对应位置设为真值
    mask[:, :, 1] = img > thick_line_threshold
    # 对粗线条掩码进行中值滤波去噪
    for _ in range(denoising_rounds):
        mask[:, :, 1] = median_filter(mask[:, :, 1], size=denoising_kernel_size)

    return mask[:, :, 0], mask[:, :, 1]


def brightness_anomaly_of_(
    img: np.ndarray,
    brightness_anomaly_num: int,
    brightness_anomaly_type: int = -1,
    anomaly_area_direction: int = -1,
    short_side_length: int = -1,
    short_side_min: float = 0.01,
    short_side_max: float = 0.05,
    long_side_length: int = -1,
    long_side_min: float = 0.2,
    long_side_max: float = 0.8,
    thin_line_threshold: int = 50,
    thick_line_threshold: int = 130,
    denoising_rounds: int = 5,
    denoising_kernel_size: int = 5,
    brightness_anomaly_level: float = 1.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成含有指定数量的亮度异常区域的图片和对应掩码。

    当存在多个异常区域时，异常区域可能重叠，但异常效果不叠加。

    Args:
        img (np.ndarray): 要添加异常区域的图片（灰度图）
        brightness_anomaly_num (int): 亮度异常区域数量（`>0`：指定此种异常的数量）
        brightness_anomaly_type (int, optional): 亮度异常类型（`-1`：随机；`1`：细线条异常变亮；`2`：粗线条异常变暗）
        anomaly_area_direction (int, optional): 亮度异常区域的方向（`-1`：随机；`1`：横向；`2`：纵向）
        short_side_length (int, optional): 亮度异常区域的短边长度（`-1`：从下限到上限的范围中随机；`otherwise`：异常区域直径占图片短边的比例，若超出上下限则取上下限）
        short_side_min (float, optional): （谨慎调整）短边边长的随机下限
        short_side_max (float, optional): （谨慎调整）短边边长的随机上限
        long_side_length (int, optional): 亮度异常区域的长边长度（`-1`：从下限到上限的范围中随机；`otherwise`：异常区域直径占图片短边的比例，若超出上下限则取上下限）
        long_side_min (float, optional): （谨慎调整）长边边长的随机下限
        long_side_max (float, optional): （谨慎调整）长边边长的随机上限
        thin_line_threshold (int, optional): （谨慎调整）细线条识别阈值（亮度值）
        thick_line_threshold (int, optional): （谨慎调整）粗线条识别阈值（亮度值）
        denoising_rounds (int, optional): （谨慎调整）线条识别时的中值滤波去噪轮数
        denoising_kernel_size (int, optional): （谨慎调整）线条识别去噪时的中值滤波核大小
        brightness_anomaly_level (float, optional): （谨慎调整）亮度异常程度（更亮的情况相比更暗的情况的亮度值倍率）

    Returns:
        Tuple:
            - np.ndarray: 包含指定数量的亮度异常区域的图片
            - np.ndarray: 指示对应异常区域的掩码
    """
    # 获取图片的宽高并确定短边长度
    img_h, img_w = img.shape
    img_short_side = min(img_h, img_w)
    # 创建对应大小的双层布尔类型全零矩阵作为掩码，分别记录两种异常情况
    mask = np.zeros((img_h, img_w, 2), dtype=bool)
    # 根据异常区域的数量进行循环
    for _ in range(brightness_anomaly_num):
        # 确定矩形的短边长度、长边长度和方向，以及异常类型
        if short_side_length == -1:
            s = random.uniform(short_side_min, short_side_max) * img_short_side
        elif short_side_length > short_side_max:
            s = short_side_max * img_short_side
        elif short_side_length < short_side_min:
            s = short_side_min * img_short_side
        else:
            s = short_side_length * img_short_side

        if long_side_length == -1:
            l = random.uniform(long_side_min, long_side_max) * img_short_side
        elif long_side_length > long_side_max:
            l = long_side_max * img_short_side
        elif long_side_length < long_side_min:
            l = long_side_min * img_short_side
        else:
            l = long_side_length * img_short_side

        if anomaly_area_direction == -1:
            d = random.randint(1, 2)
        else:
            d = anomaly_area_direction

        if brightness_anomaly_type == -1:
            t = random.randint(1, 2)
        else:
            t = brightness_anomaly_type

        # 根据方向确定矩形的宽度和高度（向下取整）
        if d == 1:
            rect_w = int(l)
            rect_h = int(s)
        else:
            rect_w = int(s)
            rect_h = int(l)
        # 在可行的范围内随机选取矩形左上角的坐标
        rect_x = random.randint(0, img_w - rect_w)
        rect_y = random.randint(0, img_h - rect_h)
        # 将矩形对应区域的对应层掩码设置为真值
        if t == 1:
            mask[rect_y : rect_y + rect_h, rect_x : rect_x + rect_w, 0] = True
        else:
            mask[rect_y : rect_y + rect_h, rect_x : rect_x + rect_w, 1] = True
    # 提取图片的粗细线条区域
    thin_line_mask, thick_line_mask = line_mask_of_(
        img,
        thin_line_threshold=thin_line_threshold,
        thick_line_threshold=thick_line_threshold,
        denoising_rounds=denoising_rounds,
        denoising_kernel_size=denoising_kernel_size,
    )
    # 根据第一层掩码提取图片对应区域的细线条并使其异常变亮
    final_thin_line_mask = mask[:, :, 0] & thin_line_mask
    img[final_thin_line_mask] = (
        brightness_anomaly_level * img[final_thin_line_mask]
    ).astype(np.uint8)
    img[img > 255] = 255
    # 根据第二层掩码提取图片对应区域的粗线条并使其异常变暗
    final_thick_line_mask = mask[:, :, 1] & thick_line_mask
    img[final_thick_line_mask] = (
        img[final_thick_line_mask] / brightness_anomaly_level
    ).astype(np.uint8)
    # 合并两层掩码
    final_mask = final_thin_line_mask | final_thick_line_mask

    return img, final_mask


# 根据距离计算高斯权重
def gaussian(distance, sigma):
    return np.exp(-(distance**2) / (2 * sigma**2)) / sigma


def distortion_anomaly_of_(
    img: np.ndarray,
    distortion_anomaly_num: int,
    diameter: int = -1,
    diameter_min: float = 0.1,
    diameter_max: float = 0.16,
    padding: int = 0.05,
    distortion_level: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成含有指定数量的形状异常区域的图片和对应掩码。

    当存在多个异常区域时，异常区域可能重叠，但异常效果不叠加。

    Args:
        img (np.ndarray): 要添加异常区域的图片（灰度图）
        distortion_anomaly_num (int): 形状异常区域数量（`>0`：指定此种异常的数量）
        diameter (int, optional): 形状异常区域的直径（`-1`：从下限到上限的范围中随机；`otherwise`：异常区域直径占图片短边的比例，若超出上下限则取上下限）
        diameter_min (float, optional): （谨慎调整）直径的随机下限
        diameter_max (float, optional): （谨慎调整）直径的随机上限
        padding (int, optional): （谨慎调整）形状异常区域的过渡距离占图片短边的比例，若大于异常区域的半径则直接取半径值
        distortion_level (int, optional): （谨慎调整）形状异常区域的扭曲程度（越大越膨胀，越小越收缩，取`0`时无扭曲）

    Returns:
        Tuple:
            - np.ndarray: 包含指定数量的形状异常区域的图片
            - np.ndarray: 指示对应异常区域的掩码
    """
    # 获取图片的宽高并确定短边长度
    img_h, img_w = img.shape
    img_short_side = min(img_h, img_w)
    # 创建对应大小的全零矩阵记录扭曲程度
    distortion = np.zeros((img_h, img_w))
    # 根据异常区域的数量进行循环
    padding = padding * img_short_side
    sigma = padding / 2
    for _ in range(distortion_anomaly_num):
        # 确定异常区域的直径大小
        if diameter == -1:
            d = random.uniform(diameter_min, diameter_max) * img_short_side
        elif diameter > diameter_max:
            d = diameter_max * img_short_side
        elif diameter < diameter_min:
            d = diameter_min * img_short_side
        else:
            d = diameter * img_short_side
        # 计算异常外半径（2_sigma处）内半径（0_sigma处）和图像处理外半径（3_sigma处）
        r_2_sigma = d / 2
        if padding > r_2_sigma:
            padding = r_2_sigma
        r_0_sigma = r_2_sigma - padding
        r_3_sigma = r_2_sigma + sigma
        # 在可行的范围内随机选取圆心的坐标
        mu_x = random.randint(int(r_2_sigma), img_w - int(r_2_sigma))
        mu_y = random.randint(int(r_2_sigma), img_h - int(r_2_sigma))
        # 计算矩阵元素到圆心距离
        x_coords, y_coords = np.meshgrid(np.arange(img_w), np.arange(img_h))
        distances = np.sqrt((x_coords - mu_x) ** 2 + (y_coords - mu_y) ** 2)
        # 如果距离小于内半径，将距离直接设为内半径
        distances[distances < r_0_sigma] = r_0_sigma
        # 对于距离小于图像处理外半径的部分，根据距离减内半径的值计算高斯权重作为扭曲程度
        distortion[distances < r_3_sigma] += gaussian(
            distances[distances < r_3_sigma] - r_0_sigma, sigma
        )
    # 对扭曲程度进行限制和归一化
    allowed_maximum = gaussian(0, sigma)
    distortion[distortion > allowed_maximum] = allowed_maximum
    distortion[distortion != 0] -= np.min(distortion[distortion != 0])
    distortion /= np.max(distortion)
    # 对于扭曲程度大于2_sigma处的，将掩码对应位置设为True
    mask = np.zeros((img_h, img_w), dtype=bool)
    mask[distortion > gaussian(2 * sigma, sigma) / allowed_maximum] = True
    # 使用扭曲程度参数控制图片的扭曲
    distortion *= distortion_level
    # 对图片进行扭曲
    virtual_camera = vcam(H=img_h, W=img_w)
    plane = meshGen(H=img_h, W=img_w)
    plane.Z += distortion.reshape(-1, 1)
    pts3d = plane.getPlane()
    pts2d = virtual_camera.project(pts3d)
    map_x, map_y = virtual_camera.getMaps(pts2d)
    distortion_img = cv2.remap(img, map_x, map_y, cv2.INTER_AREA)

    return np.fliplr(distortion_img), np.fliplr(mask)


# 生成异常样本
def hd_anomaly_of_(
    img: np.ndarray,
    anomaly_area_num: int = 10,
    num_min: int = 1,
    num_max: int = 5,
    distortion_anomaly_num: int = -1,
    brightness_anomaly_num: int = -1,
    diameter: int = -1,
    diameter_min: float = 0.1,
    diameter_max: float = 0.16,
    padding: int = 0.05,
    distortion_level: int = 20,
    brightness_anomaly_type: int = -1,
    anomaly_area_direction: int = -1,
    short_side_length: int = -1,
    short_side_min: float = 0.01,
    short_side_max: float = 0.05,
    long_side_length: int = -1,
    long_side_min: float = 0.2,
    long_side_max: float = 0.8,
    thin_line_threshold: int = 50,
    thick_line_threshold: int = 130,
    denoising_rounds: int = 5,
    denoising_kernel_size: int = 5,
    brightness_anomaly_level: float = 1.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """使用正常图片生成人工仿制的高难度异常图片和对应掩码。

    基本用法参考：
        - 生成含有一个随机类型的异常区域的图片：`hd_anomaly_of_(img)`
        - 生成含有两个随机类型的异常区域的图片：`hd_anomaly_of_(img,anomaly_area_num=2)`
        - 生成含有两个形状异常区域的图片：`hd_anomaly_of_(img,distortion_anomaly_num=2)`
        - 生成含有三个亮度异常区域的图片：`hd_anomaly_of_(img,brightness_anomaly_num=3)`
        - 生成含有三个异常区域，且至少一处亮度异常的图片：`hd_anomaly_of_(img,anomaly_area_num=3,brightness_anomaly_num=1)`

    当存在多个异常区域时，优先仿制亮度异常，然后仿制形状异常。异常区域可能重叠，但同类型的异常效果不叠加。

    更细致的异常仿制调整请阅读参数说明。

    Args:
        img (np.ndarray): 要添加异常区域的图片（灰度图）
        anomaly_area_num (int, optional): 异常区域数量（`-1`：随机；`>0`：指定总数量，如果指定的特定异常数量超出此限制则特定异常的要求优先）
        num_min (int, optional): （谨慎调整）异常区域数量的随机下限
        num_max (int, optional): （谨慎调整）异常区域数量的随机上限
        distortion_anomaly_num (int, optional): 形状异常区域数量（`-1`：随机；`>0`：指定此种异常的数量下限）
        brightness_anomaly_num (int, optional): 亮度异常区域数量（`-1`：随机；`>0`：指定此种异常的数量下限）
        diameter (int, optional): 形状异常区域的直径（`-1`：从下限到上限的范围中随机；`otherwise`：异常区域直径占图片短边的比例，若超出上下限则取上下限）
        diameter_min (float, optional): （谨慎调整）直径的随机下限
        diameter_max (float, optional): （谨慎调整）直径的随机上限
        padding (int, optional): （谨慎调整）形状异常区域的过渡距离占图片短边的比例，若大于异常区域的半径则直接取半径值
        distortion_level (int, optional): （谨慎调整）形状异常区域的扭曲程度（越大越膨胀，越小越收缩，取`0`时无扭曲）
        brightness_anomaly_type (int, optional): 亮度异常类型（`-1`：随机；`1`：细线条异常变亮；`2`：粗线条异常变暗）
        anomaly_area_direction (int, optional): 亮度异常区域的方向（`-1`：随机；`1`：横向；`2`：纵向）
        short_side_length (int, optional): 亮度异常区域的短边长度（`-1`：从下限到上限的范围中随机；`otherwise`：异常区域直径占图片短边的比例，若超出上下限则取上下限）
        short_side_min (float, optional): （谨慎调整）短边边长的随机下限
        short_side_max (float, optional): （谨慎调整）短边边长的随机上限
        long_side_length (int, optional): 亮度异常区域的长边长度（`-1`：从下限到上限的范围中随机；`otherwise`：异常区域直径占图片短边的比例，若超出上下限则取上下限）
        long_side_min (float, optional): （谨慎调整）长边边长的随机下限
        long_side_max (float, optional): （谨慎调整）长边边长的随机上限
        thin_line_threshold (int, optional): （谨慎调整）细线条识别阈值（亮度值）
        thick_line_threshold (int, optional): （谨慎调整）粗线条识别阈值（亮度值）
        denoising_rounds (int, optional): （谨慎调整）线条识别时的中值滤波去噪轮数
        denoising_kernel_size (int, optional): （谨慎调整）线条识别去噪时的中值滤波核大小
        brightness_anomaly_level (float, optional): （谨慎调整）亮度异常程度（更亮的情况相比更暗的情况的亮度值倍率，参考取值大约在 1.15~1.25 之间）

    Returns:
        Tuple:
            - np.ndarray: 包含异常区域的图片
            - np.ndarray: 指示异常区域的掩码
    """
    # 数量预处理
    if anomaly_area_num == -1:
        anomaly_area_num = random.randint(num_min, num_max)
    if brightness_anomaly_num == -1:
        brightness_anomaly_num = 0
    if distortion_anomaly_num == -1:
        distortion_anomaly_num = 0
    # 确定不同类型异常区域的数量
    rest_anomaly_num = (
        anomaly_area_num - brightness_anomaly_num - distortion_anomaly_num
    )
    if rest_anomaly_num > 0:
        for i in range(rest_anomaly_num):
            if random.random() < 0.5:
                brightness_anomaly_num += 1
            else:
                distortion_anomaly_num += 1
    # 仿制异常区域
    brightness_anomaly_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    distortion_anomaly_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    if brightness_anomaly_num > 0:
        img, brightness_anomaly_mask = brightness_anomaly_of_(
            img,
            brightness_anomaly_num=brightness_anomaly_num,
            brightness_anomaly_type=brightness_anomaly_type,
            anomaly_area_direction=anomaly_area_direction,
            short_side_length=short_side_length,
            short_side_min=short_side_min,
            short_side_max=short_side_max,
            long_side_length=long_side_length,
            long_side_min=long_side_min,
            long_side_max=long_side_max,
            thin_line_threshold=thin_line_threshold,
            thick_line_threshold=thick_line_threshold,
            denoising_rounds=denoising_rounds,
            denoising_kernel_size=denoising_kernel_size,
            brightness_anomaly_level=brightness_anomaly_level,
        )
    if distortion_anomaly_num > 0:
        img, distortion_anomaly_mask = distortion_anomaly_of_(
            img,
            distortion_anomaly_num=distortion_anomaly_num,
            diameter=diameter,
            diameter_min=diameter_min,
            diameter_max=diameter_max,
            padding=padding,
            distortion_level=distortion_level,
        )
    # 使用逻辑或运算合并异常区域
    anomaly_mask = brightness_anomaly_mask | distortion_anomaly_mask
    return img, anomaly_mask


# 当直接运行该脚本时，遍历指定目录下的所有图片，并生成异常图片存放在另一个指定的目录下
source_folder = "./train/good/"
target_folder = "./train/defined_anomaly/"

if __name__ == "__main__":
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    counter = 0  # 限制生成数量（测试用）
    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg"):
            file_path = os.path.join(source_folder, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # 设置异常构造方法
            anomaly_img, mask = hd_anomaly_of_(img)  # 默认尝试生成10个左右的异常区域
            # 其他例子：
            # anomaly_img, mask = hd_anomaly_of_(img, anomaly_area_num=10)  # 尝试生成8个左右的异常区域
            # anomaly_img, mask = hd_anomaly_of_(img,distortion_anomaly_num=6) # 尝试生成6个左右的异常区域，仅包含扭曲异常
            # anomaly_img, mask = hd_anomaly_of_(img,brightness_anomaly_num=5) # 尝试生成5个左右的异常区域，仅包含亮度异常
            # anomaly_img, mask = hd_anomaly_of_(img,anomaly_area_num=9,distortion_anomaly_num=2) # 尝试生成9个左右的异常区域，且至少2个为扭曲异常

            # 设置文件命名规则
            anomaly_target_path = os.path.join(
                target_folder, filename.replace(".jpg", "-anomaly.png")
            )
            mask_target_path = os.path.join(
                target_folder, filename.replace(".jpg", "-label.png")
            )

            # 调整文件类型和压缩程度等
            mask_img = np.uint8(mask * 255)
            cv2.imwrite(
                anomaly_target_path, anomaly_img, [cv2.IMWRITE_PNG_COMPRESSION, 0]
            )
            cv2.imwrite(mask_target_path, mask_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"Processed and saved: {anomaly_target_path} and {mask_target_path}")

            # 限制生成数量（测试用）
            counter += 1
            if counter >= 5:
                print(
                    "当前为测试模式，仅生成少数异常图片。如要生成更多，请在文件中删除或注释这部分代码。"
                )
                break
