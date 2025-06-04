# utils/image_processing.py
import cv2
import numpy as np
import os # Though not directly used in process_image_local, good to have for utils related to paths if added later

def _apply_preprocessing(img_data: np.ndarray, target_size: int,
                        mean_rgb: list[float], std_rgb: list[float],
                        is_bgr: bool = True) -> np.ndarray:
    """
    对图像数据应用预处理步骤 (缩放、归一化、标准化、HWC转CHW)。
    """
    if is_bgr:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    img_data = cv2.resize(img_data, (target_size, target_size))
    img_data = img_data.astype('float32') / 255.0
    mean = np.array(mean_rgb, dtype='float32').reshape((1, 1, 3))
    std = np.array(std_rgb, dtype='float32').reshape((1, 1, 3))
    img_normalized = (img_data - mean) / std
    img_chw = img_normalized.transpose((2, 0, 1))
    img_expanded = np.expand_dims(img_chw, axis=0)
    return img_expanded.astype('float32')

def process_image(img_path: str | None = None, img_data: np.ndarray | None = None,
                  target_size: int = 64, mean_rgb: list[float] = [0.485, 0.456, 0.406],
                  std_rgb: list[float] = [0.229, 0.224, 0.225],
                  is_bgr_input: bool = True) -> np.ndarray:
    """
    通用的图像预处理函数，可从文件路径或numpy数组加载图像。

    Args:
        img_path (str, optional): 输入图像的文件路径。与 img_data 互斥。
        img_data (np.ndarray, optional): 输入图像的numpy数组数据。与 img_path 互斥。
        target_size (int, optional): 图像将被缩放到的目标正方形尺寸。默认为 64。
        mean_rgb (list[float], optional): RGB三通道的均值。
        std_rgb (list[float], optional): RGB三通道的标准差。
        is_bgr_input (bool, optional): 如果 img_data 是 BGR 格式，设置为 True。默认为 True。

    Returns:
        np.ndarray: 预处理后的图像数据 (1, 3, target_size, target_size)，float32类型。

    Raises:
        ValueError: 如果 img_path 和 img_data 同时指定或同时未指定。
        FileNotFoundError: 如果 img_path 指定但文件无法读取。
    """
    if img_path is not None and img_data is not None:
        raise ValueError("只能同时指定 img_path 或 img_data 中的一个。")
    if img_path is None and img_data is None:
        raise ValueError("必须指定 img_path 或 img_data 中的一个。")

    if img_path:
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(f"错误: 无法读取图像文件 {img_path}")
        return _apply_preprocessing(img, target_size, mean_rgb, std_rgb, is_bgr=True) # 从文件读取默认是BGR
    elif img_data is not None:
        return _apply_preprocessing(img_data, target_size, mean_rgb, std_rgb, is_bgr=is_bgr_input)


# 以下是旧的 process_image_local 函数，已废弃或内部使用
# def process_image_local(img_path: str, target_size: int = 64,
#                         mean_rgb: list[float] = [0.485, 0.456, 0.406],
#                         std_rgb: list[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
#     # 此函数已被新的 process_image 函数替代，现在仅作为内部辅助函数
#     # 如果它仍然被外部调用，则需要兼容性处理或更新调用方
#     # 为了兼容性，将其内部逻辑修改为调用 _apply_preprocessing
#     img = cv2.imread(img_path)
#     if img is None: raise FileNotFoundError(f"错误: 无法读取图像文件 {img_path}")
#     return _apply_preprocessing(img, target_size, mean_rgb, std_rgb) 