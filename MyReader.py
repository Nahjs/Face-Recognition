# coding:utf-8
import os
import cv2
import random
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
import paddle.vision.transforms as T
import json
from config_utils import ConfigObject

class ImageAugmentation:
    """图像数据增强和预处理类。

    该类封装了多种常见的图像增强技术，如随机翻转、颜色抖动、亮度调整等，
    以及图像的缩放和归一化操作。
    它区分训练模式和评估/测试模式，在训练时应用数据增强以提高模型泛化能力，
    在评估/测试时仅进行必要的尺寸调整和归一化。
    """

    def __init__(self, config: ConfigObject, is_train: bool = True):
        """
        初始化图像增强器。

        Args:
            config (ConfigObject): 包含数据增强配置的ConfigObject对象。
            is_train (bool): 是否处于训练模式，决定是否应用随机增强。
        """
        self.is_train = is_train
        self.image_size = config.image_size # 训练和评估/验收时图像尺寸应一致
        self.mean_value = config.dataset_params.mean
        self.std_value = config.dataset_params.std
        self.data_augmentation_config = config.dataset_params.data_augmentation # 获取增强配置块

        transform_list = []

        # 仅在训练模式且配置文件启用时应用随机增强
        if self.is_train and self.data_augmentation_config.get('use_train_augmentation', False):
            if self.data_augmentation_config.get('transforms'):
                for transform_name, params in self.data_augmentation_config.transforms.items():
                    transform_class = getattr(T, transform_name, None)
                    if transform_class:
                        # Paddle transforms 可以作用于 PIL Image 或 numpy.ndarray (HWC uint8)
                        current_params = params if params is not None else {}
                        # For RandomResizedCrop, explicitly add the required 'size' parameter from config.image_size
                        if transform_name == 'RandomResizedCrop':
                            current_params['size'] = (self.image_size, self.image_size) # size is a tuple (h, w)
                            print(f"Info: Added size {self.image_size} to RandomResizedCrop params.") # Add a print for debugging
                        transform_list.append(transform_class(**current_params))
                    else:
                        print(f"Warning: Unknown transform {transform_name} specified in config.")

        # 所有模式都需要进行的后处理：Resize, ToTensor, Normalize
        # 注意：这里的 Resize 应该是在应用随机crop等之后，如果配置中已经有 Resize/RandomResizedCrop，这里的可能需要调整逻辑
        # 为了简单，我们假设配置中的transforms是先应用的随机增强，最后再进行统一的Resize和标准化
        # 如果配置中已经包含了最终Resize，这里可以跳过
        if not any(isinstance(t, T.Resize) for t in transform_list):
             # 注意：这里添加的 Resize 是一个固定的 Resize，不是随机增强
             transform_list.append(T.Resize((self.image_size, self.image_size)))

        # ToTensor 会将图片转为 Tensor 并除以255归一化到[0, 1]
        transform_list.append(T.ToTensor())
        # Normalize 将数据标准化到指定均值和标准差
        transform_list.append(T.Normalize(mean=self.mean_value, std=self.std_value))

        # 将所有 transforms 组合成一个 Compose 对象
        self.transforms = T.Compose(transform_list)

    def process_image(self, image_path):
        """
        读取、处理（增强和标准化）单张图像。

        Args:
            image_path (str): 图像文件的完整路径。

        Returns:
            paddle.Tensor: 处理后的图像 Tensor，shape为 [C, H, W]。
        """
        # 使用 OpenCV 读取图像，确保通道顺序为 RGB
        # OpenCV 读取的是 BGR 格式，需要转换为 RGB
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 将 NumPy 图像转换为 PIL Image，因为 PaddlePaddle transforms 通常作用于 PIL Image
        # 如果 transforms 设计为作用于 uint8 HWC numpy 数组，则不需要转换为 PIL
        # 检查 Paddle transforms 文档，大多数支持 numpy HWC (uint8) 或 PIL Image
        # 为了更好的兼容性，转换为 PIL Image 是一个稳妥的选择
        from PIL import Image
        img = Image.fromarray(img)

        # 应用构建好的 transforms 序列
        img = self.transforms(img)

        return img

# Assuming MyDataset class is defined below this or in the same file
class MyDataset(Dataset):
    """
    自定义数据集类，用于加载图像列表和对应的标签。
    """
    def __init__(self, data_list_file: str, image_augmentor: ImageAugmentation, data_root_path: str = ''):
        """
        初始化数据集。

        Args:
            data_list_file (str): 包含图像路径和标签的列表文件路径。
            image_augmentor (ImageAugmentation): 图像增强器实例。
            data_root_path (str): 数据集根目录路径，列表文件中的路径是相对于此目录的。
        """
        super(MyDataset, self).__init__()
        self.data_list_file = data_list_file
        self.data_root_path = data_root_path
        self.image_augmentor = image_augmentor
        self.data_list = self._load_data_list()
        # total_classes 在 create_data_loader 中从 readme.json 读取并返回，Dataset 内部不需要存储

    def _load_data_list(self):
        """
        从列表文件加载图像路径和标签。
        列表文件格式: image_path\tlabel
        """
        data_list = []
        try:
            with open(self.data_list_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        # 假设是 tab 分隔，且第一列是相对路径，第二列是标签
                        image_relative_path, label = line.split('\t')
                        # 构建完整的图像路径
                        full_image_path = os.path.join(self.data_root_path, image_relative_path)
                        # 检查文件是否存在，避免加载不存在的文件导致后续错误
                        if not os.path.exists(full_image_path):
                             print(f"Warning: Image file not found, skipping: {full_image_path}")
                             continue
                        data_list.append((full_image_path, int(label)))
                    except ValueError:
                        print(f"Warning: Skipping malformed line in {self.data_list_file}: {line}")
                        continue
        except FileNotFoundError:
            print(f"Error: Data list file not found at {self.data_list_file}")
            # 在 DataLoader 创建时检查文件存在性可能更好，这里作为二次检查
            return []
        except Exception as e:
            print(f"Error reading data list file {self.data_list_file}: {e}")
            return []

        if not data_list:
             print(f"Warning: No valid data loaded from {self.data_list_file}.")

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label = self.data_list[index]
        # image_augmentor.process_image 返回的是 Paddle Tensor
        img_tensor = self.image_augmentor.process_image(image_path)

        # image_augmentor.process_image 已经处理了加载失败的情况并返回 None
        # DataLoader 会自动处理 None，可能会跳过该样本或抛出错误，取决于 DataLoader 的实现
        # 更安全的做法是在 _load_data_list 中过滤掉加载失败的图片路径
        if img_tensor is None:
             # 可以返回一个 dummy tensor 和 label，但可能会影响训练稳定性
             # 或者依赖 DataLoader 的默认行为
             # 暂时依赖 _load_data_list 过滤和 DataLoader 的默认行为
             pass

        label_tensor = paddle.to_tensor(label, dtype='int64')
        return img_tensor, label_tensor

# Assuming create_data_loader function is defined below this or in the same file
# This function is likely called from train.py

# Example of how create_data_loader might be implemented (based on train.py's call):
def create_data_loader(config: ConfigObject, mode: str = 'train'):
    """
    根据配置和模式创建数据加载器。

    Args:
        config (ConfigObject): 配置对象。
        mode (str): 数据加载模式 ('train', 'eval', 'acceptance')。

    Returns:
        paddle.io.DataLoader: 数据加载器实例。
        int: 数据集的总类别数。
    """
    dataset_params = config.dataset_params
    data_root_from_config = config.data_dir # 数据集根目录，例如 'data'

    if mode == 'train':
        list_file_name = dataset_params.train_list # 例如 'face/train.list'
        is_train = True
        batch_size = config.batch_size
        shuffle = True
    elif mode == 'validation': # 验证模式 (取代 eval/acceptance)
        list_file_name = dataset_params.validation_list # 例如 'face/validation.list'
        is_train = False # 验证时关闭随机增强
        batch_size = config.batch_size # 验证时batch_size也可以独立设置
        shuffle = False # 验证时通常不打乱
    elif mode == 'test': # 测试模式
        list_file_name = dataset_params.test_list # 例如 'face/test.list'
        is_train = False # 测试时关闭随机增强
        batch_size = config.batch_size # 测试时batch_size也可以独立设置
        shuffle = False # 测试时通常不打乱
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 构建完整的列表文件路径
    list_file_path = os.path.join(data_root_from_config, config.class_name, list_file_name)

    # 检查列表文件是否存在
    if not os.path.exists(list_file_path):
        print(f"Error: Data list file for mode '{mode}' not found at {list_file_path}.")
        # Consider raising an error or returning None/empty DataLoader
        raise FileNotFoundError(f"Data list file for mode '{mode}' not found at {list_file_path}.")

    # 获取数据列表文件所在的目录，这将作为 MyDataset 的 data_root_path
    # 因为列表文件中的路径是相对于此目录的
    # 例如，如果 list_file_path 是 'data/face/train.list'，那么列表文件内容可能是 'person1/abc.jpg\t0'
    # Data root for MyDataset 应该是 'data/face'
    data_root_for_dataset = os.path.dirname(list_file_path)

    # 创建图像增强器实例
    image_augmentor_instance = ImageAugmentation(config=config, is_train=is_train)

    # 创建数据集实例
    dataset = MyDataset(
        data_list_file=list_file_path,
        image_augmentor=image_augmentor_instance,
        data_root_path=os.path.join(data_root_from_config, config.class_name) # 将 data_dir/class_name 作为数据根目录传递
    )

    # 获取总类别数
    # 从与列表文件同目录的 readme.json 中获取总类别数
    readme_path = os.path.join(data_root_for_dataset, "readme.json")
    total_classes = 0
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_data = json.load(f)
            # 修改这里，正确访问嵌套的 total_classes
            data_statistics = readme_data.get("data_statistics", {})
            total_classes = data_statistics.get("total_classes", 0)
            if total_classes == 0:
                 print(f"Warning: 'total_classes' not found or is 0 in {readme_path} (under data_statistics).")
    except FileNotFoundError:
        print(f"Warning: readme.json not found at {readme_path}. Cannot determine total classes.")
    except json.JSONDecodeError:
        print(f"Warning: Error decoding readme.json at {readme_path}.")
    except Exception as e:
        print(f"Error reading readme.json at {readme_path}: {e}")

    if total_classes == 0 and mode == 'train':
         print(f"Error: Total classes could not be determined for training mode. This is required. Check readme.json or config.num_classes.")
         # Fallback: try to get from config.num_classes if readme.json fails, but readme.json is preferred
         total_classes = config.get('num_classes', 0)
         if total_classes == 0:
              raise ValueError("Total classes could not be determined from readme.json or config.num_classes.")
         else:
              print(f"Using total_classes from config.num_classes: {total_classes} as fallback.")


    # --- Add a custom collate_fn for batching Tensors ---
    def custom_collate_fn(batch):
        """Manually collate data into batches."""
        # Filter out None values if any samples failed to load
        batch = [data for data in batch if data is not None]
        if not batch:
            return None, None # Return None or handle empty batch appropriately
        images, labels = zip(*batch) # Unzip the batch of (img, label) tuples
        # Stack individual tensors into batch tensors
        images = paddle.stack(images, axis=0)
        labels = paddle.stack(labels, axis=0)
        return images, labels
    # --- End custom_collate_fn ---

    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=dataset_params.num_workers,
        drop_last=mode == 'train',
        collate_fn=custom_collate_fn # Always use custom collate for explicit batching
    )

    print(f"Created DataLoader for mode '{mode}' with {len(dataset)} samples and {total_classes} classes.")
    return data_loader, total_classes


# Add test functions here if needed, like the ones in the original snippet
# if __name__ == '__main__':
#     # Add test logic here
#     pass