# coding=utf-8
"""
CreateDataList.py

本模块包含 CreateDataList 类，专门用于扫描指定的数据集根目录，
并根据其中的图像文件自动生成用于模型训练和测试的数据列表文件（通常是 .list 格式）
以及一个元数据文件（通常是 readme.json）。

数据列表文件的格式通常是：`图像文件相对路径\\t类别标签ID`。
元数据文件 (readme.json) 则包含了数据集的整体信息，如总类别数、总图像数、
以及每个类别的详细统计（类别名、标签ID、图像数量等）。

该脚本对于准备和管理机器学习（尤其是图像分类、人脸识别等任务）所需的数据集非常有用。
它规范了数据列表的生成过程，确保了标签分配的一致性，并提供了数据集的概览信息。
"""
import os
import json
import random
import argparse # 导入 argparse
from tqdm import tqdm # For progress bar, if not used, can be removed

class CreateDataList:
    """
    遍历指定数据集的根目录，为每个类别创建图像列表，
    并将它们划分为训练集、测试集和新增的验收集。
    同时，它会生成一个元数据文件 (readme.json)，包含类别到ID的映射和划分统计。
    """

    def __init__(self):
        self.label_id_counter = 0 # 用于为每个类别分配唯一的整数ID
        self.class_to_id_map = {} # 存储类别名称到整数ID的映射
        self.data_statistics = { # 存储每个集合的样本数量
            "total_classes": 0,
            "total_images": 0,
            "train_set_count": 0,
            "validation_set_count": 0, # 新增
            "test_set_count": 0,
            "images_per_class": {}
        }

    def _reset_state(self):
        """重置内部状态以便处理新的数据集或多次调用。"""
        self.label_id_counter = 0
        self.class_to_id_map = {}
        self.data_statistics = {
            "total_classes": 0,
            "total_images": 0,
            "train_set_count": 0,
            "validation_set_count": 0, # 新增
            "test_set_count": 0,
            "images_per_class": {}
        }

    def create_data_list(self, data_root_path: str,
                         train_list_name: str = "trainer.list",
                         test_list_name: str = "test.list",
                         validation_list_name: str = "validation.list", # 新增参数
                         meta_file_name: str = "readme.json",
                         train_ratio: float = 0.7, # 调整默认训练集比例
                         validation_ratio: float = 0.2, # 新增参数
                         test_ratio: float = 0.1,  # 调整默认测试集比例
                         output_num_classes_file: str = None,
                         output_dir: str = None) -> None:
        """
        生成训练、测试数据列表文件，以及一个包含类别映射的元数据JSON文件。

        Args:
            data_root_path (str): 数据集根目录的路径。
                                  期望的结构是: data_root_path/class_name/image_files
            train_list_name (str, optional): 输出的训练列表文件名。
            test_list_name (str, optional): 输出的测试列表文件名。
            validation_list_name (str, optional): 输出的验证列表文件名。
            meta_file_name (str, optional): 输出的元数据JSON文件名。
            train_ratio (float, optional): 训练集所占的比例 (0.0 到 1.0)。
            validation_ratio (float, optional): 验证集所占的比例 (0.0 到 1.0)。
            test_ratio (float, optional): 测试集所占的比例 (0.0 到 1.0)。
                                            训练集比例将是 1.0 - test_ratio。
            output_num_classes_file (str, optional): 如果提供，将类别总数写入此文件。
            output_dir (str, optional): 如果提供，将列表文件保存到此目录。

        Raises:
            ValueError: 如果 train_ratio, validation_ratio 和 test_ratio 的和不接近 1.0 或比例不合法。
            FileNotFoundError: 如果 data_root_path 不存在。
        """
        self._reset_state() #确保每次调用都是干净的状态

        # 确保训练集、验证集和测试集比例之和接近 1
        if not (0 < train_ratio < 1 and 0 < validation_ratio < 1 and 0 < test_ratio < 1 and
                0.99 <= (train_ratio + validation_ratio + test_ratio) <= 1.01):
            raise ValueError("train_ratio, validation_ratio 和 test_ratio 必须在 (0,1) 开区间，且它们的和必须接近 1.0。")

        if not os.path.exists(data_root_path):
            raise FileNotFoundError(f"指定的数据根目录 '{data_root_path}' 不存在。")

        # Use the provided output_dir, or default to data_root_path
        final_output_dir = output_dir if output_dir is not None else data_root_path

        # Ensure the output directory exists
        os.makedirs(final_output_dir, exist_ok=True)

        output_dir = data_root_path
        train_list_path = os.path.join(final_output_dir, train_list_name)
        test_list_path = os.path.join(final_output_dir, test_list_name)
        validation_list_path = os.path.join(final_output_dir, validation_list_name) # 新增路径
        meta_file_path = os.path.join(final_output_dir, meta_file_name)

        all_images_by_class = {} 
        
        print(f"正在扫描数据目录: {data_root_path} ...")
        class_names = sorted([d for d in os.listdir(data_root_path) if os.path.isdir(os.path.join(data_root_path, d))])
        
        if not class_names:
            print(f"警告: 在 '{data_root_path}' 中没有找到子目录（类别）。列表文件将为空。")
            open(train_list_path, 'w').close()
            open(test_list_path, 'w').close()
            open(validation_list_path, 'w').close() # 新增
            with open(meta_file_path, 'w', encoding='utf-8') as f_meta:
                json.dump(self.data_statistics, f_meta, indent=4, ensure_ascii=False)
            return

        self.data_statistics["total_classes"] = len(class_names)

        for class_name in tqdm(class_names, desc="处理类别"):
            if class_name not in self.class_to_id_map:
                self.class_to_id_map[class_name] = self.label_id_counter
                current_class_id = self.label_id_counter
                self.label_id_counter += 1
            else:
                current_class_id = self.class_to_id_map[class_name]

            class_path = os.path.join(data_root_path, class_name)
            images_in_class = []
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_relative_path = os.path.join(class_name, img_file)
                    images_in_class.append(img_relative_path)
            
            if not images_in_class:
                print(f"警告: 类别 '{class_name}' 为空，已跳过。")
                continue

            random.shuffle(images_in_class)
            num_images_in_class = len(images_in_class)
            self.data_statistics["total_images"] += num_images_in_class
            self.data_statistics["images_per_class"][class_name] = num_images_in_class

            num_train = int(num_images_in_class * train_ratio)
            num_validation = int(num_images_in_class * validation_ratio) # 新增
            num_test = num_images_in_class - num_train - num_validation # 调整测试集计算

            # 确保在样本极少时至少有一个训练样本，并调整其他集合大小
            if num_images_in_class > 0 and num_train == 0:
                num_train = 1
                if num_validation >= num_images_in_class - num_train:
                    num_validation = num_images_in_class - num_train
                num_test = num_images_in_class - num_train - num_validation

            if num_test < 0:
                num_test = 0 # 确保测试集不为负
            if num_validation < 0:
                num_validation = 0 # 确保验证集不为负
            if num_train < 0:
                num_train = 0 # 确保训练集不为负


            if current_class_id not in all_images_by_class:
                 all_images_by_class[current_class_id] = {'train': [], 'validation': [], 'test': []} # 新增 'validation'

            all_images_by_class[current_class_id]['train'].extend(images_in_class[:num_train])
            all_images_by_class[current_class_id]['validation'].extend(images_in_class[num_train : num_train + num_validation]) # 新增
            all_images_by_class[current_class_id]['test'].extend(images_in_class[num_train + num_validation :]) # 调整测试集起始索引
            
            self.data_statistics["train_set_count"] += len(all_images_by_class[current_class_id]['train'])
            self.data_statistics["validation_set_count"] += len(all_images_by_class[current_class_id]['validation']) # 新增
            self.data_statistics["test_set_count"] += len(all_images_by_class[current_class_id]['test'])

        print(f"正在写入列表文件到: {final_output_dir}")
        with open(train_list_path, 'w', encoding='utf-8') as f_train, \
             open(test_list_path, 'w', encoding='utf-8') as f_test, \
             open(validation_list_path, 'w', encoding='utf-8') as f_validation: # 新增文件句柄
            
            sorted_class_ids = sorted(all_images_by_class.keys())

            for class_id in sorted_class_ids:
                for img_path in all_images_by_class[class_id]['train']:
                    f_train.write(f"{img_path}\t{class_id}\n")
                for img_path in all_images_by_class[class_id]['validation']: # 新增写入验证集
                    f_validation.write(f"{img_path}\t{class_id}\n")
                for img_path in all_images_by_class[class_id]['test']:
                    f_test.write(f"{img_path}\t{class_id}\n")
        
        print("列表文件写入完成。")
        
        meta_data_to_save = {
            "class_to_id_map": self.class_to_id_map,
            "data_statistics": self.data_statistics,
            "generation_parameters": {
                "data_root_path": data_root_path,
                "train_list_name": train_list_name,
                "test_list_name": test_list_name,
                "validation_list_name": validation_list_name, # 新增
                "meta_file_name": meta_file_name,
                "train_ratio_config": train_ratio,
                "validation_ratio_config": validation_ratio, # 新增
                "test_ratio_config": test_ratio,
                # 计算实际的测试集比例用于记录
                "effective_test_ratio": (self.data_statistics['test_set_count'] / self.data_statistics['total_images'] 
                                         if self.data_statistics['total_images'] > 0 else 0),
                "effective_validation_ratio": (self.data_statistics['validation_set_count'] / self.data_statistics['total_images'] 
                                         if self.data_statistics['total_images'] > 0 else 0) # 新增
            }
        }
        try:
            with open(meta_file_path, 'w', encoding='utf-8') as f_meta:
                json.dump(meta_data_to_save, f_meta, indent=4, ensure_ascii=False)
            print(f"元数据文件 '{meta_file_name}' 已保存到 '{final_output_dir}'.")
        except Exception as e:
            print(f"错误: 保存元数据文件到 {meta_file_path} 失败: {e}")

        if output_num_classes_file:
            try:
                with open(output_num_classes_file, 'w', encoding='utf-8') as f_num_classes:
                    f_num_classes.write(str(self.data_statistics["total_classes"]))
                print(f"类别总数 ({self.data_statistics['total_classes']}) 已写入到: {output_num_classes_file}")
            except Exception as e:
                print(f"错误: 写入类别总数到文件 {output_num_classes_file} 失败: {e}")
        
        print("\n--- 数据集划分统计 ---")
        print(f"总类别数: {self.data_statistics['total_classes']}")
        print(f"总图片数: {self.data_statistics['total_images']}")
        print(f"训练集图片数: {self.data_statistics['train_set_count']}")
        print(f"验证集图片数: {self.data_statistics['validation_set_count']}") # 新增
        print(f"测试集图片数: {self.data_statistics['test_set_count']}")
        print("-----------------------\n")
        print("数据列表创建完成。")

# --- 脚本主入口 ---
if __name__ == '__main__':
    """
    当该脚本作为主程序直接运行时，执行此处的代码。
    这里提供了一个简单的示例，演示如何实例化 CreateDataList 类并调用其方法。
    """
    print("CreateDataList.py 脚本正在作为主程序运行...")
    
    # 实例化数据列表创建器
    data_lister_instance = CreateDataList()
    
    # --- 用户配置区域 ---
    # 指定你的数据集根目录。
    # 请将此路径替换为你的实际数据集所在的根目录。
    # 例如: target_dataset_root = 'data/face' 
    #      或者 target_dataset_root = 'path/to/your/image_dataset'
    target_dataset_root = 'data/face'  # 默认示例数据集路径
    
    # (可选) 自定义输出文件名
    # custom_train_list_filename = "my_train.txt"
    # custom_test_list_filename = "my_test.txt"
    # custom_meta_filename = "dataset_info.json"
    
    print(f"\n将为数据集位于 '{target_dataset_root}' 的数据创建或更新数据列表和元数据文件。")
    print("重要提示: 如果目标目录下已存在同名列表文件和元数据文件，它们的内容将会被覆盖。")
    
    # 调用核心方法来创建数据列表和元数据文件
    # 如果需要自定义输出文件名，可以传递相应的参数给 create_data_list 方法，例如：
    # data_lister_instance.create_data_list(
    #     target_dataset_root,
    #     train_list_name=custom_train_list_filename,
    #     test_list_name=custom_test_list_filename,
    #     meta_file_name=custom_meta_filename
    # )
    parser = argparse.ArgumentParser(description="生成 PaddlePaddle 模型训练所需的数据列表文件。")
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录路径。此目录下应包含代表不同类别的子目录，子目录中包含图片文件。')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='生成的 train_list.txt, validation_list.txt 和 test_list.txt 文件的输出目录。' # 更新说明
                             '如果未指定，则默认使用 data_root 目录。')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集所占的比例 (0.0 到 1.0)。默认为 0.7。')
    parser.add_argument('--validation_ratio', type=float, default=0.2, # 新增参数
                        help='验证集所占的比例 (0.0 到 1.0)。默认为 0.2。')
    parser.add_argument('--test_ratio', type=float, default=0.1, # 调整默认值
                        help='测试集所占的比例 (0.0 到 1.0)。训练集和验证集比例将是 1.0 - train_ratio - validation_ratio。默认为 0.1。')
    parser.add_argument('--output_num_classes_file', type=str, default=None,
                        help='可选参数。指定一个文件路径，用于写入计算出的类别总数。'
                             '例如: /path/to/project/latest_num_classes.txt')

    args = parser.parse_args()

    # 如果未指定输出目录，则默认为 data_root
    if args.output_dir is None:
        args.output_dir = args.data_root

    # 执行列表创建
    data_lister_instance.create_data_list(
        data_root_path=args.data_root,
        train_list_name="trainer.list",
        test_list_name="test.list",
        validation_list_name="validation.list", # 新增
        meta_file_name="readme.json",
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio, # 新增
        test_ratio=args.test_ratio,
        output_num_classes_file=args.output_num_classes_file,
        output_dir=args.output_dir
    )