# create_face_library.py
# 该脚本的目的是为基于特征比对的人脸识别模型 (如 ArcFace) 创建一个人脸特征库。
# 特征库是一个字典，其中键是人物的身份标签 (通常是整数ID)，值是该人物的多张图像的平均特征向量。
#
# 主要流程：
# 1. 加载配置文件和命令行参数，获取模型路径、数据集路径、运行设备等信息。
# 2. 设置运行设备 (CPU/GPU)。
# 3. 加载预训练好的人脸识别模型（通常只加载骨干网络部分，因为特征提取是主要任务）。
#    模型文件应包含训练时的配置信息 (如 model_type, image_size) 以便正确恢复骨干网络。
# 4. 使用 model_factory.py 动态实例化骨干网络。
# 5. 加载骨干网络权重到实例化的网络中，并设置为评估模式。
# 6. 遍历指定的数据集目录 (由 config.data_dir 和 config.class_name 定义)。
#    数据集目录结构应为：
#      data_dir/
#      └── class_name/  (例如: "face_dataset")
#          ├── person1_label_X/ (X是分配给person1的整数ID)
#          │   ├── image1.jpg
#          │   ├── image2.png
#          │   └── ...
#          ├── person2_label_Y/
#          │   ├── imageA.jpg
#          │   └── ...
#          └── readme.json (由 CreateDataList.py 生成，包含类别ID到真实姓名的映射)
#    脚本会解析子目录名中的标签ID (例如 "person1_label_X" 中的 X)。
# 7. 对每个身份的每张图像进行以下操作：
#    a. 图像预处理 (缩放、归一化、标准化)。
#    b. 使用加载的骨干网络提取图像的人脸特征向量。
# 8. 对每个身份，将其所有图像的特征向量进行平均，得到该身份的平均特征向量。
# 9. 将所有身份的平均特征向量存储在一个字典中 {label_id: avg_feature_vector}。
# 10. 使用 pickle 将这个特征库字典序列化并保存到磁盘文件 (通常是 .pkl 文件)。
#     保存路径由配置中的 `infer.face_library_path` 指定。
# 11. 同时，确保 `readme.json` (包含标签ID到真实名称的映射) 与特征库一起提供，
#     以便在后续推理时可以将识别出的标签ID转换为可读的人名。
#
# 注意: 此脚本中的 process_image_local 函数与 infer.py 和 compare.py 中的类似函数功能相同，
#       理想情况下应移至共享的 utils.py 模块。

import os
import cv2
import argparse
import numpy as np
import paddle
import pickle # 用于序列化和保存特征库字典
import json   # 用于读取 readme.json 以获取标签ID到真实姓名的映射
from tqdm import tqdm # 用于显示处理进度条
from config_utils import load_config, ConfigObject # 导入配置加载工具和配置对象类型
from model_factory import get_backbone # 仅需骨干网络进行特征提取
from paddle.io import Dataset, DataLoader # Ensure DataLoader is imported if MyReader.create_data_loader is used indirectly
import MyReader # Needed for MyReader.create_data_loader
from utils.image_processing import process_image # 从共享模块导入
import paddle.nn.functional as F

def create_face_library(config: ConfigObject, cmd_args: argparse.Namespace):
    """
    使用预训练模型为指定数据集中的图像提取特征并保存为人脸特征库。
    """
    # --- 设置设备 ---
    use_gpu_flag = cmd_args.use_gpu if cmd_args.use_gpu is not None else config.get('use_gpu', True) # 提供默认值
    use_gpu_flag = use_gpu_flag and paddle.is_compiled_with_cuda()
    paddle.set_device('gpu' if use_gpu_flag else 'cpu')
    print(f"使用 {'GPU' if use_gpu_flag else 'CPU'} 创建人脸库。")

    # --- 确定模型权重路径 ---
    model_weights_path = cmd_args.model_path or \
                         config.get('model_path') or \
                         (config.create_library.get('model_path') if hasattr(config, 'create_library') and config.create_library else None)

    if not model_weights_path:
        raise ValueError("错误: 缺少模型文件路径。请通过 --model_path 或YAML配置提供。")
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"错误: 指定的模型权重文件未找到: {model_weights_path}")
    print(f"将从模型文件 {model_weights_path} 加载模型用于创建人脸库。")
    
    # --- 确定输出特征库的路径 ---
    output_path_from_config = config.create_library.get('output_library_path') if hasattr(config, 'create_library') and config.create_library else "face_library.pkl"
    
    final_output_library_path = None
    source_of_output_path = ""

    if cmd_args.face_library_path:
        final_output_library_path = cmd_args.face_library_path
        source_of_output_path = "command line (--face_library_path)"
        if os.path.isdir(final_output_library_path): # 如果命令行提供的是目录
            lib_filename_to_append = "face_library.pkl" # 默认文件名
            if isinstance(output_path_from_config, str) and not os.path.sep in output_path_from_config and output_path_from_config:
                lib_filename_to_append = output_path_from_config # 使用配置中的文件名
            final_output_library_path = os.path.join(final_output_library_path, lib_filename_to_append)
            source_of_output_path += f" (appended filename '{lib_filename_to_append}')"
    elif output_path_from_config:
        if os.path.isabs(output_path_from_config) or os.path.sep in output_path_from_config:
            final_output_library_path = output_path_from_config
            source_of_output_path = "config (absolute/relative path)"
        else: # 配置中的是文件名
            model_dir = os.path.dirname(model_weights_path)
            if not model_dir: # model_weights_path 可能是当前目录下的文件名
                model_dir = os.getcwd()
            final_output_library_path = os.path.join(model_dir, output_path_from_config)
            source_of_output_path = f"config (filename only, resolved to model dir: {model_dir})"
    else: # 命令行和配置中均未指定，或配置中 output_library_path 为空
        model_dir = os.path.dirname(model_weights_path)
        if not model_dir: model_dir = os.getcwd()
        final_output_library_path = os.path.join(model_dir, "face_library.pkl") # 默认名
        source_of_output_path = "default (model dir, face_library.pkl)"

    print(f"特征库将保存至: {final_output_library_path} (路径来源: {source_of_output_path})")
    # 创建输出目录（如果尚不存在）
    output_dir_for_lib = os.path.dirname(final_output_library_path)
    if output_dir_for_lib: # 确保目录名不为空（例如当输出到当前工作目录时）
        os.makedirs(output_dir_for_lib, exist_ok=True)


    # --- 尝试从模型元数据加载配置 ---
    loaded_model_type = None
    loaded_image_size = None
    loaded_model_specific_params = {}
    source_of_model_config = "N/A"
    metadata_path = model_weights_path.replace('.pdparams', '.json')
    using_metadata_config = False
    metadata = {} # 初始化

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            temp_model_type = metadata.get('model_type')
            temp_image_size = metadata.get('image_size')
            temp_model_specific_params = metadata.get('model_specific_params')

            if all([temp_model_type, temp_image_size is not None, temp_model_specific_params is not None]):
                loaded_model_type = temp_model_type
                loaded_image_size = temp_image_size
                loaded_model_specific_params = temp_model_specific_params if isinstance(temp_model_specific_params, dict) else {}
                source_of_model_config = f"Metadata file ({metadata_path})"
                using_metadata_config = True
                print(f"已从元数据文件 {metadata_path} 加载构建骨干网络的配置。")
            else:
                print(f"警告: 模型元数据文件 {metadata_path} 中缺少部分骨干网络构建所需配置。")
        except Exception as e:
            print(f"警告: 加载或解析模型元数据文件 {metadata_path} 失败: {e}。")
    
    if not using_metadata_config:
        print(f"将使用 create_face_library.py 的全局/命令行配置文件中的配置构建骨干网络。")
        loaded_model_type = config.model_type
        loaded_image_size = cmd_args.image_size if cmd_args.image_size is not None else config.image_size
        loaded_model_specific_params_co = config.model.get(f'{loaded_model_type}_params', ConfigObject({}))
        loaded_model_specific_params = loaded_model_specific_params_co.to_dict() if isinstance(loaded_model_specific_params_co, ConfigObject) else loaded_model_specific_params_co

        source_of_model_config = "Global/CMD Config (fallback)"
        if not all([loaded_model_type, loaded_image_size is not None]):
             raise ValueError("错误: 无法从全局/命令行配置中确定骨干网络构建所需的核心配置 (model_type, image_size)。")
    
    print(f"--- 骨干网络构建配置来源: {source_of_model_config} ---")
    print(f"  Model Type: {loaded_model_type}")
    print(f"  Image Size (for model preproc): {loaded_image_size}")
    print(f"  Model Params: {loaded_model_specific_params}")
    print("--------------------------------------------------")

    model_backbone, backbone_out_dim = get_backbone(
        config_model_params=loaded_model_specific_params,
        model_type_str=loaded_model_type,
        image_size=loaded_image_size
    )
    print(f"骨干网络 ({loaded_model_type.upper()}) 构建成功，输出特征维度: {backbone_out_dim}")

    full_state_dict = paddle.load(model_weights_path)
    backbone_state_dict_to_load = {k.replace('backbone.', '', 1): v for k, v in full_state_dict.items() if k.startswith('backbone.')}
    
    if backbone_state_dict_to_load:
        model_backbone.set_state_dict(backbone_state_dict_to_load)
        print(f"骨干网络权重从 {model_weights_path} (提取 'backbone.' 部分) 加载成功。")
    else:
        try:
            model_backbone.set_state_dict(full_state_dict)
            print(f"骨干网络权重 (尝试直接加载整个文件) 从 {model_weights_path} 加载成功。")
        except Exception as e_direct_bb_load:
            raise RuntimeError(f"错误: 在模型文件 {model_weights_path} 中未找到 'backbone.' 前缀的权重，且直接加载整个状态字典到骨干网络失败: {e_direct_bb_load}。")
    
    model_backbone.eval()

    # --- 数据准备 ---
    # 确定用于建库的数据列表文件名
    # 优先级: 命令行参数 > config.dataset_params.train_list > 默认值 "trainer.list" (忽略 config.create_library)
    data_list_for_library_name = cmd_args.data_list_for_library
    if data_list_for_library_name is None:
        if hasattr(config, 'dataset_params') and config.dataset_params and config.dataset_params.get('train_list'):
            data_list_for_library_name = config.dataset_params.get('train_list')
        else:
            data_list_for_library_name = "trainer.list" # 最终默认值

    data_root_for_lists = cmd_args.data_dir or config.data_dir
    class_name_for_lists = config.class_name # 通常是 'face'
    
    # 根据 data_root 和 class_name 构建完整路径
    # 注意: dataset_params.train_list/eval_list 已经是相对于 data_dir/class_name 的路径 (例如 trainer.list)
    # create_face_library.py 的 data_list_for_library 参数/配置项 也预期是这样的文件名
    actual_library_data_list_path = os.path.join(data_root_for_lists, class_name_for_lists, data_list_for_library_name)

    if not os.path.exists(actual_library_data_list_path):
        raise FileNotFoundError(f"错误: 用于创建人脸库的数据列表文件 '{actual_library_data_list_path}' 未找到。")
    print(f"将使用数据列表 '{actual_library_data_list_path}' 中的图像创建人脸库。")

    image_mean = config.dataset_params.mean
    image_std = config.dataset_params.std

    library_features = []
    library_labels_ids = []

    print("开始从数据列表提取特征...")
    try:
        with open(actual_library_data_list_path, 'r', encoding='utf-8') as f_list:
            # 使用 strip() 去除每行末尾的换行符，然后使用制表符分割
            image_paths_and_labels = [line.strip().split('\t') for line in f_list if line.strip()]

    except Exception as e:
        raise IOError(f"读取或解析库数据列表 {actual_library_data_list_path} 失败: {e}")


    with paddle.no_grad():
        for item in tqdm(image_paths_and_labels, desc="处理图像建库"):
            if len(item) != 2:
                print(f"警告: 跳过格式不正确的行: {item} (在列表 {actual_library_data_list_path} 中)")
                continue
            
            img_relative_path, label_id_str = item
            
            # CreateDataList.py 生成的列表路径已经是相对于 data_dir 的 'class_name/image.jpg' 形式
            # 所以我们直接用 data_dir 和这个相对路径拼接
            # 修正：img_relative_path 是相对于 data_dir/class_name 的，所以拼接时需要包含 class_name
            # data_root_for_lists 指向 config.data_dir (例如 "data")
            # class_name_for_lists 指向 config.class_name (例如 "face")
            # img_relative_path 来自列表文件 (例如 "dilireba/some_image.jpg")
            full_img_path = os.path.join(data_root_for_lists, class_name_for_lists, img_relative_path)

            if not os.path.exists(full_img_path):
                print(f"警告: 图像文件 {full_img_path} (来自列表项: {img_relative_path}) 未找到，已跳过。")
                continue 
            
            try:
                img_processed_np = process_image(
                    img_path=full_img_path,
                    target_size=loaded_image_size,
                    mean_rgb=image_mean,
                    std_rgb=image_std,
                    is_bgr_input=True
                )
                img_tensor = paddle.to_tensor(img_processed_np)
                feature_vector = model_backbone(img_tensor).numpy().flatten()
                
                library_features.append(feature_vector)
                library_labels_ids.append(int(label_id_str))
            except Exception as e_proc:
                print(f"警告: 处理图像 {full_img_path} 时发生错误: {e_proc}，已跳过。")
                continue
    
    if not library_features:
        print("错误: 未能从数据列表中成功提取任何特征。人脸库为空，无法保存。")
        return

    library_features_np = np.array(library_features, dtype='float32')
    library_labels_ids_np = np.array(library_labels_ids, dtype='int64')

    try:
        with open(final_output_library_path, 'wb') as f_lib:
            pickle.dump((library_features_np, library_labels_ids_np), f_lib)
            # 尝试强制文件系统同步，确保数据已写入磁盘
            f_lib.flush() # 先刷新缓冲区
            os.fsync(f_lib.fileno()) # 再强制同步到磁盘
        print(f"人脸特征库已成功保存到: {final_output_library_path} (包含 {library_features_np.shape[0]} 个特征)。")
    except Exception as e_save:
        raise IOError(f"保存人脸特征库到 {final_output_library_path} 失败: {e_save}")

    print("创建人脸库完成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用预训练的人脸识别模型创建人脸特征库。")
    parser.add_argument('--config_path', type=str, default='configs/default_config.yaml',
                        help='配置文件的路径。')
    parser.add_argument('--active_config', type=str, default=None, 
                        help='要激活的配置块名称 (覆盖YAML中的 active_config)。选择包含ArcFace等需要建库的模型配置。')
    
    # 核心输入：模型路径
    parser.add_argument('--model_path', type=str, default=None,
                        help='必需：训练好的人脸识别模型 (.pdparams) 的路径。脚本将从此模型加载权重。')
    
    # 数据相关 (通常由配置文件管理，但允许覆盖)
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据集根目录 (覆盖YAML中的 data_dir)。')
    # parser.add_argument('--class_name', type=str, default=None, # class_name 通常全局一致
    #                     help='数据集类别名 (覆盖YAML)')
    parser.add_argument('--data_list_for_library', type=str, default=None,
                        help='(可选) 用于建库的特定图像列表文件名称 (位于 data_dir/class_name/ 下)。默认为配置文件中指定或 "train.list"。')

    # 输出相关
    parser.add_argument('--face_library_path', type=str, default=None,
                        help='(可选) 输出人脸库文件的完整路径或目录。如果只提供目录，将使用配置文件中的文件名或默认名 "face_library.pkl"。'
                             '如果未指定，将根据配置文件或模型路径推断。')
    
    # 运行参数
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=None, help='是否使用GPU (覆盖YAML)。')
    parser.add_argument('--image_size', type=int, default=None,
                        help='(可选) 强制指定模型加载和图像预处理时使用的图像尺寸。强烈建议让脚本从模型元数据推断此值。')

    cmd_line_args = parser.parse_args()

    # 检查 --model_path 是否提供 (因为后续逻辑依赖它)
    # 虽然 load_config 会加载，但这里做一个早期检查
    effective_model_path = cmd_line_args.model_path
    if not effective_model_path:
        # 尝试从配置文件加载 model_path (如果 active_config 指定的块里有)
        temp_config_for_model_path_check = load_config(cmd_line_args.config_path, cmd_line_args)
        effective_model_path = temp_config_for_model_path_check.get('model_path') or \
                               (temp_config_for_model_path_check.create_library.get('model_path') if hasattr(temp_config_for_model_path_check, 'create_library') and temp_config_for_model_path_check.create_library else None)
    
    if not effective_model_path:
        parser.error("错误: 缺少模型文件路径。请通过 --model_path 参数或在YAML配置的相应块中 (如 active_config 指向的块的 model_path 或 create_library.model_path) 提供。")


    final_config = load_config(
        default_yaml_path=cmd_line_args.config_path,
        cmd_args_namespace=cmd_line_args # 传递命令行参数以覆盖
    )
    
    # 再次确保 model_path 在 final_config 中，因为它是后续逻辑的关键
    if not final_config.get('model_path') and not (hasattr(final_config, 'create_library') and final_config.create_library and final_config.create_library.get('model_path')):
        if cmd_line_args.model_path: # 如果命令行有，但没合并到 config 对象中
            final_config['model_path'] = cmd_line_args.model_path # 手动注入
        else:
            # 此处不应到达，因为前面有 parser.error
            print("严重错误: model_path 未能设置到 final_config。")
            exit(1)


    # 将命令行中可能覆盖 create_library 子配置的参数合并进去
    if not hasattr(final_config, 'create_library') or final_config.create_library is None:
        final_config.create_library = ConfigObject({})

    if cmd_line_args.data_list_for_library:
        final_config.create_library['data_list_for_library'] = cmd_line_args.data_list_for_library
    # output_library_path 的处理已在 create_face_library 函数内部完成 (基于 cmd_args.face_library_path)

    create_face_library(final_config, cmd_line_args) 