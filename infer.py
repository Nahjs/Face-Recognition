# infer.py
# 该脚本负责人脸识别的推理 (inference) 功能。
# 主要流程包括：
# 1. 加载配置文件和命令行参数。
# 2. 根据配置设置运行设备 (CPU/GPU)。
# 3. 加载预训练好的模型文件 (.pdparams)，这包括模型权重和训练时的配置信息。
# 4. 根据模型文件中保存的配置 (model_type, loss_type, image_size等) 和当前脚本的配置，
#    使用 model_factory.py 动态实例化骨干网络 (backbone) 和必要的头部模块 (head)。
#    - 对于CrossEntropy模型，会实例化分类头 (CrossEntropyHead)。
#    - 对于ArcFace模型，推理时通常只使用骨干网络提取特征，然后与特征库比对，不直接实例化ArcFaceHead进行前向计算。
# 5. 加载模型权重到实例化的网络中。
# 6. 加载类别标签映射文件 (readme.json)，用于将预测的ID转换为可读的名称。
# 7. 对于ArcFace模型，加载预先构建的人脸特征库 (.pkl)。
# 8. 对输入的单张图像进行预处理 (缩放、归一化、标准化)。
# 9. 执行推理：
#    - ArcFace模型: 提取输入图像的特征，然后与特征库中的每个已知身份的特征计算余弦相似度，
#                   找出最相似的身份，并根据阈值判断是否为已知人物。
#    - CrossEntropy模型: 将图像输入到完整的 模型(骨干+头) 中，获取分类的logits，
#                       通过softmax得到概率，取概率最高的类别作为预测结果。
# 10. 可视化推理结果：在输入图像上标注预测的类别名称和置信度/相似度，并保存到 results/ 目录。

import os
import cv2
import argparse
import numpy as np
import paddle
import paddle.nn.functional as F
import json # 用于加载标签映射文件 (readme.json)
import matplotlib
matplotlib.use('Agg') # 切换到非交互式后端，防止在无GUI服务器上出错
import matplotlib.pyplot as plt
import pickle # 用于加载ArcFace模型所需的人脸特征库 (.pkl文件)
from config_utils import load_config, ConfigObject # 导入配置加载工具和配置对象类型
from model_factory import get_backbone, get_head # 导入模型构建的工厂函数
from model_factory import get_backbone, get_head   # 导入模型工厂函数
from utils.image_processing import process_image_local # 从共享模块导入
import time # 导入 time 模块

# 全局变量，用于保存加载的标签映射，避免重复读取文件
loaded_label_map = None
# 全局变量，用于保存加载的人脸库特征，避免重复加载和计算
loaded_face_library_features = None

def compute_similarity(feature_vec1: np.ndarray, feature_vec2: np.ndarray) -> float:
    """计算两个一维特征向量之间的余弦相似度。

    余弦相似度衡量两个向量在方向上的相似程度，值域为 [-1, 1]。
    值越接近1，表示两个向量越相似。

    Args:
        feature_vec1 (np.ndarray): 第一个特征向量 (一维numpy数组)。
        feature_vec2 (np.ndarray): 第二个特征向量 (一维numpy数组)。

    Returns:
        float: 计算得到的余弦相似度。如果任一向量的范数为0，则返回0.0以避免除零错误。
    """
    # 确保输入是一维向量 (如果已经是，flatten操作无影响)
    f1 = feature_vec1.flatten()
    f2 = feature_vec2.flatten()
    
    # 计算各自的L2范数 (模长)
    norm_f1 = np.linalg.norm(f1)
    norm_f2 = np.linalg.norm(f2)
    
    # 防止除以零错误：如果任一向量的范数为0，则认为它们不相似 (相似度为0)
    if norm_f1 == 0 or norm_f2 == 0:
        return 0.0
        
    # 计算点积 / (范数之积)
    similarity = np.dot(f1, f2) / (norm_f1 * norm_f2)
    return float(similarity) # 确保返回的是标准的float类型

def infer(config: ConfigObject, cmd_args: argparse.Namespace):
    """
    主推理函数，加载模型、标签和目标图片，进行预测。
    """
    # --- 设置设备 ---
    use_gpu_flag = cmd_args.use_gpu if cmd_args.use_gpu is not None else config.use_gpu
    use_gpu_flag = use_gpu_flag and paddle.is_compiled_with_cuda()
    paddle.set_device('gpu' if use_gpu_flag else 'cpu')
    print(f"使用 {'GPU' if use_gpu_flag else 'CPU'} 进行推理")

    # --- 确定模型权重路径 ---
    model_weights_path = cmd_args.model_path or config.model_path
    if not model_weights_path:
        raise ValueError("错误: 必须通过 --model_path 或在配置文件中通过 model_path 指定模型权重文件路径。")
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"错误: 指定的模型权重文件未找到: {model_weights_path}")
    
    print(f"将从模型文件 {model_weights_path} 加载模型。")

    # --- 尝试从模型元数据加载配置 ---
    loaded_model_type = None
    loaded_loss_type = None
    loaded_image_size = None
    loaded_num_classes = None
    loaded_model_specific_params = {}
    loaded_loss_specific_params = {}

    metadata_path = model_weights_path.replace('.pdparams', '.json')
    using_metadata_config = False
    source_of_config = "Global infer.py Config"

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            temp_model_type = metadata.get('model_type')
            temp_loss_type = metadata.get('loss_type')
            temp_image_size = metadata.get('image_size')
            temp_num_classes = metadata.get('num_classes')
            temp_model_specific_params = metadata.get('model_specific_params')
            temp_loss_specific_params = metadata.get('loss_specific_params')

            if all([temp_model_type, temp_loss_type, 
                    temp_image_size is not None, temp_num_classes is not None,
                    temp_model_specific_params is not None, temp_loss_specific_params is not None]):
                loaded_model_type = temp_model_type
                loaded_loss_type = temp_loss_type
                loaded_image_size = temp_image_size
                loaded_num_classes = temp_num_classes
                loaded_model_specific_params = temp_model_specific_params if isinstance(temp_model_specific_params, dict) else {}
                loaded_loss_specific_params = temp_loss_specific_params if isinstance(temp_loss_specific_params, dict) else {}
                source_of_config = f"Metadata file ({metadata_path})"
                print(f"已从元数据文件 {metadata_path} 加载完整配置。")
                using_metadata_config = True
            else:
                print(f"警告: 模型元数据文件 {metadata_path} 中缺少部分关键配置项。")
        except Exception as e:
            print(f"警告: 加载或解析模型元数据文件 {metadata_path} 失败: {e}。")

    if not using_metadata_config:
        print(f"将使用 infer.py 的全局配置文件中的配置 (回退或元数据加载失败/不完整)。")
        loaded_model_type = config.model_type
        loaded_loss_type = config.loss_type
        loaded_image_size = config.image_size
        loaded_num_classes = config.num_classes
        # 从全局配置加载详细参数
        loaded_model_specific_params = config.model.get(f'{loaded_model_type}_params', {}).to_dict() \
                                     if isinstance(config.model.get(f'{loaded_model_type}_params', {}), ConfigObject) \
                                     else config.model.get(f'{loaded_model_type}_params', {})
        loaded_loss_specific_params = config.loss.get(f'{loaded_loss_type}_params', {}).to_dict() \
                                    if isinstance(config.loss.get(f'{loaded_loss_type}_params', {}), ConfigObject) \
                                    else config.loss.get(f'{loaded_loss_type}_params', {})
        source_of_config = "Global infer.py Config (fallback)"
        if not all([loaded_model_type, loaded_loss_type, loaded_image_size is not None, loaded_num_classes is not None]):
             raise ValueError("错误: 无法从全局配置中确定模型构建所需的核心配置。请检查 infer.py 的配置文件。")
    
    print(f"--- 模型构建配置来源: {source_of_config} ---")
    print(f"  Model Type: {loaded_model_type}")
    print(f"  Loss Type: {loaded_loss_type}")
    print(f"  Image Size: {loaded_image_size}")
    print(f"  Num Classes: {loaded_num_classes}")
    print(f"  Model Params: {loaded_model_specific_params}")
    print(f"  Loss Params: {loaded_loss_specific_params}")
    print("--------------------------------------------------")

    # --- 构建模型 ---
    model_backbone, backbone_out_dim = get_backbone(
        config_model_params=loaded_model_specific_params, 
        model_type_str=loaded_model_type,
        image_size=loaded_image_size
    )
    print(f"骨干网络 ({loaded_model_type.upper()}) 构建成功，期望输入图像尺寸: {loaded_image_size}, 输出特征维度: {backbone_out_dim}")

    model_head = None
    if loaded_loss_type == 'cross_entropy': 
        model_head = get_head(
            config_loss_params=loaded_loss_specific_params, 
            loss_type_str=loaded_loss_type,
            in_features=backbone_out_dim,
            num_classes=loaded_num_classes
        )
        print(f"头部模块 ({loaded_loss_type.upper()}) 构建成功，输入特征维度: {backbone_out_dim}, 输出类别数: {loaded_num_classes}")

    # --- 加载模型权重 ---
    if not model_weights_path or not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"错误: 模型权重文件 '{model_weights_path}' 未找到或未指定。")

    # 如果代码执行到这里，说明 model_weights_path 是有效的并且文件存在
    full_state_dict = paddle.load(model_weights_path)
    
    # 提取骨干网络权重
    backbone_state_dict = {k.replace('backbone.', '', 1): v for k, v in full_state_dict.items() if k.startswith('backbone.')}
    if backbone_state_dict:
        model_backbone.set_state_dict(backbone_state_dict)
        print(f"骨干网络权重从 {model_weights_path} 加载成功。")
    else:
        # 如果没有 'backbone.' 前缀，尝试直接加载整个 state_dict 到 backbone (可能模型只保存了骨干)
        try:
            model_backbone.set_state_dict(full_state_dict)
            print(f"骨干网络权重 (可能为直接保存的骨干模型) 从 {model_weights_path} 加载成功。")
        except Exception as e_direct_bb_load:
            raise RuntimeError(f"错误: 在模型文件 {model_weights_path} 中未找到 'backbone.' 前缀的权重，并且直接加载整个状态字典到骨干网络失败: {e_direct_bb_load}。请确保模型文件与期望的结构一致。")

    # 如果存在头部模型 (例如 CrossEntropy 模式)，则加载头部权重
    if model_head:
        head_state_dict = {k.replace('head.', '', 1): v for k, v in full_state_dict.items() if k.startswith('head.')}
        if head_state_dict:
            model_head.set_state_dict(head_state_dict)
            print(f"头部模块 ({loaded_loss_type}) 权重从 {model_weights_path} 加载成功。")
        else: # Head is instantiated, but no 'head.' prefixed weights found.
            print(f"警告: 头部模块 ({loaded_loss_type}) 已实例化，但在模型文件 {model_weights_path} 中未找到 'head.' 前缀的权重。头部将使用其默认初始化权重。")
    # If model_head is None (e.g., for ArcFace feature extraction path as currently coded in infer.py),
    # no head loading is attempted, and no warnings about missing head weights are printed here.

    model_backbone.eval()
    if model_head:
        model_head.eval()

    # --- 加载类别标签文件 ---
    label_file_path = None
    source_for_label_file = None

    if cmd_args.label_file:
        if os.path.isabs(cmd_args.label_file) and os.path.exists(cmd_args.label_file):
            label_file_path = cmd_args.label_file
            source_for_label_file = "command line (absolute)"
        elif os.path.exists(cmd_args.label_file): # Relative path from CWD
            label_file_path = cmd_args.label_file
            source_for_label_file = "command line (relative to CWD)"
        else: # Filename or non-existing relative path from cmd_args
            # Attempt to resolve against dataset dir
            if config.data_dir and config.class_name:
                prospective_path = os.path.join(config.data_dir, config.class_name, cmd_args.label_file)
                if os.path.exists(prospective_path):
                    label_file_path = prospective_path
                    source_for_label_file = "command line (resolved to dataset dir)"
            if not label_file_path:
                 print(f"警告: 命令行提供的标签文件路径 '{cmd_args.label_file}' 未找到。")


    if not label_file_path and config.infer.get('label_file'):
        config_label_file = config.infer.get('label_file')
        if os.path.isabs(config_label_file) and os.path.exists(config_label_file):
            label_file_path = config_label_file
            source_for_label_file = "config file (absolute)"
        elif os.path.exists(config_label_file): # Relative path from CWD
            label_file_path = config_label_file
            source_for_label_file = "config file (relative to CWD)"
        else: # Filename or non-existing relative path
             # Attempt to resolve against dataset dir
            if config.data_dir and config.class_name:
                prospective_path = os.path.join(config.data_dir, config.class_name, config_label_file)
                if os.path.exists(prospective_path):
                    label_file_path = prospective_path
                    source_for_label_file = "config file (resolved to dataset dir)"
            if not label_file_path:
                print(f"警告: 配置文件中的标签文件路径 '{config_label_file}' 未找到。")

    # Fallback: If still no valid path, try the default dataset location for "readme.json"
    if not label_file_path:
        if config.data_dir and config.class_name:
            default_dataset_readme_path = os.path.join(config.data_dir, config.class_name, "readme.json")
            if os.path.exists(default_dataset_readme_path):
                label_file_path = default_dataset_readme_path
                source_for_label_file = "default dataset location (readme.json)"
                print(f"标签文件未在命令行或配置中明确找到，使用默认的数据集位置: {label_file_path}")
            else:
                # Last resort: try the model experiment directory (less likely for CreateDataList's readme.json)
                if model_weights_path:
                    inferred_path_model_dir = os.path.join(os.path.dirname(os.path.dirname(model_weights_path)), "readme.json")
                    if os.path.exists(inferred_path_model_dir):
                        label_file_path = inferred_path_model_dir
                        source_for_label_file = "model experiment directory (readme.json)"
                        print(f"尝试从模型实验目录推断标签文件: {label_file_path}")


    if not label_file_path:
        raise FileNotFoundError("错误: 最终未能确定类别标签文件 (readme.json) 的有效路径。请通过 --label_file 指定或确保其在预期位置 (如 data/face/readme.json)。")
    
    print(f"最终用于加载的类别标签文件: {label_file_path} (来源: {source_for_label_file})")
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            # Assuming readme.json contains class_to_id_map
            full_meta = json.load(f)
            class_to_id_map = full_meta.get('class_to_id_map')
            if class_to_id_map is None:
                raise ValueError("readme.json 中未找到 'class_to_id_map'。")
            id_to_class_map = {str(v): k for k, v in class_to_id_map.items()} # Invert for easy lookup
        print(f"类别标签文件 {label_file_path} 加载成功 ({len(id_to_class_map)} 个类别)。")
    except Exception as e:
        raise RuntimeError(f"加载或解析类别标签文件 {label_file_path} 失败: {e}")

    # --- 图像预处理 ---
    target_image_path = cmd_args.image_path
    if not target_image_path or not os.path.exists(target_image_path):
        raise FileNotFoundError(f"错误: 输入图像 --image_path '{target_image_path}' 未指定或未找到。")
    
    # 使用从元数据或配置中加载的 image_size, mean, std
    image_mean = config.dataset_params.mean # Assuming these are globally consistent for now
    image_std = config.dataset_params.std
    
    preprocessed_image_np = process_image_local(
        target_image_path, 
        target_size=loaded_image_size,
        mean_rgb=image_mean,
        std_rgb=image_std
    )
    img_tensor = paddle.to_tensor(preprocessed_image_np)

    # --- 执行推理 ---
    predicted_label_name = "未知"
    confidence_or_similarity = 0.0

    inference_start_time = time.time() # 记录推理开始时间

    with paddle.no_grad():
        features = model_backbone(img_tensor)

        if loaded_loss_type == 'arcface':
            print("ArcFace 推理模式: 提取特征并与人脸库比较...")
            face_lib_path = None
            source_for_face_lib = "未确定"
            potential_paths_tried = []

            # 1. 尝试从命令行参数获取
            if cmd_args.face_library_path:
                potential_paths_tried.append(f"Command Line (--face_library_path): '{cmd_args.face_library_path}'")
                if os.path.isfile(cmd_args.face_library_path):
                    face_lib_path = cmd_args.face_library_path
                    source_for_face_lib = "Command Line"
                else:
                    print(f"  提示: 命令行提供的 --face_library_path '{cmd_args.face_library_path}' 不是一个有效的文件。")

            # 2. 尝试从配置文件 (config.infer.face_library_path) 获取
            if not face_lib_path and config.infer.get('face_library_path'):
                config_lib_path_str = str(config.infer.face_library_path) # Ensure it's a string
                potential_paths_tried.append(f"Config (infer.face_library_path): '{config_lib_path_str}'")
                if os.path.isfile(config_lib_path_str):
                    face_lib_path = config_lib_path_str
                    source_for_face_lib = "Config File (infer.face_library_path)"
                else:
                    print(f"  提示: 配置文件提供的 infer.face_library_path '{config_lib_path_str}' 不是一个有效的文件。")
            
            # 3. 回退：尝试在模型权重文件所在目录查找
            if not face_lib_path and model_weights_path:
                # 确定预期的库文件名
                # 优先使用 config.create_library.output_library_path (如果存在)
                # 否则默认为 "face_library.pkl"
                expected_lib_filename = "face_library.pkl"
                if hasattr(config, 'create_library') and isinstance(config.create_library, ConfigObject) and config.create_library.get('output_library_path'):
                    expected_lib_filename = config.create_library.get('output_library_path')
                
                default_path_near_model = os.path.join(os.path.dirname(model_weights_path), expected_lib_filename)
                potential_paths_tried.append(f"Default (model directory + '{expected_lib_filename}'): '{default_path_near_model}'")
                if os.path.isfile(default_path_near_model):
                    face_lib_path = default_path_near_model
                    source_for_face_lib = f"Default (model directory, found '{expected_lib_filename}')"
                else:
                    # 如果默认的 "face_library.pkl" 找不到，并且 expected_lib_filename 不是 "face_library.pkl"
                    # 也尝试一下 "face_library.pkl" 以防万一 (例如配置指定了其他名字但实际是默认名保存的)
                    if expected_lib_filename != "face_library.pkl":
                        fallback_default_path = os.path.join(os.path.dirname(model_weights_path), "face_library.pkl")
                        potential_paths_tried.append(f"Fallback Default (model directory + 'face_library.pkl'): '{fallback_default_path}'")
                        if os.path.isfile(fallback_default_path):
                             face_lib_path = fallback_default_path
                             source_for_face_lib = "Fallback Default (model directory, found 'face_library.pkl')"
            
            if not face_lib_path:
                 print(f"  未能成功加载人脸特征库。尝试过的特征库路径包括 (按优先级):")
                 for i, p_path in enumerate(potential_paths_tried):
                    print(f"    {i+1}. {p_path}")
                 raise FileNotFoundError("错误: ArcFace模式需要人脸特征库 (.pkl)，但最终未能确定其有效路径。请通过 --face_library_path 指定，或确保其存在于预期的位置 (通常与模型文件同目录，名为 face_library.pkl 或由 create_library.output_library_path 定义)。")

            print(f"最终用于加载的人脸库文件: {face_lib_path} (来源: {source_for_face_lib})")
            try:
                with open(face_lib_path, 'rb') as f:
                    library_features, library_labels_ids = pickle.load(f)
                print(f"人脸特征库 {face_lib_path} 加载成功 (包含 {library_features.shape[0]} 个特征)。")
            except Exception as e:
                raise RuntimeError(f"加载人脸特征库 {face_lib_path} 失败: {e}")

            input_feature_vec = features.numpy().flatten()
            similarities = np.dot(library_features, input_feature_vec) / (np.linalg.norm(library_features, axis=1) * np.linalg.norm(input_feature_vec))
            
            best_match_idx = np.argmax(similarities)
            confidence_or_similarity = similarities[best_match_idx]
            
            recognition_thresh = cmd_args.recognition_threshold if cmd_args.recognition_threshold is not None else config.infer.get('recognition_threshold', 0.5)

            if confidence_or_similarity >= recognition_thresh:
                predicted_id = library_labels_ids[best_match_idx]
                predicted_label_name = id_to_class_map.get(str(predicted_id), f"ID_{predicted_id}_未知")
            else:
                predicted_label_name = "图库外人员 (低于阈值)"
            print(f"输入: {target_image_path}, 预测: {predicted_label_name}, 余弦相似度: {confidence_or_similarity:.4f}, 阈值: {recognition_thresh}")

        elif loaded_loss_type == 'cross_entropy':
            print("CrossEntropy 推理模式: 进行分类...")
            if not model_head:
                raise RuntimeError("错误: CrossEntropy模式下模型头部 (model_head) 未正确初始化。")
            
            _, logits = model_head(features) # CrossEntropyHead 返回 (loss, logits)
            probabilities = paddle.nn.functional.softmax(logits, axis=1)
            
            confidence_or_similarity = float(paddle.max(probabilities, axis=1).numpy()[0])
            predicted_id = int(paddle.argmax(probabilities, axis=1).numpy()[0])
            predicted_label_name = id_to_class_map.get(str(predicted_id), f"ID_{predicted_id}_未知")
            print(f"输入: {target_image_path}, 预测: {predicted_label_name}, 置信度: {confidence_or_similarity:.4f}")
        
        else:
            raise ValueError(f"不支持的推理模式 (基于loss_type): {loaded_loss_type}")

    inference_end_time = time.time() # 记录推理结束时间
    inference_duration = inference_end_time - inference_start_time # 计算推理耗时
    print(f"推理耗时: {inference_duration:.4f} 秒")

    # --- 可视化结果 ---
    should_visualize = cmd_args.infer_visualize if cmd_args.infer_visualize is not None else config.infer.get('infer_visualize', True)

    if should_visualize:
        try:
            img_display = cv2.imread(target_image_path)
            if img_display is None:
                print(f"警告: 无法读取图像文件 {target_image_path}，跳过可视化。")
            
            text_lines = []
            text_lines.append(f"名称: {predicted_label_name}")
            if config.infer.get('display_confidence', True):
                if loaded_loss_type == 'arcface':
                    text_lines.append(f"相似度: {confidence_or_similarity:.4f}")
                else:
                    text_lines.append(f"置信度: {confidence_or_similarity:.4f}")
            if config.infer.get('display_inference_time', True):
                text_lines.append(f"耗时: {inference_duration:.4f}s")
            if config.infer.get('display_model_info', True):
                text_lines.append(f"模型: {loaded_model_type.upper()} ({loaded_loss_type.upper()})")

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            line_height = int(cv2.getTextSize("测试", font, font_scale, thickness)[0][1] * 1.5) # 预估行高
            text_x = 10
            text_y_start = line_height # 从第一行开始，预留顶部空间

            # 获取配置的颜色，并转换为 OpenCV 的 BGR 格式
            text_color_rgb = config.infer.get('text_color_rgb', [0, 255, 0]) # 默认为绿色
            background_color_rgb = config.infer.get('background_color_rgb', [0, 0, 0]) # 默认为黑色
            text_color_bgr = (text_color_rgb[2], text_color_rgb[1], text_color_rgb[0])
            background_color_bgr = (background_color_rgb[2], background_color_rgb[1], background_color_rgb[0])

            for i, line in enumerate(text_lines):
                text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
                current_text_y = text_y_start + i * line_height
                
                # 绘制带背景的文本
                cv2.rectangle(img_display, (text_x - 2, current_text_y - text_size[1] - 2), 
                              (text_x + text_size[0] + 2, current_text_y + 2), background_color_bgr, -1) 
                cv2.putText(img_display, line, (text_x, current_text_y), font, 
                            font_scale, text_color_bgr, thickness, cv2.LINE_AA)

            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            
            base_img_name = os.path.splitext(os.path.basename(target_image_path))[0]
            model_name_tag = f"{loaded_model_type}_{loaded_loss_type}"
            output_filename = f"infer_{model_name_tag}_{base_img_name}_{predicted_label_name.replace(' ', '_')}.png"
            output_path = os.path.join(results_dir, output_filename)
            
            cv2.imwrite(output_path, img_display)
            print(f"推理结果图像已保存到: {output_path}")

        except Exception as e_vis:
            print(f"可视化推理结果时发生错误: {e_vis}")

    print("推理完成。")

if __name__ == '__main__':
    # --- 命令行参数解析 --- 
    parser = argparse.ArgumentParser(description='人脸识别单图推理脚本')
    
    # 核心控制参数
    parser.add_argument('--config_path', type=str, default=None, 
                        help='指定YAML配置文件的路径。如果未提供，则使用脚本内部定义的默认路径。')
    parser.add_argument('--image_path', type=str, # required=True, 但由config_utils处理，允许从YAML加载
                        help='待识别的单张输入图像路径 (必需，除非在YAML中指定)。')
    parser.add_argument('--model_path', type=str, # required=True, 但由config_utils处理
                        help='训练好的模型文件路径 (.pdparams) (必需，除非在YAML中指定)。') 
    
    # 其他可覆盖配置文件的参数
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=None,
                        help='是否使用GPU进行推理。此命令行开关会覆盖配置文件中的 global_settings.use_gpu 设置。')
    parser.add_argument('--image_size', type=int, default=None,
                        help='输入图像预处理后的统一大小。此命令行参数会覆盖配置文件或模型自带的image_size设置。')
    
    # 推理特定参数 (覆盖配置文件中 infer: {...} 下的同名项)
    parser.add_argument('--label_file', type=str, default=None,
                        help='类别标签ID到名称映射的JSON文件路径 (覆盖 infer.label_file)。')
    parser.add_argument('--face_library_path', type=str, default=None,
                        help='ArcFace模型所需的人脸特征库文件路径 (.pkl) (覆盖 infer.face_library_path)。')
    parser.add_argument('--recognition_threshold', type=float, default=None,
                        help='ArcFace模型识别时的相似度阈值 (覆盖 infer.recognition_threshold)。')
    parser.add_argument('--infer_visualize', action=argparse.BooleanOptionalAction, default=None,
                        help='是否可视化识别结果并保存图像。此命令行开关会覆盖配置文件中的 infer.infer_visualize 设置。')

    cmd_line_args = parser.parse_args()

    # --- 配置加载与合并 --- 
    # 使用 config_utils.load_config 函数加载和合并配置。
    final_config = load_config(
        default_yaml_path='configs/default_config.yaml', 
        cmd_args_namespace=cmd_line_args
    )

    # 检查关键路径是否已配置 (model_path 和 image_path)
    if not final_config.model_path:
        parser.error("错误: 缺少模型文件路径。请通过 --model_path 命令行参数或在YAML配置文件中提供 model_path。")
    if not final_config.image_path:
        parser.error("错误: 缺少待识别图像路径。请通过 --image_path 命令行参数或在YAML配置文件中提供 image_path。")

    # 尝试设置matplotlib中文字体，以便在可视化结果中正确显示中文名称
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用 SimHei 字体 (黑体)
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    except Exception as e_font:
        print(f"提示: 设置matplotlib中文字体SimHei失败: {e_font}。可视化结果中的中文可能显示为乱码。")
        print(f"       请确保系统中安装了SimHei字体，或者在代码中指定其他可用的中文字体。")

    # 执行推理
    try:
        infer(final_config, cmd_line_args)
    except FileNotFoundError as e:
        print(f"推理失败: {e}")
    except RuntimeError as e:
        print(f"推理时发生运行时错误: {e}")
    except ValueError as e:
        print(f"配置错误: {e}")
    except Exception as e:
        print(f"发生意外错误: {e}")