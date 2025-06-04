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
from model_factory import get_backbone, get_head   # 导入模型工厂函数
from utils.image_processing import process_image # 从共享模块导入
import time # 导入 time 模块
import platform # 导入 platform 模块用于判断操作系统
from datetime import datetime # 导入 datetime 模块

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

def _process_frame_and_infer(frame: np.ndarray, config: ConfigObject, id_to_class_map: dict, loaded_model_type: str, loaded_loss_type: str, loaded_image_size: int, backbone_instance: paddle.nn.Layer, head_module_instance: paddle.nn.Layer | None, library_features: paddle.Tensor | None, library_labels: list | None, recognition_threshold: float) -> tuple[np.ndarray, str, float, float]:
    """
    处理单个图像帧，执行推理，并返回带标注的图像、预测结果、置信度和推理耗时。
    这个函数将被 infer 主函数和实时捕获逻辑调用。
    """
    if frame is None:
        print("警告: 接收到空帧，跳过处理。")
        return None, "", 0.0, 0.0
    
    # 将OpenCV读取的BGR图像转换为RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 使用从元数据或配置中加载的 image_size, mean, std
    image_mean = config.dataset_params.mean 
    image_std = config.dataset_params.std
    
    # 对图像进行预处理
    preprocessed_image_np = process_image(
        img_data=frame_rgb, # 直接传递 numpy 数组
        target_size=loaded_image_size,
        mean_rgb=image_mean,
        std_rgb=image_std,
        is_bgr_input=False # 传入的已经是RGB
    )
    img_tensor = paddle.to_tensor(preprocessed_image_np)

    predicted_label_name = "未知"
    confidence_or_similarity = 0.0

    inference_start_time = time.time() 

    with paddle.no_grad():
        features = backbone_instance(img_tensor)

        if loaded_loss_type == 'arcface':
            # ArcFace 推理逻辑 (与原 infer.py 相似，但移除了特征库加载，因为它在外部已加载)
            if library_features is None or library_labels is None:
                print("错误: ArcFace模式下，特征库未加载或为空。")
                return frame, "无法推理", 0.0, 0.0

            input_feature_vec = features.numpy().flatten()
            similarities = np.dot(library_features, input_feature_vec) / (np.linalg.norm(library_features, axis=1) * np.linalg.norm(input_feature_vec))
            
            best_match_idx = np.argmax(similarities)
            confidence_or_similarity = similarities[best_match_idx]
            
            if confidence_or_similarity >= recognition_threshold:
                predicted_id = library_labels[best_match_idx]
                predicted_label_name = id_to_class_map.get(str(predicted_id), f"ID_{predicted_id}_未知")
            else:
                predicted_label_name = "图库外人员 (低于阈值)"
            # print(f"预测: {predicted_label_name}, 余弦相似度: {confidence_or_similarity:.4f}, 阈值: {recognition_threshold}")

        elif loaded_loss_type == 'cross_entropy':
            # CrossEntropy 推理逻辑 (与原 infer.py 相似)
            if not head_module_instance:
                print("错误: CrossEntropy模式下模型头部 (head_module_instance) 未正确初始化。")
                return frame, "无法推理", 0.0, 0.0
            
            outputs = head_module_instance(features)
            logits = None
            if isinstance(outputs, tuple) and len(outputs) == 2:
                 _, logits = outputs
            elif isinstance(outputs, paddle.Tensor):
                 logits = outputs

            if logits is None:
                 print("错误: CrossEntropy 头部未返回 logits。")
                 return frame, "无法推理", 0.0, 0.0

            probabilities = paddle.nn.functional.softmax(logits, axis=1)
            
            confidence_or_similarity = float(paddle.max(probabilities, axis=1).numpy()[0])
            predicted_id = int(paddle.argmax(probabilities, axis=1).numpy()[0])
            predicted_label_name = id_to_class_map.get(str(predicted_id), f"ID_{predicted_id}_未知")
            # print(f"预测: {predicted_label_name}, 置信度: {confidence_or_similarity:.4f}")
        
        else:
            print(f"不支持的推理模式 (基于loss_type): {loaded_loss_type}")
            return frame, "不支持的模型", 0.0, 0.0

    inference_duration = time.time() - inference_start_time # 计算推理耗时
    # print(f"推理耗时: {inference_duration:.4f} 秒")

    # --- 可视化结果 (移入此函数，对帧进行标注) ---
    if config.infer.get('infer_visualize', True):
        img_display = frame.copy() # 复制原始帧进行标注
        
        text_lines = []
        text_lines.append(f"Name: {predicted_label_name}")
        if config.infer.get('display_confidence', True):
            if loaded_loss_type == 'arcface':
                text_lines.append(f"Similarity: {confidence_or_similarity:.4f}")
            else:
                text_lines.append(f"Confidence: {confidence_or_similarity:.4f}")
        if config.infer.get('display_inference_time', True):
            text_lines.append(f"Time: {inference_duration:.4f}s")
        if config.infer.get('display_model_info', True):
            text_lines.append(f"Model: {loaded_model_type.upper()} ({loaded_loss_type.upper()})")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0 # 增大字体大小
        thickness = 2    # 增大字体粗细
        line_height = int(cv2.getTextSize("Test", font, font_scale, thickness)[0][1] * 1.5) 
        text_x = 10
        text_y_start = line_height 

        # 获取配置的颜色，并转换为 OpenCV 的 BGR 格式
        text_color_rgb = config.infer.get('text_color_rgb', [0, 255, 0]) 
        background_color_rgb = config.infer.get('background_color_rgb', [0, 0, 0]) 
        text_color_bgr = (text_color_rgb[2], text_color_rgb[1], text_color_rgb[0])
        background_color_bgr = (background_color_rgb[2], background_color_rgb[1], background_color_rgb[0])

        for i, line in enumerate(text_lines):
            text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
            current_text_y = text_y_start + i * line_height
            
            cv2.rectangle(img_display, (text_x - 2, current_text_y - text_size[1] - 2), 
                          (text_x + text_size[0] + 2, current_text_y + 2), background_color_bgr, -1) 
            cv2.putText(img_display, line, (text_x, current_text_y), font, 
                        font_scale, text_color_bgr, thickness, cv2.LINE_AA)

        return img_display, predicted_label_name, confidence_or_similarity, inference_duration
    else:
        return frame, predicted_label_name, confidence_or_similarity, inference_duration

def infer(config: ConfigObject, cmd_args: argparse.Namespace):
    """
    主推理函数，加载模型、标签和目标图片，进行预测。支持单图推理或实时摄像头捕获。
    """
    # --- 设置设备 ---
    use_gpu_flag = cmd_args.use_gpu if cmd_args.use_gpu is not None else config.use_gpu
    use_gpu_flag = use_gpu_flag and paddle.is_compiled_with_cuda()
    paddle.set_device('gpu' if use_gpu_flag else 'cpu')

    # --- 确定模型和特征库路径 (基于 active_infer_config 或命令行) ---
    logs_base_dir = config.get('visualdl_log_dir', 'logs')

    # 获取当前推理的配置类型，从命令行或配置文件中 active_infer_config 字段读取
    active_infer_config_name = cmd_args.active_infer_config # 优先从命令行参数获取
    if active_infer_config_name is None:
        # 如果命令行未指定，则从配置文件中获取 (config_utils.py 已确保其存在于顶层)
        active_infer_config_name = config.get('active_infer_config', None)

    # 所有的模型、损失、优化器、学习率调度器类型都已在 load_config 中处理，
    # 并扁平化到 config 对象的顶层。
    # 因此，无论是否指定了 active_infer_config，直接从 config 对象获取即可。
    infer_model_type = config.get('model_type')
    infer_loss_type = config.get('loss_type')
    infer_optimizer_type = config.get('optimizer_type')
    infer_lr_scheduler_type = config.get('lr_scheduler_type')
    
    if not all([infer_model_type, infer_loss_type, infer_optimizer_type, infer_lr_scheduler_type]):
        raise ValueError("错误: 无法确定用于推理的模型类型、损失类型、优化器类型或学习率调度器类型。请检查 active_infer_config 或 global_settings。")

    # 动态构建模型组合目录名称，用于查找模型和特征库
    # 假设模型保存路径模式为 logs/{model_type}__{loss_type}__{optimizer}__{scheduler}__{lr_formatted}__{wd_formatted}/{timestamp}/checkpoints/best_model_model_checkpoint.pdparams
    # 需要从 config 中获取 learning_rate 和 weight_decay 来构建完整的目录名
    lr_value = config.get('learning_rate', 0.001)
    wd_value = config.optimizer_params.get('weight_decay', 0.0) if hasattr(config, 'optimizer_params') else 0.0

    lr_formatted = f"lr{str(lr_value).replace('0.', '')}"
    wd_formatted = f"wd{str(wd_value).replace('0.', '')}"
    
    # 构建训练脚本使用的完整组合目录名称前缀
    # 注意：这里假设 infer.py 会查找与某个训练配置完全匹配的模型目录
    if active_infer_config_name:
        # 如果指定了 active_infer_config，直接使用其名称作为目标目录前缀
        target_combo_dir_prefix = active_infer_config_name
    else:
        # 否则，根据参数构建组合目录前缀
        target_combo_dir_prefix = f"{infer_model_type}__{infer_loss_type}__{infer_optimizer_type}__{infer_lr_scheduler_type}__{lr_formatted}__{wd_formatted}"
    
    print(f"正在搜索匹配的模型和特征库 (目标组合: {target_combo_dir_prefix})...")

    found_model_path = None
    found_face_library_path = None
    search_base_path = logs_base_dir # 通常是 'logs'

    # 遍历 logs 目录下的所有训练组合目录
    for combo_dir in sorted(os.listdir(search_base_path), reverse=True): # 倒序，优先最新训练的组合
        # 精确匹配目录名称，确保找到正确的模型
        if combo_dir.startswith(target_combo_dir_prefix):
            full_combo_path = os.path.join(search_base_path, combo_dir)
            if os.path.isdir(full_combo_path):
                # 遍历每个组合目录下的时间戳目录
                for timestamp_dir in sorted(os.listdir(full_combo_path), reverse=True): # 倒序，优先最新时间戳
                    full_timestamp_path = os.path.join(full_combo_path, timestamp_dir)
                    checkpoints_path = os.path.join(full_timestamp_path, "checkpoints")
                    expected_model_filename = "best_model_model_checkpoint.pdparams" # 训练脚本中硬编码的名称
                    potential_model_path = os.path.join(checkpoints_path, expected_model_filename)

                    if os.path.exists(potential_model_path):
                        found_model_path = potential_model_path
                        print(f"已找到匹配的模型: {found_model_path}")
                        
                        # 如果是 ArcFace 模型，还需要查找特征库
                        if infer_loss_type == 'arcface':
                            expected_lib_filename = config.create_library.get('output_library_path', 'face_library.pkl')
                            # 特征库通常保存在模型文件所在的目录
                            potential_lib_path = os.path.join(os.path.dirname(found_model_path), expected_lib_filename)
                            if os.path.exists(potential_lib_path):
                                found_face_library_path = potential_lib_path
                                print(f"已找到匹配的特征库: {found_face_library_path}")
                            else:
                                print(f"警告: ArcFace 模型 {found_model_path} 找到了，但预期的特征库 {potential_lib_path} 未找到。请确保已运行 create_face_library.py。")
                        break # 找到最新模型，跳出时间戳循环
                if found_model_path: # 如果在这个组合下找到了模型，就不用再搜索其他组合了
                    break

    # 命令行参数 model_path 优先级最高
    model_weights_path = cmd_args.model_path or found_model_path
    if not model_weights_path:
        raise ValueError("错误: 未能找到或指定模型权重文件路径。请确保已训练模型且 logs 目录结构正确，或者通过 --model_path 指定。")
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"错误: 指定的模型权重文件未找到: {model_weights_path}")

    print(f"将从模型文件 {model_weights_path} 加载模型。")

    loaded_face_library_features = None
    loaded_face_library_labels = None
    
    # 只有当损失类型是 ArcFace 时，才需要处理特征库
    if infer_loss_type == 'arcface':
        # 命令行参数 face_library_path 优先级最高
        face_lib_path_candidate = cmd_args.face_library_path or found_face_library_path

        if not face_lib_path_candidate:
            print("\n错误: ArcFace 模型推理需要人脸特征库，但未能找到或指定。\n")
            print("请尝试运行以下命令来创建特征库 (如果您的模型已经训练完成):\n")
            # 确保 cmd_args.config_path 属性安全访问
            config_path_for_cmd = getattr(cmd_args, 'config_path', 'configs/48_config.yaml')
            suggested_cmd = (
                f"python create_face_library.py "
                f"--config_path {config_path_for_cmd} "
                f"--active_config {active_infer_config_name} "
                f"--model_path {model_weights_path}" # 使用实际确定的模型路径
            )
            print(f"\n   {suggested_cmd}\n")
            raise ValueError("ArcFace 模型推理需要特征库。")

        if not os.path.exists(face_lib_path_candidate):
            print(f"\n错误: ArcFace 模型推理指定的特征库文件未找到: {face_lib_path_candidate}\n")
            print("请检查文件路径是否正确，或者尝试运行以下命令来创建特征库 (如果您的模型已经训练完成):\n")
            config_path_for_cmd = getattr(cmd_args, 'config_path', 'configs/48_config.yaml')
            suggested_cmd = (
                f"python create_face_library.py "
                f"--config_path {config_path_for_cmd} "
                f"--active_config {active_infer_config_name} "
                f"--model_path {model_weights_path}"
            )
            print(f"\n   {suggested_cmd}\n")
            raise FileNotFoundError("ArcFace 模型推理指定的特征库文件未找到。")
        
        print(f"将从特征库文件 {face_lib_path_candidate} 加载人脸特征库。")
        try:
            with open(face_lib_path_candidate, 'rb') as f:
                feature_library_data = pickle.load(f)
            loaded_face_library_features = paddle.to_tensor(feature_library_data[0], dtype='float32') # numpy array to Paddle Tensor
            loaded_face_library_labels = feature_library_data[1].tolist() # numpy array to list
            print(f"人脸特征库 {face_lib_path_candidate} 加载成功 (包含 {loaded_face_library_features.shape[0]} 个特征)。")
        except Exception as e:
            raise RuntimeError(f"加载人脸特征库 {face_lib_path_candidate} 失败: {e}")
    # End of ArcFace-specific feature library loading

    # --- 尝试从模型元数据加载配置 (作为模型构建的权威来源) ---
    model_type_from_metadata = None
    loss_type_from_metadata = None
    image_size_from_metadata = None
    num_classes_from_metadata = None
    model_specific_params_from_metadata = {}
    loss_specific_params_from_metadata = {}
    
    metadata_path = model_weights_path.replace('.pdparams', '.json')
    source_of_model_config = ""

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 严格检查并从元数据加载所有模型构建所需的核心参数
            if metadata.get('model_type') and metadata.get('loss_type') and \
               metadata.get('image_size') is not None and metadata.get('num_classes') is not None:
                model_type_from_metadata = metadata['model_type']
                loss_type_from_metadata = metadata['loss_type']
                image_size_from_metadata = metadata['image_size']
                num_classes_from_metadata = metadata['num_classes']
                
                # 安全地获取嵌套的参数字典
                model_specific_params_from_metadata = metadata.get('model_specific_params', {})
                if not isinstance(model_specific_params_from_metadata, dict): # Ensure it's a dict
                    model_specific_params_from_metadata = {}

                loss_specific_params_from_metadata = metadata.get('loss_specific_params', {})
                if not isinstance(loss_specific_params_from_metadata, dict): # Ensure it's a dict
                    loss_specific_params_from_metadata = {}

                source_of_model_config = f"元数据文件 ({metadata_path})"
                print(f"已成功从元数据文件 {metadata_path} 加载模型构建参数。")
            else:
                print(f"警告: 模型元数据文件 {metadata_path} 不完整，缺少核心配置项。将回退到YAML配置。")
        except Exception as e:
            print(f"警告: 加载或解析模型元数据文件 {metadata_path} 失败: {e}。将回退到YAML配置。")

    # 如果元数据加载不成功或不完整，则回退到YAML配置 (全局设置或 active_infer_config)
    if not model_type_from_metadata:
        print("信息: 未能从模型元数据加载完整的模型构建参数，将使用YAML配置作为补充或主要来源。")
        # 从当前ConfigObject获取参数，这些参数已经过优先级合并
        model_type_to_use = config.get('model_type')
        loss_type_to_use = config.get('loss_type')
        image_size_to_use = config.get('image_size')
        num_classes_to_use = config.get('num_classes')
        
        # 从ConfigObject中获取嵌套参数，确保是字典
        backbone_params_obj = config.model.get(f'{model_type_to_use}_params', ConfigObject({}))
        model_specific_params_to_use = backbone_params_obj.to_dict() if isinstance(backbone_params_obj, ConfigObject) else backbone_params_obj
        
        head_params_obj = config.loss.get(f'{loss_type_to_use}_params', ConfigObject({}))
        loss_specific_params_to_use = head_params_obj.to_dict() if isinstance(head_params_obj, ConfigObject) else head_params_obj
        
        source_of_model_config = f"YAML 配置 ({config.get('active_infer_config', 'global_settings')})"
        print(f"信息: 模型参数的最终来源或补充来源: {source_of_model_config}")
        
        # Final check for essential parameters after fallback
        if not all([model_type_to_use, loss_type_to_use, image_size_to_use is not None, num_classes_to_use is not None]):
             raise ValueError("错误: 无法从元数据或YAML配置中确定模型构建所需的核心配置。请检查模型文件和配置。")
    else: # Metadata was successfully loaded and complete
        model_type_to_use = model_type_from_metadata
        loss_type_to_use = loss_type_from_metadata
        image_size_to_use = image_size_from_metadata
        num_classes_to_use = num_classes_from_metadata
        model_specific_params_to_use = model_specific_params_from_metadata
        loss_specific_params_to_use = loss_specific_params_from_metadata
        print(f"信息: 模型参数已成功从元数据 ({source_of_model_config}) 确定。")
        
    print(f"--- 模型构建配置来源: {source_of_model_config} ---")
    print(f"  Model Type: {model_type_to_use}")
    print(f"  Loss Type: {loss_type_to_use}")
    print(f"  Image Size: {image_size_to_use}")
    print(f"  Num Classes: {num_classes_to_use}")
    print(f"  Model Params: {model_specific_params_to_use}")
    print(f"  Loss Params: {loss_specific_params_to_use}")
    print("--------------------------------------------------")

    # --- 构建模型 ---
    model_backbone, backbone_out_dim = get_backbone(
        config_model_params=model_specific_params_to_use, 
        model_type_str=model_type_to_use,
        image_size=image_size_to_use
    )
    print(f"骨干网络 ({model_type_to_use.upper()}) 构建成功，期望输入图像尺寸: {image_size_to_use}, 输出特征维度: {backbone_out_dim}")

    model_head = None
    if loss_type_to_use == 'cross_entropy': 
        model_head = get_head(
            config_loss_params=loss_specific_params_to_use, 
            loss_type_str=loss_type_to_use,
            in_features=backbone_out_dim,
            num_classes=num_classes_to_use
        )
        print(f"头部模块 ({loss_type_to_use.upper()}) 构建成功，输入特征维度: {backbone_out_dim}, 输出类别数: {num_classes_to_use}")

    # --- 加载模型权重 ---
    full_state_dict = paddle.load(model_weights_path)
    
    backbone_state_dict = {k.replace('backbone.', '', 1): v for k, v in full_state_dict.items() if k.startswith('backbone.')}
    if backbone_state_dict:
        model_backbone.set_state_dict(backbone_state_dict)
        print(f"骨干网络权重从 {model_weights_path} 加载成功。")
    else:
        try:
            model_backbone.set_state_dict(full_state_dict)
            print(f"骨干网络权重 (可能为直接保存的骨干模型) 从 {model_weights_path} 加载成功。")
        except Exception as e_direct_bb_load:
            raise RuntimeError(f"错误: 在模型文件 {model_weights_path} 中未找到 'backbone.' 前缀的权重，并且直接加载整个状态字典到骨干网络失败: {e_direct_bb_load}。请确保模型文件与期望的结构一致。")

    if model_head:
        head_state_dict = {k.replace('head.', '', 1): v for k, v in full_state_dict.items() if k.startswith('head.')}
        if head_state_dict:
            model_head.set_state_dict(head_state_dict)
            print(f"头部模块 ({loss_type_to_use}) 权重从 {model_weights_path} 加载成功。")
        else:
            print(f"警告: 头部模块 ({loss_type_to_use}) 已实例化，但在模型文件 {model_weights_path} 中未找到 'head.' 前缀的权重。头部将使用其默认初始化权重。")

    model_backbone.eval()
    if model_head:
        model_head.eval()

    # --- 加载类别标签文件 ---
    label_file_from_cmd = getattr(cmd_args, 'label_file', None)
    label_file_to_load = label_file_from_cmd or config.infer.get('label_file', 'readme.json')
    actual_label_file_path = os.path.join(config.data_dir, config.class_name, label_file_to_load)
    if not os.path.exists(actual_label_file_path):
         raise FileNotFoundError(f"错误: 类别标签文件 {actual_label_file_path} 未找到。")

    try:
        with open(actual_label_file_path, 'r', encoding='utf-8') as f:
            full_meta = json.load(f)
            class_to_id_map = full_meta.get('class_to_id_map')
            if class_to_id_map is None:
                raise ValueError("readme.json 中未找到 'class_to_id_map'。")
            id_to_class_map = {str(v): k for k, v in class_to_id_map.items()} # Invert for easy lookup
        print(f"类别标签文件 {actual_label_file_path} 加载成功 ({len(id_to_class_map)} 个类别)。")
    except Exception as e:
        raise RuntimeError(f"加载或解析类别标签文件 {actual_label_file_path} 失败: {e}")

    # --- 执行推理 --- 
    if cmd_args.live_capture:
        print("信息: 启用实时摄像头捕获模式。按 'q' 键退出。")
        # 摄像头索引，0通常是默认摄像头
        camera_index = cmd_args.camera_index if cmd_args.camera_index is not None else 0 
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"错误: 无法打开摄像头 {camera_index}。请检查摄像头连接或权限。")
        
        # 设置捕获帧的分辨率（可选，可能需要根据摄像头支持的调整）
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("警告: 无法从摄像头读取帧，可能已断开连接。")
                    break
                
                # 调用统一的处理和推理函数
                processed_frame, pred_name, conf_sim, infer_time = _process_frame_and_infer(
                    frame=frame,
                    config=config,
                    id_to_class_map=id_to_class_map,
                    loaded_model_type=model_type_to_use,
                    loaded_loss_type=loss_type_to_use,
                    loaded_image_size=image_size_to_use,
                    backbone_instance=model_backbone,
                    head_module_instance=model_head,
                    library_features=loaded_face_library_features,
                    library_labels=loaded_face_library_labels,
                    recognition_threshold=config.infer.get('recognition_threshold', 0.5)
                )

                if processed_frame is not None:
                    cv2.imshow('Live Inference (Press Q to Quit)', processed_frame)

                    # 保存实时捕获的帧（可选，或仅保存识别成功的帧）
                    if cmd_args.save_live_results and pred_name != "未知" and pred_name != "图库外人员 (低于阈值)":
                        results_dir = "results_live_capture"
                        os.makedirs(results_dir, exist_ok=True)
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] # 毫秒级时间戳
                        output_filename = f"live_infer_{pred_name.replace(' ', '_')}_{timestamp_str}.png"
                        output_path = os.path.join(results_dir, output_filename)
                        cv2.imwrite(output_path, processed_frame)
                        print(f"信息: 实时推理结果图像已保存到: {output_path}")

                if cv2.waitKey(1) & 0xFF == ord('q'): # 按 'q' 键退出
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    else: # 单张图片推理模式 (现有逻辑) - 调整为调用 _process_frame_and_infer
        target_image_path = cmd_args.image_path or config.infer.image_path
        if not target_image_path or not os.path.exists(target_image_path):
            raise FileNotFoundError(f"错误: 输入图像 --image_path '{target_image_path}' 未指定或未找到。")
        
        print(f"信息: 正在处理单张图片: {target_image_path}")
        img_to_infer = cv2.imread(target_image_path)
        if img_to_infer is None:
            raise IOError(f"错误: 无法读取图像文件 {target_image_path}。")

        processed_img, pred_name, conf_sim, infer_time = _process_frame_and_infer(
            frame=img_to_infer,
            config=config,
            id_to_class_map=id_to_class_map,
            loaded_model_type=model_type_to_use,
            loaded_loss_type=loss_type_to_use,
            loaded_image_size=image_size_to_use,
            backbone_instance=model_backbone,
            head_module_instance=model_head,
            library_features=loaded_face_library_features,
            library_labels=loaded_face_library_labels,
            recognition_threshold=config.infer.get('recognition_threshold', 0.5)
        )

        # ---- 新增打印，确保 acceptance.sh 可以捕获 ----
        print(f"Name: {pred_name}")
        if loss_type_to_use == 'arcface':
            print(f"Similarity: {conf_sim:.4f}")
        else: # Assuming cross_entropy or similar
            print(f"Confidence: {conf_sim:.4f}")
        # ---- 结束新增打印 ----

        if processed_img is not None:
            # 单图模式下直接保存结果
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            
            base_img_name = os.path.splitext(os.path.basename(target_image_path))[0]
            model_name_tag = f"{model_type_to_use}_{loss_type_to_use}"
            output_filename = f"infer_{model_name_tag}_{base_img_name}_{pred_name.replace(' ', '_')}.png"
            output_path = os.path.join(results_dir, output_filename)
            
            cv2.imwrite(output_path, processed_img)
            print(f"信息: 推理结果图像已保存到: {output_path}")

    print("推理完成。")

if __name__ == '__main__':
    print("信息: 主程序块开始，准备解析命令行参数。")
    parser = argparse.ArgumentParser(description='人脸识别单图推理脚本')
    
    parser.add_argument('--config_path', type=str, default='configs/48_config.yaml', # 将默认配置文件路径改为 48_config.yaml
                        help='指定YAML配置文件的路径。默认值为 configs/48_config.yaml。')
    parser.add_argument('--active_infer_config', type=str, default=None,
                        help='指定要激活的推理配置块名称 (覆盖YAML中的 active_infer_config)。')
    parser.add_argument('--image_path', type=str, default=None,
                        help='待识别的单张输入图像路径。在 --live_capture 模式下可选，否则必需。')
    parser.add_argument('--model_path', type=str, default=None,
                        help='训练好的模型文件路径 (.pdparams)。如果未指定，将根据 active_infer_config 自动搜索。') 
    parser.add_argument('--face_library_path', type=str, default=None,
                        help='[ArcFace Only] 用于比对的特征库文件 (.pkl) 的路径。如果未指定，将根据 active_infer_config 自动搜索。')
    
    # 新增摄像头和实时捕获参数
    parser.add_argument('--live_capture', action='store_true', 
                        help='启用摄像头实时捕获并进行推理。启用此模式时，--image_path 可选。')
    parser.add_argument('--camera_index', type=int, default=0, 
                        help='指定要使用的摄像头设备索引（通常0是默认摄像头）。')
    parser.add_argument('--save_live_results', action='store_true', 
                        help='在实时捕获模式下，保存每一帧的推理结果图像到 results_live_capture 目录。')

    # 其他可覆盖配置文件的参数 (保持不变)
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=None,
                        help='是否使用GPU进行推理。此命令行开关会覆盖配置文件中的 global_settings.use_gpu 设置。')
    parser.add_argument('--image_size', type=int, default=None,
                        help='输入图像预处理后的统一大小。此命令行参数会覆盖配置文件或模型自带的image_size设置。')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='类别数。覆盖配置文件, 影响模型加载。')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据集根目录。覆盖配置文件。')
    parser.add_argument('--class_name', type=str, default=None,
                        help='数据集子目录名。覆盖配置文件。')
    parser.add_argument('--log_interval', type=int, default=None,
                        help='打印日志的间隔批次数。覆盖配置文件。')
    parser.add_argument('--recognition_threshold', type=float, default=None,
                        help='[ArcFace Only] 人脸识别比对的相似度阈值 (覆盖配置文件)。')
    parser.add_argument('--infer_visualize', action=argparse.BooleanOptionalAction, default=None,
                        help='是否可视化识别结果并保存图像。此命令行开关会覆盖配置文件中的 infer.infer_visualize 设置。')
    
    cmd_line_args = parser.parse_args()

    # --- 配置加载与合并 --- 
    final_config = load_config(
        default_yaml_path=cmd_line_args.config_path, 
        cmd_args_namespace=cmd_line_args
    )

    print(f"信息: infer函数内, 使用的 active_infer_config 名称: {cmd_line_args.active_infer_config}")

    # 检查关键路径是否已配置
    if not cmd_line_args.live_capture and (not cmd_line_args.image_path and not final_config.infer.get('image_path')):
        parser.error("错误: 缺少待识别图像路径。在非实时捕获模式下，必须通过 --image_path 命令行参数或在YAML配置文件中提供 image_path。")
    
    # 即使在实时捕获模式下，如果 active_infer_config 未指定，仍需要模型类型和损失类型来自动加载模型
    if not final_config.get('active_infer_config') and not (final_config.get('model_type') and final_config.get('loss_type')):
        parser.error("错误: 无法确定用于推理的模型类型或损失类型。请在 YAML 中设置 active_infer_config 或 global_settings.model_type/loss_type。")

    # 执行推理
    try:
        print(f"信息: 即将调用 infer() 函数处理推理任务。")
        infer(final_config, cmd_line_args)
    except FileNotFoundError as e:
        print(f"错误: 推理失败: {e}")
    except RuntimeError as e:
        print(f"错误: 推理时发生运行时错误: {e}")
    except ValueError as e:
        print(f"错误: 配置错误: {e}")
    except Exception as e:
        print(f"错误: 发生意外错误: {e}")
    
    print("推理脚本执行完毕。")