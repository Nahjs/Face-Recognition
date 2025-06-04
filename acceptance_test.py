# acceptance_test.py
# 该脚本用于在"验收集" (acceptance set) 上评估已训练的人脸识别模型的性能。
#
# 主要流程:
# 1.  加载配置文件 (YAML + 命令行参数)。
# 2.  设置运行设备 (CPU/GPU)。
# 3.  根据配置实例化骨干网络 (backbone) 和模型头部 (head)。
# 4.  从指定的检查点文件加载训练好的模型权重。
# 5.  使用 MyReader.py 创建验收集的数据加载器。
# 6.  执行评估循环：
#     - 将模型设置为评估模式。
#     - 在 paddle.no_grad() 上下文中进行。
#     - 遍历验收数据加载器中的所有批次。
#     - 执行前向传播，获取特征和头部输出。
#     - 计算损失值 (可选，但有助于了解模型在验收集上的表现)。
#     - 计算准确率。
# 7.  汇总并打印在整个验收集上的平均损失和准确率。
# 8.  (可选) 可以扩展此脚本以保存每个样本的预测结果或失败案例。

import os
import argparse
import paddle
import numpy as np # type: ignore
import json # 用于加载模型元数据或保存结果
import time
import pickle # 导入 pickle 用于加载特征库
import inspect # 导入 inspect 用于检查 forward 函数参数
import sys # 导入 sys 用于脚本退出状态

import MyReader # 数据读取模块
from config_utils import load_config, ConfigObject # 配置加载工具
from model_factory import get_backbone, get_head # 同时导入 backbone 和 head
# from model_factory import get_head # 不再需要头部模块
# from .compare import calculate_cosine_similarity # 假设 compare.py 中有相似度计算函数，或自己实现
import paddle.nn.functional as F # 用于可能的相似度计算（如Cosine Similarity）和分类损失计算

def run_acceptance_evaluation(config: ConfigObject, cmd_args: argparse.Namespace):
    """
    执行基于特征比对 (ArcFace) 或分类 (CrossEntropy) 的模型的验收测试。

    Args:
        config (ConfigObject): 合并后的配置对象。
        cmd_args (argparse.Namespace): 命令行参数，应包含 trained_model_path，ArcFace 需要 feature_library_path。
    """
    # 1. 设置运行设备
    if config.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
        print("使用 GPU 进行验收测试。")
    else:
        paddle.set_device('cpu')
        print("使用 CPU 进行验收测试。")

    # 2. 创建测试数据加载器 (使用原 eval 列表，现在是合并的测试集)
    print("正在创建测试数据加载器 (用于验收测试)...")
    try:
        # 使用 mode='eval' 来加载配置中 dataset_params.eval_list 指定的文件
        # CreateDataList.py 修改后，eval_list (test.list) 现在是 30% 的测试集
        test_loader = MyReader.create_data_loader(
            config=config,
            mode='test' # 修改为 'test' 以加载测试集
        )
        print(f"测试数据加载器创建成功，共 {len(test_loader.dataset)} 个样本。")
    except Exception as e:
        print(f"创建测试数据加载器失败: {e}")
        return

    if len(test_loader.dataset) == 0:
        print("警告: 测试数据集为空，无法进行验收。请检查测试列表文件和路径配置。")
        return

    # 3. 加载模型权重并获取模型配置
    #    trained_model_path 应指向包含骨干网络权重的 .pdparams 文件。
    model_weights_path = cmd_args.trained_model_path # model_weights_path 必须通过命令行指定
    if not model_weights_path:
        print("错误: 未通过命令行参数 --trained_model_path 指定模型权重路径。")
        return
    if not os.path.exists(model_weights_path):
        print(f"错误: 指定的模型权重文件不存在: {model_weights_path}")
        return

    print(f"将从以下路径加载模型权重 (仅骨干网络): {model_weights_path}")
    try:
        checkpoint_data = paddle.load(model_weights_path)
        if not isinstance(checkpoint_data, dict): # 确保加载的是 state_dict 字典
             # 如果不是字典，可能是旧格式或其他问题，尝试加载为整个模型（但不推荐）
             # 或者直接退出，因为我们预期加载的是 CombinedModel 的 state_dict
             print(f"错误: 加载的模型权重文件 {model_weights_path} 不是预期的 state_dict 格式。")
             return
    except Exception as e:
        print(f"加载模型权重文件 {model_weights_path} 失败: {e}")
        return

    # --- 获取模型构建参数 (同训练脚本逻辑，优先从元数据加载) ---
    model_type_to_use = None
    loss_type_to_use = None
    num_classes_to_use = None
    image_size_to_use = None
    backbone_specific_params_to_use = {}
    head_specific_params_to_use = {}
    source_of_model_config = "未确定"

    # 尝试加载与模型权重配对的 .json 元数据文件
    metadata_json_path = model_weights_path.replace('.pdparams', '.json')
    if os.path.exists(metadata_json_path):
        print(f"尝试从元数据文件加载模型配置: {metadata_json_path}")
        try:
            with open(metadata_json_path, 'r', encoding='utf-8') as f_meta:
                metadata = json.load(f_meta)
            # 从元数据提取所需配置 (与 train.py 中 CheckpointManager 保存的元数据对应)
            model_type_to_use = metadata.get('model_type')
            loss_type_to_use = metadata.get('loss_type')
            num_classes_to_use = metadata.get('num_classes')
            image_size_to_use = metadata.get('image_size')
            # model_specific_params 和 loss_specific_params 在元数据中可能是嵌套的字典
            # config.model.get(f'{final_config.model_type}_params', {})
            if isinstance(metadata.get('model_specific_params'), dict):
                backbone_specific_params_to_use = metadata['model_specific_params']
            if isinstance(metadata.get('loss_specific_params'), dict):
                head_specific_params_to_use = metadata['loss_specific_params']
            
            if all([model_type_to_use, loss_type_to_use, num_classes_to_use is not None, image_size_to_use is not None]):
                source_of_model_config = f"元数据文件 ({metadata_json_path})"
                print(f"成功从元数据文件加载模型配置。")
            else:
                print(f"警告: 元数据文件 {metadata_json_path} 不完整，缺少核心模型配置。将回退。")
                # 部分加载成功也可能导致后续使用 fallback 值，所以重置确保一致性
                model_type_to_use, loss_type_to_use, num_classes_to_use, image_size_to_use = None, None, None, None 
        except Exception as e_meta_load:
            print(f"加载或解析元数据文件 {metadata_json_path} 失败: {e_meta_load}。将回退。")
            model_type_to_use, loss_type_to_use, num_classes_to_use, image_size_to_use = None, None, None, None 

    # 如果从元数据文件加载失败或不完整，则回退到当前脚本的配置
    if source_of_model_config == "未确定":
        source_of_model_config = "当前脚本的 final_config (回退机制)"
        print(f"警告: 未能从配对的 .json 文件成功加载完整模型配置。将使用 {source_of_model_config}。"
              f"请确保此配置与被测模型 {os.path.basename(model_weights_path)} 的训练时配置严格匹配，否则可能导致错误或不准确的评估。")
        model_type_to_use = config.model_type
        loss_type_to_use = config.loss_type
        num_classes_to_use = config.num_classes
        image_size_to_use = config.image_size
        # 从当前 config 对象安全地获取参数
        backbone_params_obj = config.model.get(f'{model_type_to_use}_params', ConfigObject({}))
        backbone_specific_params_to_use = backbone_params_obj.to_dict() if isinstance(backbone_params_obj, ConfigObject) else backbone_params_obj
        
        head_params_obj = config.loss.get(f'{loss_type_to_use}_params', ConfigObject({}))
        head_specific_params_to_use = head_params_obj.to_dict() if isinstance(head_params_obj, ConfigObject) else head_params_obj

    print(f"--- 模型实例化配置来源: {source_of_model_config} ---")
    print(f"  模型类型: {model_type_to_use}, 损失类型: {loss_type_to_use}, 类别数: {num_classes_to_use}, 图像尺寸: {image_size_to_use}")
    print(f"  骨干参数: {backbone_specific_params_to_use}")
    print(f"  头部参数: {head_specific_params_to_use}")
    print("---------------------------------------------------")

    if not all([model_type_to_use, loss_type_to_use, num_classes_to_use is not None, image_size_to_use is not None]):
        print("错误: 无法最终确定模型构建所需的核心配置 (model_type, loss_type, num_classes, image_size)。请检查元数据或脚本配置。")
        return

    # --- 根据模型类型执行不同的验收逻辑 ---
    if loss_type_to_use.lower() == 'arcface':
        print("\n检测到 ArcFace 模型，执行基于特征比对的识别验收...")

        # 实例化骨干网络
        try:
            backbone_instance, feature_dim_from_backbone = get_backbone(
                config_model_params=backbone_specific_params_to_use,
                model_type_str=model_type_to_use,
                image_size=image_size_to_use
            )
            if isinstance(checkpoint_data, dict): # 确保 checkpoint_data 是 state_dict
                backbone_state_dict = {k.replace('backbone.', '', 1): v for k, v in checkpoint_data.items() if k.startswith('backbone.')}
                if backbone_state_dict:
                    backbone_instance.set_state_dict(backbone_state_dict)
                    print(f"骨干网络 ({model_type_to_use}) 实例化并从检查点加载 'backbone.' 权重成功。输出特征维度: {feature_dim_from_backbone}")
                else:
                    # 尝试将整个 state_dict 加载到 backbone (如果模型只包含 backbone)
                    try:
                        backbone_instance.set_state_dict(checkpoint_data)
                        print(f"骨干网络 ({model_type_to_use}) 实例化并尝试直接加载整个检查点权重成功。")
                    except Exception as e_direct_bb_load:
                        print(f"警告: 在检查点中未找到 'backbone.' 前缀的权重，且直接加载整个状态字典到骨干网络失败: {e_direct_bb_load}。骨干网络将使用随机初始化权重。")
            else:
                print(f"警告: 加载的检查点数据不是预期的字典格式。骨干网络 ({model_type_to_use}) 将使用随机初始化权重。")

        except Exception as e:
            print(f"创建或加载骨干网络 ({model_type_to_use}) 失败: {e}")
            return

        # 4. 加载特征库
        feature_library_path = cmd_args.feature_library_path # 特征库路径 ArcFace 必需通过命令行指定
        if not feature_library_path:
            print("错误: ArcFace 模型验收需要指定特征库文件路径。未通过命令行参数 --feature_library_path 指定。")
            return

        print(f"当前工作目录: {os.getcwd()}")
        print(f"尝试加载特征库的完整路径: {feature_library_path}")
        print(f"正在加载特征库文件: {feature_library_path}")

        load_success = False
        retry_count = 0

        # 增加重试机制以应对文件系统同步延迟问题，尤其是在 WSL/网络文件系统等环境下
        max_retries = 10 # 增加重试次数
        retry_delay_seconds = 2.0 # 增加每次重试的延迟到 2.0 秒
        initial_delay_seconds = 2.0 # 在第一次尝试加载前增加初始延迟

        print(f"加载文件前等待 {initial_delay_seconds} 秒...")
        time.sleep(initial_delay_seconds)

        print(f"开始尝试加载特征库文件 (最多重试 {max_retries} 次, 间隔 {retry_delay_seconds} 秒)... ")

        while retry_count < max_retries:
            try:
                # 直接尝试打开文件，让操作系统抛出 FileNotFoundError
                with open(feature_library_path, 'rb') as f:
                    feature_library = pickle.load(f) # 加载得到的是元组 (features_array, labels_array)

                # 假设特征库格式为 (features_array, labels_array)
                library_features_np = feature_library[0] # 获取特征数组 (numpy array)
                library_labels = feature_library[1].tolist() # 获取标签数组并转换为列表

                # 将特征数组转换为 Paddle Tensor
                library_features = paddle.to_tensor(library_features_np, dtype='float32')

                # 检查特征库是否为空
                if library_features.shape[0] == 0:
                    print("错误: 加载的特征库为空，无法进行比对识别验收。")
                    return None # 返回 None 表示验收失败，让调用者处理

                # 如果try块执行到这里，说明加载成功
                print(f"特征库加载成功，包含 {len(library_labels)} 个样本 (在第 {retry_count + 1} 次尝试时加载)。") # 使用标签数量作为样本数更准确
                load_success = True
                break # 加载成功，退出重试循环

            except FileNotFoundError as fnf_error:
                # 明确捕获 FileNotFoundError
                print(f"警告: 特征库文件 {feature_library_path} 在第 {retry_count + 1} 次尝试时报告 FileNotFoundError: {fnf_error}. 等待 {retry_delay_seconds} 秒后重试...")
                # 在重试前进行额外的文件系统检查和日志记录
                try:
                    # 尝试获取文件状态
                    file_stat = os.stat(feature_library_path)
                    print(f"  -> Python os.stat() 成功获取文件状态: {file_stat}")
                except FileNotFoundError:
                    print(f"  -> Python os.stat() 在尝试 {retry_count + 1} 时也报告 FileNotFoundError。")
                except Exception as stat_e:
                    print(f"  -> Python os.stat() 在尝试 {retry_count + 1} 时遇到其他错误: {stat_e}")
                
                # 尝试列出目录内容
                try:
                    parent_dir = os.path.dirname(feature_library_path)
                    if parent_dir and os.path.exists(parent_dir):
                        dir_contents = os.listdir(parent_dir)
                        print(f"  -> 父目录 {parent_dir} 内容 (前10项): {dir_contents[:10]}")
                    elif parent_dir:
                        print(f"  -> 父目录 {parent_dir} 不存在或无法访问。")
                    else:
                        print(f"  -> 无法确定父目录以进行列表。")
                except Exception as listdir_e:
                    print(f"  -> Python os.listdir() 在尝试 {retry_count + 1} 时遇到错误: {listdir_e}")

                time.sleep(retry_delay_seconds)
                retry_count += 1
            except Exception as e:
                # 捕获其他可能的加载错误 (如 pickle.UnpicklingError 等)
                print(f"警告: 在第 {retry_count + 1} 次尝试加载或解析特征库文件 {feature_library_path} 时发生非文件找不到错误: {e}. 等待 {retry_delay_seconds} 秒后重试...")
                time.sleep(retry_delay_seconds)
                retry_count += 1

        if not load_success:
            print(f"错误: 达到最大重试次数 ({max_retries}) 后，仍无法加载特征库文件 {feature_library_path}. 验收测试终止。")
            return None

        # 5. 执行比对识别循环
        backbone_instance.eval()

        correct_identifications = 0
        total_samples = 0

        # 从 config 获取识别阈值，如果命令行指定了则覆盖
        recognition_threshold = cmd_args.recognition_threshold if cmd_args.recognition_threshold is not None else config.infer.get('recognition_threshold', 0.5)
        print(f"使用识别阈值: {recognition_threshold:.4f}")

        print(f"\n开始在测试集上进行比对识别验收 (共 {len(test_loader.dataset)} 个样本)...")
        eval_start_time = time.time()

        # 实现 Cosine Similarity 计算函数 (如果 compare.py 不可用)
        def calculate_cosine_similarity(vec1: paddle.Tensor, matrix: paddle.Tensor) -> paddle.Tensor:
            # vec1: [N, feature_dim]
            # matrix: [M, feature_dim]
            # 输出: [N, M]
            # 归一化
            vec1 = F.normalize(vec1, axis=1)
            matrix = F.normalize(matrix, axis=1)
            # 计算相似度：[N, feature_dim] @ [feature_dim, M] = [N, M]
            similarity_scores = paddle.matmul(vec1, matrix, transpose_y=True)
            return similarity_scores


        with paddle.no_grad():
            # Batch processing for inference
            for batch_id, data_batch in enumerate(test_loader):
                images, labels = data_batch
                if labels.ndim > 1 and labels.shape[1] == 1:
                    labels = paddle.squeeze(labels, axis=1)

                # 提取查询图片的特征 (一个批次)
                query_features_batch = backbone_instance(images) # [batch_size, feature_dim]

                # 计算批次内所有查询图片与特征库的相似度
                # similarity_scores_batch: [batch_size, num_library_features]
                similarity_scores_batch = calculate_cosine_similarity(query_features_batch, library_features)

                # 处理批次内的每个样本
                for i in range(query_features_batch.shape[0]):
                    total_samples += 1
                    true_label = labels[i].item()

                    # 获取当前样本的相似度分数 [num_library_features]
                    current_sample_scores = similarity_scores_batch[i, :]

                    # 找到最高相似度及其索引
                    max_similarity, max_similarity_idx = paddle.max(current_sample_scores, axis=0).numpy()

                    # 获取最高相似度对应的库中样本的标签
                    predicted_label = library_labels[max_similarity_idx]

                    # 判断是否识别正确: 最高相似度大于阈值 AND 预测标签与真实标签一致
                    if max_similarity > recognition_threshold and predicted_label == true_label:
                        correct_identifications += 1

                    # 可选：打印每张图片的识别结果 (详细模式)
                    # if config.get('verbose_acceptance', False):
                    #     print(f"  样本 {total_samples}: 真\t标签 {true_label}, 最\t高相似度 {max_similarity:.4f}, 预测标签 {predicted_label}. {'正确' if max_similarity > recognition_threshold and predicted_label == true_label else '错误'}")

                if (batch_id + 1) % config.get('log_interval', 20) == 0:
                    current_accuracy = correct_identifications / total_samples if total_samples > 0 else 0
                    print(f"  批次 {batch_id + 1}/{len(test_loader)}: 已处理 {total_samples} 样本，当前识别准确率: {current_accuracy:.4f}")

        eval_duration = time.time() - eval_start_time

        # 6. 汇总并打印最终结果
        final_accuracy = correct_identifications / total_samples if total_samples > 0 else 0
        print("\n--- 验收测试结果 (ArcFace 识别) ---")
        print(f"总测试样本数: {total_samples}")
        print(f"正确识别样本数: {correct_identifications}")
        print(f"识别准确率 (阈值 > {recognition_threshold:.4f}): {final_accuracy:.4f}")
        print(f"验收测试总耗时: {eval_duration:.2f} 秒")
        print("--------------------------------")

        # --- 将结果保存到 JSON 文件 ---
        results = {
            "model_path": cmd_args.trained_model_path,
            "loss_type": loss_type_to_use,
            "total_samples": total_samples,
            "accuracy": final_accuracy,
            "duration_seconds": eval_duration,
        }

        results["correct_identifications"] = correct_identifications
        results["recognition_threshold"] = recognition_threshold
        results["feature_library_path"] = feature_library_path

        output_json_path = cmd_args.output_json_path
        if not output_json_path:
            # 如果没有指定输出路径，默认保存到当前目录下的一个文件
            output_json_path = os.path.join(".", "acceptance_results.json")
            print(f"警告: 未指定验收结果输出路径 (--output_json_path)，将保存到默认路径: {output_json_path}")

        try:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"验收结果已保存到: {output_json_path}")
        except Exception as e:
            print(f"错误: 保存验收结果到 {output_json_path} 失败: {e}")
            # 即使保存失败，也尝试返回准确率，但标记出错
            # sys.exit(1) # 退出，标记失败

        return final_accuracy # 虽然保存到JSON，仍然返回准确率，方便脚本解析或打印

    elif loss_type_to_use.lower() == 'cross_entropy':
        print("\n检测到 Cross-Entropy 模型，执行基于分类的验收...")

        # 实例化骨干网络和头部
        try:
            backbone_instance, feature_dim_from_backbone = get_backbone(
                config_model_params=backbone_specific_params_to_use,
                model_type_str=model_type_to_use,
                image_size=image_size_to_use
            )
            head_module_instance = get_head(
                config_loss_params=head_specific_params_to_use,
                loss_type_str=loss_type_to_use,
                in_features=feature_dim_from_backbone,
                num_classes=num_classes_to_use
            )

            # 加载模型权重
            if isinstance(checkpoint_data, dict): # 确保 checkpoint_data 是 state_dict
                 try:
                     # 尝试加载到 CombinedModel 结构
                     class CombinedModel(paddle.nn.Layer):
                         def __init__(self, backbone, head=None):
                             super().__init__()
                             self.backbone = backbone
                             self.head = head
                         def forward(self, x, label=None):
                             features = self.backbone(x)
                             if self.head:
                                  head_forward_params = inspect.signature(self.head.forward).parameters
                                  if 'label' in head_forward_params:
                                       return self.head(features, label)
                                  else:
                                       return self.head(features)
                             else:
                                 # CrossEntropy 模型应该有 head，此处不会返回纯 features
                                 raise ValueError("CrossEntropy 模型需要头部模块进行分类。")

                     combined_model = CombinedModel(backbone_instance, head_module_instance)
                     combined_model.set_state_dict(checkpoint_data)
                     print("CombinedModel 实例化并加载权重成功。")
                     # 使用 combined_model 进行评估
                     model_to_evaluate = combined_model

                 except Exception as e_combined_load:
                      print(f"警告: 加载 CombinedModel 权重失败: {e_combined_load}. 将尝试单独加载 backbone 和 head 权重.")
                      # 回退到单独加载 backbone 和 head
                      try:
                          backbone_state_dict = {k.replace('backbone.', '', 1): v for k, v in checkpoint_data.items() if k.startswith('backbone.')}
                          if backbone_state_dict:
                              backbone_instance.set_state_dict(backbone_state_dict)
                              print(f"骨干网络 ({model_type_to_use}) 单独加载权重成功。")
                          else:
                              print(f"警告: 在检查点中未找到 'backbone.' 权重。骨干网络将使用随机初始化权重。")

                          head_state_dict = {k.replace('head.', '', 1): v for k, v in checkpoint_data.items() if k.startswith('head.')}
                          if head_state_dict:
                              head_module_instance.set_state_dict(head_state_dict)
                              print(f"头部模块 ({loss_type_to_use}) 单独加载权重成功。")
                          else:
                              print(f"警告: 在检查点中未找到 'head.' 权重。头部模块将使用随机初始化权重。")

                          # 使用单独加载权重的 backbone 和 head 进行评估
                          class SeparatedModel(paddle.nn.Layer):
                              def __init__(self, backbone, head):
                                  super().__init__()
                                  self.backbone = backbone
                                  self.head = head
                              def forward(self, x, label=None):
                                   features = self.backbone(x)
                                   return self.head(features, label) # CrossEntropyHead forward 接受可选的 label 参数

                          model_to_evaluate = SeparatedModel(backbone_instance, head_module_instance)
                          print("使用单独加载权重的模型进行评估。")

                      except Exception as e_separate_load:
                          print(f"错误: 单独加载 backbone 和 head 权重失败: {e_separate_load}. 无法进行验收。")
                          return

            else:
                 print(f"错误: 加载的模型权重文件 {model_weights_path} 不是预期的 state_dict 格式。无法加载模型。")
                 return

            print(f"模型 ({model_type_to_use} + {loss_type_to_use} 头部) 实例化并加载权重成功。")

        except Exception as e:
            print(f"创建或加载模型 ({model_type_to_use} + {loss_type_to_use} 头部) 失败: {e}")
            return

        # 5. 执行分类评估循环
        model_to_evaluate.eval()

        correct_predictions = 0
        total_samples = 0
        total_loss = 0.0 # 可以选择计算并报告损失

        print(f"\n开始在测试集上进行分类验收 (共 {len(test_loader.dataset)} 个样本)...")
        eval_start_time = time.time()

        with paddle.no_grad():
            for batch_id, data_batch in enumerate(test_loader):
                images, labels = data_batch
                if labels.ndim > 1 and labels.shape[1] == 1:
                    labels = paddle.squeeze(labels, axis=1)

                # 前向传播
                # CombinedModel 或 SeparatedModel 的 forward 会调用 head.forward
                outputs = model_to_evaluate(images, labels) # 传递 labels 以便 CrossEntropyHead 计算损失
                loss = None
                logits = None

                if isinstance(outputs, tuple) and len(outputs) == 2: # 假设 head 返回 (loss, logits)
                    loss, logits = outputs
                elif isinstance(outputs, paddle.Tensor): # 假设 head 只返回 logits
                    logits = outputs
                    # 手动计算损失
                    try:
                        loss = F.cross_entropy(logits, labels)
                    except Exception as e_loss_calc:
                         print(f"警告: 无法计算分类损失: {e_loss_calc}")
                         loss = None

                if loss is not None:
                    total_loss += loss.item() * images.shape[0]

                if logits is not None:
                    # 计算准确率
                    predicted_labels = paddle.argmax(logits, axis=1)
                    correct_predictions += paddle.sum(predicted_labels == labels).item()
                
                total_samples += labels.shape[0]

                if (batch_id + 1) % config.get('log_interval', 20) == 0:
                    current_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
                    current_avg_loss = total_loss / total_samples if total_samples > 0 else 0
                    print(f"  批次 {batch_id + 1}/{len(test_loader)}: 已处理 {total_samples} 样本，当前准确率: {current_accuracy:.4f}", end='')
                    if loss is not None: print(f", 当前平均损失: {current_avg_loss:.4f}", end='')
                    print("")

        eval_duration = time.time() - eval_start_time

        # 6. 汇总并打印最终结果
        final_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        final_avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')

        print("\n--- 验收测试结果 (Cross-Entropy 分类) ---")
        print(f"总测试样本数: {total_samples}")
        print(f"正确预测样本数: {correct_predictions}")
        print(f"分类准确率: {final_accuracy:.4f}")
        if loss is not None: print(f"平均损失: {final_avg_loss:.4f}")
        print(f"验收测试总耗时: {eval_duration:.2f} 秒")
        print("---------------------------------------")

        # --- 将结果保存到 JSON 文件 ---
        results = {
            "model_path": cmd_args.trained_model_path,
            "loss_type": loss_type_to_use,
            "total_samples": total_samples,
            "accuracy": final_accuracy,
            "duration_seconds": eval_duration,
        }

        results["correct_predictions"] = correct_predictions
        results["average_loss"] = final_avg_loss # 添加平均损失

        output_json_path = cmd_args.output_json_path
        if not output_json_path:
            # 如果没有指定输出路径，默认保存到当前目录下的一个文件
            output_json_path = os.path.join(".", "acceptance_results.json")
            print(f"警告: 未指定验收结果输出路径 (--output_json_path)，将保存到默认路径: {output_json_path}")

        try:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"验收结果已保存到: {output_json_path}")
        except Exception as e:
            print(f"错误: 保存验收结果到 {output_json_path} 失败: {e}")
            # 即使保存失败，也尝试返回准确率，但标记出错
            # sys.exit(1) # 退出，标记失败

        return final_accuracy # 虽然保存到JSON，仍然返回准确率，方便脚本解析或打印

    else:
        print(f"错误: 不支持的损失类型 '{loss_type_to_use}' 用于验收测试。")
        return None

# --- 脚本主入口 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸识别模型验收测试脚本')

    # 核心控制参数
    parser.add_argument('--config_path', type=str, default=None,
                        help='指定YAML配置文件的路径。')
    parser.add_argument('--active_config', type=str, default=None,
                        help='通过命令行指定要激活的配置块名称，覆盖YAML文件中的active_config设置。')
    parser.add_argument('--trained_model_path', type=str, required=True,
                        help='必需：指定已训练模型的权重文件路径 (.pdparams)。如果配置文件中也指定了，命令行优先。')
    parser.add_argument('--output_json_path', type=str, default=None,
                        help='指定验收结果 JSON 文件的输出路径。如果不指定，将保存到当前目录下的 acceptance_results.json。')
    
    # 可覆盖配置文件的参数
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=None,
                        help='是否使用GPU。覆盖配置文件设置。')
    parser.add_argument('--batch_size', type=int, help='验收测试批大小 (覆盖配置文件)。')
    parser.add_argument('--image_size', type=int, help='输入图像尺寸 (覆盖配置文件)。')
    parser.add_argument('--num_classes', type=int, help='类别数 (覆盖配置文件, 影响模型加载)。')
    parser.add_argument('--data_dir', type=str, help='数据集根目录 (覆盖配置文件)。')
    parser.add_argument('--class_name', type=str, help='数据集子目录名 (覆盖配置文件)。')
    parser.add_argument('--log_interval', type=int, help='打印日志的间隔批次数 (覆盖配置文件)。')
    parser.add_argument('--results_save_dir', type=str, help='验收结果保存目录 (覆盖配置文件)。')
    parser.add_argument('--feature_library_path', type=str, default=None,
                        help='[ArcFace Only] 用于比对的特征库文件 (.pkl) 的路径。ArcFace 模型验收时必需指定。')
    parser.add_argument('--recognition_threshold', type=float, default=None,
                        help='[ArcFace Only] 人脸识别比对的相似度阈值 (覆盖配置文件)。')

    cmd_line_args = parser.parse_args()

    # 加载配置
    final_config = load_config(
        default_yaml_path=cmd_line_args.config_path,
        cmd_args_namespace=cmd_line_args
    )

    # 确保 trained_model_path 已提供
    if not cmd_line_args.trained_model_path and not final_config.get('trained_model_path'):
        parser.error("错误: 必须通过 --trained_model_path 命令行参数或在配置文件中 "
                     "(e.g., global_settings.trained_model_path or active_config.trained_model_path) "
                     "指定模型权重路径。")
    
    print("\\n--- 最终生效的验收测试配置 ---")
    print(f"  模型路径: {cmd_line_args.trained_model_path or final_config.get('trained_model_path')}")
    print(f"  使用GPU: {final_config.use_gpu}")
    print(f"  批大小: {final_config.batch_size}")
    print(f"  类别数: {final_config.num_classes}") # 确保这里能正确打印
    print(f"  图像尺寸: {final_config.image_size}")
    print(f"  结果保存目录: {final_config.get('results_save_dir', 'acceptance_results')}")
    print("---------------------------------\\n")
    
    if final_config.get('num_classes') is None: # 再次检查 num_classes
        parser.error("错误: 最终配置中未能确定 'num_classes'。请检查YAML文件和命令行参数 --num_classes。")

    # 运行验收评估
    accuracy = run_acceptance_evaluation(final_config, cmd_line_args)
    
    # 脚本的退出状态可以反映成功与否，以及（可选）准确率是否达到阈值
    if accuracy is not None:
        # 可以根据准确率是否大于某个阈值来设置退出状态
        # 例如：如果准确率低于 0.9，脚本退出码为 1
        # if accuracy < 0.9:
        #     print("验收准确率未达到期望阈值 (0.9)。")
        #     sys.exit(1)
        # else:
        #     print("验收通过。")
        sys.exit(0) # 成功执行并获得结果
    else:
        sys.exit(1) # 验收执行失败 