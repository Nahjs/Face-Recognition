# coding:utf-8
# compare.py
# 该脚本负责人脸对比 (face comparison) 功能。
# 它接收两张人脸图像作为输入，使用预训练的人脸识别模型分别提取这两张图像的特征向量，
# 然后计算这两个特征向量之间的余弦相似度，以此来判断这两张图像是否属于同一个人。
#
# 主要流程包括：
# 1. 加载配置文件和命令行参数，用于获取模型路径、图像路径、运行设备等设置。
# 2. 根据配置设置运行设备 (CPU/GPU)。
# 3. 加载预训练好的模型文件 (.pdparams)。这个模型文件通常只包含骨干网络 (backbone) 的权重，
#    因为人脸对比主要依赖于高质量的特征表示，而不需要分类头。
#    模型文件也应包含训练时的配置信息 (如 model_type, image_size)，以便正确恢复骨干网络。
# 4. 根据模型文件中保存的配置和当前脚本的配置，使用 model_factory.py 动态实例化骨干网络。
# 5. 加载骨干网络权重到实例化的网络中。
# 6. 对输入的两张图像分别进行预处理 (缩放、归一化、标准化)。
# 7. 使用加载的骨干网络分别提取两张图像的人脸特征向量。
# 8. 计算两个特征向量之间的余弦相似度。
# 9. 根据预设的相似度阈值，判断两张图像是否属于同一个人。
# 10. 打印对比结果，包括相似度得分和判断结论。
# 11. （可选）如果配置了可视化，则将两张图像并排显示，并在标题中注明相似度和判断结果，
#      然后将结果图像保存到 results/ 目录。
#
# 注意: 此脚本中的 process_image_local 和 compute_similarity_fc 函数与 infer.py 中的类似函数功能相同，
#       理想情况下，这些通用工具函数可以移至一个共享的 utils.py 模块中以避免代码重复。
#       但为了脚本的独立性，暂时在此处保留了本地副本。

import json
import os
import cv2
import argparse
import numpy as np
import matplotlib

import heads
import model_factory
matplotlib.use('Agg')  # 使用非交互式后端，防止在无显示环境的服务器上出错
import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F # 用于余弦相似度计算
from config_utils import load_config, ConfigObject
from model_factory import get_backbone
from utils.image_processing import process_image # 从共享模块导入

# 全局变量，用于在 compare_faces 和 show_comparison_result 之间传递模型训练时的配置信息，
# 以便在保存结果图像时，文件名能反映所用模型的类型。
# 更好的做法可能是将这些信息作为参数传递，或通过一个类来封装状态。
saved_model_config_dict_for_filename: dict = {}

def compute_similarity_fc(feature1: np.ndarray, feature2: np.ndarray) -> float: 
    """
    计算两个一维特征向量之间的余弦相似度。
    
    Args:
        feature1 (numpy.ndarray): 第一个特征向量 (应为一维或可展平为一维)。
        feature2 (numpy.ndarray): 第二个特征向量 (应为一维或可展平为一维)。
    
    Returns:
        float: 余弦相似度得分。范围通常在-1到1之间。
               如果任一向量为零向量 (范数为0)，则返回0.0以避免除零错误。
    """
    #展平向量以确保它们是一维的
    f1 = feature1.flatten()
    f2 = feature2.flatten()
    
    # 计算余弦相似度: dot_product / (norm(f1) * norm(f2))
    norm_f1 = np.linalg.norm(f1)
    norm_f2 = np.linalg.norm(f2)
    if norm_f1 == 0 or norm_f2 == 0:
        return 0.0 # 如果任一向量为零向量，相似度为0
    similarity = np.dot(f1, f2) / (norm_f1 * norm_f2)
    return float(similarity)

def show_comparison_result(img1_path: str, img2_path: str, similarity: float, 
                           model_type_loaded: str, loss_type_loaded_from_model_cfg: str, 
                           threshold: float, visualize_flag: bool, results_dir_path: str):
    """可视化人脸对比结果并保存图像。

    如果 `visualize_flag` 为 True，则执行以下操作：
    1. 读取两张原始图像。
    2. 使用 Matplotlib 将两张图像并排显示。
    3. 在图像上方居中显示标题，包含计算得到的相似度、判断阈值以及是/否为同一人的结论。
    4. 根据判断结果（是否为同一人）设置标题颜色（绿色表示同一人，红色表示不同人）。
    5. 将结果图像保存到 `results/` 目录下。
       保存的文件名会包含模型类型 (如 resnet, vgg)、模型训练时使用的损失类型 (如 arcface, cross_entropy)、
       以及两张输入图像的文件名，以便区分和追溯。

    Args:
        img1_path (str): 第一张原始图像的文件路径。
        img2_path (str): 第二张原始图像的文件路径。
        similarity (float): 计算得到的两张图像特征的余弦相似度。
        model_type_loaded (str): 加载的模型的骨干网络类型 (如 'resnet', 'vgg')。
                                  用于生成结果图像的文件名。
        loss_type_loaded_from_model_cfg (str): 从模型配置文件中解析出的训练时使用的损失类型
                                            (如 'arcface', 'cross_entropy')。用于文件名。
        threshold (float): 判断两张图像是否为同一人的相似度阈值。
        visualize_flag (bool): 控制是否执行可视化并保存图像的标志。如果为 False，则此函数不执行任何操作。
        results_dir_path (str): 保存结果图像的目录路径。
    """
    if not visualize_flag: return

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print(f"错误: 无法读取图像进行对比可视化 (图像1: {img1_path}, 图像2: {img2_path})。跳过可视化。")
        return

    # OpenCV 读取的是 BGR 格式，Matplotlib 显示需要 RGB 格式
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 6)) # 创建一个图窗，设置大小
    plt.subplot(1, 2, 1); plt.imshow(img1_rgb); plt.title("图像1", fontproperties="SimHei"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(img2_rgb); plt.title("图像2", fontproperties="SimHei"); plt.axis('off')
    
    is_same_person = similarity >= threshold
    judgment_text = "是同一个人" if is_same_person else "不是同一个人"
    title_color = 'green' if is_same_person else 'red' # 结果为真时绿色，否则红色
    # 准备图窗的超级标题文本
    result_text = f"相似度: {similarity:.4f}\n判断: {judgment_text} (阈值: {threshold:.2f})"
    plt.suptitle(result_text, fontsize=16, color=title_color, fontproperties="SimHei")
    
    # 准备结果保存目录和文件名
    result_dir = results_dir_path
    if not os.path.exists(result_dir): 
        os.makedirs(result_dir)
        print(f"已创建对比结果保存目录: {result_dir}")
    
    base1 = os.path.basename(img1_path).split('.')[0]
    base2 = os.path.basename(img2_path).split('.')[0]
    
    # 使用从全局变量 saved_model_config_dict_for_filename 获取的模型配置中的损失类型来命名
    # 这里的 loss_type_loaded_from_model_cfg 已经由 compare_faces 函数正确传递
    active_loss_type_str = loss_type_loaded_from_model_cfg 
    # 构建文件名后缀，包含模型类型和损失类型
    model_suffix = f"{model_type_loaded}_{active_loss_type_str}"

    result_filename = f"compare_{model_suffix}_{base1}_vs_{base2}.png"
    result_file_path = os.path.join(result_dir, result_filename)
    
    try:
        plt.savefig(result_file_path)
        print(f"对比结果图像已保存至: {result_file_path}")
    except Exception as e_save:
        print(f"错误: 保存对比结果图像到 {result_file_path} 失败: {e_save}")
    finally:
        plt.close() # 关闭图窗，释放资源

def compare_faces(config: ConfigObject, cmd_args: argparse.Namespace):
    """
    主比较函数，加载模型、处理两张图片，进行比较并显示结果。
    """
    # --- 设置设备 ---
    use_gpu_flag = cmd_args.use_gpu if cmd_args.use_gpu is not None else config.use_gpu
    use_gpu_flag = use_gpu_flag and paddle.is_compiled_with_cuda()
    paddle.set_device('gpu' if use_gpu_flag else 'cpu')
    print(f"使用 {'GPU' if use_gpu_flag else 'CPU'} 进行比较")

    # --- 确定模型权重路径 ---
    # 优先使用命令行提供的 trained_model_path，然后是配置文件中的，最后是 compare.py 自身的 trained_model_path
    model_weights_path = cmd_args.trained_model_path or config.get('trained_model_path') or config.compare.get('trained_model_path')

    if not model_weights_path:
        raise ValueError("错误: 必须通过 --trained_model_path 或在配置文件中指定模型权重文件路径。")
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"错误: 指定的模型权重文件未找到: {model_weights_path}")
    
    print(f"将从模型文件 {model_weights_path} 加载模型。")

    # --- 尝试从模型元数据加载配置 ---
    loaded_model_type = None
    loaded_image_size = None
    loaded_model_specific_params = {} # 确保有默认值
    # 对于 compare.py，loss_type 和 num_classes 可能不那么重要，主要关注骨干网络
    source_of_config = "Global compare.py Config"

    metadata_path = model_weights_path.replace('.pdparams', '.json')
    using_metadata_config = False

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
                source_of_config = f"Metadata file ({metadata_path})"
                print(f"已从元数据文件 {metadata_path} 加载构建骨干网络的配置。")
                using_metadata_config = True
            else:
                print(f"警告: 模型元数据文件 {metadata_path} 中缺少部分骨干网络构建所需的配置项。")
        except Exception as e:
            print(f"警告: 加载或解析模型元数据文件 {metadata_path} 失败: {e}。")

    if not using_metadata_config:
        print(f"将使用 compare.py 的全局配置文件中的配置构建骨干网络 (回退或元数据加载失败/不完整)。")
        loaded_model_type = config.model_type
        loaded_image_size = config.image_size
        # 从全局配置加载详细模型参数
        loaded_model_specific_params = config.model.get(f'{loaded_model_type}_params', {}).to_dict() 
        if isinstance(config.model.get(f'{loaded_model_type}_params', {}), ConfigObject):
            loaded_model_specific_params = config.model.get(f'{loaded_model_type}_params', {}).to_dict()
        else:
            loaded_model_specific_params = config.model.get(f'{loaded_model_type}_params', {})
        source_of_config = "Global compare.py Config (fallback)"
        if not all([loaded_model_type, loaded_image_size is not None]):
             raise ValueError("错误: 无法从全局配置中确定骨干网络构建所需的核心配置 (model_type, image_size)。")

    print(f"--- 骨干网络构建配置来源: {source_of_config} ---")
    print(f"  Model Type: {loaded_model_type}")
    print(f"  Image Size: {loaded_image_size}")
    print(f"  Model Params: {loaded_model_specific_params}")
    print("--------------------------------------------------")
    
    # 更新全局变量以便 show_comparison_result 使用模型类型 (如果需要从元数据更新)
    # 考虑更好的传递方式，暂时保留
    global saved_model_config_dict_for_filename 
    saved_model_config_dict_for_filename['model_type_loaded'] = loaded_model_type
    # loss_type 从元数据加载可能对文件名有用，但对比较逻辑本身不关键
    if using_metadata_config and metadata.get('loss_type'):
        saved_model_config_dict_for_filename['loss_type_loaded'] = metadata.get('loss_type')
    else:
        saved_model_config_dict_for_filename['loss_type_loaded'] = config.loss_type


    # --- 构建骨干网络 ---
    model_backbone, backbone_out_dim = get_backbone(
        config_model_params=loaded_model_specific_params,
        model_type_str=loaded_model_type,
        image_size=loaded_image_size
    )
    print(f"骨干网络 ({loaded_model_type.upper()}) 构建成功，输出特征维度: {backbone_out_dim}")

    # --- 加载骨干网络权重 ---
    full_state_dict = paddle.load(model_weights_path)
    backbone_state_dict_to_load = {k.replace('backbone.', '', 1): v for k, v in full_state_dict.items() if k.startswith('backbone.')}
    
    if backbone_state_dict_to_load:
        model_backbone.set_state_dict(backbone_state_dict_to_load)
        print(f"骨干网络权重从 {model_weights_path} (提取 'backbone.' 部分) 加载成功。")
    else:
        # 如果没有 'backbone.' 前缀，尝试直接加载整个 state_dict (可能模型只保存了骨干)
        try:
            model_backbone.set_state_dict(full_state_dict)
            print(f"骨干网络权重 (尝试直接加载整个文件) 从 {model_weights_path} 加载成功。")
        except Exception as e_direct_bb_load:
            raise RuntimeError(f"错误: 在模型文件 {model_weights_path} 中未找到 'backbone.' 前缀的权重，且直接加载整个状态字典到骨干网络失败: {e_direct_bb_load}。请确保模型文件包含可用的骨干网络权重。")

    model_backbone.eval()

    # --- 图像预处理 ---
    # 注意：预处理参数 (mean, std) 应与模型训练时一致。
    # 这些参数通常不在每个模型的元数据中，而是作为全局数据集参数。
    image_mean = config.dataset_params.mean
    image_std = config.dataset_params.std

    print(f"正在处理图片: {cmd_args.img1_path}")
    img1_processed = process_image(img_path=cmd_args.img1_path, target_size=loaded_image_size, mean_rgb=image_mean, std_rgb=image_std, is_bgr_input=True)
    print(f"正在处理图片: {cmd_args.img2_path}")
    img2_processed = process_image(img_path=cmd_args.img2_path, target_size=loaded_image_size, mean_rgb=image_mean, std_rgb=image_std, is_bgr_input=True)

    img1_tensor = paddle.to_tensor(img1_processed)
    img2_tensor = paddle.to_tensor(img2_processed)

    # --- 特征提取 ---
    with paddle.no_grad():
        feature1 = model_backbone(img1_tensor).numpy()
        feature2 = model_backbone(img2_tensor).numpy()

    # --- 计算相似度 ---
    similarity = compute_similarity_fc(feature1, feature2)
    print(f"图像1 ({cmd_args.img1_path}) 与 图像2 ({cmd_args.img2_path}) 的余弦相似度: {similarity:.4f}")

    # --- 判断并显示结果 ---
    threshold = cmd_args.compare_threshold if cmd_args.compare_threshold is not None else config.compare.compare_threshold
    is_same = similarity >= threshold
    print(f"判断阈值: {threshold:.2f}")
    if is_same:
        print("结论: 两张图像很可能为同一个人。")
    else:
        print("结论: 两张图像不太可能为同一个人。")

    # --- 可视化 ---
    visualize = cmd_args.compare_visualize if cmd_args.compare_visualize is not None else config.compare.compare_visualize
    if visualize:
        # Determine results_dir
        results_dir_to_pass = config.compare.get('results_dir') # Try from compare block
        if results_dir_to_pass is None:
            results_dir_to_pass = config.get('results_dir', "results") # Try from global, default to "results"

        show_comparison_result(
            img1_path=cmd_args.img1_path,
            img2_path=cmd_args.img2_path,
            similarity=similarity,
            # 使用从元数据或配置加载的model_type 和 loss_type
            model_type_loaded=saved_model_config_dict_for_filename.get('model_type_loaded', loaded_model_type), 
            loss_type_loaded_from_model_cfg=saved_model_config_dict_for_filename.get('loss_type_loaded', config.loss_type),
            threshold=threshold,
            visualize_flag=True, # Redundant, but explicit
            results_dir_path=results_dir_to_pass # Pass the determined path
        )
    
    print("比较完成。")

# --- Helper 函数 (如果还没定义) ---
def preprocess_image(image, target_size):
    """简单的图像预处理，需要与训练时保持一致。"""
    # 示例: 调整大小, 归一化
    image = cv2.resize(image, (target_size, target_size))
    image = image.astype('float32') / 255.0
    # 可能需要调整通道顺序 BGR -> RGB
    # image = image[:, :, ::-1]
    # 转换为 CHW 格式 (PaddlePaddle 需要)
    image = image.transpose((2, 0, 1))
    return image

def calculate_cosine_similarity(vector, matrix):
    """计算一个向量与一个矩阵中所有行向量的余弦相似度。"""
    # 标准化向量和矩阵的行
    vector_norm = vector / np.linalg.norm(vector)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    # 计算点积 (即余弦相似度)
    similarity = np.dot(matrix_norm, vector_norm)
    return similarity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸比较脚本')

    # 核心控制参数
    parser.add_argument('--config_path', type=str, default=None, help='YAML配置文件路径。')
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=None, help='是否使用GPU。')
    parser.add_argument('--active_config', type=str, default=None, help='覆盖YAML中的active_config。')

    # 输入参数
    parser.add_argument('--img1_path', type=str, required=True, help='第一张人脸图片路径。')
    parser.add_argument('--img2_path', type=str, required=True, help='第二张人脸图片路径。')
    parser.add_argument('--trained_model_path', type=str, default=None, help='指定要加载的训练好的骨干网络模型 (.pdparams) 文件路径。如果提供，将优先于配置文件中的设置。')
    
    # 可覆盖配置文件的参数
    parser.add_argument('--model_type', type=str, choices=['vgg', 'resnet'], help='骨干网络类型 (用于回退配置)')
    parser.add_argument('--image_size', type=int, help='图像尺寸 (用于回退配置)')
    parser.add_argument('--compare_threshold', type=float, help='覆盖配置文件中的比较阈值。')
    parser.add_argument('--compare_visualize', action=argparse.BooleanOptionalAction, default=None, help='是否可视化对比结果并保存图片 (覆盖配置文件)。')

    cmd_line_args = parser.parse_args()

    # --- 配置加载与合并 ---
    final_config = load_config(
        default_yaml_path='configs/default_config.yaml',
        cmd_args_namespace=cmd_line_args
    )

    # 更新 final_config 中可能受命令行影响的顶层参数
    if cmd_line_args.trained_model_path:
        final_config['trained_model_path'] = cmd_line_args.trained_model_path
    if cmd_line_args.model_type:
        final_config['model_type'] = cmd_line_args.model_type
    if cmd_line_args.image_size:
        final_config['image_size'] = cmd_line_args.image_size
    
    # 安全地更新 compare 子配置
    if not hasattr(final_config, 'compare') or final_config.compare is None:
        final_config.compare = ConfigObject({}) # 如果不存在或为None，则创建一个空的

    if cmd_line_args.compare_threshold is not None:
        final_config.compare['compare_threshold'] = cmd_line_args.compare_threshold
    if cmd_line_args.compare_visualize is not None:
        final_config.compare['compare_visualize'] = cmd_line_args.compare_visualize

    print("\n--- 最终生效的比较配置 (YAML与命令行合并后) ---") # 使用单反斜杠
    print(f"  模型权重路径: {final_config.get('trained_model_path', '未指定')}")
    
    # 从 final_config.compare 中获取值，如果 compare 不存在或键不存在，则提供默认值
    compare_threshold_to_print = final_config.compare.get('compare_threshold', '使用默认(配置文件)')
    compare_visualize_to_print = final_config.compare.get('compare_visualize', '使用默认(配置文件)')
    results_dir_to_print = final_config.compare.get('results_dir', final_config.get('results_dir', 'results')) # For printing
    
    print(f"  比较阈值: {compare_threshold_to_print}")
    print(f"  可视化: {compare_visualize_to_print}")
    print(f"  结果保存目录: {results_dir_to_print}") # Print the actual path to be used
    print("---------------------------------------------------")

    try:
        compare_faces(final_config, cmd_line_args)
    except FileNotFoundError as e:
        print(f"比较失败 (文件未找到): {e}")
    except RuntimeError as e:
        print(f"比较时发生运行时错误: {e}")
    except ValueError as e:
        print(f"配置或输入错误: {e}")
    except Exception as e:
        print(f"发生意外错误: {e}") 