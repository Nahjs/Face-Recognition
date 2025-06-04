# train.py
# 本脚本是人脸识别项目的核心训练模块。
# 它负责根据配置文件中的设定，执行模型的训练流程，包括：
# 1. 设置运行环境（CPU/GPU）和随机种子。
# 2. 使用 MyReader.py 中的 create_dataloader 创建训练和测试数据加载器。
# 3. 使用 model_factory.py 中的工厂函数 (get_backbone, get_head) 构建骨干网络和模型头部。
# 4. 使用 utils.lr_scheduler_factory.py 中的工厂函数 (get_lr_scheduler) 构建学习率调度器。
# 5. 定义优化器 (AdamW, Momentum)。
# 6. 实现检查点加载逻辑，支持从断点恢复训练或加载预训练模型（部分权重）。
# 7. 执行训练循环 (epochs)，包括前向传播、损失计算、反向传播、参数优化。
# 8. 在每个epoch结束后，在测试集上进行评估。
# 9. 根据评估结果保存最佳模型和训练检查点。
# 10. 详细的日志输出，记录训练过程中的损失、准确率、学习率等信息。
#
# 脚本通过解析命令行参数和YAML配置文件来获取所有必要的参数，
# 由 config_utils.py 中的 load_config 函数统一管理配置的加载和合并。

import os
import argparse
import paddle
import paddle.nn as nn
import paddle.optimizer as opt # 使用 opt 别名
import paddle.nn.functional as F
import MyReader                 # 导入数据读取和预处理模块
from config_utils import load_config, ConfigObject # 导入配置加载工具
from model_factory import get_backbone, get_head   # 导入模型工厂函数
from utils.lr_scheduler_factory import get_lr_scheduler # 导入学习率调度器工厂函数
from utils.checkpoint_manager import CheckpointManager # 导入检查点管理器
import time
from datetime import datetime
from visualdl import LogWriter # 引入 LogWriter
import inspect
import random
import numpy as np # 确保导入 numpy

# 引入用于模型导出的模块
import paddle.jit
import paddle.static
import json # 导入json用于保存配置概要

import random

def set_seed(seed: int):
    """设置全局随机种子以确保实验的可重复性。"""
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"全局随机种子已设置为: {seed}")

def evaluate(model: nn.Layer, head: nn.Layer | None, eval_loader: paddle.io.DataLoader, config: ConfigObject, epoch: int, log_writer: LogWriter | None = None, base_tag_prefix: str = "") -> tuple[float, float]:
    """
    在评估数据集上评估模型性能。
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    start_time = time.time()

    print(f"\n信息: 开始评估 Epoch {epoch + 1}/{config.epochs}...") # 简化评估开始打印

    with paddle.no_grad():
        for batch_id, (images, labels) in enumerate(eval_loader):
            outputs = model(images, labels)
            loss = None
            logits = None

            if isinstance(outputs, tuple) and len(outputs) == 2:
                 loss, logits = outputs
            elif isinstance(outputs, paddle.Tensor):
                 if config.loss_type == 'cross_entropy' and model.head:
                     logits = outputs
                     loss = F.cross_entropy(logits, labels)
                 else:
                     print(f"警告: 评估模式下，模型输出不包含损失和分类logits，跳过损失和准确率计算。")
                     total_loss += 0
                     continue

            if loss is not None:
                total_loss += loss.item() * labels.shape[0]

            if logits is not None:
                 predicted_labels = paddle.argmax(logits, axis=1)
                 correct_predictions += paddle.sum(predicted_labels == labels).item()
                 total_samples += labels.shape[0]

            if log_writer and batch_id == 0 and hasattr(config, 'log_eval_image_interval') and config.log_eval_image_interval is not None and (epoch + 1) % config.log_eval_image_interval == 0 :
                num_images_to_log = min(4, images.shape[0])
                # 随机选择要记录的图片索引
                if images.shape[0] > 0:
                    random_indices = random.sample(range(images.shape[0]), min(num_images_to_log, images.shape[0]))
                else:
                    random_indices = []

                for i, idx in enumerate(random_indices): # Iterate over random indices
                    img_tensor = images[idx] # Use random index
                    img_np_chw = img_tensor.numpy()
                    mean_unnorm = np.array(config.dataset_params.mean if hasattr(config.dataset_params, 'mean') and config.dataset_params.mean else [0.5, 0.5, 0.5]).reshape([3,1,1])
                    std_unnorm = np.array(config.dataset_params.std if hasattr(config.dataset_params, 'std') and config.dataset_params.std else [0.5, 0.5, 0.5]).reshape([3,1,1])
                    img_unnormalized_chw = img_np_chw * std_unnorm + mean_unnorm
                    img_uint8_chw = np.clip(img_unnormalized_chw * 255.0, 0, 255).astype(np.uint8)
                    img_hwc_uint8 = np.transpose(img_uint8_chw, (1, 2, 0))
                    task_name = config.class_name if hasattr(config, 'class_name') and config.class_name else "unknown_dataset"
                    log_writer.add_image(
                        tag=f"Eval/{task_name}/Epoch{epoch + 1}_RandomSample{i}", # Update tag to indicate random sample
                        img=img_hwc_uint8,
                        step=epoch + 1
                    )
                log_writer.flush()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    end_time = time.time()
    print(f"信息: 评估 Epoch {epoch + 1} 完成。耗时: {end_time - start_time:.2f}s, 损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}") # 简化评估结束打印

    if log_writer:
        log_writer.add_scalar(tag='Loss/Eval_Epoch', value=avg_loss, step=epoch + 1)
        log_writer.add_scalar(tag='Metric/Eval_Accuracy_Epoch', value=accuracy, step=epoch + 1)
        log_writer.flush()

    model.train()
    return avg_loss, accuracy

def train_one_epoch(model: nn.Layer, head: nn.Layer | None, train_loader: paddle.io.DataLoader, optimizer: opt.Optimizer, lr_scheduler, config: ConfigObject, epoch: int, log_writer: LogWriter | None = None, base_tag_prefix: str = ""):
    """
    执行一个训练周期的逻辑。
    """
    model.train()
    total_loss_this_epoch = 0.0
    total_correct_this_epoch = 0
    total_samples_this_epoch = 0
    epoch_start_time = time.time()

    print(f"\n信息: 训练 Epoch {epoch + 1}/{config.epochs} (当前LR: {optimizer.get_lr():.6f})...") # 简化训练开始打印

    for batch_id, (images, labels) in enumerate(train_loader):
        batch_start_time = time.time()
        global_step = epoch * len(train_loader) + batch_id

        outputs = model(images, labels)
        loss = None
        logits = None

        if isinstance(outputs, tuple) and len(outputs) == 2:
             loss, logits = outputs
        elif isinstance(outputs, paddle.Tensor):
             if config.loss_type == 'cross_entropy' and model.head:
                 logits = outputs
                 loss = F.cross_entropy(logits, labels)
             else:
                 # 这通常不应该在 train_one_epoch 中发生，因为模型设计为返回损失或logits
                 raise ValueError(f"错误: 训练模式下，模型输出不包含损失。请检查 CombinedModel 的 forward 方法和损失类型 '{config.loss_type}'。")

        if loss is None:
             raise ValueError(f"错误: 损失计算失败。请检查 CombinedModel 的 forward 方法和损失类型 '{config.loss_type}'。")
        
        # --- 检查损失是否为 NaN ---

        if paddle.isnan(loss):
            print(f"警告: Epoch {epoch + 1}, Batch {batch_id + 1}: 检测到 NaN 损失。这通常表示训练不稳定，请检查学习率、数据或模型配置。") # 调整警告措辞
            # 可以在这里添加 break 或其他处理逻辑，但目前只打印警告
            # continue # 暂时不跳过批次，以便观察更多 NaN 情况
        # --- 结束检查 ---


        total_loss_this_epoch += loss.item() * labels.shape[0]

        current_batch_acc = 0.0
        if logits is not None:
             predicted_labels = paddle.argmax(logits, axis=1)
             total_correct_this_epoch += paddle.sum(predicted_labels == labels).item()
             current_batch_acc = paddle.sum(predicted_labels == labels).item() / labels.shape[0] if labels.shape[0] > 0 else 0.0
        total_samples_this_epoch += labels.shape[0]

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if log_writer and (batch_id + 1) % config.log_interval == 0:
            current_batch_loss = loss.item()
            avg_loss_so_far_this_epoch = total_loss_this_epoch / total_samples_this_epoch if total_samples_this_epoch > 0 else 0.0
            avg_acc_so_far_this_epoch = total_correct_this_epoch / total_samples_this_epoch if total_samples_this_epoch > 0 else 0.0

            log_writer.add_scalar(tag='Loss/Train_Batch', value=current_batch_loss, step=global_step)
            if logits is not None:
                 log_writer.add_scalar(tag='Metric/Train_Accuracy_Batch', value=current_batch_acc, step=global_step)
                 log_writer.add_scalar(tag='Metric/Train_AvgAcc_EpochSoFar', value=avg_acc_so_far_this_epoch, step=global_step)
            log_writer.add_scalar(tag='Loss/Train_AvgLoss_EpochSoFar', value=avg_loss_so_far_this_epoch, step=global_step)
            log_writer.add_scalar(tag='LR/Batch', value=optimizer.get_lr(), step=global_step)
            log_writer.flush()

            batch_time = time.time() - batch_start_time
            print(f"  Epoch: {epoch + 1}, Batch: {batch_id + 1}/{len(train_loader)}, Loss: {current_batch_loss:.4f}, Acc: {current_batch_acc:.4f}, LR: {optimizer.get_lr():.6f}, Time: {batch_time:.2f}s") # 简化输出

        if log_writer and batch_id == 0 and hasattr(config, 'log_train_image_interval') and config.log_train_image_interval is not None and (epoch + 1) % config.log_train_image_interval == 0:
            num_images_to_log = min(4, images.shape[0])
             # 随机选择要记录的图片索引
            if images.shape[0] > 0:
                random_indices = random.sample(range(images.shape[0]), min(num_images_to_log, images.shape[0]))
            else:
                random_indices = []

            for i, idx in enumerate(random_indices): # Iterate over random indices
                img_tensor = images[idx] # Use random index
                img_np_chw = img_tensor.numpy()
                mean_unnorm = np.array(config.dataset_params.mean if hasattr(config.dataset_params, 'mean') and config.dataset_params.mean else [0.5, 0.5, 0.5]).reshape([3,1,1])
                std_unnorm = np.array(config.dataset_params.std if hasattr(config.dataset_params, 'std') and config.dataset_params.std else [0.5, 0.5, 0.5]).reshape([3,1,1])
                img_unnormalized_chw = img_np_chw * std_unnorm + mean_unnorm
                img_uint8_chw = np.clip(img_unnormalized_chw * 255.0, 0, 255).astype(np.uint8)
                img_hwc_uint8 = np.transpose(img_uint8_chw, (1, 2, 0))
                task_name = config.class_name if hasattr(config, 'class_name') and config.class_name else "unknown_dataset"
                log_writer.add_image(
                    tag=f"Train/{task_name}/Epoch{epoch+1}_RandomSample{i}", # Update tag to indicate random sample
                    img=img_hwc_uint8,
                    step=epoch + 1
                )
            log_writer.flush()

    if lr_scheduler and hasattr(lr_scheduler, 'step') and config.lr_scheduler_type.lower() != 'reduceonplateau':
        pass

    avg_train_loss = total_loss_this_epoch / total_samples_this_epoch if total_samples_this_epoch > 0 else 0.0
    avg_train_acc = total_correct_this_epoch / total_samples_this_epoch if total_samples_this_epoch > 0 else 0.0

    epoch_duration = time.time() - epoch_start_time
    print(f"信息: 训练 Epoch {epoch + 1} 概览: 损失: {avg_train_loss:.4f}, 准确率: {avg_train_acc:.4f}, 耗时: {epoch_duration:.2f}s") # 简化摘要

    if log_writer:
        log_writer.add_scalar(tag='Loss/Train_Epoch', value=avg_train_loss, step=epoch + 1)
        log_writer.add_scalar(tag='Metric/Train_Accuracy_Epoch', value=avg_train_acc, step=epoch + 1)
        log_writer.add_scalar(tag='LR/EpochEnd', value=optimizer.get_lr(), step=epoch + 1)
        log_writer.flush()

    if log_writer and hasattr(config, 'log_histogram_interval') and config.log_histogram_interval is not None and (epoch + 1) % config.log_histogram_interval == 0:
        print(f"信息: 记录参数直方图 Epoch {epoch + 1}/{config.epochs}...") # 简化直方图打印
        for name, param in model.named_parameters():
            if param.trainable:
                try:
                    log_writer.add_histogram(
                        tag=f"Hist/{name.replace('.', '/')}/Parameters",
                        values=param.numpy(),
                        step=epoch + 1,
                        buckets=10
                    )
                except Exception as e_hist_param:
                    print(f"警告: 记录参数直方图失败 (参数: {name}): {e_hist_param}")
                if param.grad is not None:
                    try:
                        log_writer.add_histogram(
                            tag=f"Hist/{name.replace('.', '/')}/Gradients",
                            values=param.grad.numpy(),
                            step=epoch + 1,
                            buckets=10
                        )
                    except Exception as e_hist_grad:
                        print(f"警告: 记录梯度直方图失败 (参数: {name}): {e_hist_grad}")
        log_writer.flush()
        print(f"信息: 参数直方图记录完毕 Epoch {epoch + 1}.") # 简化直方图结束打印

def train(final_config: ConfigObject, cmd_line_args: argparse.Namespace):
    place = paddle.CUDAPlace(0) if final_config.use_gpu else paddle.CPUPlace()
    paddle.set_device('gpu:0' if final_config.use_gpu else 'cpu')
    print(f"信息: 使用 {'GPU' if final_config.use_gpu else 'CPU'} 进行训练。随机种子: {final_config.seed}") # 简化并合并种子信息
    set_seed(final_config.seed)

    backbone_type = final_config.model_type
    loss_fn_type = final_config.loss_type
    lr_scheduler_name = final_config.lr_scheduler_type
    active_config_name = getattr(final_config, '_active_config_name', None) # Get active config name for logging

    # 获取配置的主要组件，包括学习率和权重衰减
    lr_value = final_config.learning_rate
    wd_value = final_config.optimizer_params.get('weight_decay', 0.0) if hasattr(final_config, 'optimizer_params') else 0.0
    
    # 格式化学习率和权重衰减为简洁形式
    lr_formatted = f"lr{str(lr_value).replace('0.', '')}"
    wd_formatted = f"wd{str(wd_value).replace('0.', '')}"
    
    # 创建更详细的组合目录名称，包含所有关键参数
    combo_dir_name = f"{backbone_type}__{loss_fn_type}__{final_config.optimizer_type}__{lr_scheduler_name}__{lr_formatted}__{wd_formatted}"
    
    # 如果active_config_name存在，可以直接使用它，因为它已经包含了所有组件
    if active_config_name:
        combo_dir_name = active_config_name
        
    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    current_run_dir_name = timestamp_str
    
    base_log_dir = final_config.visualdl_log_dir if hasattr(final_config, 'visualdl_log_dir') and final_config.visualdl_log_dir else "logs"
    current_logdir = os.path.join(base_log_dir, combo_dir_name, current_run_dir_name)

    os.makedirs(current_logdir, exist_ok=True)
    log_writer = LogWriter(logdir=current_logdir)
    print(f"信息: VisualDL 日志将保存到: {current_logdir}") # 简化日志路径打印

    hparams_dict = {
        'backbone': backbone_type,
        'loss': loss_fn_type,
        'lr_scheduler': lr_scheduler_name,
        'learning_rate': final_config.learning_rate,
        'batch_size': final_config.batch_size,
        'optimizer': final_config.optimizer_type,
        'epochs': final_config.epochs,
        'seed': final_config.seed,
        'image_size': final_config.image_size,
        'num_classes': final_config.num_classes,
        'log_dir_combo': combo_dir_name,
        'run_timestamp': timestamp_str
    }
    if hasattr(final_config, 'optimizer_params') and final_config.optimizer_params:
        for opt_param_key, opt_param_val in final_config.optimizer_params.items():
            hparams_dict[f"opt_{opt_param_key}"] = opt_param_val
            
    if active_config_name:
        hparams_dict['active_config_yaml'] = active_config_name
    
    tracked_metrics_for_hparams = [
        'Loss/Train_Epoch',
        'Metric/Train_Accuracy_Epoch',
        'Loss/Eval_Epoch',
        'Metric/Eval_Accuracy_Epoch',
        'LR/EpochEnd'
    ]
    try:
        log_writer.add_hparams(hparams_dict=hparams_dict, metrics_list=tracked_metrics_for_hparams)
        print("信息: 超参数已记录到 VisualDL HParams。") # 调整为信息
    except Exception as e_hparam:
        print(f"警告: 记录超参数到 VisualDL HParams 失败: {e_hparam}。将保存为JSON。") # 调整为警告
        config_summary_path = os.path.join(current_logdir, "hparams_summary.json")
        with open(config_summary_path, 'w', encoding='utf-8') as f_cfg:
            json.dump({"hparams": hparams_dict, "tracked_metrics_for_hparams": tracked_metrics_for_hparams}, f_cfg, indent=4, ensure_ascii=False)
        print(f"信息: 超参数概要已保存为JSON: {config_summary_path}") # 调整为信息

    log_writer.flush()

    print("\n信息: --- 数据加载器创建 ---") # 新增
    print("信息: 正在创建训练数据加载器...")
    train_loader, num_train_classes = MyReader.create_data_loader(final_config, mode='train')
    # 更新配置中的类别数量，以确保模型头部使用实际的数据类别数
    final_config.num_classes = num_train_classes 
    print("信息: 正在创建验证数据加载器...")
    validation_loader, _ = MyReader.create_data_loader(final_config, mode='validation') # 第二个返回值可以忽略，因为训练类别数已更新
    print("信息: 正在创建测试数据加载器...") # 简化
    test_loader, _ = MyReader.create_data_loader(final_config, mode='test')

    model_backbone, backbone_out_dim = get_backbone(
        config_model_params=final_config.model.get(f'{final_config.model_type}_params', {}),
        model_type_str=final_config.model_type,
        image_size=final_config.image_size
    )
    print(f"信息: 骨干网络 ({final_config.model_type.upper()}) 加载成功，输出特征维度: {backbone_out_dim}") # 调整为信息

    model_head = None
    if final_config.loss_type in ['cross_entropy', 'arcface']:
        model_head = get_head(
            config_loss_params=final_config.loss.get(f'{final_config.loss_type}_params', {}),
            loss_type_str=final_config.loss_type,
            in_features=backbone_out_dim,
            num_classes=final_config.num_classes
        )
        print(f"信息: 头部模块 ({final_config.loss_type.upper()}) 加载成功，输入特征维度: {backbone_out_dim}, 输出类别数: {final_config.num_classes}") # 调整为信息

    class CombinedModel(nn.Layer):
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
                return features
    model = CombinedModel(model_backbone, model_head)

    total_steps_per_epoch = len(train_loader)
    total_training_steps = total_steps_per_epoch * final_config.epochs

    # --- 新增：打印实际加载的warmup配置 ---
    print(f"调试信息: final_config.lr_scheduler_params.warmup: {final_config.lr_scheduler_params.warmup}")
    # --- 结束新增 ---

    lr_scheduler = get_lr_scheduler(
        config=final_config,
        initial_learning_rate=final_config.learning_rate,
        total_steps=total_training_steps,
        epochs=final_config.epochs,
        steps_per_epoch=total_steps_per_epoch,
    )

    optimizer_params = final_config.optimizer_params if final_config.optimizer_params is not None else {}
    optimizer = None
    if final_config.optimizer_type.lower() == 'adamw':
         optimizer = opt.AdamW(
             learning_rate=lr_scheduler,
             parameters=model.parameters(),
             weight_decay=optimizer_params.get('weight_decay', 0.0)
         )
    elif final_config.optimizer_type.lower() == 'momentum':
         optimizer = opt.Momentum(
             learning_rate=lr_scheduler,
             parameters=model.parameters(),
             momentum=optimizer_params.get('momentum', 0.9),
             weight_decay=optimizer_params.get('weight_decay', 0.0005)
         )
    else:
        raise ValueError(f"错误: 不支持的优化器类型: {final_config.optimizer_type}")
    print(f"信息: 优化器 ({final_config.optimizer_type.upper()}) 已构建。") # 新增

    # 定义固定的检查点文件名
    fixed_checkpoint_filename_base = "model_checkpoint"

    # --- 新的检查点加载逻辑 ---
    start_epoch = 0
    best_acc = 0.0
    loaded_meta_data = {} # Initialize with an empty dict

    if final_config.resume: # final_config.resume is True by default from YAML (after change)
        combo_logs_path = os.path.join(base_log_dir, combo_dir_name)
        latest_prev_run_dir = None
        if os.path.exists(combo_logs_path):
            all_prev_runs = sorted([
                d for d in os.listdir(combo_logs_path)
                if os.path.isdir(os.path.join(combo_logs_path, d)) and d != current_run_dir_name # Exclude current run
            ]) # Sorts alphabetically, which works for YYYYMMDD-HHMMSS
            if all_prev_runs:
                latest_prev_run_dir_name = all_prev_runs[-1]
                latest_prev_run_dir = os.path.join(combo_logs_path, latest_prev_run_dir_name)
        
        if latest_prev_run_dir:
            previous_checkpoint_save_subdir = os.path.join(latest_prev_run_dir, "checkpoints")
            # Correctly form the expected paths including the "checkpoint_" prefix used by CheckpointManager
            expected_pdparams_path = os.path.join(previous_checkpoint_save_subdir, f"checkpoint_{fixed_checkpoint_filename_base}.pdparams")
            expected_metadata_path = os.path.join(previous_checkpoint_save_subdir, f"checkpoint_{fixed_checkpoint_filename_base}.json")

            if os.path.exists(expected_pdparams_path) and os.path.exists(expected_metadata_path):
                print(f"信息: 检测到先前运行的检查点，尝试从 {previous_checkpoint_save_subdir} 恢复训练。") # 调整为信息
                # 使用临时的 CheckpointManager 从先前运行加载
                loader_checkpoint_manager = CheckpointManager(
                    model_save_dir=previous_checkpoint_save_subdir,
                    model_name=fixed_checkpoint_filename_base # Use fixed name for loading
                )
                # load_checkpoint should correctly set its internal paths based on its init params
                start_epoch, best_acc, loaded_meta_data = loader_checkpoint_manager.load_checkpoint(
                    model, optimizer, lr_scheduler, resume=True # resume=True here to force loading attempt
                )
                if start_epoch > 0: # Indicates successful load
                    print(f"信息: 成功从 {latest_prev_run_dir} 恢复。起始 epoch: {start_epoch}, 上次运行的最佳准确率: {best_acc:.4f}") # 调整为信息
                else:
                    print(f"警告: 尝试从 {latest_prev_run_dir} 加载检查点但未能恢复有效状态 (起始 epoch 为 0)。将从头开始训练。") # 调整为警告
                    # Ensure best_acc is reset if load "failed" to produce a resume state
                    best_acc = 0.0
                    loaded_meta_data = {} 
            else:
                print(f"警告: 配置了恢复训练 (resume=True)，但在最新的先前运行目录 {latest_prev_run_dir} 中未找到有效检查点文件。将从头开始训练。") # 调整为警告
        else:
            print(f"警告: 配置了恢复训练 (resume=True)，但未找到相同超参数组合的先前运行记录。将从头开始训练。") # 调整为警告
    else: # final_config.resume is False
        print("信息: 未配置恢复训练 (resume=False)。将从头开始训练。") # 调整为信息

    # --- 初始化用于保存到当前运行目录的 CheckpointManager ---

    checkpoint_save_subdir = os.path.join(current_logdir, "checkpoints")
    # checkpoint_manager_model_name = active_config_name if active_config_name else combo_dir_name # 旧的动态命名
    checkpoint_manager = CheckpointManager(
        model_save_dir=checkpoint_save_subdir,
        model_name=fixed_checkpoint_filename_base # 使用固定的基础名称
    )
    # 注意: loaded_meta_data['previous_best_acc'] 会由 CheckpointManager.load_checkpoint 内部设置。
    # 如果是全新训练，best_acc 是 0.0， save_checkpoint 将正确处理。

    # 旧的加载逻辑 (直接加载到当前目录的CheckpointManager，如果存在的话)
    # start_epoch, best_acc, loaded_meta_data = checkpoint_manager.load_checkpoint(model, optimizer, lr_scheduler, final_config.resume)

    training_start_time = time.time()
    print(f"开始训练，总共 {final_config.epochs} 个 epochs... 从 epoch {start_epoch} 开始")

    for epoch in range(start_epoch, final_config.epochs):
        train_one_epoch(model, model_head, train_loader, optimizer, lr_scheduler, final_config, epoch, log_writer, base_tag_prefix="")
        # 在每个 epoch 结束后，使用验证集进行评估
        validation_avg_loss, validation_accuracy = evaluate(model, model_head, validation_loader, final_config, epoch, log_writer, base_tag_prefix="Validation")

        if final_config.lr_scheduler_type.lower() == 'reduceonplateau' and lr_scheduler:
             lr_scheduler.step(validation_avg_loss)

        is_best = validation_accuracy > best_acc # 根据验证集准确率决定是否保存最佳模型

        current_metadata = {
            'epoch': epoch + 1,
            'best_acc': max(validation_accuracy, best_acc),
            'config_name': active_config_name if active_config_name else combo_dir_name,
            'model_type': final_config.model_type,
            'loss_type': final_config.loss_type,
            'lr_scheduler_type': final_config.lr_scheduler_type,
            'optimizer_type': final_config.optimizer_type,
            'batch_size': final_config.batch_size,
            'image_size': final_config.image_size,
            'num_classes': final_config.num_classes,
            'seed': final_config.seed,
            'last_eval_accuracy': validation_accuracy,
            'last_eval_loss': validation_avg_loss,
            'previous_best_acc': best_acc,
            'model_specific_params': final_config.model.get(f'{final_config.model_type}_params', {}).to_dict() 
                                     if isinstance(final_config.model.get(f'{final_config.model_type}_params', {}), ConfigObject) 
                                     else final_config.model.get(f'{final_config.model_type}_params', {}),
            'loss_specific_params': final_config.loss.get(f'{final_config.loss_type}_params', {}).to_dict()
                                    if isinstance(final_config.loss.get(f'{final_config.loss_type}_params', {}), ConfigObject)
                                    else final_config.loss.get(f'{final_config.loss_type}_params', {})
        }
        checkpoint_manager.save_checkpoint(model, optimizer, lr_scheduler, current_metadata, is_best=is_best)
        if is_best:
            best_acc = validation_accuracy

    print("\n信息: --- 训练结束 ---") # 简化
    print(f"信息: 总训练耗时: {time.time() - training_start_time:.2f}s") # 简化
    print(f"信息: 最佳验证集准确率: {best_acc:.4f}") # 简化
    print(f"信息: 最新检查点保存于: {checkpoint_manager.checkpoint_path}") # 简化
    print(f"信息: 性能最佳的模型保存于: {checkpoint_manager.best_model_path}") # 简化

    # 训练结束后，使用测试集进行最终评估
    print("\n信息: --- 在测试集上进行最终评估 ---") # 简化
    final_test_loss, final_test_accuracy = evaluate(model, model_head, test_loader, final_config, final_config.epochs, log_writer, base_tag_prefix="FinalTest")
    print(f"信息: 最终测试集结果: 损失: {final_test_loss:.4f}, 准确率: {final_test_accuracy:.4f}") # 简化
    if log_writer:
        log_writer.add_scalar(tag='Metric/Final_Test_Accuracy', value=final_test_accuracy, step=final_config.epochs)
        log_writer.add_scalar(tag='Loss/Final_Test_Loss', value=final_test_loss, step=final_config.epochs)
        log_writer.flush()

    print("\n信息: --- 模型导出 ---") # 新增
    export_model_name_prefix = "model_for_graph" 
    export_path_prefix = os.path.join(current_logdir, export_model_name_prefix)

    try:
        model_to_export, _ = get_backbone(
            config_model_params=final_config.model.get(f'{final_config.model_type}_params', {}),
            model_type_str=final_config.model_type,
            image_size=final_config.image_size
        )

        if not os.path.exists(checkpoint_manager.best_model_path):
            print(f"警告: 最佳模型文件 {checkpoint_manager.best_model_path} 未找到，无法导出用于Graph的骨干网络。")
            model_to_export = None
        else:
            print(f"信息: 从最佳模型加载骨干网络权重: {checkpoint_manager.best_model_path}")
            full_model_state_dict = paddle.load(checkpoint_manager.best_model_path)
            backbone_state_dict = {k.replace('backbone.', '', 1): v for k, v in full_model_state_dict.items() if k.startswith('backbone.')}

            if not backbone_state_dict:
                print(f"警告: 在最佳模型 {checkpoint_manager.best_model_path} 中未找到 'backbone.' 前缀的权重。将尝试直接加载整个状态字典到 model_to_export。")
                try:
                    model_to_export.set_state_dict(full_model_state_dict)
                    print("信息: 直接加载权重到 model_to_export 成功。")
                except RuntimeError as e_load_direct:
                    print(f"错误: 直接加载权重到 model_to_export 失败: {e_load_direct}。Graph导出将跳过。")
                    model_to_export = None
            else:
                model_to_export.set_state_dict(backbone_state_dict)

        if model_to_export:
            model_to_export.eval()
            dummy_input = paddle.randn([1, 3, final_config.image_size, final_config.image_size], dtype='float32')
            paddle.jit.save(
                layer=model_to_export,
                path=export_path_prefix,
                input_spec=[paddle.static.InputSpec(shape=[None, 3, final_config.image_size, final_config.image_size], dtype='float32')]
            )
            print(f"信息: 用于Graph可视化的骨干网络模型已成功导出到: {export_path_prefix}.pdmodel 和 {export_path_prefix}.pdiparams。") # 简化
        else:
            print("警告: 由于权重加载问题或 model_to_export 为 None，跳过Graph模型导出。") # 调整为警告

    except Exception as e:
        print(f"错误: 导出用于Graph可视化的模型失败: {e}")

    if log_writer:
        log_writer.close()
        print("信息: VisualDL LogWriter 已关闭。") # 调整为信息

    # --- 训练完成后，如果使用的是ArcFace，自动创建特征库 ---

    if final_config.loss_type.lower() == 'arcface':
        print("\n--- ArcFace 模型训练完成，尝试自动创建特征库 ---")
        try:
            # 确保导入 create_face_library 函数
            try:
                from create_face_library import create_face_library as build_feature_lib
                print("信息: 成功导入 create_face_library.py 中的 build_feature_lib 函数。") # 调整为信息
            except ImportError as e_import_cfl:
                print(f"错误: 导入 create_face_library 模块失败: {e_import_cfl}。自动建库将跳过。")
                build_feature_lib = None

            if build_feature_lib:
                # 确保最佳模型路径存在
                if not checkpoint_manager.best_model_path or not os.path.exists(checkpoint_manager.best_model_path):
                    print(f"错误: 最佳模型路径 {checkpoint_manager.best_model_path} 未找到或无效。无法自动创建特征库。")
                else:
                    print(f"准备使用最佳模型 {checkpoint_manager.best_model_path} 自动创建特征库。")
                    pseudo_cmd_args_for_lib_creation = argparse.Namespace(
                        config_path=cmd_line_args.config_path, # 复用原始的config_path
                        active_config=active_config_name, # 使用当前运行的 active_config
                        model_path=checkpoint_manager.best_model_path, # 使用当前运行的最佳模型
                        data_dir=final_config.data_dir, 
                        # class_name 将从 final_config 获取
                        # face_library_path 设为 None，让 create_face_library 决定默认输出路径
                        face_library_path=None, 
                        use_gpu=final_config.use_gpu,
                        image_size=final_config.image_size, # 从 final_config 获取
                        # data_list_for_library: create_face_library 会从 config 中查找或使用默认
                    )
                    
                    # 预期特征库的保存路径
                    expected_lib_output_filename = final_config.create_library.get('output_library_path', 'face_library.pkl') \
                                                 if hasattr(final_config, 'create_library') and final_config.create_library else 'face_library.pkl'
                    expected_lib_full_path = os.path.join(os.path.dirname(checkpoint_manager.best_model_path), expected_lib_output_filename)
                    
                    print(f"调用 build_feature_lib。预期特征库将保存在: {expected_lib_full_path}")
                    try:
                        build_feature_lib(config=final_config, cmd_args=pseudo_cmd_args_for_lib_creation)
                        print("build_feature_lib 调用完成。")
                        if os.path.exists(expected_lib_full_path):
                            print(f"成功: 特征库已创建于 {expected_lib_full_path}")
                        else:
                            print(f"警告: build_feature_lib 调用后，预期的特征库文件 {expected_lib_full_path} 未找到。请检查 create_face_library.py 的逻辑。")
                    except Exception as e_build_lib_call:
                        print(f"错误: 调用 build_feature_lib 时发生异常: {e_build_lib_call}")
                        print(f"       您可以稍后手动运行 create_face_library.py 脚本，并指定模型路径为: {checkpoint_manager.best_model_path}")

            else: # build_feature_lib is None due to import error
                print("警告: 由于导入 create_face_library 失败，跳过自动创建特征库。") # 调整为警告

        except Exception as e_create_lib_outer:
            print(f"错误: 自动创建特征库的准备阶段失败: {e_create_lib_outer}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸识别模型训练脚本')
    parser.add_argument('--config_path', type=str, default='configs/default_config.yaml',
                        help='配置文件的路径')
    parser.add_argument('--active_config', type=str, default=None,
                        help='要激活的配置块名称 (覆盖YAML中的 active_config)')
    
    parser.add_argument('--log_train_image_interval', type=int, default=None, help='训练时记录图片的 epoch 间隔 (覆盖YAML中同名参数)')
    parser.add_argument('--log_eval_image_interval', type=int, default=None, help='评估时记录图片的 epoch 间隔 (覆盖YAML中同名参数)')
    parser.add_argument('--log_histogram_interval', type=int, default=None, help='记录直方图的 epoch 间隔 (覆盖YAML中同名参数)')
    parser.add_argument('--visualdl_log_dir', type=str, default=None, help='VisualDL 日志的基础保存目录 (覆盖YAML中同名参数)')

    parser.add_argument('--use_gpu', action='store_true', help='是否使用 GPU (覆盖YAML)')
    parser.add_argument('--resume', action='store_true', help='是否从检查点恢复 (覆盖YAML)')
    parser.add_argument('--data_dir', type=str, default=None, help='数据集根目录 (覆盖YAML)')
    parser.add_argument('--model_save_dir', type=str, default=None, help='模型检查点保存目录的父目录 (此参数作用已减弱, 实际路径由 visualdl_log_dir 和组合名决定)')
    parser.add_argument('--epochs', type=int, default=None, help='训练 epoch 数 (覆盖YAML)')
    parser.add_argument('--batch_size', type=int, default=None, help='批处理大小 (覆盖YAML)')
    parser.add_argument('--learning_rate', type=float, default=None, help='初始学习率 (覆盖YAML)')
    parser.add_argument('--log_interval', type=int, default=None, help='控制台打印日志的 batch 间隔 (覆盖YAML)')
    parser.add_argument('--seed', type=int, default=None, help='随机种子 (覆盖YAML)')
    parser.add_argument('--source', type=str, default='manual', help='训练来源标识 (覆盖YAML)')

    cmd_line_args = parser.parse_args()
    final_config = load_config(
        default_yaml_path=cmd_line_args.config_path,
        cmd_args_namespace=cmd_line_args
    )
    train(final_config, cmd_line_args)