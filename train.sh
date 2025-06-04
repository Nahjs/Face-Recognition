#!/bin/bash
echo "LD_LIBRARY_PATH when train.sh starts: $LD_LIBRARY_PATH"
set -e # 任何命令失败立即退出

# --- 用户配置区域 (WSL 本地环境) ---
PROJECT_DIR="$(pwd)" # 使用当前目录作为项目目录，或者您可以硬编码一个绝对路径
VENV_PATH="paddle/bin/activate" # 虚拟环境激活脚本的相对路径 (相对于PROJECT_DIR)
CONFIG_FILE="configs/48_config.yaml"
LOG_DIR="${PROJECT_DIR}/logs_Terminal-4" # 日志文件保存目录

# 需要在本地 GPU 训练的配置块名称列表
CONFIG_NAMES_TO_TRAIN=(
    # ResNet 作为骨干网络通常性能较强；ArcFace Loss 适合识别任务；AdamW 是一种常用的优化器；
    #CosineAnnealingDecay 是一种使学习率先快速下降然后缓慢下降的调度器，常能获得不错的效果。
    resnet_arcface_adamw_cosineannealingdecay_warmup_config
    #ResNet 骨干网络；ArcFace Loss；Momentum 是另一种经典的优化器，常用于图像任务；
    #StepDecay 是一个简单的学习率调度器，在预设的 epoch 节点降低学习率。与 AdamW+CosineDecay 组合形成对比。
    resnet_arcface_momentum_stepdecay_config
    # 下面两个用来做对比，ArcFace 和 CrossEntropy 的对比
    vgg_arcface_adamw_cosineannealingdecay_config
    vgg_arcface_momentum_stepdecay_config

    # --- 新增交叉熵模型组合 (共4种) ---
    # ResNet 骨干网络；CrossEntropy Loss；AdamW 优化器；CosineAnnealingDecay 调度器
    resnet_ce_adamw_cosineannealingdecay_config
    # ResNet 骨干网络；CrossEntropy Loss；Momentum 优化器；StepDecay 调度器
    resnet_ce_momentum_stepdecay_config
    # VGG 骨干网络；CrossEntropy Loss；AdamW 优化器；CosineAnnealingDecay 调度器
    vgg_ce_adamw_cosineannealingdecay_config
    # VGG 骨干网络；CrossEntropy Loss；Momentum 优化器；StepDecay 调度器
    vgg_ce_momentum_stepdecay_config
    
    # "vgg_ce_steplr_config"
    # "vgg_ce_multistep_config"
    # "vgg_ce_cosine_config"
    # "vgg_ce_reduce_lr_config"
    # "vgg_ce_warm_restarts_config"
    # "vgg_arcface_steplr_config"
    # "vgg_arcface_multistep_config"
    # "vgg_arcface_cosine_config"
    # "vgg_arcface_reduce_lr_config"
    # "vgg_arcface_warm_restarts_config"
    # "resnet_ce_steplr_config"
    # "resnet_ce_multistep_config"
    # "resnet_ce_cosine_config"
    # "resnet_ce_reduce_lr_config"
    # "resnet_ce_warm_restarts_config"
    # "resnet_arcface_steplr_config"
    # "resnet_arcface_multistep_config"
    # "resnet_arcface_cosine_config"
    # "resnet_arcface_reduce_lr_config"
    # "resnet_arcface_warm_restarts_config"
    # vgg_ce_adamw_stepdecay_config
    # vgg_ce_adamw_multistepdecay_config
    # vgg_ce_adamw_cosineannealingdecay_config
    # vgg_ce_adamw_reduceonplateau_config
    # vgg_ce_adamw_cosineannealingwarmrestarts_config
    # vgg_ce_adamw_polynomialdecay_config
    # vgg_ce_momentum_stepdecay_config
    # vgg_ce_momentum_multistepdecay_config
    # vgg_ce_momentum_cosineannealingdecay_config
    # vgg_ce_momentum_reduceonplateau_config
    # vgg_ce_momentum_cosineannealingwarmrestarts_config
    # vgg_ce_momentum_polynomialdecay_config
    # vgg_arcface_adamw_stepdecay_config
    # vgg_arcface_adamw_multistepdecay_config
    # vgg_arcface_adamw_cosineannealingdecay_config
    # vgg_arcface_adamw_reduceonplateau_config
    # vgg_arcface_adamw_cosineannealingwarmrestarts_config
    # vgg_arcface_adamw_polynomialdecay_config
    # vgg_arcface_momentum_stepdecay_config
    # vgg_arcface_momentum_multistepdecay_config
    # vgg_arcface_momentum_cosineannealingdecay_config
    # vgg_arcface_momentum_reduceonplateau_config
    # vgg_arcface_momentum_cosineannealingwarmrestarts_config
    # vgg_arcface_momentum_polynomialdecay_config
    # resnet_ce_adamw_stepdecay_config
    # resnet_ce_adamw_multistepdecay_config
    # resnet_ce_adamw_reduceonplateau_config
    # resnet_ce_adamw_cosineannealingwarmrestarts_config
    # resnet_ce_adamw_polynomialdecay_config
    # resnet_ce_momentum_stepdecay_config
    # resnet_ce_momentum_multistepdecay_config
    # resnet_ce_momentum_cosineannealingdecay_config
    # resnet_ce_momentum_reduceonplateau_config
    # resnet_ce_momentum_cosineannealingwarmrestarts_config
    # resnet_ce_momentum_polynomialdecay_config
    # resnet_arcface_adamw_stepdecay_config 
    # resnet_arcface_adamw_cosineannealingdecay_warmup_config # 推荐组合，已在列表开头
    # resnet_arcface_adamw_multistepdecay_config
    # resnet_arcface_adamw_reduceonplateau_config
    # resnet_arcface_adamw_cosineannealingwarmrestarts_config
    # resnet_arcface_adamw_polynomialdecay_config
    
    # resnet_arcface_momentum_multistepdecay_config
    # resnet_arcface_momentum_cosineannealingdecay_config
    # resnet_arcface_momentum_reduceonplateau_config
    # resnet_arcface_momentum_cosineannealingwarmrestarts_config
    # resnet_arcface_momentum_polynomialdecay_config

)

# --- 脚本开始 ---
echo "===================================================="
echo "本地 GPU 自动化训练脚本启动..."
echo "项目目录: ${PROJECT_DIR}"
echo "配置文件: ${CONFIG_FILE}"
echo "日志目录: ${LOG_DIR}"
echo "===================================================="

# 进入项目目录 (如果 PROJECT_DIR 是相对路径，此步骤可能多余，但为保险起见保留)
# 如果 PROJECT_DIR 已是绝对路径或 $(pwd) 结果正确，则此cd可能不需要
# cd "${PROJECT_DIR}" || { echo "错误: 无法进入项目目录 ${PROJECT_DIR}"; exit 1; }

# 创建日志目录 (如果不存在)
mkdir -p "${LOG_DIR}" || { echo "错误: 无法创建日志目录 ${LOG_DIR}"; exit 1; }

# 激活虚拟环境
VENV_FULL_PATH="${PROJECT_DIR}/${VENV_PATH}"
echo "--> 尝试激活虚拟环境: ${VENV_FULL_PATH}"
if [ -f "${VENV_FULL_PATH}" ]; then
    source "${VENV_FULL_PATH}" || { echo "错误: 无法激活虚拟环境 ${VENV_FULL_PATH}"; exit 1; }
    echo "Python环境: $(which python)"
else
    echo "警告: 虚拟环境激活脚本未找到于 ${VENV_FULL_PATH}。"
    echo "     将尝试在当前Python环境执行，请确保依赖已安装。"
fi


# --- 主训练循环 ---
# 如果您希望脚本在所有模型达到目标epoch后停止，请移除或注释掉外层的 'while true; do' 和 'done'
# 以及相关的 sleep 和 .stop_local_train 检查。
# 当前默认行为是只执行一轮所有配置的训练，直到它们各自达到目标epoch。
# 要实现持续运行，请取消注释以下 'while true; do' 和其对应的 'done' 以及内部的控制逻辑。

echo "----------------- 开始新一轮训练检查/执行 -----------------"

# 可选：从Git拉取最新的代码和配置
# echo "--> 正在从Git仓库拉取最新代码 (origin main)..."
# if git pull origin main; then
#     echo "代码拉取成功。"
# else
#     echo "警告: git pull 失败。将使用本地现有代码继续。"
# fi

all_configs_completed_this_round=true # 假设初始为true，如果任何配置失败则设为false

for config_name in "${CONFIG_NAMES_TO_TRAIN[@]}"; do
    echo ""
    echo "===================================================="
    echo "准备训练配置: ${config_name} (GPU, 手动来源)"
    echo "===================================================="

    # 使用更规范的日志文件名，包含日期和时间戳
    LOG_FILE_TRAIN="${LOG_DIR}/train_${config_name}_gpu_manual_$(date +%Y%m%d_%H%M%S).log"
    echo "训练日志将保存到: ${LOG_FILE_TRAIN}"

    # 构建 train.py 命令
    TRAIN_CMD="python train.py --config_path "${CONFIG_FILE}" --active_config "${config_name}" --use_gpu --resume --source manual" 

    echo "执行命令: ${TRAIN_CMD}"
    # 使用 eval 执行命令，并将标准输出和错误都重定向到日志文件
    # PIPESTATUS[0] 用于获取管道中第一个命令 (python train.py) 的退出码
    eval ${TRAIN_CMD} 2>&1 | tee "${LOG_FILE_TRAIN}"
    train_exit_code=${PIPESTATUS[0]} 

    if [ $train_exit_code -ne 0 ]; then
        echo "===================================================="
        echo "警告: 配置 '${config_name}' 的训练过程失败 (退出码: ${train_exit_code})。"
        echo "详情请查看日志: '${LOG_FILE_TRAIN}'"
        echo "===================================================="
        all_configs_completed_this_round=false # 标记本轮有配置未完成
        # 决定是继续下一个配置还是怎样。当前是继续。
        # break # 如果一个失败就停止整个 for 循环
        continue # 跳过当前失败的配置，继续下一个
    else
         echo "===================================================="
         echo "配置 '${config_name}' 的训练过程已完成或已达到目标epoch。"
         echo "日志: '${LOG_FILE_TRAIN}'"
         echo "===================================================="
         # 训练成功后，可以考虑触发上传脚本同步模型到服务器
         # echo "--> 训练完成，正在同步模型到服务器..."
         # UPLOAD_SCRIPT_PATH="${PROJECT_DIR}/upload_to_server.sh"
         # if [ -f "${UPLOAD_SCRIPT_PATH}" ]; then
         #    bash "${UPLOAD_SCRIPT_PATH}" # 调用上传脚本
         #    if [ $? -ne 0 ]; then
         #        echo "警告: 同步模型到服务器失败！"
         #    fi
         # else
         #    echo "警告: 上传脚本 ${UPLOAD_SCRIPT_PATH} 未找到。"
         # fi
    fi
done # 结束 for 循环 (遍历所有配置)

echo "----------------- 本轮所有配置训练检查/执行完毕 -----------------"

# --- 持续运行逻辑 (如果需要，取消注释以下 'while true; do' 块) ---
# while true; do
#    echo "----------------- 开始新一轮训练检查/执行 -----------------"
#    # ... (复制上面 for 循环的逻辑到这里) ...
#
#    if ! $all_configs_completed_this_round; then
#        echo "本轮训练中存在失败的配置，将在一分钟后重试..."
#        sleep 60
#    else
#        echo "本轮所有配置均成功执行 (或已达目标epoch)。"
#        # 可选：完成一轮所有配置训练后暂停一段时间
#        echo "暂停 600 秒 (10 分钟)... 按 Ctrl+C 提前中止。"
#        sleep 600
#    fi
#    
#    # 检查停止旗标 (仅当 while true 循环激活时有意义)
#    STOP_FLAG_FILE="${PROJECT_DIR}/.stop_local_train"
#    if [ -f "${STOP_FLAG_FILE}" ]; then
#        echo "检测到停止旗标 (${STOP_FLAG_FILE})，脚本将退出。"
#        rm -f "${STOP_FLAG_FILE}" # 删除旗标，以便下次正常启动
#        break # 退出 while true 循环
#    fi
# done # 结束 while true 循环

# 尝试停用虚拟环境 (如果之前成功激活)
if [[ -n "${VIRTUAL_ENV}" ]]; then # VIRTUAL_ENV 变量由 source activate 设置
    echo "--> 尝试停用虚拟环境..."
    deactivate &>/dev/null || echo "停用虚拟环境命令 (deactivate) 执行遇到问题或未找到。"
else
    echo "--> 未检测到活动的虚拟环境，无需停用。"
fi

echo "===================================================="
echo "本地 GPU 自动化训练脚本结束。"
echo "===================================================="
