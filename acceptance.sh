#!/bin/bash
echo "LD_LIBRARY_PATH when train.sh starts: $LD_LIBRARY_PATH"
set -e # 任何命令失败立即退出

# --- 用户配置区域 (WSL 本地环境) ---
PROJECT_DIR="$(pwd)" # 使用当前目录作为项目目录，或者您可以硬编码一个绝对路径
VENV_PATH="paddle/bin/activate" # 虚拟环境激活脚本的相对路径 (相对于PROJECT_DIR)
CONFIG_FILE="configs/48_config.yaml"
LOGS_BASE_DIR="logs" # 日志文件保存目录

# 新增命令行参数解析
MODE="acceptance_test" # 默认模式
INPUT_IMAGE_PATH=""

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift ;;
        --image_path) INPUT_IMAGE_PATH="$2"; shift ;;
        # 可以在这里添加更多全局参数的解析，如果需要覆盖其他变量
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

# --- 自动推断模式 (如果提供了 --image_path 且未明确指定模式) ---
if [ -n "${INPUT_IMAGE_PATH}" ] && [ "$MODE" == "acceptance_test" ]; then
    echo "信息: 检测到 --image_path 参数，且未明确指定运行模式，自动切换到 'compare_single_image' 模式。"
    MODE="compare_single_image"
fi

# --- 脚本开始 --- (保留原有日志和虚拟环境激活，但放在参数解析之后)
echo "=================================================="
echo "自动化验收/模型比较脚本启动..."
echo "项目目录: ${PROJECT_DIR}"
echo "配置文件: ${CONFIG_FILE}"
echo "日志基础目录: ${LOGS_BASE_DIR}"
echo "当前运行模式: ${MODE}"
if [ -n "${INPUT_IMAGE_PATH}" ]; then
    echo "输入图片路径: ${INPUT_IMAGE_PATH}"
fi
echo "=================================================="

# 创建日志目录 (如果不存在) - 虽然这里主要用于 train.sh，但为一致性保留
mkdir -p "${PROJECT_DIR}/${LOGS_BASE_DIR}" || { echo "错误: 无法创建日志基础目录 ${PROJECT_DIR}/${LOGS_BASE_DIR}"; exit 1; }

# 激活虚拟环境
VENV_FULL_PATH="${PROJECT_DIR}/${VENV_PATH}"
echo "--> 尝试激活虚拟环境: ${VENV_FULL_PATH}"
if [ -f "${VENV_FULL_PATH}" ]; then
    source "${VENV_FULL_PATH}" || { echo "错误: 无法激活虚拟环境 ${VENV_FULL_PATH}"; exit 1; }
    echo "Python环境: $(which python)"
else
    echo "警告: 虚拟环境激活脚本未找到于 ${VENV_FULL_PATH}。
     将尝试在当前Python环境执行，请确保依赖已安装。"
fi

# Check for jq (needed to parse JSON metadata and results) and check for Bash version (for associative arrays)
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install it (e.g., sudo apt-get install jq on Ubuntu/Debian)."
    exit 1
fi

# Check for Bash version >= 4 for associative arrays
if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    echo "Error: Bash version 4 or higher is required for associative arrays. Your version is ${BASH_VERSION}."
    echo "Please upgrade Bash (e.g., sudo apt-get update && sudo apt-get install --only-upgrade bash)."
    exit 1
fi

# 声明一个关联数组来存储ArcFace模型的特征库路径
declare -A ARCFACE_LIBRARY_PATHS
# 存储所有找到的模型路径 (使用 mapfile 避免子shell问题)
declare -a ALL_MODELS_FOUND

# 在任何模式下都先查找模型并准备特征库路径映射
echo "\n--- 查找所有模型并准备特征库路径 ---"
# 使用 mapfile (Bash 4+) 将 find 的结果直接读取到数组，避免子shell问题
mapfile -t ALL_MODELS_FOUND < <(find "${PROJECT_DIR}/${LOGS_BASE_DIR}" -type f -name "best_model_*.pdparams" 2>/dev/null)

# 遍历 ALL_MODELS_FOUND 并处理特征库
# 注意：此处遍历 ALL_MODELS_FOUND 是在父shell中，不会影响其内容。
# 但我们仍需根据模型路径查找元数据和构建特征库路径。
# 优化：将特征库的检查和路径存储逻辑移到这个循环中
for MODEL_PATH in "${ALL_MODELS_FOUND[@]}"; do
    MODEL_DIR=$(dirname "${MODEL_PATH}")
    METADATA_FILENAME=$(basename "${MODEL_PATH}" .pdparams).json
    METADATA_PATH="${MODEL_DIR}/${METADATA_FILENAME}"

    if [ ! -f "${METADATA_PATH}" ]; then
        echo "警告: 未找到模型 ${MODEL_PATH} 的元数据文件。跳过此模型在特征库准备阶段的检查。"
        continue
    fi

    LOSS_TYPE=$(jq -r '.loss_type' "${METADATA_PATH}")
    if [ -z "${LOSS_TYPE}" ] || [ "${LOSS_TYPE}" == "null" ]; then
        echo "警告: 无法从元数据 ${METADATA_PATH} 中提取 loss_type。跳过此模型在特征库准备阶段的检查。"
        continue
    fi

    if [ "${LOSS_TYPE}" == "arcface" ]; then
        FEATURE_LIBRARY_DIR="${MODEL_DIR}" # 特征库通常与模型文件同目录
        FEATURE_LIBRARY_PATH="${FEATURE_LIBRARY_DIR}/face_library.pkl" # 标准 pkl 名称
        if [ -f "${FEATURE_LIBRARY_PATH}" ]; then
            ARCFACE_LIBRARY_PATHS[${MODEL_PATH}]="${FEATURE_LIBRARY_PATH}"
            echo "信息: 已找到ArcFace模型 ${MODEL_PATH} 的特征库: ${FEATURE_LIBRARY_PATH}"
        else
            echo "警告: ArcFace模型 ${MODEL_PATH} 的特征库 ${FEATURE_LIBRARY_PATH} 未找到。在需要时可能无法进行推理。"
        fi
    fi
done

# 检查 ALL_MODELS_FOUND 是否真的被填充了
if [ ${#ALL_MODELS_FOUND[@]} -eq 0 ]; then
    echo "警告: 未找到任何训练好的模型。请确保 logs 目录下有模型检查点 (.pdparams 文件)。"
fi

echo "\n--- 开始执行脚本模式: ${MODE} ---"

if [ "$MODE" == "acceptance_test" ]; then
    echo "执行：数据集验收测试模式。"
    # 保持原有 acceptance.sh 的逻辑

    ACCEPTANCE_RESULTS_DIR="${PROJECT_DIR}/acceptance_summary_results"
    mkdir -p "${ACCEPTANCE_RESULTS_DIR}"
    RESULTS_CSV="${ACCEPTANCE_RESULTS_DIR}/acceptance_results_$(date +%Y%m%d-%H%M%S).csv"

    # Write CSV header
    echo "Combo_Name,Timestamp,Model_Type,Loss_Type,Train_Accuracy_EpochEnd,Eval_Accuracy_EpochEnd,Acceptance_Accuracy,Recognition_Threshold,Model_Path,Metadata_Path,Feature_Library_Path,Status,Notes" > "${RESULTS_CSV}"

    # Check for default data list files (optional, as Python scripts will check from config)
DEFAULT_TRAIN_LIST_NAME="trainer.list"
DEFAULT_TEST_LIST_NAME="test.list"
DEFAULT_DATA_ROOT_DIR="${PROJECT_DIR}/data"

if [ ! -f "${DEFAULT_DATA_ROOT_DIR}/${DEFAULT_TRAIN_LIST_NAME}" ]; then
    echo "Error: Default training data list not found at ${DEFAULT_DATA_ROOT_DIR}/${DEFAULT_TRAIN_LIST_NAME}. Please run CreateDataList.py first or update your config/checks."
    exit 1
fi
if [ ! -f "${DEFAULT_DATA_ROOT_DIR}/${DEFAULT_TEST_LIST_NAME}" ]; then
    echo "Error: Default test data list not found at ${DEFAULT_DATA_ROOT_DIR}/${DEFAULT_TEST_LIST_NAME}. Please run CreateDataList.py first or update your config/checks."
    exit 1
fi

    # Iterate through all models found and run acceptance test
    for MODEL_PATH in "${ALL_MODELS_FOUND[@]}"; do
        # 重新精确提取相关路径和名称
        CHECKPOINTS_DIR=$(dirname "${MODEL_PATH}")
        TIMESTAMP_DIR=$(dirname "${CHECKPOINTS_DIR}")
        COMBO_CONFIG_DIR=$(dirname "${TIMESTAMP_DIR}")

        TIMESTAMP=$(basename "${TIMESTAMP_DIR}")
        COMBO_NAME=$(basename "${COMBO_CONFIG_DIR}") # 正确的组合名称

    METADATA_FILENAME=$(basename "${MODEL_PATH}" .pdparams).json
        METADATA_PATH="${CHECKPOINTS_DIR}/${METADATA_FILENAME}"

        MODEL_TYPE_FROM_META="Unknown"
        LOSS_TYPE_FROM_META="Unknown"
        if [ -f "${METADATA_PATH}" ]; then
            MODEL_TYPE_FROM_META=$(jq -r '.model_type' "${METADATA_PATH}")
            LOSS_TYPE_FROM_META=$(jq -r '.loss_type' "${METADATA_PATH}")
        fi

        ACCEPTANCE_ACC="N/A"
        FEATURE_LIBRARY_PATH_CSV="N/A"
    STATUS="Unknown"
    NOTES=""
        TRAIN_ACC="N/A"
        EVAL_ACC="N/A"

        echo "\n--- 运行验收测试 for Model: ${COMBO_NAME} (Timestamp: ${TIMESTAMP}) ---"
    echo "  Model path: ${MODEL_PATH}"
    echo "  Metadata path: ${METADATA_PATH}"

    if [ -f "${METADATA_PATH}" ]; then
            TRAIN_ACC_FROM_META=$(jq -r '.last_eval_accuracy' "${METADATA_PATH}") # 假设 last_eval_accuracy 是训练准确率
            EVAL_ACC_FROM_META=$(jq -r '.best_acc' "${METADATA_PATH}") # 假设 best_acc 是验证准确率

        if [ "${TRAIN_ACC_FROM_META}" != "null" ] && [ -n "${TRAIN_ACC_FROM_META}" ]; then TRAIN_ACC="${TRAIN_ACC_FROM_META}"; fi
        if [ "${EVAL_ACC_FROM_META}" != "null" ] && [ -n "${EVAL_ACC_FROM_META}" ]; then EVAL_ACC="${EVAL_ACC_FROM_META}"; fi
    fi

        if [ "${LOSS_TYPE_FROM_META}" == "arcface" ]; then
        echo "  Loss is ArcFace. Using feature library for identification."
            CURRENT_FEATURE_LIBRARY_PATH=${ARCFACE_LIBRARY_PATHS[${MODEL_PATH}]}

        if [ -z "${CURRENT_FEATURE_LIBRARY_PATH}" ]; then
                echo "错误: ArcFace模型 ${COMBO_NAME} 的特征库未找到或未在 Phase 1 创建。跳过验收测试。"
            STATUS="Skipped"
            NOTES="Feature library not created or path not recorded in Phase 1"
                FEATURE_LIBRARY_PATH_CSV="N/A"
            else
            echo "  Using feature library: ${CURRENT_FEATURE_LIBRARY_PATH}"
                FEATURE_LIBRARY_PATH_CSV="\"${CURRENT_FEATURE_LIBRARY_PATH}\""

                ACCEPTANCE_RESULTS_MODEL_DIR="${CHECKPOINTS_DIR}/acceptance_results"
                mkdir -p "${ACCEPTANCE_RESULTS_MODEL_DIR}" 2>/dev/null
            ACCEPTANCE_LOG_FILE="${ACCEPTANCE_RESULTS_MODEL_DIR}/acceptance_test_log.txt"
                ACCEPTANCE_JSON_OUTPUT_FILE="${ACCEPTANCE_RESULTS_MODEL_DIR}/acceptance_results.json"

             PYTHON_ACCEPTANCE_CMD=(
               python "${PROJECT_DIR}/acceptance_test.py" \
                --trained_model_path "${MODEL_PATH}" \
                --config_path "${CONFIG_FILE}" \
                    --output_json_path "${ACCEPTANCE_JSON_OUTPUT_FILE}" \
               --feature_library_path "${CURRENT_FEATURE_LIBRARY_PATH}" \
                    --use_gpu # 假设这里直接传递 use_gpu
             )

            ("${PYTHON_ACCEPTANCE_CMD[@]}" 2>&1) | tee "${ACCEPTANCE_LOG_FILE}"

            if [ ${PIPESTATUS[0]} -ne 0 ]; then
                    echo "错误: acceptance_test.py failed for ${COMBO_NAME} (ArcFace). See log file ${ACCEPTANCE_LOG_FILE}." | tee -a "${RESULTS_CSV}"
                 STATUS="Failed"
                NOTES="Acceptance test failed. See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
                    ACCEPTANCE_ACC="Failed"
             else
                  STATUS="Success"
                 NOTES="See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
                    if [ -f "${ACCEPTANCE_JSON_OUTPUT_FILE}" ]; then
                        ACCEPTANCE_ACC=$(jq -r '.accuracy' "${ACCEPTANCE_JSON_OUTPUT_FILE}")
                      if [ "${ACCEPTANCE_ACC}" == "null" ]; then
                            echo "警告: 'accuracy' not found or is null in ${ACCEPTANCE_JSON_OUTPUT_FILE}. Check acceptance_test.py output JSON." | tee -a "${RESULTS_CSV}"
                            ACCEPTANCE_ACC="ParseError"
                           NOTES="JSON parse error or missing key. See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
                           STATUS="PartialSuccess"
                       fi
                  else
                        echo "警告: Acceptance results JSON file not found: ${ACCEPTANCE_JSON_OUTPUT_FILE}" | tee -a "${RESULTS_CSV}"
                       ACCEPTANCE_ACC="FileNotFound"
                      NOTES="JSON result file missing. See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
                      STATUS="PartialSuccess"
                  fi
                fi
            fi

        elif [ "${LOSS_TYPE_FROM_META}" == "cross_entropy" ]; then
        echo "  Loss is Cross-Entropy. Running classification test."
            FEATURE_LIBRARY_PATH_CSV="N/A"

            ACCEPTANCE_RESULTS_MODEL_DIR="${CHECKPOINTS_DIR}/acceptance_results"
            mkdir -p "${ACCEPTANCE_RESULTS_MODEL_DIR}" 2>/dev/null
        ACCEPTANCE_LOG_FILE="${ACCEPTANCE_RESULTS_MODEL_DIR}/acceptance_test_log.txt"
            ACCEPTANCE_JSON_OUTPUT_FILE="${ACCEPTANCE_RESULTS_MODEL_DIR}/acceptance_results.json"

        PYTHON_ACCEPTANCE_CMD=(
             python "${PROJECT_DIR}/acceptance_test.py" \
             --trained_model_path "${MODEL_PATH}" \
             --config_path "${CONFIG_FILE}" \
                --output_json_path "${ACCEPTANCE_JSON_OUTPUT_FILE}" \
                --use_gpu # 假设这里直接传递 use_gpu
        )

        ("${PYTHON_ACCEPTANCE_CMD[@]}" 2>&1) | tee "${ACCEPTANCE_LOG_FILE}"

        if [ ${PIPESTATUS[0]} -ne 0 ]; then
                echo "错误: acceptance_test.py failed for ${COMBO_NAME} (Cross-Entropy). See log file ${ACCEPTANCE_LOG_FILE}." | tee -a "${RESULTS_CSV}"
            STATUS="Failed"
            NOTES="Acceptance test failed. See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
                ACCEPTANCE_ACC="Failed"
        else
             STATUS="Success"
             NOTES="See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
                if [ -f "${ACCEPTANCE_JSON_OUTPUT_FILE}" ]; then
                    ACCEPTANCE_ACC=$(jq -r '.accuracy' "${ACCEPTANCE_JSON_OUTPUT_FILE}")
                    if [ "${ACCEPTANCE_ACC}" == "null" ]; then
                        echo "警告: 'accuracy' not found or is null in ${ACCEPTANCE_JSON_OUTPUT_FILE}. Check acceptance_test.py output JSON." | tee -a "${RESULTS_CSV}"
                       ACCEPTANCE_ACC="ParseError"
                       NOTES="JSON parse error or missing key. See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
                        STATUS="PartialSuccess"
                  fi
             else
                    echo "警告: Acceptance results JSON file not found: ${ACCEPTANCE_JSON_OUTPUT_FILE}" | tee -a "${RESULTS_CSV}"
                  ACCEPTANCE_ACC="FileNotFound"
                  NOTES="JSON result file missing. See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
                    STATUS="PartialSuccess"
                fi
             fi

        else
            echo "  警告: 不支持的损失类型 '${LOSS_TYPE_FROM_META}' 用于验收测试。跳过。"
        STATUS="Skipped"
            NOTES="Unsupported loss type: ${LOSS_TYPE_FROM_META}"
        ACCEPTANCE_ACC="Skipped"
        FEATURE_LIBRARY_PATH_CSV="N/A"
        fi

        echo "\"${COMBO_NAME}\",\"${TIMESTAMP}\",\"${MODEL_TYPE_FROM_META}\",\"${LOSS_TYPE_FROM_META}\",\"${TRAIN_ACC}\",\"${EVAL_ACC}\",\"${ACCEPTANCE_ACC}\",N/A,\"${MODEL_PATH}\",\"${METADATA_PATH}\",${FEATURE_LIBRARY_PATH_CSV},\"${STATUS}\",\"${NOTES}\"" >> "${RESULTS_CSV}"
        echo "--- 完成处理 ${COMBO_NAME} (状态: ${STATUS}) ---"
    done

    echo "\n--- 自动化验收测试完成。结果保存到 ${RESULTS_CSV} ---"

elif [ "$MODE" == "compare_single_image" ]; then
    echo "执行：单张图片多模型比较模式。"
    if [ -z "$INPUT_IMAGE_PATH" ]; then
        echo "错误: 在 'compare_single_image' 模式下，必须通过 --image_path 指定输入图片。"
        exit 1
    fi
    if [ ! -f "$INPUT_IMAGE_PATH" ]; then
        echo "错误: 指定的输入图片未找到: ${INPUT_IMAGE_PATH}"
        exit 1
    fi

    # 为单张图片比较结果创建 CSV 文件
    SINGLE_IMAGE_RESULTS_DIR="${PROJECT_DIR}/single_image_comparison_results"
    mkdir -p "${SINGLE_IMAGE_RESULTS_DIR}"
    SINGLE_IMAGE_CSV="${SINGLE_IMAGE_RESULTS_DIR}/single_image_comparison_results_$(date +%Y%m%d-%H%M%S).csv"
    echo "模型组合名称,模型类型,损失类型,预测标签,分数" > "${SINGLE_IMAGE_CSV}"
    echo "信息: 单张图片比较结果将保存到: ${SINGLE_IMAGE_CSV}"

    echo "--------------------------------------------------------------------------------------"
    printf "%-40s | %-10s | %-12s | %-15s | %-10s\n" "模型组合名称" "模型类型" "损失类型" "预测标签" "分数"
    echo "--------------------------------------------------------------------------------------"

    if [ ${#ALL_MODELS_FOUND[@]} -eq 0 ]; then
        echo "警告: 未找到任何训练好的模型。请确保 logs 目录下有模型检查点。"
    fi

    for MODEL_PATH in "${ALL_MODELS_FOUND[@]}"; do
        # 重新精确提取相关路径和名称
        CHECKPOINTS_DIR=$(dirname "${MODEL_PATH}")
        TIMESTAMP_DIR=$(dirname "${CHECKPOINTS_DIR}")
        COMBO_CONFIG_DIR=$(dirname "${TIMESTAMP_DIR}")

        TIMESTAMP=$(basename "${TIMESTAMP_DIR}")
        COMBO_NAME=$(basename "${COMBO_CONFIG_DIR}") # 正确的组合名称

        METADATA_FILENAME=$(basename "${MODEL_PATH}" .pdparams).json
        METADATA_PATH="${CHECKPOINTS_DIR}/${METADATA_FILENAME}"
        
        MODEL_TYPE_FROM_META="Unknown"
        LOSS_TYPE_FROM_META="Unknown"
        if [ -f "${METADATA_PATH}" ]; then
            MODEL_TYPE_FROM_META=$(jq -r '.model_type' "${METADATA_PATH}")
            LOSS_TYPE_FROM_META=$(jq -r '.loss_type' "${METADATA_PATH}")
        fi

        echo "DEBUG: MODEL_PATH=${MODEL_PATH}"
        echo "DEBUG: COMBO_NAME=${COMBO_NAME}"
        echo "DEBUG: TIMESTAMP=${TIMESTAMP}"
        echo "DEBUG: METADATA_PATH=${METADATA_PATH}"
        echo "DEBUG: MODEL_TYPE_FROM_META=${MODEL_TYPE_FROM_META}"
        echo "DEBUG: LOSS_TYPE_FROM_META=${LOSS_TYPE_FROM_META}"

        # 在调用 infer.py 之前检查元数据文件是否存在
        if [ ! -f "${METADATA_PATH}" ]; then
            echo "警告: 模型 ${COMBO_NAME} (Timestamp: ${TIMESTAMP}) 的元数据文件 ${METADATA_PATH} 未找到。将跳过此模型的推理。"
            PREDICTED_LABEL="SKIPPED"
            SCORE="METADATA_MISSING"
            ERROR_NOTE="元数据文件缺失"
            printf "%-40s | %-10s | %-12s | %-15s | %-10s\n" "${COMBO_NAME}" "${MODEL_TYPE_FROM_META}" "${LOSS_TYPE_FROM_META}" "${PREDICTED_LABEL}" "${SCORE}"
            if [ -n "${ERROR_NOTE}" ]; then
                 echo "   - 注意: ${ERROR_NOTE}"
            fi
            echo "\""${COMBO_NAME}\"",\""${MODEL_TYPE_FROM_META}\"",\""${LOSS_TYPE_FROM_META}\"",\""${PREDICTED_LABEL}\"",\""${SCORE}\""" >> "${SINGLE_IMAGE_CSV}"
            continue # 跳到循环中的下一个模型
        fi

        INFER_CMD_ARRAY=(
            python -u infer.py 
            --config_path "${CONFIG_FILE}"
            --active_infer_config "${COMBO_NAME}"
            --model_path "${MODEL_PATH}"
            --image_path "${INPUT_IMAGE_PATH}"
            --use_gpu
        )
        
        if [ "${LOSS_TYPE_FROM_META}" == "arcface" ] && [ -n "${ARCFACE_LIBRARY_PATHS[${MODEL_PATH}]}" ]; then
            INFER_CMD_ARRAY+=(--face_library_path "${ARCFACE_LIBRARY_PATHS[${MODEL_PATH}]}")
        fi

        echo "正在使用模型 ${COMBO_NAME} (Timestamp: ${TIMESTAMP}) 进行预测..."
        # 运行 infer.py 并捕获输出
        INFER_OUTPUT=$("${INFER_CMD_ARRAY[@]}" 2>&1)
        INFER_EXIT_CODE=$?

        echo "--- 来自 infer.py 的完整输出 (模型: ${COMBO_NAME}, 退出码: ${INFER_EXIT_CODE}) ---"
        echo "${INFER_OUTPUT}"
        echo "--------------------------------------------------------------------------"

        # 恢复原始的标签和分数提取逻辑
        PREDICTED_LABEL="N/A"
        SCORE="N/A"
        ERROR_NOTE=""

        if [ $INFER_EXIT_CODE -eq 0 ]; then
            PREDICTED_LABEL=$(echo "${INFER_OUTPUT}" | grep -m 1 "Name:" | awk '{print $2}' || echo "N/A")
            SCORE_LINE=$(echo "${INFER_OUTPUT}" | grep -m 1 -E "Similarity:|Confidence:" || echo "")
            if [ -n "${SCORE_LINE}" ]; then
                SCORE=$(echo "${SCORE_LINE}" | awk '{print $2}')
            else
                SCORE="N/A"
            fi
        else
            PREDICTED_LABEL="ERROR"
            SCORE="ERROR"
            ERROR_NOTE="推理失败 (退出码: ${INFER_EXIT_CODE})"
            echo "--- infer.py 完整错误输出 (模型: ${COMBO_NAME}) ---"
            echo "${INFER_OUTPUT}" # 打印完整错误输出
            echo "---------------------------------------------------"
        fi
        
        printf "%-40s | %-10s | %-12s | %-15s | %-10s\n" "${COMBO_NAME}" "${MODEL_TYPE_FROM_META}" "${LOSS_TYPE_FROM_META}" "${PREDICTED_LABEL}" "${SCORE}"
        if [ -n "${ERROR_NOTE}" ]; then
             echo "   - 注意: ${ERROR_NOTE}"
        fi

        # 将结果写入 CSV 文件 (确保字段正确引用，以防包含逗号或空格)
        echo "\"${COMBO_NAME}\",\"${MODEL_TYPE_FROM_META}\",\"${LOSS_TYPE_FROM_META}\",\"${PREDICTED_LABEL}\",\"${SCORE}\"" >> "${SINGLE_IMAGE_CSV}"

    done
    echo "--------------------------------------------------------------------------------------"
    echo "单张图片多模型比较完成。结果已保存到: ${SINGLE_IMAGE_CSV}"

else
    echo "错误: 未知的运行模式: ${MODE}。请使用 --mode acceptance_test 或 --mode compare_single_image。"
    exit 1
fi

# 尝试停用虚拟环境 (如果之前成功激活)
if [[ -n "${VIRTUAL_ENV}" ]]; then # VIRTUAL_ENV 变量由 source activate 设置
    echo "--> 尝试停用虚拟环境..."
    deactivate &>/dev/null || echo "停用虚拟环境命令 (deactivate) 执行遇到问题或未找到。"
else
    echo "--> 未检测到活动的虚拟环境，无需停用。"
fi

echo "=================================================="
echo "脚本执行完毕。"
echo "=================================================="