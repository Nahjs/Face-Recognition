#!/bin/bash
set -e # Any command failing will cause the script to exit.

# --- User Configuration --- (Moved most data/path/setting config to YAML)
# Set the project directory to the directory where this script is located.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to the main configuration file (used by create_face_library.py and acceptance_test.py)
CONFIG_FILE="${PROJECT_DIR}/configs/48_config.yaml"

# Base directory for logs and checkpoints (relative to PROJECT_DIR)
# Find models recursively in this directory structure.
LOGS_BASE_DIR="logs"

# Output directory for the overall acceptance test results summary CSV file
ACCEPTANCE_RESULTS_DIR="${PROJECT_DIR}/acceptance_summary_results"

# --- Removed hardcoded data paths, list names, threshold, GPU flag here ---
# These settings should now be managed in the CONFIG_FILE (e.g., 432_config.yaml)

# --- Setup ---
echo "--- Starting Automated Acceptance Tests ---"

# Ensure acceptance results directory exists
mkdir -p "${ACCEPTANCE_RESULTS_DIR}"
RESULTS_CSV="${ACCEPTANCE_RESULTS_DIR}/acceptance_results_$(date +%Y%m%d-%H%M%S).csv"

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

# --- Data List Checks (Keeping these basic checks for early feedback, but Python scripts will read paths from config) ---
# Note: The actual list paths and data_root will be read by the Python scripts from the config.
# These checks just verify the existence of default list files in a default data location.
# If your data lists are elsewhere, you might need to adjust these checks or rely solely on Python script errors.

# Default list names and a potential default data dir for the check
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


# Write CSV header
# Combo_Name,Timestamp,Model_Type,Loss_Type,Train_Accuracy_EpochEnd,Eval_Accuracy_EpochEnd,Acceptance_Accuracy,Recognition_Threshold,Model_Path,Metadata_Path,Feature_Library_Path,Status,Notes
echo "Combo_Name,Timestamp,Model_Type,Loss_Type,Train_Accuracy_EpochEnd,Eval_Accuracy_EpochEnd,Acceptance_Accuracy,Recognition_Threshold,Model_Path,Metadata_Path,Feature_Library_Path,Status,Notes" > "${RESULTS_CSV}"

# Declare an associative array to store feature library paths for ArcFace models
declare -A ARCFACE_LIBRARY_PATHS
declare -a ALL_MODELS_FOUND # Array to store paths of all models found

# --- Phase 1: Create Feature Libraries for ArcFace Models ---
echo "\n--- Phase 1: Creating Feature Libraries for ArcFace Models ---"

# Find all best_model_*.pdparams files recursively and populate ALL_MODELS_FOUND
find "${PROJECT_DIR}/${LOGS_BASE_DIR}" -type f -name "best_model_*.pdparams" 2>/dev/null | while read MODEL_PATH;
do
    ALL_MODELS_FOUND+=("${MODEL_PATH}")

    # Extract relevant paths and names
    MODEL_DIR=$(dirname "${MODEL_PATH}")
    METADATA_FILENAME=$(basename "${MODEL_PATH}" .pdparams).json
    METADATA_PATH="${MODEL_DIR}/${METADATA_FILENAME}"

    # Check if the corresponding metadata file exists
    if [ ! -f "${METADATA_PATH}" ]; then
        echo "警告: Metadata file not found for model ${MODEL_PATH}. Cannot determine loss type for library creation. Skipping library creation for this model."
        continue # Skip library creation for this model, but keep it in ALL_MODELS_FOUND
    fi

    # Read metadata to get loss_type
    LOSS_TYPE=$(jq -r '.loss_type' "${METADATA_PATH}")
     if [ -z "${LOSS_TYPE}" ] || [ "${LOSS_TYPE}" == "null" ]; then
        echo "警告: Could not extract loss_type from metadata ${METADATA_PATH}. Skipping library creation for this model."
        continue # Skip library creation for this model
    fi

    if [ "${LOSS_TYPE}" == "arcface" ]; then
        # Extract COMBO_NAME and TIMESTAMP for logging
        TIMESTAMP=$(basename "${MODEL_DIR}")
        CONFIG_DIR=$(dirname "${MODEL_DIR}")
        COMBO_NAME=$(basename "${CONFIG_DIR}")

        echo "\n--- Creating Feature Library for ArcFace Model: ${COMBO_NAME} (Timestamp: ${TIMESTAMP}) ---"
        echo "  Model path: ${MODEL_PATH}"

        FEATURE_LIBRARY_DIR="${MODEL_DIR}/feature_library"
        mkdir -p "${FEATURE_LIBRARY_DIR}"
        FEATURE_LIBRARY_PATH="${FEATURE_LIBRARY_DIR}/feature_library.pkl" # Standard pkl name

        # Call create_face_library.py
        echo "Executing: python ${PROJECT_DIR}/create_face_library.py --model_path ${MODEL_PATH} --config_path ${CONFIG_FILE} --face_library_path ${FEATURE_LIBRARY_DIR}"
        python "${PROJECT_DIR}/create_face_library.py" \
            --model_path "${MODEL_PATH}" \
            --config_path "${CONFIG_FILE}" \
            --face_library_path "${FEATURE_LIBRARY_DIR}"

        # Check create_face_library.py exit status
        if [ $? -ne 0 ]; then
            echo "错误: create_face_library.py failed for ${COMBO_NAME} (ArcFace). Feature library not created."
            # We don't add it to ARCFACE_LIBRARY_PATHS if creation failed
        else
            echo "成功创建特征库: ${FEATURE_LIBRARY_PATH}"
            # Store the path for Phase 2
            ARCFACE_LIBRARY_PATHS["${MODEL_PATH}"]="${FEATURE_LIBRARY_PATH}"
            # Optional: Add a small delay after successful creation if desired
            sleep 1
        fi
    fi # End if loss_type is arcface

done # End of Phase 1 find loop

echo "\n--- Phase 1 Complete. Created ${#ARCFACE_LIBRARY_PATHS[@]} feature libraries. ---"
echo "Proceeding to Phase 2 after a brief pause..."
sleep 5 # Add a deliberate pause between phases

# --- Phase 2: Run Acceptance Tests for All Found Models ---
echo "\n--- Phase 2: Running Acceptance Tests ---"

# Iterate through all models found in Phase 1
for MODEL_PATH in "${ALL_MODELS_FOUND[@]}"; do

    # Re-extract relevant paths and names (or pass them from Phase 1 if preferred)
    MODEL_DIR=$(dirname "${MODEL_PATH}")
    TIMESTAMP=$(basename "${MODEL_DIR}")
    CONFIG_DIR=$(dirname "${MODEL_DIR}")
    COMBO_NAME=$(basename "${CONFIG_DIR}")
    MODEL_FILENAME=$(basename "${MODEL_PATH}")
    MODEL_TYPE=$(echo "${MODEL_FILENAME}" | sed 's/^best_model_//' | sed 's/\.pdparams$//')
    METADATA_FILENAME="best_model_${MODEL_TYPE}.json"
    METADATA_PATH="${MODEL_DIR}/${METADATA_FILENAME}"

    # Initialize results variables for this model
    ACCEPTANCE_ACC="N/A" # Initialize accuracy
    FEATURE_LIBRARY_PATH_CSV="N/A" # Initialize feature library path for CSV
    STATUS="Unknown"
    NOTES=""
    LOSS_TYPE="Unknown" # Initialize loss type
    TRAIN_ACC="N/A" # Initialize Train Acc
    EVAL_ACC="N/A"  # Initialize Eval Acc

    echo "\n--- Running Acceptance Test for Model: ${COMBO_NAME} (Timestamp: ${TIMESTAMP}, Type: ${MODEL_TYPE}) ---"
    echo "  Model path: ${MODEL_PATH}"
    echo "  Metadata path: ${METADATA_PATH}"

    # Read metadata to get loss_type and train/eval accuracy
    if [ -f "${METADATA_PATH}" ]; then
        LOSS_TYPE=$(jq -r '.loss_type' "${METADATA_PATH}")
        # Extract Train/Eval Accuracy from Metadata (Optional)
        TRAIN_ACC_FROM_META=$(jq -r '.metrics.train_acc' "${METADATA_PATH}")
        EVAL_ACC_FROM_META=$(jq -r '.metrics.eval_acc' "${METADATA_PATH}")
        if [ "${TRAIN_ACC_FROM_META}" != "null" ] && [ -n "${TRAIN_ACC_FROM_META}" ]; then TRAIN_ACC="${TRAIN_ACC_FROM_META}"; fi
        if [ "${EVAL_ACC_FROM_META}" != "null" ] && [ -n "${EVAL_ACC_FROM_META}" ]; then EVAL_ACC="${EVAL_ACC_FROM_META}"; fi
    fi

    if [ "${LOSS_TYPE}" == "arcface" ]; then
        echo "  Loss is ArcFace. Using feature library for identification."

        # Get the feature library path from the associative array
        CURRENT_FEATURE_LIBRARY_PATH=${ARCFACE_LIBRARY_PATHS["${MODEL_PATH}"]}

        if [ -z "${CURRENT_FEATURE_LIBRARY_PATH}" ]; then
            echo "错误: Feature library path not found in recorded list for model ${MODEL_PATH}. Skipping acceptance test."
            STATUS="Skipped"
            NOTES="Feature library not created or path not recorded in Phase 1"
            FEATURE_LIBRARY_PATH_CSV="N/A" # No library used/found
        else # Feature library path was found and recorded in Phase 1
            echo "  Using feature library: ${CURRENT_FEATURE_LIBRARY_PATH}"
            FEATURE_LIBRARY_PATH_CSV="\"${CURRENT_FEATURE_LIBRARY_PATH}\"" # Escape path for CSV

             # --- ArcFace: Run Acceptance Test (Identification using Library) ---
             echo "  Running acceptance test (ArcFace - Library Identification)..."
            ACCEPTANCE_RESULTS_MODEL_DIR="${MODEL_DIR}/acceptance_results"
            mkdir -p "${ACCEPTANCE_RESULTS_MODEL_DIR}" 2>/dev/null # Create output directory for this model's results, suppress errors if exists
            ACCEPTANCE_LOG_FILE="${ACCEPTANCE_RESULTS_MODEL_DIR}/acceptance_test_log.txt"

            # Call acceptance_test.py for ArcFace
             PYTHON_ACCEPTANCE_CMD=(
               python "${PROJECT_DIR}/acceptance_test.py" \
                --trained_model_path "${MODEL_PATH}" \
                --config_path "${CONFIG_FILE}" \
                --results_save_dir "${ACCEPTANCE_RESULTS_MODEL_DIR}" \
               --feature_library_path "${CURRENT_FEATURE_LIBRARY_PATH}" \
                ${USE_GPU_FLAG} # Pass GPU flag if configured
             )

            # Execute the command and redirect output to log file and tee to stderr (to avoid mixing with stdout)
            ("${PYTHON_ACCEPTANCE_CMD[@]}" 2>&1) | tee "${ACCEPTANCE_LOG_FILE}"

             # Check acceptance_test.py exit status
            if [ ${PIPESTATUS[0]} -ne 0 ]; then
                echo "错误: acceptance_test.py failed for ${COMBO_NAME} (ArcFace). See log file ${ACCEPTANCE_LOG_FILE}." | tee -a "${RESULTS_CSV}" # Log error to main CSV too
                 STATUS="Failed"
                NOTES="Acceptance test failed. See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
                 ACCEPTANCE_ACC="Failed" # Mark accuracy as failed
             else
                  # Status is Success if test script exited with 0
                  STATUS="Success"
                 NOTES="See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
                  # Extract Acceptance Accuracy from JSON output
                  ACCEPTANCE_RESULTS_JSON="${ACCEPTANCE_RESULTS_MODEL_DIR}/acceptance_results.json"
                  if [ -f "${ACCEPTANCE_RESULTS_JSON}" ]; then
                      ACCEPTANCE_ACC=$(jq -r '.accuracy' "${ACCEPTANCE_RESULTS_JSON}")
                      if [ "${ACCEPTANCE_ACC}" == "null" ]; then
                           echo "警告: 'accuracy' not found or is null in ${ACCEPTANCE_RESULTS_JSON}. Check acceptance_test.py output JSON." | tee -a "${RESULTS_CSV}"
                            ACCEPTANCE_ACC="ParseError"
                           NOTES="JSON parse error or missing key. See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
                           STATUS="PartialSuccess"
                       fi
                  else
                      echo "警告: Acceptance results JSON file not found: ${ACCEPTANCE_RESULTS_JSON}" | tee -a "${RESULTS_CSV}"
                       ACCEPTANCE_ACC="FileNotFound"
                      NOTES="JSON result file missing. See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
                      STATUS="PartialSuccess"
                  fi
             fi # End check acceptance_test.py exit status for ArcFace
        fi # End check for feature library path validity (exists in Phase 1 list)

    elif [ "${LOSS_TYPE}" == "cross_entropy" ]; then
        echo "  Loss is Cross-Entropy. Running classification test."
        FEATURE_LIBRARY_PATH_CSV="N/A" # No feature library for CE

        # --- Cross-Entropy: Run Acceptance Test (Classification) ---
        echo "  Running acceptance test (Classification)..."
        ACCEPTANCE_RESULTS_MODEL_DIR="${MODEL_DIR}/acceptance_results"
        mkdir -p "${ACCEPTANCE_RESULTS_MODEL_DIR}" # Create output directory for this model's results
        ACCEPTANCE_LOG_FILE="${ACCEPTANCE_RESULTS_MODEL_DIR}/acceptance_test_log.txt"

        # Call acceptance_test.py for Cross-Entropy
        PYTHON_ACCEPTANCE_CMD=(
             python "${PROJECT_DIR}/acceptance_test.py" \
             --trained_model_path "${MODEL_PATH}" \
             --config_path "${CONFIG_FILE}" \
             --results_save_dir "${ACCEPTANCE_RESULTS_MODEL_DIR}" \
             ${USE_GPU_FLAG} # Pass GPU flag if configured
        )
        # Execute the command and redirect output
        ("${PYTHON_ACCEPTANCE_CMD[@]}" 2>&1) | tee "${ACCEPTANCE_LOG_FILE}"

         # Check acceptance_test.py exit status
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "错误: acceptance_test.py failed for ${COMBO_NAME} (Cross-Entropy). See log file ${ACCEPTANCE_LOG_FILE}." | tee -a "${RESULTS_CSV}" # Log error to main CSV too
            STATUS="Failed"
            NOTES="Acceptance test failed. See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
            ACCEPTANCE_ACC="Failed" # Mark accuracy as failed
        else
             # Status is Success if test script exited with 0
             STATUS="Success"
             NOTES="See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
             # Extract Acceptance Accuracy from JSON output
             ACCEPTANCE_RESULTS_JSON="${ACCEPTANCE_RESULTS_MODEL_DIR}/acceptance_results.json"
             if [ -f "${ACCEPTANCE_RESULTS_JSON}" ]; then
                  # jq -r '.accuracy' as acceptance_test.py saves results with key 'accuracy'
                  ACCEPTANCE_ACC=$(jq -r '.accuracy' "${ACCEPTANCE_RESULTS_JSON}")
                   if [ "${ACCEPTANCE_ACC}" == "null" ]; then # jq returns "null" if key not found or value is null
                       echo "警告: 'accuracy' not found or is null in ${ACCEPTANCE_RESULTS_JSON}. Check acceptance_test.py output JSON." | tee -a "${RESULTS_CSV}"
                       ACCEPTANCE_ACC="ParseError"
                       NOTES="JSON parse error or missing key. See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
                       STATUS="PartialSuccess" # Test succeeded, but result parsing failed
                  fi
             else
                  echo "警告: Acceptance results JSON file not found: ${ACCEPTANCE_RESULTS_JSON}" | tee -a "${RESULTS_CSV}"
                  ACCEPTANCE_ACC="FileNotFound"
                  NOTES="JSON result file missing. See log: $(basename "${ACCEPTANCE_LOG_FILE}")"
                  STATUS="PartialSuccess" # Test succeeded, but result file missing
             fi
        fi # End check acceptance_test.py exit status for Cross-Entropy

    else # Unsupported Loss Type
        echo "  Warning: Unsupported loss type '${LOSS_TYPE}' for acceptance test. Skipping."
        STATUS="Skipped"
        NOTES="Unsupported loss type: ${LOSS_TYPE}"
        ACCEPTANCE_ACC="Skipped"
        FEATURE_LIBRARY_PATH_CSV="N/A"

    fi # End Loss Type Check

    # Log results to CSV
    # Ensure fields are quoted if they might contain commas or spaces, especially paths and notes.
    # Combo_Name,Timestamp,Model_Type,Loss_Type,Train_Accuracy_EpochEnd,Eval_Accuracy_EpochEnd,Acceptance_Accuracy,Recognition_Threshold,Model_Path,Metadata_Path,Feature_Library_Path,Status,Notes
    # Note: recognition_threshold is only relevant for ArcFace and is handled within acceptance_test.py now, not directly passed via shell.
    # We can extract it from acceptance_results.json if needed, but for now, keep it N/A unless explicitly passed or stored differently.
    echo "\"${COMBO_NAME}\",\"${TIMESTAMP}\",\"${MODEL_TYPE}\",\"${LOSS_TYPE}\",\"${TRAIN_ACC}\",\"${EVAL_ACC}\",\"${ACCEPTANCE_ACC}\",N/A,\"${MODEL_PATH}\",\"${METADATA_PATH}\",${FEATURE_LIBRARY_PATH_CSV},\"${STATUS}\",\"${NOTES}\"" >> "${RESULTS_CSV}"

    echo "--- Finished processing ${COMBO_NAME} (Status: ${STATUS}) ---"

done # End of Phase 2 model loop

echo "\n--- Automated Acceptance Tests Complete. Results saved to ${RESULTS_CSV} ---"