#!/usr/bin/env bash
set -euo pipefail

# ==============================
# GPU scheduling config
# ==============================
NUM_GPUS=2
MAX_TASKS_PER_GPU=4

# ==============================
# Paths (YOU MAY EDIT)
# ==============================
TEST_PY="test.py"

FEATURE_BAG_DIR="./feature_bags"
CHECKPOINT_DIR="./runs"
CONFIG_YAML="./config.yaml"

ROUND_NUM=10
WORKERS=4

# ==============================
# Manifests to evaluate
# ==============================
label_files=(
  ./Data_Split/Datasplit_0_10_fold_by_patient.csv
  ./Data_Split/Datasplit_1_10_fold_by_patient.csv
  ./Data_Split/Datasplit_2_10_fold_by_patient.csv
  ./Data_Split/Datasplit_3_10_fold_by_patient.csv
  ./Data_Split/Datasplit_4_10_fold_by_patient.csv
)

# ==============================
# GPU task counters
# ==============================
for ((i=0; i<NUM_GPUS; i++)); do
  echo 0 > gpu_${i}_task_count.txt
done

get_free_gpu() {
  while true; do
    for ((i=0; i<NUM_GPUS; i++)); do
      current=$(cat gpu_${i}_task_count.txt)
      if (( current < MAX_TASKS_PER_GPU )); then
        echo $i
        return
      fi
    done
    sleep 5
  done
}

# ==============================
# Main loop
# ==============================
for manifest in "${label_files[@]}"; do
  dataset_name=$(basename "$manifest" .csv)
  echo "[INFO] Start testing dataset: ${dataset_name}"

  gpu_idx=$(get_free_gpu)
  echo "[INFO] Assigning ${dataset_name} to GPU ${gpu_idx}"

  current=$(cat gpu_${gpu_idx}_task_count.txt)
  echo $((current+1)) > gpu_${gpu_idx}_task_count.txt

  (
    export CUDA_VISIBLE_DEVICES=${gpu_idx}

    python -u ${TEST_PY} \
      --manifest "${manifest}" \
      --feature_bag_dir "${FEATURE_BAG_DIR}" \
      --checkpoint_dir "${CHECKPOINT_DIR}" \
      --round_num ${ROUND_NUM} \
      --workers ${WORKERS} \
      --config "${CONFIG_YAML}" \
      &> "${dataset_name}_test.log"

    current=$(cat gpu_${gpu_idx}_task_count.txt)
    echo $((current-1)) > gpu_${gpu_idx}_task_count.txt
  ) &

done

wait
echo "[INFO] All testing jobs completed."
