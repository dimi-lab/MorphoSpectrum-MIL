#!/usr/bin/env bash

# Total number of GPUs
NUM_GPUS=2
# Maximum number of concurrent tasks per GPU
MAX_TASKS_PER_GPU=2

# label_file paths (assume 5 different datasets, paths can be stored in an array)
label_files=(
  Datasplit_0_10_fold_by_patient.csv
  Datasplit_1_10_fold_by_patient.csv
  Datasplit_2_10_fold_by_patient.csv
  Datasplit_3_10_fold_by_patient.csv
  Datasplit_4_10_fold_by_patient.csv
)

# Output directory
base_output_dir="./Train_Output/"
mkdir -p "${base_output_dir}"

# Clear master log file (optional)
: > "${base_output_dir}/master_log.txt"

# Initialize GPU task counters
for ((i=0; i<NUM_GPUS; i++)); do
  echo 0 > gpu_${i}_task_count.txt
done

# Define a function to get the index of a GPU with fewer than MAX_TASKS_PER_GPU tasks
# If all GPUs are fully occupied, the task waits here until one becomes available
get_free_gpu() {
  while true; do
    for ((i=0; i<NUM_GPUS; i++)); do
      current_tasks=$(cat gpu_${i}_task_count.txt)
      if (( current_tasks < MAX_TASKS_PER_GPU )); then
        echo $i
        return
      fi
    done
    sleep 5
  done
}


# Main loop: iterate over each label_file, each dataset, each round
for label_file in "${label_files[@]}"; do
  
  # Extract dataset name (without extension)
  dataset_name=$(basename "$label_file" .csv)
  # Create output directory for this dataset
  dataset_output_dir="${base_output_dir}/${dataset_name}"
  mkdir -p "${dataset_output_dir}"


  for round in {0..9}; do
    # Get an available GPU index
    gpu_idx=$(get_free_gpu)
    echo "Launching dataset ${label_file}, round ${round}, using GPU ${gpu_idx}" >> ${base_output_dir}/master_log.txt

    # Increase task counter
    current=$(cat gpu_${gpu_idx}_task_count.txt)
    echo $((current+1)) > gpu_${gpu_idx}_task_count.txt

    # Launch training task in a subshell
    # The subshell runs independently, so the main loop continues to the next task
    (
      export CUDA_VISIBLE_DEVICES=${gpu_idx}
      python -u train_early_stopping.py \
          --feature_bag_dir feature_bags \
          --input_feature_size 512 \
          --manifest "${label_file}" \
          --round ${round} \
          --config ./config.yaml \
          &> "${dataset_output_dir}/train_round${round}.txt" &
      
      # Get python process pid
      pid=$!
      # Wait for the task to finish and then decrease GPU task counter
      wait $pid
      current=$(cat gpu_${gpu_idx}_task_count.txt)
      echo $((current-1)) > gpu_${gpu_idx}_task_count.txt
    ) &
  
  done
done

# Wait for all background tasks to finish
wait
echo "All tasks completed!" >> ${base_output_dir}/master_log.txt
