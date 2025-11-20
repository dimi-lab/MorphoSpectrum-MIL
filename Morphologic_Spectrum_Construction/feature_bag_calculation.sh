gpu_ids=("0" "0" "1" "1")
data_files=("slide_id_missing_features.txt")
# Loop through each data file and run the command in the background
for i in "${!data_files[@]}"; do
    data_file="${data_files[$i]}"
    gpu_id="${gpu_ids[$i]}"
    
    # 生成独立日志文件，以便区分不同任务的输出
    log_file="high_endo_sbt_features.txt"
    
    # 把 CUDA_VISIBLE_DEVICES 放在同一行里，让该命令只看到指定 GPU
    cat "./$data_file" | \
    xargs -I WSIFILE bash -c \
    'export CUDA_VISIBLE_DEVICES='"${gpu_id}"';
    python -u preprocess.py \
        --input_slide WSIFILE \
        --output_dir ./feature_bags \
        --tile_size 360 \
        --out_size 224 \
        --batch_size 256 \
        --workers 8' \
    &> "$log_file" &
done

# Wait for all background processes to finish
wait