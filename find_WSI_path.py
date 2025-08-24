from pathlib import Path

# 设置文件夹路径
folder = Path("/home/peiliang/projects/Dr_Huang_Project/im4MEC/wholeslides/Clear_Cell_Carcinoma/unzipped")

# 输出文件
output_file = "svs_files2.txt"

# 找到所有 .svs 文件并写入
with open(output_file, "w") as f:
    for svs_file in folder.rglob("*.svs"):  # rglob递归搜索
        f.write(str(svs_file.resolve()) + "\n")

print(f"All .svs file paths written to {output_file}")