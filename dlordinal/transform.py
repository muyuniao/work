import os
from PIL import Image


def convert_jpg_to_png_recursive(source_root, target_root):
    """
    递归遍历 source_root，将 JPG 转换为 PNG，
    并在 target_root 中保持原有的子文件夹结构。
    """
    success_count = 0

    # os.walk 会遍历所有子目录
    # root: 当前正在遍历的文件夹路径
    # dirs: 当前文件夹下的子文件夹列表
    # files: 当前文件夹下的文件列表
    for root, dirs, files in os.walk(source_root):

        # 1. 计算当前文件夹相对于源根目录的相对路径
        # 例如: source_root是 "A", 当前 root 是 "A/B/C", relative_path 就是 "B/C"
        relative_path = os.path.relpath(root, source_root)

        # 2. 构建对应的目标文件夹路径
        target_dir = os.path.join(target_root, relative_path)

        # 3. 确保目标文件夹存在（如果不存在则创建）
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            # print(f"创建目录: {target_dir}") # 如果不想看太多日志，可以注释掉这行

        # 4. 遍历当前文件夹内的文件
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg')):
                try:
                    # 源文件完整路径
                    source_file_path = os.path.join(root, filename)

                    # 目标文件名（替换后缀）
                    file_name_without_ext = os.path.splitext(filename)[0]
                    target_file_path = os.path.join(target_dir, f"{file_name_without_ext}.png")

                    # 转换并保存
                    with Image.open(source_file_path) as img:
                        # 可以在这里添加 img = img.convert("RGB") 防止某些特殊模式报错
                        img.save(target_file_path, 'PNG')

                    print(f"转换成功: {os.path.join(relative_path, filename)} -> PNG")
                    success_count += 1

                except Exception as e:
                    print(f"转换出错 {filename}: {e}")

    print(f"\n全部完成！共转换了 {success_count} 张图片。")


# --- 使用示例 ---
if __name__ == "__main__":
    # 请修改为你的实际路径
    source_folder = r"C:\Users\yunheishere\Desktop\work\dlordinal\dataset"  # 里面可能有子文件夹 "Day1", "Day2"...
    target_folder = r"C:\Users\yunheishere\Desktop\work\dlordinal\dataset_trans"  # 生成后会有同样的结构 "Day1", "Day2"...

    convert_jpg_to_png_recursive(source_folder, target_folder)