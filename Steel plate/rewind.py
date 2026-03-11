import os
import re
from PIL import Image
from tqdm import tqdm

# ================= 配置区 =================
# 输入路径：你切分好的数据集根目录（里面有 1, 2, 3, 4 类别文件夹）
INPUT_DIR = r"D:\work\Steel plate\dataset_sliced_tmp"
# 输出路径：复原后的完整大图存放位置
OUTPUT_DIR = r"D:\work\Steel plate\dataset_rewind"


# ==========================================

def reconstruct_dataset():
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    classes = ['1', '2', '3', '4']

    for cls in classes:
        cls_in_dir = os.path.join(INPUT_DIR, cls)
        cls_out_dir = os.path.join(OUTPUT_DIR, cls)

        # 确保输出类别文件夹存在
        os.makedirs(cls_out_dir, exist_ok=True)

        if not os.path.exists(cls_in_dir):
            continue

        # 获取该类别下所有的子文件夹（即每一张原图对应的文件夹）
        subfolders = [f for f in os.listdir(cls_in_dir) if os.path.isdir(os.path.join(cls_in_dir, f))]

        if not subfolders:
            continue

        print(f"\n正在复原类别 [{cls}] 的图片...")

        for folder_name in tqdm(subfolders):
            folder_path = os.path.join(cls_in_dir, folder_name)
            meta_path = os.path.join(folder_path, 'meta.txt')

            if not os.path.exists(meta_path):
                print(f"[警告] 找不到 meta.txt，跳过文件夹: {folder_name}")
                continue

            try:
                # 1. 读取并解析 meta.txt 获取复原密钥
                meta = {}
                with open(meta_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            meta[key] = value

                W = int(meta['Original_Width'])
                H = int(meta['Original_Height'])
                width_action = meta['Width_Action']
                action_value = int(meta['Action_Value'])
                patch_size = int(meta.get('Patch_Size', 224))

                # 2. 建立一个 224xH 的临时长条画布（用于贴合所有的正方形切片）
                strip_canvas = Image.new('RGB', (patch_size, H), (0, 0, 0))

                # 3. 找到所有切片并提取 y 坐标
                patches = []
                for f_name in os.listdir(folder_path):
                    # 使用正则匹配文件名中的 y 坐标，例如 1-1_y112.jpg
                    match = re.search(r'_y(\d+)\.(jpg|png|bmp|jpeg)$', f_name, re.IGNORECASE)
                    if match:
                        y_coord = int(match.group(1))
                        patches.append((y_coord, f_name))

                # 核心：必须按照 y 坐标从小到大排序，保证重叠部分按照从上往下的物理顺序覆盖
                patches.sort(key=lambda x: x[0])

                # 4. 逐个将切片拼接到长条画布上
                for y, fname in patches:
                    patch_path = os.path.join(folder_path, fname)
                    img_patch = Image.open(patch_path).convert('RGB')
                    # 粘贴时，横坐标永远为 0，纵坐标为读取到的 y
                    strip_canvas.paste(img_patch, (0, y))

                # 5. 逆向处理宽度，还原出最真实的原始形态
                if width_action == "padded_left":
                    # 如果当初是补了黑边，现在就把两边的黑边切掉，提取中间的钢板
                    final_img = strip_canvas.crop((action_value, 0, action_value + W, H))

                elif width_action == "cropped_left":
                    # 物理坦白局：如果原图宽度超过 224，当初被裁掉的边缘像素已经物理丢失了。
                    # 我们能做的，是创建一个真实宽度 (W) 的纯黑背景，把核心的 224 长条精准贴在它原来的位置。
                    final_img = Image.new('RGB', (W, H), (0, 0, 0))
                    final_img.paste(strip_canvas, (action_value, 0))

                else:
                    final_img = strip_canvas

                # 6. 保存最终复原的图片
                save_name = f"{folder_name}_reconstructed.bmp"
                save_path = os.path.join(cls_out_dir, save_name)
                final_img.save(save_path)

            except Exception as e:
                print(f"[错误] 复原 {folder_name} 时失败: {e}")

    print("\n✅ 所有数据集已完美复原完毕！请前往 dataset_reconstructed 文件夹查看。")


if __name__ == "__main__":
    reconstruct_dataset()