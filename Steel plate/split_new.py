import os
from PIL import Image
from tqdm import tqdm

# ================= 配置区 =================
INPUT_DIR = r"D:\work\Steel plate\original_datasets"  # 原始数据集路径 (包含 1, 2, 3, 4 文件夹)
OUTPUT_DIR = r"D:\work\Steel plate\dataset_sliced_new"  # 切片后的输出路径
PATCH_SIZE = 224  # 切片大小 224x224
STRIDE = 112  # 纵向步长 (112 表示 50% 重叠)


# ==========================================

def process_vertical_dataset():
    classes = ['1', '2', '3', '4']

    for cls in classes:
        cls_in_dir = os.path.join(INPUT_DIR, cls)
        cls_out_dir = os.path.join(OUTPUT_DIR, cls)

        if not os.path.exists(cls_in_dir):
            continue

        img_files = [f for f in os.listdir(cls_in_dir) if f.lower().endswith(('.bmp', '.jpg', '.png'))]
        print(f"\n正在处理类别 [{cls}] 的图片...")

        for img_name in tqdm(img_files):
            img_path = os.path.join(cls_in_dir, img_name)

            try:
                img = Image.open(img_path).convert('RGB')
                W, H = img.size

                # 1. 创建子文件夹
                base_name = os.path.splitext(img_name)[0]
                subfolder = os.path.join(cls_out_dir, base_name)
                os.makedirs(subfolder, exist_ok=True)

                # ================= 核心：宽度处理规则 =================
                width_action = "none"
                action_value = 0

                if W < PATCH_SIZE:
                    # 不足 224：创建 224 宽的纯黑背景，将原图居中贴上去
                    new_img = Image.new('RGB', (PATCH_SIZE, H), (0, 0, 0))
                    pad_left = (PATCH_SIZE - W) // 2
                    new_img.paste(img, (pad_left, 0))
                    img = new_img
                    width_action = "padded_left"
                    action_value = pad_left

                elif W > PATCH_SIZE:
                    # 超过 224：两边等量裁掉 (Center Crop)
                    crop_left = (W - PATCH_SIZE) // 2
                    crop_right = crop_left + PATCH_SIZE
                    img = img.crop((crop_left, 0, crop_right, H))
                    width_action = "cropped_left"
                    action_value = crop_left

                # 此时 img 的宽度已经严格等于 224，高度保持不变
                new_W, new_H = img.size

                # 2. 写入 metadata 记录操作，这是后期“完美复原”的钥匙
                with open(os.path.join(subfolder, 'meta.txt'), 'w') as f:
                    f.write(f"Original_Width:{W}\nOriginal_Height:{H}\n")
                    f.write(f"Width_Action:{width_action}\nAction_Value:{action_value}\n")
                    f.write(f"Patch_Size:{PATCH_SIZE}\n")

                # ================= 核心：从上往下纵向切分 =================
                for y in range(0, new_H, STRIDE):
                    upper = y
                    lower = upper + PATCH_SIZE

                    # 处理底部边缘：如果到底了不够 224，就把框向上平移
                    if lower > new_H:
                        lower = new_H
                        upper = max(0, new_H - PATCH_SIZE)

                    # 裁剪切片 (此时 x 轴永远是 0 到 224)
                    patch = img.crop((0, upper, PATCH_SIZE, lower))

                    # 3. 命名规则：只需记录 y 坐标
                    patch_filename = f"{base_name}_y{upper}.bmp"
                    patch.save(os.path.join(subfolder, patch_filename))

                    # 如果已经切到了最底部，退出循环
                    if lower == new_H:
                        break

            except Exception as e:
                print(f"[错误] 处理 {img_name} 失败: {e}")

    print("\n✅ 所有竖向图片切分完成！已按要求处理宽度并记录坐标。")


if __name__ == "__main__":
    process_vertical_dataset()