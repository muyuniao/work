import os
from PIL import Image
from collections import defaultdict

# ==========================================
#              用户配置区域
# ==========================================

# 1. 文件夹路径配置
INPUT_FOLDER = r'D:\work\dlordinal\dataset\Level1'  # 输入：源图片文件夹 (只读取第一层，不进子目录)
SLICE_OUTPUT_FOLDER = r'D:\work\dlordinal\sliced datasets\Level1'  # 输出：切片存放文件夹
RESTORE_OUTPUT_FOLDER = r'D:\work\dlordinal\restoretest'  # 输出：还原后的图片文件夹

# 2. 滑动窗口参数
WINDOW_WIDTH = 224  # 窗口宽度 (切出来的图片宽度严格等于此值)
STEP_SIZE = 112 # 步长

# 3. 命名配置
SEPARATOR = "__loc_"  # 文件名分隔符

# 4. 功能开关
RUN_SLICING = True  # 运行切分
RUN_RESTORING = False  # 运行还原


# ==========================================
#              核心逻辑代码
# ==========================================

def slice_images_root_only():
    """
    只处理 INPUT_FOLDER 根目录下的图片，忽略子文件夹。
    采用固定尺寸切分，最后一块如果不够则回退对齐。
    """
    print(f"\n--- [1] 开始切分 (仅处理根目录文件) ---")

    if not os.path.exists(SLICE_OUTPUT_FOLDER):
        os.makedirs(SLICE_OUTPUT_FOLDER)

    count = 0

    # --- 修改点：使用 listdir 代替 walk，只获取第一层文件 ---
    # os.listdir 返回文件名列表，不包含路径
    all_items = os.listdir(INPUT_FOLDER)

    for filename in all_items:
        # 构建完整路径
        src_path = os.path.join(INPUT_FOLDER, filename)

        # 1. 确保它是文件而不是文件夹
        if not os.path.isfile(src_path):
            continue

        # 2. 检查图片后缀
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            continue

        try:
            with Image.open(src_path) as img:
                w, h = img.size

                # 检查图片是否小于窗口宽度
                if w < WINDOW_WIDTH:
                    print(f"[跳过] {filename} 宽度({w}) 小于窗口宽度({WINDOW_WIDTH})")
                    continue

                x = 0
                slice_idx = 0

                while True:
                    # 计算当前窗口理论右边界
                    current_right = x + WINDOW_WIDTH

                    # 越界判断与回退逻辑
                    if current_right > w:
                        # 强制回退，对齐最右边
                        x = w - WINDOW_WIDTH
                        is_last_block = True
                    else:
                        is_last_block = False

                    # 裁剪
                    box = (x, 0, x + WINDOW_WIDTH, h)
                    crop_img = img.crop(box)

                    # 保存: 原文件名__loc_X坐标.png
                    new_filename = f"{filename}{SEPARATOR}{x}.png"
                    save_path = os.path.join(SLICE_OUTPUT_FOLDER, new_filename)
                    crop_img.save(save_path, "PNG")

                    slice_idx += 1

                    # 退出条件
                    if is_last_block:
                        break
                    if x + WINDOW_WIDTH == w:
                        break

                    x += STEP_SIZE

                print(f"[切分] {filename} -> {slice_idx} 张")
                count += 1

        except Exception as e:
            print(f"[错误] 处理 {filename} 异常: {e}")

    print(f"切分完成，共处理 {count} 张图片。")


def restore_images():
    """
    还原程序 (逻辑保持不变)
    """
    print(f"\n--- [2] 开始还原 ---")

    if not os.path.exists(RESTORE_OUTPUT_FOLDER):
        os.makedirs(RESTORE_OUTPUT_FOLDER)

    image_groups = defaultdict(list)
    files = os.listdir(SLICE_OUTPUT_FOLDER)

    for fname in files:
        if SEPARATOR in fname and fname.endswith('.png'):
            try:
                original_name, rest = fname.rsplit(SEPARATOR, 1)
                x_coord = int(rest.replace('.png', ''))
                full_path = os.path.join(SLICE_OUTPUT_FOLDER, fname)
                image_groups[original_name].append((x_coord, full_path))
            except ValueError:
                continue

    if not image_groups:
        print("没有找到符合规则的切片文件。")
        return

    restore_count = 0
    for original_name, slices in image_groups.items():
        try:
            # 必须排序，确保回退的那一张最后贴
            slices.sort(key=lambda s: s[0])

            # 扫描尺寸
            total_width = 0
            img_height = 0
            for x, path in slices:
                with Image.open(path) as s_img:
                    s_w, s_h = s_img.size
                    img_height = s_h
                    if x + s_w > total_width:
                        total_width = x + s_w

            # 粘贴还原
            canvas = Image.new('RGB', (total_width, img_height))
            for x, path in slices:
                with Image.open(path) as s_img:
                    canvas.paste(s_img, (x, 0))

            save_name = f"Restored_{original_name}.png"
            canvas.save(os.path.join(RESTORE_OUTPUT_FOLDER, save_name))
            print(f"[还原] {save_name}")
            restore_count += 1

        except Exception as e:
            print(f"[错误] 还原 {original_name} 失败: {e}")

    print(f"还原完成，共生成 {restore_count} 张图片。")


# ==========================================
#              程序入口
# ==========================================
if __name__ == "__main__":
    if RUN_SLICING:
        slice_images_root_only()

    if RUN_RESTORING:
        restore_images()