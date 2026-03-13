import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# 配置路径
IMAGE_DIR = r"/home/duomeitinrfx/data/WuGang_new"  # 你切分好的按文件夹存放的图片
FEATURE_DIR = r"/home/duomeitinrfx/users/yunhe/dataset_features"  # 提取出的特征存放处
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs(FEATURE_DIR, exist_ok=True)

    # 1. 加载预训练的 ResNet18，并砍掉最后的分类层 (fc)
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()  # 将全连接层替换为空操作，直接输出 512 维特征
    resnet = resnet.to(DEVICE)
    resnet.eval()  # 必须设为评估模式

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for cls in ['1', '2', '3', '4']:
        cls_dir = os.path.join(IMAGE_DIR, cls)
        out_cls_dir = os.path.join(FEATURE_DIR, cls)
        os.makedirs(out_cls_dir, exist_ok=True)
        if not os.path.exists(cls_dir): continue

        folders = [f for f in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, f))]

        print(f"正在提取类别 {cls} 的特征...")
        for folder in tqdm(folders):
            folder_path = os.path.join(cls_dir, folder)
            img_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.bmp')]

            if len(img_files) == 0: continue

            features_list = []
            # 2. 逐张读取切片，提取特征
            with torch.no_grad():
                for img_name in img_files:
                    img_path = os.path.join(folder_path, img_name)
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # 增加 batch 维度

                    feat = resnet(img_tensor)  # 形状: (1, 512)
                    features_list.append(feat.cpu())

            # 3. 拼接成 (N, 512) 的矩阵并保存
            bag_feature = torch.cat(features_list, dim=0)
            torch.save(bag_feature, os.path.join(out_cls_dir, f"{folder}.pt"))


if __name__ == '__main__':
    main()