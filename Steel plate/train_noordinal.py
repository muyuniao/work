import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy
import numpy as np
import time

# --- 1. 核心参数配置 ---
DATA_DIR = '/home/duomeitinrfx/data/WuGang'
NUM_CLASSES = 4        # 类别: 1, 2, 3, 4
BATCH_SIZE = 256
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4    # [新增] L2 正则化系数，防止过拟合
PATIENCE = 10          # [新增] 早停机制忍耐轮数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"当前运行设备: {DEVICE}")

    # --- 2. 数据预处理 (增强版) ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            # [新增] 数据增强策略：防止过拟合的核心手段
            transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
            transforms.RandomVerticalFlip(p=0.5),   # 随机垂直翻转 (钢板缺陷通常没有方向性)
            transforms.RandomRotation(15),          # 随机旋转 +/- 15度
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- 3. 加载数据集 ---
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=4)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(f"样本数量: 训练集 {dataset_sizes['train']} | 验证集 {dataset_sizes['val']}")

    # --- 4. 搭建模型 ---
    model = models.resnet18(pretrained=True)

    # 修改全连接层，加入 Dropout
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),              # [新增] Dropout层，丢弃50%神经元防止过拟合
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    # --- 5. 定义标准分类 Loss 和 优化器 ---
    # [修改] 使用标准的交叉熵损失，不再使用有序损失
    criterion = nn.CrossEntropyLoss()

    # [修改] 加入 weight_decay (L2正则化)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 学习率调整策略
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 6. 训练循环 (含早停机制) ---
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0          # [修改] 目标改为让准确率(Acc)越高越好
    epochs_no_improve = 0   # 早停计数器

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 20)
        start_time = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计状态
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            # --- 核心逻辑: 保存 Acc 最高的模型 ---
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'best_resnet18_standard.pth')
                    print(f" -> 发现更优模型 (Acc: {best_acc:.4f}) 已保存")
                    epochs_no_improve = 0 # 重置早停计数器
                else:
                    epochs_no_improve += 1
                    print(f" -> 验证集未提升 ({epochs_no_improve}/{PATIENCE})")

        # --- 早停判断 ---
        if epochs_no_improve >= PATIENCE:
            print(f'\n早停触发! 在连续 {PATIENCE} 个 Epoch 内验证集准确率未提升。')
            break

        print(f"耗时: {time.time() - start_time:.0f}s")

    print(f'\n训练全部完成! 验证集最佳 Acc: {best_acc:.4f}')

if __name__ == '__main__':
    main()