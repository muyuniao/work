import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy
import numpy as np
from dlordinal.losses import TriangularCrossEntropyLoss
from dlordinal.metrics import amae

# --- 1. 核心参数配置 ---
DATA_DIR = '/home/duomeitinrfx/data/WuGang'
NUM_CLASSES = 4        # 类别: 1, 2, 3, 4
BATCH_SIZE = 256        
NUM_EPOCHS = 50        
LEARNING_RATE = 0.001
DEVICE = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
def main():
    print(f"当前运行设备: {DEVICE}")

    # --- 2. 数据预处理 ---
    # 已经把图切成 224x224 了，所以这里是转 Tensor 和 归一化
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
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

    # 封装 DataLoader
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=8)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(f"样本数量: 训练集 {dataset_sizes['train']} | 验证集 {dataset_sizes['val']}")

    # --- 4. 搭建模型  ---
    model = models.resnet18(pretrained=True)

    # 修改全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)

    # --- 5. 定义有序分类 Loss ---
    # 替换为dlordinal中的 Triangular Loss
    # alpha 参数控制分布的尖锐程度，1.0 是标准三角形
    criterion = TriangularCrossEntropyLoss(num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 学习率调整策略: 每 7 轮衰减一次
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 6. 训练循环 ---
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = float(10000000) # 我们的目标是让 MAE 越小越好,设置一个很大的初值

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            # 用于计算整个 Epoch 的指标
            epoch_probs = []
            epoch_targets = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) # dlordinal 会自动处理软标签

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                # 收集预测概率 (为了算 MAE，必须先做 Softmax)
                probs = torch.softmax(outputs, dim=1)
                epoch_probs.append(probs.detach().cpu().numpy())
                epoch_targets.append(labels.detach().cpu().numpy())

            if phase == 'train':
                scheduler.step()

            # 计算平均 Loss
            epoch_loss = running_loss / dataset_sizes[phase]

            # 拼接所有 Batch 的结果
            all_probs = np.concatenate(epoch_probs)
            all_targets = np.concatenate(epoch_targets)

            # --- 指标计算 ---
            # 1. 常规准确率 (Accuracy)
            preds = np.argmax(all_probs, axis=1)
            acc = np.mean(preds == all_targets)

            # 2. 有序指标 (MAE)
            # 衡量平均偏离了几个等级
            mae_score = amae(all_targets, all_probs)

            print(f'{phase} Loss: {epoch_loss:.4f} | Acc: {acc:.4f} | MAE: {mae_score:.4f}')

            # --- 核心逻辑: 保存 MAE 最低的模型 ---
            if phase == 'val' and mae_score < best_mae:
                best_mae = mae_score
                best_model_wts = copy.deepcopy(model.state_dict())
                # 保存检查点
                torch.save(model.state_dict(), 'best_ordinal_resnet18.pth')
                print(f" -> 发现更优模型 (MAE: {best_mae:.4f}) 已保存")

    print(f'\n训练全部完成! 验证集最佳 MAE: {best_mae:.4f}')

if __name__ == '__main__':
    main()