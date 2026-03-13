import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from MIL import AttentionMIL  # 导入我们刚刚写的网络模型

# ================= 配置区 =================
FEATURE_DIR = '/home/duomeitinrfx/users/yunhe/dataset_features'  # 填你服务器上 .pt 文件的路径
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================

# 1. 定义包级别数据集 (Bag Dataset)
class BagDataset(Dataset):
    def __init__(self, feature_dir):
        self.files = []
        self.labels = []
        classes = ['1', '2', '3', '4']
        for i, cls in enumerate(classes):
            cls_dir = os.path.join(feature_dir, cls)
            if not os.path.exists(cls_dir): continue

            for f in os.listdir(cls_dir):
                if f.endswith('.pt'):
                    self.files.append(os.path.join(cls_dir, f))
                    # 注意：PyTorch 里的标签索引是从 0 开始的 (0,1,2,3 对应 1,2,3,4级)
                    self.labels.append(i)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 读取 (N, 512) 的特征张量
        feature = torch.load(self.files[idx]).to(DEVICE)
        label = torch.tensor(self.labels[idx], dtype=torch.long).to(DEVICE)
        return feature, label


def main():
    print(f"当前运行设备: {DEVICE}")
    dataset = BagDataset(FEATURE_DIR)

    # ⚠️ 极度关键：Batch Size 必须为 1！
    # 因为每块钢板切出来的张数 (N) 不一样，无法拼接成规则的 Batch 矩阵。
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"总计加载了 {len(dataset)} 张钢板的特征矩阵。")

    # 2. 初始化网络、损失函数和优化器
    model = AttentionMIL(feature_dim=512, num_classes=4).to(DEVICE)

    # 这里你可以随时替换成 dlordinal 的 TriangularCrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # 3. 极速训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        corrects = 0

        for features, labels in dataloader:
            # features 形状: (1, N, 512), 需要 squeeze 掉 batch 维度变成 (N, 512)
            features = features.squeeze(0)

            optimizer.zero_grad()

            # 前向传播：返回类别概率和注意力权重
            outputs, attention_weights = model(features)

            # 计算 Loss 并反向传播
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(dataset)
        epoch_acc = corrects / len(dataset)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | 原图 Acc: {epoch_acc:.4f}")

    print("训练完成！")


if __name__ == '__main__':
    main()