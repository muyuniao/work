import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from MIL import AttentionMIL  # 导入你的模型

# ================= 配置区 =================
FEATURE_DIR = '/home/duomeitinrfx/users/yunhe/dataset_features'  # .pt 特征文件夹路径
NUM_EPOCHS = 30  # 因为数据集变小，收敛会更快，可以适当调小
K_FOLDS = 5  # 5 折交叉验证
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================

# 1. 升级版的数据集类 (支持传入指定的文件列表)
class BagDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feature = torch.load(self.files[idx]).to(DEVICE)
        label = torch.tensor(self.labels[idx], dtype=torch.long).to(DEVICE)
        return feature, label


def get_all_data(feature_dir):
    """从文件夹中一次性读取所有文件路径和标签"""
    all_files = []
    all_labels = []
    classes = ['1', '2', '3', '4']

    for i, cls in enumerate(classes):
        cls_dir = os.path.join(feature_dir, cls)
        if not os.path.exists(cls_dir): continue
        for f in os.listdir(cls_dir):
            if f.endswith('.pt'):
                all_files.append(os.path.join(cls_dir, f))
                all_labels.append(i)  # 0, 1, 2, 3

    return np.array(all_files), np.array(all_labels)


def main():
    print(f"当前运行设备: {DEVICE}")

    # 获取全部数据
    X_files, y_labels = get_all_data(FEATURE_DIR)
    print(f"总计加载了 {len(X_files)} 张钢板的特征矩阵。")
    print(f"准备进行 {K_FOLDS} 折分层交叉验证...\n")

    # 定义分层交叉验证器
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_results = []  # 记录每一折的最佳验证集准确率

    # ================= K-Fold 核心循环 =================
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_files, y_labels)):
        print(f"{'=' * 15} 开始第 {fold + 1}/{K_FOLDS} 折训练 {'=' * 15}")

        # 划分训练集和验证集
        train_files, train_labels = X_files[train_idx], y_labels[train_idx]
        val_files, val_labels = X_files[val_idx], y_labels[val_idx]

        train_loader = DataLoader(BagDataset(train_files, train_labels), batch_size=1, shuffle=True)
        val_loader = DataLoader(BagDataset(val_files, val_labels), batch_size=1, shuffle=False)

        # 每一折都必须重新初始化一个全新的模型！
        # 绝不能用上一折训练好的模型接着练，否则会造成数据穿越
        model = AttentionMIL(feature_dim=512, num_classes=4).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

        best_val_acc = 0.0

        # 训练 Epochs
        for epoch in range(NUM_EPOCHS):
            # --- 训练阶段 ---
            model.train()
            train_loss, train_corrects = 0.0, 0
            for features, labels in train_loader:
                features = features.squeeze(0)
                optimizer.zero_grad()
                outputs, _ = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_corrects += torch.sum(preds == labels.data).item()

            train_acc = train_corrects / len(train_loader)

            # --- 验证阶段 ---
            model.eval()
            val_loss, val_corrects = 0.0, 0
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.squeeze(0)
                    outputs, _ = model(features)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_corrects += torch.sum(preds == labels.data).item()

            val_acc = val_corrects / len(val_loader)

            # 记录本折的最好成绩
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # 可选：可以把每一折的最佳模型存下来
                # torch.save(model.state_dict(), f'best_model_fold{fold+1}.pth')

            # 每 10 个 Epoch 打印一次，
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        print(f"--> 第 {fold + 1} 折验证集最高准确率: {best_val_acc:.4f}\n")
        fold_results.append(best_val_acc)

    # ================= 最终成绩汇报 =================
    print(f"{'=' * 15} 最终评估报告 {'=' * 15}")
    for i, acc in enumerate(fold_results):
        print(f"Fold {i + 1}: {acc:.4f}")

    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    print(f"平均验证准确率 (Mean ± Std): {mean_acc:.4f} ± {std_acc:.4f}")


if __name__ == '__main__':
    main()