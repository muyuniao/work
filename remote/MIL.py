import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    def __init__(self, feature_dim=512, num_classes=4):
        super(AttentionMIL, self).__init__()

        # 1. 注意力网络：判断每个切片的重要性
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # 2. 最终的分类器：根据融合后的特征判断等级
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # 防止在小样本上过拟合
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        # x 的输入形状是 (N, 512)，N 是这个钢板包含的切片数量

        # 计算注意力得分
        A = self.attention(x)  # 输出形状: (N, 1)

        # 使用 Softmax 将得分归一化为概率（总和为 1）
        A = F.softmax(A, dim=0)

        # 根据注意力权重，将 N 个 512 维特征融合成 1 个 512 维全局特征
        # A 广播后形状类似 (N, 1)，x 是 (N, 512)，逐元素相乘后在 N 维度求和
        M = torch.sum(A * x, dim=0, keepdim=True)  # 输出形状: (1, 512)
        # 进行分类预测
        Y_prob = self.classifier(M)  # 输出形状: (1, 4)

        # 返回预测结果，同时返回注意力权重 A，供你后续画热力图使用
        return Y_prob, A