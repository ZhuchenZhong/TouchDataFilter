import torch
import torch.nn as nn

class DiffPreservingLoss(nn.Module):
    """差异保留损失函数，为高差异区域提供更大权重"""

    def __init__(self, alpha=1.8, beta=0.5):
        super(DiffPreservingLoss, self).__init__()
        self.alpha = alpha  # 控制高差异区域的权重系数
        self.beta = beta  # MSE损失和差异保留损失的平衡系数
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, outputs, targets):
        # 基本MSE损失
        mse_loss = self.mse(outputs, targets)

        # 计算差异权重 - 绝对值越大的区域权重越高
        diff_weights = torch.abs(targets).pow(self.alpha) + 0.5

        # 加权MSE损失
        weighted_loss = (mse_loss * diff_weights).mean()

        # 同时考虑普通MSE和加权MSE
        total_loss = self.beta * mse_loss.mean() + (1 - self.beta) * weighted_loss

        return total_loss


class TouchFilterNet(nn.Module):
    """触摸数据滤波网络 - 增强版以保留差异性"""

    def __init__(self, in_channels=1):
        super(TouchFilterNet, self).__init__()

        # 特征提取
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 注意力机制 - 关注高差异区域
        self.attention = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # 解码器（恢复原始分辨率）
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
        )

        # 残差连接 - 使用1x1卷积学习残差映射
        self.residual = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 差异增强层 - 学习如何根据输入差异调整输出
        self.diff_enhancer = nn.Sequential(
            nn.Conv2d(in_channels * 2, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, in_channels, kernel_size=1),
            nn.Tanh(),  # 控制增强/抑制范围
        )

    def forward(self, x):
        # 保存原始输入用于残差连接和差异增强
        original = x
        residual = self.residual(x)

        # 编码
        encoded = self.encoder(x)

        # 注意力机制
        attention_map = self.attention(encoded)
        attended = encoded * attention_map

        # 解码
        decoded = self.decoder(attended)

        # 基础输出（带残差连接）
        base_output = decoded + residual

        # 差异增强: 将原始输入与基础输出拼接，让网络学习差异调整
        concat = torch.cat([original, base_output], dim=1)
        diff_scale = self.diff_enhancer(concat)

        # 最终输出: 基础输出加上差异调整
        # 对高差异区域增强，对低差异区域保持不变或轻微抑制
        return base_output * (1.0 + diff_scale)
