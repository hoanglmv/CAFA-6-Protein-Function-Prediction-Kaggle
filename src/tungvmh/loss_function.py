import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        """
        Args:
            gamma_neg (float): Hệ số giảm trọng số cho negative samples (quan trọng nhất cho CAFA).
            gamma_pos (float): Hệ số giảm trọng số cho positive samples.
            clip (float): Margin để dịch chuyển xác suất (probability shifting).
            disable_torch_grad_focal_loss (bool): Tắt grad cho phần weighting factor để ổn định hơn.
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """
        Args:
            x: Logits tensor (batch_size, num_classes) - ĐẦU RA CỦA MODEL (chưa qua Sigmoid)
            y: Targets tensor (batch_size, num_classes) - Multi-hot vector
        """

        # 1. Tính xác suất p từ logits
        x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x_sigmoid.clamp(min=self.eps, max=1.0 - self.eps)  # Tránh log(0)

        p_pos = x_sigmoid
        p_neg = 1 - x_sigmoid

        # 3. Tính Loss cho Positive (y=1)
        # Basic Cross Entropy: -log(p)
        # Weighting factor: (1-p)^gamma_pos
        if self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            # (1 - p) ** gamma
            pos_weight = (1 - p_pos).pow(self.gamma_pos)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)

            # Loss = - weight * y * log(p)
            loss_pos = -pos_weight * y * torch.log(p_pos)
        else:
            loss_pos = -y * torch.log(p_pos)

        # 4. Tính Loss cho Negative (y=0) - QUAN TRỌNG NHẤT
        if self.clip > 0:
            p_neg_shifted = (p_neg + self.clip - 1).clamp(
                max=0
            ) + 1  # Logic dịch chuyển

        else:
            p_neg_shifted = p_neg

        if self.gamma_neg > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            neg_weight = (1 - p_neg_shifted).pow(self.gamma_neg)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)

            loss_neg = -neg_weight * (1 - y) * torch.log(p_neg_shifted)
        else:
            loss_neg = -(1 - y) * torch.log(p_neg_shifted)

        loss = loss_pos + loss_neg

        return loss.sum() / x.size(0)
