import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) cho bài toán multi-label classification
    Đặc biệt phù hợp với CAFA protein function prediction (long-tailed distribution)

    Paper: "Asymmetric Loss For Multi-Label Classification"
    https://arxiv.org/abs/2009.14119

    Args:
        gamma_neg (float): Focusing parameter cho negative samples. Default: 4
        gamma_pos (float): Focusing parameter cho positive samples. Default: 1
        clip (float): Probability clipping value để tránh gradient quá lớn. Default: 0.05
        eps (float): Epsilon cho numerical stability. Default: 1e-8
        disable_torch_grad_focal_loss (bool): Nếu True, không tính gradient qua focal weight.
                                               Giúp tăng tốc training. Default: True
    """

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits, targets):
        """
        Forward pass

        Args:
            logits (torch.Tensor): Model output logits, shape: (batch_size, num_classes)
            targets (torch.Tensor): Multi-hot ground truth, shape: (batch_size, num_classes)
                                   Values phải là 0 hoặc 1

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Sigmoid để convert logits thành probabilities
        probs = torch.sigmoid(logits)

        # Probability clipping
        if self.clip is not None and self.clip > 0:
            probs = probs.clamp(min=self.clip)

        # === Tính loss cho positive samples ===
        # targets == 1
        pos_loss = targets * torch.log(probs + self.eps)

        # Focal weight cho positive samples: (1 - p)^gamma_pos
        if self.gamma_pos > 0:
            pos_focal_weight = (1 - probs) ** self.gamma_pos
            if self.disable_torch_grad_focal_loss:
                pos_focal_weight = pos_focal_weight.detach()
            pos_loss = pos_loss * pos_focal_weight

        # === Tính loss cho negative samples ===
        # targets == 0
        neg_loss = (1 - targets) * torch.log(1 - probs + self.eps)

        # Focal weight cho negative samples: p^gamma_neg
        if self.gamma_neg > 0:
            neg_focal_weight = probs**self.gamma_neg
            if self.disable_torch_grad_focal_loss:
                neg_focal_weight = neg_focal_weight.detach()
            neg_loss = neg_loss * neg_focal_weight

        # Asymmetric shifting: chỉ áp dụng clipping cho negative loss
        if self.clip is not None and self.clip > 0:
            # Điều chỉnh negative loss với clipping
            neg_loss = neg_loss * (1 - targets)

        # Tổng hợp loss
        loss = -(pos_loss + neg_loss)

        # Return mean loss across all elements
        return loss.mean()
