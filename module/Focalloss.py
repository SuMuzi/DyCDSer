class_counts = torch.tensor([12842645, 2917310, 509790, 79295], dtype=torch.float32).to(torch.device('cuda'))
alpha = torch.log(class_counts.sum() / class_counts)
alpha = alpha / alpha.sum() * len(class_counts)
criterion_cls = FocalLoss(alpha=alpha, gamma=2.0).to(torch.device('cuda'))

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (B, C) 或 (B, C, H, W)
        targets: (B, ) 或 (B, H, W)  类别索引
        """
        ce = F.cross_entropy(logits, targets, reduction='none')  # -log(pt)
        pt = torch.exp(-ce)  # pt
        focal = (1 - pt) ** self.gamma * ce  # (1-pt)^γ * -log(pt)
        if self.alpha is not None:
            alpha_t = self.alpha[targets] if isinstance(self.alpha, torch.Tensor) else self.alpha
            focal = alpha_t * focal
        return focal.mean() if self.reduction == 'mean' else focal.sum()