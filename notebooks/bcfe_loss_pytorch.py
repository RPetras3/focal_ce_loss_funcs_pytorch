import torch.nn as nn


# Custom loss function using nn.Module as a base class
class CustomLoss(nn.Module):
    def __init__(self, 
                 apply_class_balancing=False,
                 alpha=0.25,
                 gamma=2.0,
                 from_logits=False,
                 label_smoothing=0.0,
                 axis=-1,
                 reduction='sum', # 'none', 'mean', 'sum'
                 weight=None
                 ):
        super().__init__()
        self.apply_class_balancing = apply_class_balancing
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis
        self.reduction = reduction
        self.weight = weight
        self.name = 'binary_focal_crossentropy'

    def forward(self, y_pred, y_true):
        
        ce_base = nn.BCELoss(reduction=self.reduction,
                             weight=self.weight,
                             # label_smoothing=self.label_smoothing) # Not needed for Binary Cross-Entropy Loss in PyTorch, as it does not support label smoothing directly
                             )
        p_t = ce_base(y_pred, y_true)
        # CE(p_t) = − log(p_t)
        # FL(p_t) = −(1 − p_t)γ * log(p_t)
        # or equivalently:
        # FL(p_t) = (1 - p_t) ** gamma * CE(p_t)
        if self.gamma != 0:
            focal_loss = (self.alpha * (1 - p_t) ** self.gamma) * p_t
            return focal_loss
        else:
            # If gamma is 0, focal loss is just cross-entropy loss
            return p_t

