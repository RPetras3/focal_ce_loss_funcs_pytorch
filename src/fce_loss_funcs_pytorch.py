import torch.nn as nn


# Binary Focal Cross-Entropy loss function using nn.Module as a base class

class BFCELoss(nn.Module):
    """Binary Focal Cross-Entropy Loss.

    A focal loss variant of binary cross-entropy that down-weights well-classified
    examples and focuses training on hard, misclassified examples. This is
    particularly useful for addressing class imbalance in binary classification tasks.

    The focal loss is defined as:
        FL(p_t) = -alpha * (1 - p_t) ** gamma * log(p_t)

    When ``gamma = 0``, this reduces to standard binary cross-entropy loss.

    Args:
        apply_class_balancing (bool): Whether to apply class-balancing weights.
            Defaults to ``False``.
        alpha (float): Balancing factor for the focal loss, typically in [0, 1].
            Defaults to ``0.25``.
        gamma (float): Focusing parameter that controls the rate at which easy
            examples are down-weighted. Higher values increase the effect.
            Defaults to ``2.0``.
        from_logits (bool): Whether the predictions are raw logits rather than
            probabilities. Defaults to ``False``.
        axis (int): The axis along which to compute the loss. Defaults to ``-1``.
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
        weight (torch.Tensor, optional): A manual rescaling weight given to the
            loss of each batch element. Defaults to ``None``.
    """

    def __init__(self, 
                 apply_class_balancing=False,
                 alpha=0.25,
                 gamma=2.0,
                 from_logits=False,
                 axis=-1,
                 reduction='sum', # 'none', 'mean', 'sum'
                 weight=None
                 ):
        super().__init__()
        self.apply_class_balancing = apply_class_balancing
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.axis = axis
        self.reduction = reduction
        self.weight = weight
        self.name = 'binary_focal_crossentropy'

    def forward(self, y_pred, y_true):
        """Compute the binary focal cross-entropy loss.

        Args:
            y_pred (torch.Tensor): Predicted probabilities with values in [0, 1].
            y_true (torch.Tensor): Ground-truth binary labels (0 or 1), with the
                same shape as ``y_pred``.

        Returns:
            torch.Tensor: The computed focal loss. If ``gamma`` is 0, returns the
                standard binary cross-entropy loss.
        """
        ce_base = nn.BCELoss(reduction=self.reduction,
                             weight=self.weight,
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


# Focal Cross-Entropy loss function using nn.Module as a base class
class FocalCELoss(nn.Module):
    """Focal Cross-Entropy Loss.

    A focal loss variant of cross-entropy that reduces the relative loss for
    well-classified examples and puts more focus on hard, misclassified examples.
    Extends :class:`BFCELoss` with an additional ``label_smoothing`` parameter.

    The focal loss is defined as:
        FL(p_t) = -alpha * (1 - p_t) ** gamma * log(p_t)

    When ``gamma = 0``, this reduces to standard binary cross-entropy loss.

    Args:
        apply_class_balancing (bool): Whether to apply class-balancing weights.
            Defaults to ``False``.
        alpha (float): Balancing factor for the focal loss, typically in [0, 1].
            Defaults to ``0.25``.
        gamma (float): Focusing parameter that controls the rate at which easy
            examples are down-weighted. Higher values increase the effect.
            Defaults to ``2.0``.
        from_logits (bool): Whether the predictions are raw logits rather than
            probabilities. Defaults to ``False``.
        label_smoothing (float): Amount of label smoothing to apply. A value of
            ``0.0`` means no smoothing. Note: not currently used by the underlying
            ``nn.BCELoss``. Defaults to ``0.0``.
        axis (int): The axis along which to compute the loss. Defaults to ``-1``.
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
        weight (torch.Tensor, optional): A manual rescaling weight given to the
            loss of each batch element. Defaults to ``None``.
    """

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
        """Compute the focal cross-entropy loss.

        Args:
            y_pred (torch.Tensor): Predicted probabilities with values in [0, 1].
            y_true (torch.Tensor): Ground-truth binary labels (0 or 1), with the
                same shape as ``y_pred``.

        Returns:
            torch.Tensor: The computed focal loss. If ``gamma`` is 0, returns the
                standard binary cross-entropy loss.
        """
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
