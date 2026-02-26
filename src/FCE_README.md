# Focal Cross Entropy Loss

A PyTorch-compatible implementation of the Focal Cross Entropy loss function, designed to address class imbalance by down-weighting easy examples and focusing training on hard negatives.

## Install

```text
pip install focal_ce_loss_funcs_pytorch
```

## Quick start

```python
from focal_ce_loss import FocalCELoss

criterion = FocalCrossEntropyLoss()
loss = criterion(inputs, targets)
```
### FocalCELoss

A PyTorch `nn.Module` implementing the Focal Cross Entropy loss.

```python
from focal_ce_loss import FocalCELoss

criterion = FocalCELoss(alpha=0.25, gamma=2.0, reduction="sum")

# Compute loss from logits and targets
loss = criterion(inputs, targets)
loss.backward()
```

**Parameters**
- `gamma`: focusing parameter; higher values down-weight easy examples more aggressively (default: `2.0`)
- `alpha`: optional class weight tensor of shape `(C,)` for additional class balancing (default: `0.25`)
- `reduction`: specifies the reduction to apply — `"mean"`, `"sum"`, or `"none"` (default: `"sum"`)
- `ignore_index`: target value to ignore during loss computation (default: `-100`)

**Methods**
- `forward(input, target)` — accepts a torch.Tensor of shape `(N, C)` and integer targets of shape `(N,)`; returns a scalar loss tensor (or per-sample tensor if `reduction="none"`)

**Inputs**
- `input`: `torch.Tensor` of shape `(N, C)`
- `target`: `torch.Tensor` of shape `(N,)`
`
**Example**

```python
import torch
from focal_ce_loss import FocalCELoss

inputs  = torch.randn(8, 10)          # batch of 8, 10 classes
targets = torch.randint(0, 10, (8,))  # ground-truth labels

criterion = FocalCrossEntropyLoss(gamma=2.0)
loss = criterion(inputs, targets)
loss.backward()
```

### BFCELoss

A PyTorch `nn.Module` implementing the Binary Focal Cross Entropy loss for binary classification tasks.

```python
from focal_ce_loss import BFCELoss

criterion = BFCELoss(alpha=0.25, gamma=2.0, reduction="sum")

# Compute loss from logits and binary targets
loss = criterion(inputs, targets)
loss.backward()
```

**Parameters**
- `gamma`: focusing parameter; higher values down-weight easy examples more aggressively (default: `2.0`)
- `alpha`: optional scalar weight for the positive class (default: `0.25`)
- `reduction`: specifies the reduction to apply — `"mean"`, `"sum"`, or `"none"` (default: `"mean"`)
- `ignore_index`: target value to ignore during loss computation (default: `-100`)

**Methods**
- `forward(input, target)` — accepts a torch.Tensor of shape `(N,)` or `(N, 1)` and binary targets of shape `(N,)`; returns a scalar loss tensor (or per-sample tensor if `reduction="none"`)

**Inputs**
- `input`: `torch.Tensor` of shape `(N,)` or `(N, 1)` of predicted probabilities with values in [0, 1].
- `target`: `torch.Tensor` of shape `(N,)` containing binary labels `{0, 1}`

**Example**

```python
import torch
from focal_ce_loss import BFCELoss

inputs  = torch.randn(8)           # batch of 8 logits
targets = torch.randint(0, 2, (8,)).float()  # binary ground-truth labels

criterion = BFCELoss(gamma=2.0, alpha=0.25)
loss = criterion(inputs, targets)
loss.backward()
```