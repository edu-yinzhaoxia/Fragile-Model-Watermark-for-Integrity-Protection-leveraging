import torch.nn.functional as F
import torch.nn as nn


class ActivationHook:
    def __init__(self):
        self.loss = 0.0

    def hook_fn(self, module, input, output):
        s = F.relu(-1.0 * output)
        self.loss += s.sum().item()

    def register_hook(self, model):
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d,nn.Linear)):
                layer.register_forward_hook(self.hook_fn)

    def compute_loss(self):
        loss = self.loss  # Store the current value
        self.loss = 0.0  # Reset the loss for the next iteration
        return loss