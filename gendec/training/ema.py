import copy

import torch


class ModelEma:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            model_value = model_state[key].detach()
            if not torch.is_floating_point(value):
                value.copy_(model_value)
                continue
            value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)
