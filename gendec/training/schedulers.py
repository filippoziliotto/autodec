import math

from torch.optim.lr_scheduler import LambdaLR


def build_cosine_warmup_scheduler(optimizer, total_steps, warmup_steps=0, min_lr=1e-5):
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def _lr_lambda_factory(base_lr):
        min_ratio = min(float(min_lr) / max(float(base_lr), 1e-12), 1.0)

        def _lr_lambda(step):
            step = min(int(step), total_steps)
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return _lr_lambda

    return LambdaLR(optimizer, lr_lambda=[_lr_lambda_factory(base_lr) for base_lr in base_lrs])
