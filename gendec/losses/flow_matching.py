import torch
import torch.nn as nn

from gendec.losses.objectives import per_sample_exist_bce, per_sample_flow_mse
from gendec.losses.path import build_flow_batch


class FlowMatchingLoss(nn.Module):
    def __init__(self, lambda_flow=1.0, lambda_exist=0.05, exist_channel=-1):
        super().__init__()
        self.lambda_flow = float(lambda_flow)
        self.lambda_exist = float(lambda_exist)
        self.exist_channel = int(exist_channel)

    def _per_sample_metrics(self, batch, v_hat):
        per_sample = {
            "flow_loss": per_sample_flow_mse(v_hat, batch["velocity_target"]),
        }
        total = self.lambda_flow * per_sample["flow_loss"]
        if "exist" in batch and self.lambda_exist > 0:
            per_sample["exist_loss"] = per_sample_exist_bce(
                batch,
                v_hat,
                exist_channel=self.exist_channel,
            )
            total = total + self.lambda_exist * per_sample["exist_loss"]
        per_sample["all"] = total
        return per_sample

    def forward(self, batch, v_hat, return_per_sample=False):
        per_sample = self._per_sample_metrics(batch, v_hat)
        loss = per_sample["all"].mean()
        metrics = {
            "flow_loss": float(per_sample["flow_loss"].mean().detach().item()),
            "all": float(loss.detach().item()),
        }
        if "exist_loss" in per_sample:
            metrics["exist_loss"] = float(per_sample["exist_loss"].mean().detach().item())
        if return_per_sample:
            metrics["per_sample"] = {
                key: value.detach()
                for key, value in per_sample.items()
            }
        return loss, metrics


__all__ = ["FlowMatchingLoss", "build_flow_batch"]
