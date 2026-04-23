import torch.nn as nn

from gendec.losses.objectives import per_sample_exist_bce, per_sample_flow_mse
from gendec.losses.path import build_flow_batch
from gendec.tokens import TOKEN_DIM


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


class JointFlowMatchingLoss(nn.Module):
    """Phase 2 flow matching loss for joint (E, Z) tokens.

    Splits the target velocity into explicit and residual branches and computes
    separate normalized MSE losses for each.  An optional BCE existence auxiliary
    term is added on top, evaluated against the reconstructed clean explicit
    token's existence-logit channel.

    Args:
        explicit_dim:    number of explicit scaffold channels (default 15)
        lambda_e:        weight for the explicit flow loss (default 1.0)
        lambda_z:        weight for the residual flow loss (default 1.0)
        lambda_exist:    weight for the existence auxiliary loss (default 0.05)
        exist_channel:   index of the existence-logit inside the *explicit* slice
                         (default -1, i.e. channel 14)
    """

    def __init__(
        self,
        explicit_dim=TOKEN_DIM,
        lambda_e=1.0,
        lambda_z=1.0,
        lambda_exist=0.05,
        exist_channel=-1,
    ):
        super().__init__()
        self.explicit_dim = int(explicit_dim)
        self.lambda_e = float(lambda_e)
        self.lambda_z = float(lambda_z)
        self.lambda_exist = float(lambda_exist)
        self.exist_channel = int(exist_channel)

    def _split_velocity(self, v):
        return v[..., :self.explicit_dim], v[..., self.explicit_dim:]

    def _per_sample_metrics(self, batch, v_hat_e, v_hat_z, v_hat):
        v_target_e, v_target_z = self._split_velocity(batch["velocity_target"])

        flow_e = per_sample_flow_mse(v_hat_e, v_target_e)
        flow_z = per_sample_flow_mse(v_hat_z, v_target_z)

        per_sample = {
            "flow_loss_e": flow_e,
            "flow_loss_z": flow_z,
        }
        total = self.lambda_e * flow_e + self.lambda_z * flow_z

        if "exist" in batch and self.lambda_exist > 0:
            # Build a synthetic batch that looks like the Phase 1 exist-bce helper
            # but uses only the explicit portion of the joint token for reconstruction.
            e_batch = {
                "Et": batch["Et"][..., :self.explicit_dim],
                "t": batch["t"],
                "exist": batch["exist"],
            }
            if "token_mean" in batch:
                e_batch["token_mean"] = batch["token_mean"][..., :self.explicit_dim]
            if "token_std" in batch:
                e_batch["token_std"] = batch["token_std"][..., :self.explicit_dim]
            per_sample["exist_loss"] = per_sample_exist_bce(
                e_batch,
                v_hat_e,
                exist_channel=self.exist_channel,
            )
            total = total + self.lambda_exist * per_sample["exist_loss"]

        per_sample["all"] = total
        return per_sample

    def forward(self, batch, v_hat_e, v_hat_z, v_hat=None, return_per_sample=False):  # noqa: ARG002 v_hat kept for API symmetry
        """Compute Phase 2 loss.

        Args:
            batch:    flow batch dict with keys ``Et``, ``velocity_target``,
                      ``t``, and optionally ``exist``, ``token_mean``, ``token_std``
            v_hat_e:  explicit velocity prediction [B, 16, explicit_dim]
            v_hat_z:  residual velocity prediction [B, 16, residual_dim]
            v_hat:    concatenated prediction [B, 16, explicit_dim + residual_dim]
            return_per_sample: if True, include per-sample tensors in metrics

        Returns:
            loss (scalar), metrics (dict)
        """
        per_sample = self._per_sample_metrics(batch, v_hat_e, v_hat_z, v_hat)
        loss = per_sample["all"].mean()
        metrics = {
            "flow_loss_e": float(per_sample["flow_loss_e"].mean().detach().item()),
            "flow_loss_z": float(per_sample["flow_loss_z"].mean().detach().item()),
            "flow_loss": float((per_sample["flow_loss_e"] + per_sample["flow_loss_z"]).mean().detach().item()),
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


__all__ = ["FlowMatchingLoss", "JointFlowMatchingLoss", "build_flow_batch"]
