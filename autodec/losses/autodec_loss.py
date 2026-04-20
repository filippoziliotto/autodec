import torch
import torch.nn as nn

from autodec.losses.chamfer import weighted_chamfer_l2
from autodec.losses.sq_regularizer import (
    SQRegularizer,
    assignment_parsimony_loss,
    existence_loss,
)


def _metric_value(value):
    if torch.is_tensor(value):
        return value.detach().item()
    return float(value)


def _phase_number(phase):
    if isinstance(phase, str):
        text = phase.lower().replace("_", "").replace("-", "")
        if text.startswith("phase"):
            text = text[5:]
        return int(text)
    return int(phase)


def _target_points(batch, reference):
    return batch["points"].to(device=reference.device, dtype=reference.dtype)


def _primitive_mass_entropy(assign_matrix, eps=1e-8):
    mass = assign_matrix.mean(dim=1)
    probs = mass / mass.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
    return entropy.mean()


def _active_primitive_count(outdict, threshold):
    if "exist" in outdict:
        exist = outdict["exist"]
    else:
        exist = torch.sigmoid(outdict["exist_logit"])
    return (exist.squeeze(-1) > threshold).to(torch.float32).sum(dim=-1).mean()


def _offset_ratio(outdict, eps=1e-8):
    if "decoded_offsets" not in outdict or "surface_points" not in outdict:
        reference = outdict["decoded_points"]
        return reference.new_tensor(0.0)
    offset_norm = outdict["decoded_offsets"].norm(dim=-1).mean()
    scaffold_norm = outdict["surface_points"].norm(dim=-1).mean()
    return offset_norm / scaffold_norm.clamp_min(eps)


def _gated_offset_ratio(outdict, eps=1e-8):
    if (
        "decoded_offsets" not in outdict
        or "decoded_weights" not in outdict
        or "surface_points" not in outdict
    ):
        reference = outdict["decoded_points"]
        return reference.new_tensor(0.0)
    gated_offsets = outdict["decoded_offsets"] * outdict["decoded_weights"].unsqueeze(-1)
    offset_norm = gated_offsets.norm(dim=-1).mean()
    scaffold_norm = outdict["surface_points"].norm(dim=-1).mean()
    return offset_norm / scaffold_norm.clamp_min(eps)


def _offset_cap_saturation(outdict, eps=1e-8, saturation_threshold=0.95):
    if "decoded_offsets" not in outdict or "offset_limit" not in outdict:
        return None
    saturation = (
        outdict["decoded_offsets"].abs()
        / outdict["offset_limit"].clamp_min(eps)
    ).clamp(max=1.0)
    return {
        "offset_cap_saturation": saturation.mean(),
        "offset_cap_saturated_fraction": (
            saturation >= saturation_threshold
        ).to(torch.float32).mean(),
    }


class AutoDecLoss(nn.Module):
    """Phase-aware AutoDec loss wrapper."""

    def __init__(
        self,
        phase=1,
        lambda_sq=1.0,
        lambda_par=0.06,
        lambda_exist=0.01,
        lambda_cons=0.0,
        n_sq_samples=256,
        sq_tau=1.0,
        angle_sampler=None,
        sq_regularizer=None,
        exist_point_threshold=24.0,
        active_exist_threshold=0.5,
        chamfer_eps=1e-6,
        min_backward_weight=1e-3,
    ):
        super().__init__()
        self.phase = _phase_number(phase)
        self.lambda_sq = lambda_sq
        self.lambda_par = lambda_par
        self.lambda_exist = lambda_exist
        self.lambda_cons = lambda_cons
        self.exist_point_threshold = exist_point_threshold
        self.active_exist_threshold = active_exist_threshold
        self.chamfer_eps = chamfer_eps
        self.min_backward_weight = min_backward_weight
        self.sq_regularizer = sq_regularizer or SQRegularizer(
            n_samples=n_sq_samples,
            tau=sq_tau,
            angle_sampler=angle_sampler,
        )

    def _reconstruction_loss(self, outdict, target):
        return weighted_chamfer_l2(
            outdict["decoded_points"],
            target,
            outdict["decoded_weights"],
            eps=self.chamfer_eps,
            min_backward_weight=self.min_backward_weight,
            return_components=True,
        )

    def _scaffold_chamfer(self, outdict, target):
        if "surface_points" not in outdict or "decoded_weights" not in outdict:
            return None
        with torch.no_grad():
            return weighted_chamfer_l2(
                outdict["surface_points"],
                target,
                outdict["decoded_weights"],
                eps=self.chamfer_eps,
                min_backward_weight=self.min_backward_weight,
            )

    def _consistency_loss(self, outdict, target):
        if "consistency_decoded_points" not in outdict:
            raise ValueError(
                "lambda_cons > 0 requires outdict['consistency_decoded_points']; "
                "call the model with return_consistency=True"
            )
        if "decoded_weights" not in outdict:
            raise ValueError("lambda_cons > 0 requires outdict['decoded_weights']")
        return weighted_chamfer_l2(
            outdict["consistency_decoded_points"],
            target,
            outdict["decoded_weights"],
            eps=self.chamfer_eps,
            min_backward_weight=self.min_backward_weight,
        )

    def forward(self, batch, outdict):
        target = _target_points(batch, outdict["decoded_points"])
        recon, recon_components = self._reconstruction_loss(outdict, target)
        loss = recon

        metrics = {
            "recon": _metric_value(recon),
            "recon_forward": _metric_value(recon_components["forward"]),
            "recon_backward": _metric_value(recon_components["backward"]),
            "active_weight_sum": _metric_value(outdict["decoded_weights"].sum(dim=1).mean()),
            "offset_ratio": _metric_value(_offset_ratio(outdict)),
            "gated_offset_ratio": _metric_value(_gated_offset_ratio(outdict)),
        }

        cap_saturation = _offset_cap_saturation(outdict)
        if cap_saturation is not None:
            metrics.update(
                {
                    key: _metric_value(value)
                    for key, value in cap_saturation.items()
                }
            )

        scaffold_chamfer = self._scaffold_chamfer(outdict, target)
        if scaffold_chamfer is not None:
            metrics["scaffold_chamfer"] = _metric_value(scaffold_chamfer)

        if "assign_matrix" in outdict:
            metrics["primitive_mass_entropy"] = _metric_value(
                _primitive_mass_entropy(outdict["assign_matrix"])
            )
        if "exist" in outdict or "exist_logit" in outdict:
            metrics["active_primitive_count"] = _metric_value(
                _active_primitive_count(outdict, self.active_exist_threshold)
            )

        if self.phase >= 2:
            if self.lambda_sq > 0:
                sq_loss, sq_components = self.sq_regularizer(
                    batch,
                    outdict,
                    return_components=True,
                )
                loss = loss + self.lambda_sq * sq_loss
                metrics["sq_loss"] = _metric_value(sq_loss)
                metrics["sq_point_to_prim"] = _metric_value(sq_components["point_to_sq"])
                metrics["sq_prim_to_point"] = _metric_value(sq_components["sq_to_point"])

            if self.lambda_par > 0:
                par_loss = assignment_parsimony_loss(outdict["assign_matrix"])
                loss = loss + self.lambda_par * par_loss
                metrics["parsimony_loss"] = _metric_value(par_loss)

            if self.lambda_exist > 0:
                exist_loss_value = existence_loss(
                    outdict["assign_matrix"],
                    exist=outdict.get("exist"),
                    exist_logit=outdict.get("exist_logit"),
                    point_threshold=self.exist_point_threshold,
                )
                loss = loss + self.lambda_exist * exist_loss_value
                metrics["exist_loss"] = _metric_value(exist_loss_value)

        if self.lambda_cons > 0:
            consistency = self._consistency_loss(outdict, target)
            loss = loss + self.lambda_cons * consistency
            metrics["consistency_loss"] = _metric_value(consistency)

        metrics["all"] = _metric_value(loss)
        return loss, metrics
