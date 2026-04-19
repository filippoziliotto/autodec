import torch


def _to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _threshold_key(value):
    text = f"{float(value):.6g}".replace(".", "_")
    return text


def paper_chamfer_metrics(pred, target, f_score_threshold=0.01, eps=1e-8):
    """Return paper-style point-cloud reconstruction metrics.

    Chamfer-L1 uses Euclidean nearest-neighbor distances. Chamfer-L2 uses the
    squared nearest-neighbor distances. Both terms are averaged symmetrically:

        0.5 * (mean pred->target + mean target->pred)

    The x100 variants match the common table-reporting convention. F-score uses
    precision over predicted points and recall over target points at the given
    Euclidean distance threshold.
    """

    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("pred must have shape [B, M, 3]")
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError("target must have shape [B, N, 3]")
    if pred.shape[0] != target.shape[0]:
        raise ValueError("pred and target must have the same batch size")

    target = target.to(device=pred.device, dtype=pred.dtype)
    distances = torch.cdist(pred, target, p=2)
    pred_to_target = distances.min(dim=2).values
    target_to_pred = distances.min(dim=1).values

    chamfer_l1 = 0.5 * (pred_to_target.mean() + target_to_pred.mean())
    chamfer_l2 = 0.5 * (pred_to_target.pow(2).mean() + target_to_pred.pow(2).mean())
    precision = (pred_to_target <= f_score_threshold).to(pred.dtype).mean()
    recall = (target_to_pred <= f_score_threshold).to(pred.dtype).mean()
    f_score = 2.0 * precision * recall / (precision + recall).clamp_min(eps)
    threshold_key = _threshold_key(f_score_threshold)
    return {
        "paper_chamfer_l1": chamfer_l1,
        "paper_chamfer_l2": chamfer_l2,
        "paper_chamfer_l1_x100": chamfer_l1 * 100.0,
        "paper_chamfer_l2_x100": chamfer_l2 * 100.0,
        f"paper_precision_tau_{threshold_key}": precision,
        f"paper_recall_tau_{threshold_key}": recall,
        f"paper_f_score_tau_{threshold_key}": f_score,
    }


class MetricAverager:
    """Accumulate weighted means for metrics that may not appear every update."""

    def __init__(self):
        self._sums = {}
        self._counts = {}

    def update(self, metrics, batch_size=1):
        batch_size = int(batch_size)
        if batch_size <= 0:
            return
        for key, value in metrics.items():
            self._sums[key] = self._sums.get(key, 0.0) + _to_float(value) * batch_size
            self._counts[key] = self._counts.get(key, 0) + batch_size

    def compute(self):
        return {
            key: self._sums[key] / self._counts[key]
            for key in sorted(self._sums)
            if self._counts[key] > 0
        }
