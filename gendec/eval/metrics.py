import torch


def _to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _threshold_key(value):
    return f"{float(value):.6g}".replace(".", "_")


def _subsample_points(points, point_count):
    if point_count is None:
        return points
    point_count = int(point_count)
    if point_count <= 0:
        raise ValueError("point_count must be positive when provided")
    total = points.shape[1]
    if total == point_count:
        return points
    if total > point_count:
        indices = torch.linspace(0, total - 1, steps=point_count, device=points.device)
        indices = indices.round().to(torch.long)
        return points.index_select(1, indices)
    indices = torch.arange(point_count, device=points.device) % total
    return points.index_select(1, indices)


def _paper_metrics_single(pred, target, f_score_threshold=0.01, eps=1e-8):
    distances = torch.cdist(pred.unsqueeze(0), target.unsqueeze(0), p=2)[0]
    pred_to_target = distances.min(dim=1).values
    target_to_pred = distances.min(dim=0).values

    chamfer_l1 = 0.5 * (pred_to_target.mean() + target_to_pred.mean())
    chamfer_l2 = 0.5 * (pred_to_target.pow(2).mean() + target_to_pred.pow(2).mean())
    precision = (pred_to_target <= f_score_threshold).to(pred.dtype).mean()
    recall = (target_to_pred <= f_score_threshold).to(pred.dtype).mean()
    f_score = 2.0 * precision * recall / (precision + recall).clamp_min(eps)
    return {
        "chamfer_l1": chamfer_l1,
        "chamfer_l2": chamfer_l2,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
    }


class MetricAverager:
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


def active_primitive_count(exist, threshold=0.5):
    return (exist.squeeze(-1) > threshold).to(torch.float32).sum(dim=-1).mean()


def active_decoded_point_count(weights, threshold=0.5):
    if weights.ndim == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    return (weights > threshold).to(torch.float32).sum(dim=-1).mean()


def token_channel_mean_abs(tokens):
    return tokens.abs().mean()


def nearest_neighbor_paper_metrics(
    pred,
    reference,
    prefix,
    point_count=None,
    f_score_threshold=0.01,
):
    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("pred must have shape [B, M, 3]")
    if reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError("reference must have shape [R, N, 3]")
    if reference.shape[0] == 0:
        raise ValueError("reference must contain at least one point cloud")

    pred = _subsample_points(pred, point_count)
    reference = _subsample_points(reference.to(device=pred.device, dtype=pred.dtype), point_count)

    selected = []
    for pred_idx in range(pred.shape[0]):
        best = None
        best_l2 = None
        for ref_idx in range(reference.shape[0]):
            metrics = _paper_metrics_single(
                pred[pred_idx],
                reference[ref_idx],
                f_score_threshold=f_score_threshold,
            )
            chamfer_l2 = metrics["chamfer_l2"]
            if best is None or chamfer_l2 < best_l2:
                best = metrics
                best_l2 = chamfer_l2
        selected.append(best)

    threshold_key = _threshold_key(f_score_threshold)
    return {
        f"{prefix}_nn_chamfer_l1": _to_float(torch.stack([item["chamfer_l1"] for item in selected]).mean()),
        f"{prefix}_nn_chamfer_l2": _to_float(torch.stack([item["chamfer_l2"] for item in selected]).mean()),
        f"{prefix}_nn_chamfer_l1_x100": _to_float(torch.stack([item["chamfer_l1"] for item in selected]).mean() * 100.0),
        f"{prefix}_nn_chamfer_l2_x100": _to_float(torch.stack([item["chamfer_l2"] for item in selected]).mean() * 100.0),
        f"{prefix}_nn_precision_tau_{threshold_key}": _to_float(
            torch.stack([item["precision"] for item in selected]).mean()
        ),
        f"{prefix}_nn_recall_tau_{threshold_key}": _to_float(
            torch.stack([item["recall"] for item in selected]).mean()
        ),
        f"{prefix}_nn_f_score_tau_{threshold_key}": _to_float(
            torch.stack([item["f_score"] for item in selected]).mean()
        ),
    }
