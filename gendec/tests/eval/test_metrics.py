def test_metric_averager_computes_weighted_means():
    from gendec.eval.metrics import MetricAverager

    averager = MetricAverager()
    averager.update({"loss": 2.0, "active": 1.0}, batch_size=2)
    averager.update({"loss": 4.0, "active": 3.0}, batch_size=1)

    metrics = averager.compute()

    assert metrics["loss"] == (2.0 * 2 + 4.0) / 3
    assert metrics["active"] == (1.0 * 2 + 3.0) / 3


def test_nearest_neighbor_paper_metrics_selects_best_reference():
    import torch

    from gendec.eval.metrics import nearest_neighbor_paper_metrics

    pred = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    reference = torch.tensor(
        [
            [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )

    metrics = nearest_neighbor_paper_metrics(
        pred,
        reference,
        prefix="coarse_decoded",
        point_count=2,
    )

    assert metrics["coarse_decoded_nn_chamfer_l1"] == 0.0
    assert metrics["coarse_decoded_nn_chamfer_l2"] == 0.0
