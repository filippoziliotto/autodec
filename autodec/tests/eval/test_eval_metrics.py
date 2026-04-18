import torch


def test_paper_chamfer_metrics_are_symmetric_and_named():
    from autodec.eval.metrics import paper_chamfer_metrics

    pred = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ]
    )
    target = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ]
    )

    metrics = paper_chamfer_metrics(pred, target)

    assert set(metrics) == {"paper_chamfer_l1", "paper_chamfer_l2"}
    assert metrics["paper_chamfer_l1"] == torch.tensor(0.5)
    assert metrics["paper_chamfer_l2"] == torch.tensor(0.5)


def test_metric_averager_keeps_stable_float_keys():
    from autodec.eval.metrics import MetricAverager

    averager = MetricAverager()
    averager.update({"recon": 1.0, "active_primitive_count": 2.0}, batch_size=2)
    averager.update({"recon": 3.0, "offset_ratio": 4.0}, batch_size=1)

    result = averager.compute()

    assert result == {
        "active_primitive_count": 2.0,
        "offset_ratio": 4.0,
        "recon": 5.0 / 3.0,
    }

