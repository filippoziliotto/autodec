import torch


def test_paper_chamfer_metrics_include_scaled_chamfer_and_fscore():
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

    assert set(metrics) == {
        "paper_chamfer_l1",
        "paper_chamfer_l2",
        "paper_chamfer_l1_x100",
        "paper_chamfer_l2_x100",
        "paper_precision_tau_0_01",
        "paper_recall_tau_0_01",
        "paper_f_score_tau_0_01",
    }
    assert metrics["paper_chamfer_l1"] == torch.tensor(0.5)
    assert metrics["paper_chamfer_l2"] == torch.tensor(0.5)
    assert metrics["paper_chamfer_l1_x100"] == torch.tensor(50.0)
    assert metrics["paper_chamfer_l2_x100"] == torch.tensor(50.0)
    assert metrics["paper_precision_tau_0_01"] == torch.tensor(0.5)
    assert metrics["paper_recall_tau_0_01"] == torch.tensor(0.5)
    assert metrics["paper_f_score_tau_0_01"] == torch.tensor(0.5)


def test_paper_chamfer_metrics_accept_custom_fscore_threshold():
    from autodec.eval.metrics import paper_chamfer_metrics

    pred = torch.tensor([[[0.0, 0.0, 0.0]]])
    target = torch.tensor([[[0.02, 0.0, 0.0]]])

    strict = paper_chamfer_metrics(pred, target, f_score_threshold=0.01)
    loose = paper_chamfer_metrics(pred, target, f_score_threshold=0.03)

    assert strict["paper_f_score_tau_0_01"] == torch.tensor(0.0)
    assert loose["paper_f_score_tau_0_03"] == torch.tensor(1.0)


def test_paper_chamfer_metrics_accept_metric_prefix():
    from autodec.eval.metrics import paper_chamfer_metrics

    pred = torch.tensor([[[0.0, 0.0, 0.0]]])
    target = torch.tensor([[[0.0, 0.0, 0.0]]])

    metrics = paper_chamfer_metrics(pred, target, prefix="paper_sq")

    assert set(metrics) == {
        "paper_sq_chamfer_l1",
        "paper_sq_chamfer_l2",
        "paper_sq_chamfer_l1_x100",
        "paper_sq_chamfer_l2_x100",
        "paper_sq_precision_tau_0_01",
        "paper_sq_recall_tau_0_01",
        "paper_sq_f_score_tau_0_01",
    }


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
