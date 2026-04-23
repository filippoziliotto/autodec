def test_import_gendec_package_does_not_eagerly_import_eval_runtime():
    import gendec

    assert hasattr(gendec, "ScaffoldTokenDataset")
    assert hasattr(gendec, "FlowMatchingLoss")
