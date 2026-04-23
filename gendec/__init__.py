__all__ = [
    "FlowMatchingLoss",
    "Phase1Evaluator",
    "ScaffoldTokenDataset",
    "SetTransformerFlowModel",
    "postprocess_tokens",
    "render_scaffold_preview",
    "sample_scaffolds",
]


def __getattr__(name):
    if name == "ScaffoldTokenDataset":
        from gendec.data.dataset import ScaffoldTokenDataset

        return ScaffoldTokenDataset
    if name == "Phase1Evaluator":
        from gendec.eval.evaluator import Phase1Evaluator

        return Phase1Evaluator
    if name == "FlowMatchingLoss":
        from gendec.losses.flow_matching import FlowMatchingLoss

        return FlowMatchingLoss
    if name == "SetTransformerFlowModel":
        from gendec.models.set_transformer_flow import SetTransformerFlowModel

        return SetTransformerFlowModel
    if name in {"postprocess_tokens", "render_scaffold_preview", "sample_scaffolds"}:
        from gendec.sampling import postprocess_tokens, render_scaffold_preview, sample_scaffolds

        return {
            "postprocess_tokens": postprocess_tokens,
            "render_scaffold_preview": render_scaffold_preview,
            "sample_scaffolds": sample_scaffolds,
        }[name]
    raise AttributeError(f"module 'gendec' has no attribute {name!r}")
