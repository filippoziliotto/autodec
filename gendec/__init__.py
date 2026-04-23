__all__ = [
    "FlowMatchingLoss",
    "JointFlowMatchingLoss",
    "JointSetTransformerFlowModel",
    "JointTokenDataset",
    "Phase1Evaluator",
    "Phase2Evaluator",
    "ScaffoldTokenDataset",
    "SetTransformerFlowModel",
    "postprocess_joint_tokens",
    "postprocess_tokens",
    "render_scaffold_preview",
    "sample_joint_scaffolds",
    "sample_scaffolds",
]


def __getattr__(name):
    if name == "ScaffoldTokenDataset":
        from gendec.data.dataset import ScaffoldTokenDataset

        return ScaffoldTokenDataset
    if name == "JointTokenDataset":
        from gendec.data.dataset import JointTokenDataset

        return JointTokenDataset
    if name == "Phase1Evaluator":
        from gendec.eval.evaluator import Phase1Evaluator

        return Phase1Evaluator
    if name == "Phase2Evaluator":
        from gendec.eval.evaluator import Phase2Evaluator

        return Phase2Evaluator
    if name == "FlowMatchingLoss":
        from gendec.losses.flow_matching import FlowMatchingLoss

        return FlowMatchingLoss
    if name == "JointFlowMatchingLoss":
        from gendec.losses.flow_matching import JointFlowMatchingLoss

        return JointFlowMatchingLoss
    if name == "SetTransformerFlowModel":
        from gendec.models.set_transformer_flow import SetTransformerFlowModel

        return SetTransformerFlowModel
    if name == "JointSetTransformerFlowModel":
        from gendec.models.set_transformer_flow import JointSetTransformerFlowModel

        return JointSetTransformerFlowModel
    if name in {
        "postprocess_joint_tokens",
        "postprocess_tokens",
        "render_scaffold_preview",
        "sample_joint_scaffolds",
        "sample_scaffolds",
    }:
        from gendec.sampling import (
            postprocess_joint_tokens,
            postprocess_tokens,
            render_scaffold_preview,
            sample_joint_scaffolds,
            sample_scaffolds,
        )

        return {
            "postprocess_joint_tokens": postprocess_joint_tokens,
            "postprocess_tokens": postprocess_tokens,
            "render_scaffold_preview": render_scaffold_preview,
            "sample_joint_scaffolds": sample_joint_scaffolds,
            "sample_scaffolds": sample_scaffolds,
        }[name]
    raise AttributeError(f"module 'gendec' has no attribute {name!r}")
