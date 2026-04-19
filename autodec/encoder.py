import warnings

import torch
import torch.nn as nn

from autodec.models.heads import SuperDecHead as AutoDecHead
from autodec.models.residual import PartResidualProjector
from superdec.models.decoder import TransformerDecoder
from superdec.models.decoder_layer import DecoderLayer
from superdec.models.heads_mlp import SuperDecHead as SuperDecHeadMLP
from superdec.models.heads_mlps import SuperDecHead as SuperDecHeadMLPS


class AutoDecEncoder(nn.Module):
    """SuperDec-compatible encoder that also emits per-part residual tokens."""

    def __init__(self, ctx, point_encoder=None, layers=None, heads=None):
        super().__init__()
        self.n_layers = ctx.decoder.n_layers
        self.n_heads = ctx.decoder.n_heads
        self.n_queries = ctx.decoder.n_queries
        self.deep_supervision = ctx.decoder.deep_supervision
        self.pos_encoding_type = ctx.decoder.pos_encoding_type
        self.dim_feedforward = ctx.decoder.dim_feedforward
        self.emb_dims = ctx.point_encoder.l3.out_channels
        self.residual_dim = getattr(ctx, "residual_dim", 64)
        self.lm_optimization = False

        if point_encoder is None:
            from superdec.models.point_encoder import StackedPVConv

            point_encoder = StackedPVConv(ctx.point_encoder)
        self.point_encoder = point_encoder
        self.layers = layers or self._build_layers(ctx)
        self.clear_orientation_heads = getattr(ctx, "clear_orientation_heads", False)
        self.heads = heads or self._build_heads(ctx)
        self.residual_projector = PartResidualProjector(
            feature_dim=self.emb_dims,
            residual_dim=self.residual_dim,
        )

        init_queries = torch.zeros(self.n_queries + 1, self.emb_dims)
        self.register_buffer("init_queries", init_queries)

    def enable_lm_optimization(self, lm_optimizer=None):
        """Enable SuperDec LM refinement before AutoDec residual decoding."""

        if lm_optimizer is None:
            from superdec.lm_optimization.lm_optimizer import LMOptimizer

            lm_optimizer = LMOptimizer()
        self.lm_optimizer = lm_optimizer
        self.lm_optimization = True
        return self

    def disable_lm_optimization(self):
        self.lm_optimization = False
        if hasattr(self, "lm_optimizer"):
            delattr(self, "lm_optimizer")
        return self

    def _build_layers(self, ctx):
        decoder_layer = DecoderLayer(
            d_model=self.emb_dims,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
            swapped_attention=ctx.decoder.swapped_attention,
        )
        layers = TransformerDecoder(
            decoder_layer=decoder_layer,
            n_layers=self.n_layers,
            max_len=self.n_queries,
            pos_encoding_type=self.pos_encoding_type,
            masked_attention=ctx.decoder.masked_attention,
        )
        layers.project_queries = nn.Sequential(
            nn.Linear(self.emb_dims, self.emb_dims),
            nn.ReLU(),
            nn.Linear(self.emb_dims, self.emb_dims),
        )
        return layers

    def _build_heads(self, ctx):
        head_type = getattr(ctx, "head_type", "heads")
        if head_type == "heads":
            return AutoDecHead(emb_dims=self.emb_dims, ctx=ctx)
        if head_type == "heads_mlp":
            return SuperDecHeadMLP(emb_dims=self.emb_dims, ctx=ctx)
        if head_type == "heads_mlps":
            return SuperDecHeadMLPS(emb_dims=self.emb_dims, ctx=ctx)
        raise ValueError(f"Unknown head_type: {head_type}")

    def load_state_dict(self, state_dict, strict=True):
        state_dict = dict(state_dict)
        allowed_prefixes = [
            "heads.tapering_head",
            "heads.bending_k_head",
            "heads.bending_a_head",
            "residual_projector",
        ]
        if self.clear_orientation_heads:
            for key in (
                "heads.scale_head.weight",
                "heads.scale_head.bias",
                "heads.shape_head.weight",
                "heads.shape_head.bias",
                "heads.rot_head.weight",
                "heads.rot_head.bias",
            ):
                state_dict.pop(key, None)
            allowed_prefixes.extend(["heads.scale_head", "heads.shape_head", "heads.rot_head"])
            warnings.warn("Clearing all orientation dependent heads.")

        if (
            getattr(self.heads, "rotation6d", False)
            and "heads.rot_head.weight" in state_dict
            and state_dict["heads.rot_head.weight"].shape[0] == 4
        ):
            state_dict.pop("heads.rot_head.weight", None)
            state_dict.pop("heads.rot_head.bias", None)
            allowed_prefixes.append("heads.rot_head")
            warnings.warn(
                "Loaded a checkpoint with 4D rotation head into a model with "
                "6D rotation head. The rotation head weights were ignored."
            )

        result = super().load_state_dict(state_dict, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        filtered_missing = [
            key for key in missing if not any(key.startswith(prefix) for prefix in allowed_prefixes)
        ]
        if strict and (filtered_missing or unexpected):
            message = ""
            if filtered_missing:
                message += f"Missing key(s) in state_dict: {filtered_missing}. \n"
            if unexpected:
                message += f"Unexpected key(s) in state_dict: {unexpected}. \n"
            raise RuntimeError(message)
        if not strict and (missing or unexpected):
            warnings.warn(
                f"load_state_dict warnings -- missing: {missing}, unexpected: {unexpected}"
            )
        return result

    def _ensure_exist_logit(self, outdict):
        if "exist_logit" not in outdict:
            exist = outdict["exist"].clamp(1e-6, 1 - 1e-6)
            outdict["exist_logit"] = torch.logit(exist)
        return outdict

    def forward(self, x, return_features=True):
        point_features = self.point_encoder(x)
        refined_queries_list, assign_matrices = self.layers(self.init_queries, point_features)

        outdict_list = []
        for query_features, assign_logits in zip(refined_queries_list, assign_matrices):
            sq_features = query_features[:, :-1, ...]
            outdict = self.heads(sq_features)
            outdict = self._ensure_exist_logit(outdict)
            outdict["assign_matrix"] = torch.softmax(assign_logits, dim=2)
            outdict_list.append(outdict)

        outdict = outdict_list[-1]
        sq_features = refined_queries_list[-1][:, :-1, ...]

        if self.lm_optimization:
            outdict = self.lm_optimizer(outdict, x)

        residual, pooled = self.residual_projector(
            sq_features,
            point_features,
            outdict["assign_matrix"],
            return_pooled=True,
        )
        outdict["residual"] = residual
        outdict["pooled_features"] = pooled

        if return_features:
            outdict["point_features"] = point_features
            outdict["sq_features"] = sq_features

        return outdict
