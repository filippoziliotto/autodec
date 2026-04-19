import math

import torch
import torch.nn as nn

from autodec.models.offset_decoder import build_offset_decoder
from autodec.sampling.sq_surface import SQSurfaceSampler
from autodec.utils.packing import pack_decoder_primitive_features, repeat_by_part_ids


class AutoDecDecoder(nn.Module):
    """Decode AutoDec encoder outputs into a dense point cloud."""

    def __init__(
        self,
        residual_dim=64,
        primitive_dim=18,
        n_surface_samples=256,
        hidden_dim=128,
        n_heads=4,
        exist_tau=1.0,
        angle_sampler=None,
        offset_scale=None,
        positional_frequencies=6,
        component_feature_dim=None,
        n_blocks=2,
        self_attention_mode="within_primitive",
    ):
        super().__init__()
        self.residual_dim = residual_dim
        self.primitive_dim = primitive_dim
        self.n_surface_samples = n_surface_samples
        self.positional_frequencies = int(positional_frequencies)
        if self.positional_frequencies < 0:
            raise ValueError("positional_frequencies must be non-negative")
        self.position_feature_dim = 3 + 6 * self.positional_frequencies
        if component_feature_dim is None:
            component_feature_dim = max(4, hidden_dim // 4)
        self.component_feature_dim = int(component_feature_dim)
        if self.component_feature_dim < 0:
            raise ValueError("component_feature_dim must be non-negative")
        self.use_component_projection = self.component_feature_dim > 0
        if self.use_component_projection:
            self.point_feature_dim = self.component_feature_dim * 4
            self.primitive_token_dim = self.component_feature_dim * 2
            self.position_projector = self._component_projector(
                self.position_feature_dim,
                self.component_feature_dim,
            )
            self.primitive_feature_projector = self._component_projector(
                primitive_dim,
                self.component_feature_dim,
            )
            self.residual_projector = self._component_projector(
                residual_dim,
                self.component_feature_dim,
            )
            self.gate_projector = nn.Sequential(
                nn.Linear(1, self.component_feature_dim),
                nn.ReLU(),
            )
        else:
            self.point_feature_dim = self.position_feature_dim + primitive_dim + residual_dim + 1
            self.primitive_token_dim = primitive_dim + residual_dim
        self.surface_sampler = SQSurfaceSampler(
            n_samples=n_surface_samples,
            tau=exist_tau,
            angle_sampler=angle_sampler,
        )
        self.offset_decoder = build_offset_decoder(
            decoder_type="cross_attention",
            point_in_dim=self.point_feature_dim,
            primitive_in_dim=self.primitive_token_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            offset_scale=offset_scale,
            n_blocks=n_blocks,
            self_attention_mode=self_attention_mode,
        )

    @staticmethod
    def _component_projector(in_dim, out_dim):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def surface_position_features(self, points):
        if self.positional_frequencies == 0:
            return points
        frequencies = 2 ** torch.arange(
            self.positional_frequencies,
            device=points.device,
            dtype=points.dtype,
        )
        angles = points.unsqueeze(2) * frequencies.view(1, 1, -1, 1) * math.pi
        encoded = torch.cat([angles.sin(), angles.cos()], dim=-1)
        encoded = encoded.reshape(points.shape[0], points.shape[1], -1)
        return torch.cat([points, encoded], dim=-1)

    def _decoder_feature_inputs(
        self,
        position_features,
        primitive_features,
        residual,
        gates,
        part_ids,
    ):
        if not self.use_component_projection:
            point_primitive_features = repeat_by_part_ids(primitive_features, part_ids)
            point_residual = repeat_by_part_ids(residual, part_ids)
            decoder_features = torch.cat(
                [position_features, point_primitive_features, point_residual, gates],
                dim=-1,
            )
            primitive_tokens = torch.cat([primitive_features, residual], dim=-1)
            return decoder_features, primitive_tokens, {}

        projected_position = self.position_projector(position_features)
        projected_primitives = self.primitive_feature_projector(primitive_features)
        projected_residual = self.residual_projector(residual)
        projected_gates = self.gate_projector(gates)
        point_primitive_features = repeat_by_part_ids(projected_primitives, part_ids)
        point_residual = repeat_by_part_ids(projected_residual, part_ids)
        decoder_features = torch.cat(
            [projected_position, point_primitive_features, point_residual, projected_gates],
            dim=-1,
        )
        primitive_tokens = torch.cat([projected_primitives, projected_residual], dim=-1)
        components = {
            "projected_position_features": projected_position,
            "projected_E_dec": projected_primitives,
            "projected_residual": projected_residual,
            "projected_gates": projected_gates,
        }
        return decoder_features, primitive_tokens, components

    def forward(self, outdict, return_attention=False):
        sample = self.surface_sampler(outdict)
        primitive_features = pack_decoder_primitive_features(outdict)
        residual = outdict["residual"]
        position_features = self.surface_position_features(sample.flat_points)

        gates = sample.weights.unsqueeze(-1)
        decoder_features, primitive_tokens, component_features = self._decoder_feature_inputs(
            position_features,
            primitive_features,
            residual,
            gates,
            sample.part_ids,
        )

        if return_attention:
            offsets, attention = self.offset_decoder(
                decoder_features,
                primitive_tokens,
                return_attention=True,
            )
        else:
            offsets = self.offset_decoder(decoder_features, primitive_tokens)
            attention = None

        decoded_points = sample.flat_points + gates * offsets
        result = dict(outdict)
        result.update(
            {
                "surface_points": sample.flat_points,
                "surface_points_by_part": sample.surface_points,
                "canonical_surface_points": sample.canonical_points,
                "decoded_weights": sample.weights,
                "part_ids": sample.part_ids,
                "E_dec": primitive_features,
                "surface_position_features": position_features,
                "decoder_features": decoder_features,
                "primitive_tokens": primitive_tokens,
                "decoded_offsets": offsets,
                "decoded_points": decoded_points,
            }
        )
        result.update(component_features)
        if attention is not None:
            result["decoder_attention"] = attention
        return result
