import torch
import torch.nn as nn
import warnings

from superdec.models.decoder import TransformerDecoder
from superdec.models.decoder_layer import DecoderLayer
from superdec.models.point_encoder import StackedPVConv
from superdec.models.heads import SuperDecHead
from superdec.lm_optimization.lm_optimizer import LMOptimizer

class SuperDec(nn.Module):
    def __init__(self, ctx):
        super(SuperDec, self).__init__()
        self.n_layers = ctx.decoder.n_layers
        self.n_heads = ctx.decoder.n_heads
        self.n_queries = ctx.decoder.n_queries
        self.deep_supervision = ctx.decoder.deep_supervision
        self.pos_encoding_type = ctx.decoder.pos_encoding_type
        self.dim_feedforward = ctx.decoder.dim_feedforward
        self.emb_dims = ctx.point_encoder.l3.out_channels # output dimension of pvcnn
        self.lm_optimization = False
        if self.lm_optimization:
            self.lm_optimizer = LMOptimizer()

        self.point_encoder = StackedPVConv(ctx.point_encoder)

        decoder_layer = DecoderLayer(d_model=self.emb_dims, nhead=self.n_heads, dim_feedforward=self.dim_feedforward, 
                                               batch_first=True, swapped_attention=ctx.decoder.swapped_attention)
        self.layers = TransformerDecoder(decoder_layer=decoder_layer, n_layers=self.n_layers, 
                                         max_len=self.n_queries, pos_encoding_type=self.pos_encoding_type, 
                                         masked_attention=ctx.decoder.masked_attention)
        
        self.layers.project_queries = nn.Sequential(
            nn.Linear(self.emb_dims, self.emb_dims),
            nn.ReLU(),
            nn.Linear(self.emb_dims, self.emb_dims),
        )
        self.heads = SuperDecHead(emb_dims=self.emb_dims, ctx=ctx)
        init_queries = torch.zeros(self.n_queries + 1, self.emb_dims)
        self.register_buffer('init_queries', init_queries) # TODO double check -> new codebase

    def load_state_dict(self, state_dict, strict=True):
        """Wrapper around nn.Module.load_state_dict that allows older checkpoints
        missing newly added head parameters (like tapering/bending) while still
        optionally enforcing strict loading for other keys.
        """
        allowed_prefixes = ["heads.tapering_head", "heads.bending_k_head", "heads.bending_a_head"]
        
        # Handle rot_head shape mismatch (4 -> 6)
        if self.heads.rotation6d and 'heads.rot_head.weight' in state_dict and state_dict['heads.rot_head.weight'].shape[0] == 4:
            del state_dict['heads.rot_head.weight']
            del state_dict['heads.rot_head.bias']
            allowed_prefixes.append("heads.rot_head")
            
            warnings.warn(
                "Loaded a checkpoint with 4D rotation head into a model with 6D rotation head. "
                "The rotation head weights have been deleted from the state dict. "
                "You will need to fine-tune the model to learn the 6D rotation representation."
            )

        res = super(SuperDec, self).load_state_dict(state_dict, strict=False)
        missing = list(res.missing_keys)
        unexpected = list(res.unexpected_keys)

        # Allow missing keys that belong to newly added heads
        filtered_missing = [k for k in missing if not any(k.startswith(p) for p in allowed_prefixes)]
        if strict:
            if filtered_missing or unexpected:
                msg = ''
                if filtered_missing:
                    msg += f"Missing key(s) in state_dict: {filtered_missing}. \n"
                if unexpected:
                    msg += f"Unexpected key(s) in state_dict: {unexpected}. \n"
                raise RuntimeError(msg)
        else:
            if missing or unexpected:
                warnings.warn(f"load_state_dict warnings -- missing: {missing}, unexpected: {unexpected}")
    
    def forward(self, x):
        point_features = self.point_encoder(x)

        refined_queries_list, assign_matrices = self.layers(self.init_queries, point_features)
        outdict_list = []

        # TODO remove this in the final version. there is no need to compute the output for all of them   
        thred = 24
        for i, q in enumerate(refined_queries_list): 
            outdict_list += [self.heads(q[:,:-1,...])]
            assign_matrix = assign_matrices[i]
            assign_matrix = torch.softmax(assign_matrix, dim=2)
            outdict_list[i]['assign_matrix'] = assign_matrix 
            # outdict_list[i]['exist'] = (assign_matrix.sum(1) > thred).to(torch.float32).detach()[...,None]

        if self.lm_optimization:
            outdict_list[-1] = self.lm_optimizer(outdict_list[-1], x)
            
        return outdict_list[-1]
