import math
from typing import Optional

import torch
import torch.nn as nn


NON_VAR_SENTINEL = -999999


class AddressPositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_model,
        intermediate_size=64,
        dropout=0.1,
        code_role_id=1,
        initial_position_scale=10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_per_level = intermediate_size
        self.code_role_id = code_role_id
        concat_size = 2 * intermediate_size

        self.address_projection = nn.Sequential(
            nn.Linear(concat_size, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.daddr_projection = nn.Sequential(
            nn.Linear(concat_size, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

        self.address_level_weights = nn.Parameter(torch.ones(2))
        self.daddr_level_weights = nn.Parameter(torch.ones(2))
        initial_log_scale = math.log(float(initial_position_scale))
        self.binary_log_scale = nn.Parameter(torch.tensor(initial_log_scale, dtype=torch.float32))
        self.function_log_scale = nn.Parameter(torch.tensor(initial_log_scale, dtype=torch.float32))
        self.bb_log_scale = nn.Parameter(torch.tensor(initial_log_scale, dtype=torch.float32))

    def _sinusoidal_encoding(self, positions, d_model, log_scale):
        batch_size, seq_len = positions.shape
        positions_f32 = positions.float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(10000.0) / d_model)
        )
        scale = torch.exp(log_scale.float()).clamp(min=1.0, max=1e6).to(device=positions.device)
        positions_scaled = positions_f32.unsqueeze(-1) * scale
        encoding = torch.zeros(batch_size, seq_len, d_model, device=positions.device, dtype=torch.float32)
        encoding[:, :, 0::2] = torch.sin(positions_scaled * div_term)
        if d_model % 2 == 0:
            encoding[:, :, 1::2] = torch.cos(positions_scaled * div_term)
        else:
            encoding[:, :, 1::2] = torch.cos(positions_scaled * div_term[:-1])
        return encoding

    def forward(
        self,
        aa_role_ids,
        binary_pos,
        function_pos,
        bb_pos,
        address_role_id,
        daddr_role_id,
        opcode_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len = aa_role_ids.shape
        device = aa_role_ids.device
        output_dtype = self.address_projection[0].weight.dtype

        if opcode_mask is None:
            opcode_mask = torch.zeros_like(aa_role_ids, dtype=torch.bool)
        else:
            opcode_mask = opcode_mask.bool()

        valid_general_positions = (function_pos >= 0) | (bb_pos >= 0)
        explicit_address_mask = aa_role_ids == address_role_id
        opcode_route_mask = opcode_mask & (aa_role_ids == self.code_role_id)
        general_position_mask = valid_general_positions & (explicit_address_mask | opcode_route_mask)
        address_mask = general_position_mask.unsqueeze(-1)
        daddr_mask = (aa_role_ids == daddr_role_id).unsqueeze(-1)

        function_pos = torch.clamp(function_pos, min=0.0)
        bb_pos = torch.clamp(bb_pos, min=0.0)
        binary_pos = torch.clamp(binary_pos, min=0.0)

        output = torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=output_dtype)

        func_enc = self._sinusoidal_encoding(function_pos, self.d_per_level, self.function_log_scale)
        bb_enc = self._sinusoidal_encoding(bb_pos, self.d_per_level, self.bb_log_scale)
        address_weights = torch.softmax(self.address_level_weights, dim=0)
        address_concat = torch.cat([func_enc * address_weights[0], bb_enc * address_weights[1]], dim=-1)
        address_concat = address_concat.to(dtype=output_dtype)
        output = output + self.address_projection(address_concat) * address_mask.to(dtype=output_dtype)

        bin_enc = self._sinusoidal_encoding(binary_pos, self.d_per_level, self.binary_log_scale)
        func_enc_2 = self._sinusoidal_encoding(function_pos, self.d_per_level, self.function_log_scale)
        daddr_weights = torch.softmax(self.daddr_level_weights, dim=0)
        daddr_concat = torch.cat([bin_enc * daddr_weights[0], func_enc_2 * daddr_weights[1]], dim=-1)
        daddr_concat = daddr_concat.to(dtype=output_dtype)
        output = output + self.daddr_projection(daddr_concat) * daddr_mask.to(dtype=output_dtype)

        return output


class VarPositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, sentinel=NON_VAR_SENTINEL):
        super().__init__()
        self.sentinel = sentinel
        self.offset_embedding = nn.Embedding(1027, d_model, padding_idx=0)
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, var_offsets):
        var_mask = (var_offsets != self.sentinel).unsqueeze(-1)
        token_ids = torch.zeros_like(var_offsets, dtype=torch.long)

        in_range = (var_offsets >= -512) & (var_offsets <= 512)
        token_ids[in_range] = var_offsets[in_range] + 512
        token_ids[var_offsets > 512] = 1025
        token_ids[var_offsets < -512] = 1026
        token_ids[var_offsets == self.sentinel] = 0

        embedding_part = self.offset_embedding(token_ids)
        log_distance = torch.log1p(torch.abs(var_offsets.float())).unsqueeze(-1).to(dtype=embedding_part.dtype)
        distance_part = self.distance_mlp(log_distance)
        return (embedding_part + distance_part) * var_mask.to(dtype=embedding_part.dtype)


class AddressAwareEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim,
        code_role_id=1,
        address_role_id=2,
        daddr_role_id=3,
        var_role_id=4,
        intermediate_size=64,
        dropout=0.1,
    ):
        super().__init__()
        self.code_role_id = code_role_id
        self.address_role_id = address_role_id
        self.daddr_role_id = daddr_role_id
        self.var_role_id = var_role_id
        self.address_position = AddressPositionalEmbedding(
            d_model=embed_dim,
            intermediate_size=intermediate_size,
            dropout=dropout,
            code_role_id=code_role_id,
        )
        self.var_position = VarPositionalEmbedding(embed_dim, dropout=dropout)

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name else ""
        current_state = self.state_dict()
        for key, value in current_state.items():
            full_key = prefix + key
            if full_key not in state_dict:
                state_dict[full_key] = value.detach().clone()

    def forward(
        self,
        aa_role_ids,
        binary_pos,
        function_pos,
        bb_pos,
        var_offset,
        opcode_mask: Optional[torch.Tensor] = None,
    ):
        address_part = self.address_position(
            aa_role_ids=aa_role_ids,
            binary_pos=binary_pos,
            function_pos=function_pos,
            bb_pos=bb_pos,
            address_role_id=self.address_role_id,
            daddr_role_id=self.daddr_role_id,
            opcode_mask=opcode_mask,
        )
        var_part = self.var_position(var_offset)
        return address_part + var_part
