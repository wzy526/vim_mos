# Copyright (c) 2023, Tri Dao, Albert Gu.
# modified by Ziyi Wang

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj, mos_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

__all__ = ['VIM_MOS']

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act_out(x)
        return x
    
class VIM_MOS(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        d_expert=4,
        bimamba_type="none", # vim
        if_divide_out=False,
        init_layer_scale=None, # vim
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.d_expert = d_expert
        self.num_experts = self.d_state // self.d_expert
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type # vim
        self.if_divide_out = if_divide_out

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True) # vim

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.mlp = MLP(self.d_inner, self.d_inner, self.num_experts)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        if bimamba_type == "v1":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True
        elif bimamba_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        conv_state, conv_state_b, ssm_state, ssm_state_b = None, None, None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        ) # (b, d_inner, seqlen)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        A = A.view(self.d_inner, self.num_experts, self.d_expert) # (d_inner, num_experts, d_expert)

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba_type == "v1":
                A_b = -torch.exp(self.A_b_log.float())
                x, out = mos_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    self.num_experts,
                    A, 
                    A_b,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )

            elif self.bimamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())
                A_b = A_b.view(self.d_inner, self.num_experts, self.d_expert) # (d_inner, num_experts, d_expert)
                x, out_l = mos_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.num_experts,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                _, out_bl = mos_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    self.num_experts,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                # dynamic weighted sum out and out_b
                x_reshaped = rearrange(x, "b d l -> b l d") # 将x转换为(b, l, d_inner)的形状
                
                # 使用MLP生成动态权重
                dynamic_w = self.mlp(x_reshaped)  # (b, l, num_experts)
                dynamic_w = rearrange(dynamic_w, "b l h -> b h l") # 将动态权重转换为(b, num_experts, l)的形状
                dynamic_w = F.softmax(dynamic_w, dim=1) # 确保动态权重在每个头上的和为1

                #  扩展动态权重的维度以匹配y_list中每个y_i的形状
                dynamic_w = dynamic_w.unsqueeze(2)  # (b, num_experts, 1, l) 等价于num_experts个(b, 1, l)

                # 将out_l和out_bl中的每个y与对应的动态权重相乘并求和
                y, yb = torch.zeros_like(out_l[0]), torch.zeros_like(out_bl[0])  
                for i in range(self.num_experts):
                    y += out_l[i] * dynamic_w[:, i, :, :]
                    yb += out_bl[i] * dynamic_w[:, i, :, :]
                y = y.view(batch, self.d_inner, seqlen)
                yb = yb.view(batch, self.d_inner, seqlen)

                if not self.if_divide_out:
                    out = F.linear(rearrange(y + yb.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                else:
                    out = F.linear(rearrange(y + yb.flip([-1]), "b d l -> b l d") / 2, self.out_proj.weight, self.out_proj.bias)

            else: # kept original mamba_simple settings
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            A_b = -torch.exp(self.A_b_log.float())
            A_b = A_b.view(self.d_inner, self.num_experts, self.d_expert) # (d_inner, num_experts, d_expert)
            x, z = xz.chunk(2, dim=1) # x shape (b, d_inner, seqlen)
            xb, zb = xz.flip([-1]).chunk(2, dim=1) # xb shape (b, d_inner, seqlen)

            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if conv_state_b is not None:
                conv_state_b.copy_(F.pad(xb, (self.d_conv - xb.shape[-1], 0)))  # Update state (B D W)

            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
                xb = self.act(self.conv1d_b(xb)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
                xb = causal_conv1d_fn(
                    x=xb, 
                    weight=rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                    bias=self.conv1d_b.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            x_dbl_b = self.x_proj_b(rearrange(xb, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dtb, Bb, Cb = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dtb = self.dt_proj_b.weight @ dtb.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            dtb = rearrange(dtb, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous() # (b, d_state, seqlen)
            Bb = rearrange(Bb, "(b l) dstate -> b dstate l", l=seqlen).contiguous() # (b, d_state, seqlen)

            # 将A, B, C, D以及backward的d_state维度分解成num_head个
            d_expert = self.d_state // self.num_experts
            A = A.view(self.d_inner, self.num_experts, d_expert) # (d_inner, num_experts, d_expert)
            A_i, A_b_i = [], []
            for i in range(self.num_experts):
                A_i.append(A[:, i, :])  # A_i shape (d_inner, d_expert)
                A_b_i.append(A_b[:, i, :])  # A_b_i shape (d_inner, d_expert)

            A_b = -torch.exp(self.A_b_log.float())
            A_b = A_b.view(self.d_inner, self.num_experts, self.d_expert) # (d_inner, num_experts, d_expert)
                
            # 将B的d_state维度分解成num_experts个d_expert维度
            B = B.view(B.shape[0], self.num_experts, d_expert, B.shape[-1])  # (b, num_experts, d_expert, seqlen)
            Bb = Bb.view(Bb.shape[0], self.num_experts, d_expert, Bb.shape[-1])  # (b, num_experts, d_expert, seqlen)
            
            # num_experts个B_i
            B_i, B_b_i = [], []
            for i in range(self.num_experts):
                B_i.append(B[:, i, :, :])  # 每个B_i的形状为 (b, d_expert, seqlen), B_i = num_experts个(b, d_expert, seqlen)
                B_b_i.append(Bb[:, i, :, :])  # 每个B_b_i的形状为 (b, d_expert, seqlen), B_b_i = num_experts个(b, d_expert, seqlen)

            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous() # (b, d_state, seqlen)
            C = C.view(C.shape[0], self.num_experts, d_expert, C.shape[-1])  # (b, num_experts, d_expert, seqlen)
            Cb = rearrange(Cb, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            Cb = Cb.view(Cb.shape[0], self.num_experts, d_expert, Cb.shape[-1])  # (b, num_experts, d_expert, seqlen)

            # num_experts个C_i
            C_i, C_b_i = [], []
            for i in range(self.num_experts):
                C_i.append(C[:, i, :, :])  # C_i shape (b, d_expert, seqlen)
                C_b_i.append(Cb[:, i, :, :])  # C_b_i shape (b, d_expert, seqlen)

            assert self.activation in ["silu", "swish"]

            y_list, yb_list = [], []
            # iterate num_experts times to do selective scan
            for i in range(self.num_experts):
                y_i = selective_scan_fn(
                    x,
                    dt,
                    A_i[i],
                    B_i[i],
                    C_i[i],
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                )

                y_bi = selective_scan_fn(
                    xb,
                    dtb,
                    A_b_i[i],
                    B_b_i[i],
                    C_b_i[i],
                    self.D_b.float(),
                    z=zb,
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state_b is not None,
                )
                
                if ssm_state is not None and ssm_state_b is not None:
                    y_i, last_state_i = y_i
                    y_bi, last_state_bi = y_bi
                    ssm_state[:, i * d_expert:(i + 1) * d_expert].copy_(last_state_i)
                    ssm_state_b[:, i * d_expert:(i + 1) * d_expert].copy_(last_state_bi)
                
                y_list.append(y_i) # num_head个shape (b, d_inner, seqlen)
                yb_list.append(y_bi) # num_head个shape (b, d_inner, seqlen)
            
            # dynamic weights 
            # 将x转换为(b, l, d_inner)的形状
            x_reshaped = rearrange(x, "b d l -> b l d")
            
            # 使用MLP生成动态权重
            dynamic_w = self.mlp(x_reshaped)  # (b, l, num_experts)
            
            # 将动态权重转换为(b, num_experts, l)的形状
            dynamic_w = rearrange(dynamic_w, "b l h -> b h l")
            
            # 确保动态权重在每个头上的和为1
            dynamic_w = F.softmax(dynamic_w, dim=1)
            
            # 扩展动态权重的维度以匹配y_list中每个y_i的形状
            dynamic_w = dynamic_w.unsqueeze(2)  # (b, num_experts, 1, l) 等价于num_experts个(b, 1, l)

            # 将y_list中的每个y与对应的动态权重相乘并求和
            y, yb = torch.zeros_like(y_list[0]), torch.zeros_like(y_list[0])  
            for i in range(self.num_experts):
                y += y_list[i] * dynamic_w[:, i, :, :]
                yb += yb_list[i] * dynamic_w[:, i, :, :]
            
            y = y.view(batch, self.d_inner, seqlen) # (b, d_inner, seqlen)
            yb = yb.view(batch, self.d_inner, seqlen) # (b, d_inner, seqlen)
            y = rearrange(y, "b d l -> b l d") # (b, seqlen, d_inner)
            yb = rearrange(yb, "b d l -> b l d") # (b, seqlen, d_inner)

            out = y + yb.flip([-1]) # (b, seqlen, d_inner)

            if not self.if_divide_out:
                out = F.linear(out, self.out_proj.weight, self.out_proj.bias)
            else:
                out = F.linear(out / 2, self.out_proj.weight, self.out_proj.bias)
            # out = self.out_proj(y) # (b, seqlen, d_model)
            # out_b = self.out_proj(yb) # (b, seqlen, d_model)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
    
    def get_B_from_mamba_inner_ref(self, 
        xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
        out_proj_weight, out_proj_bias,
        A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
        C_proj_bias=None, delta_softplus=True
    ):
        assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
        delta = rearrange(delta, "d (b l) -> b d l", l=L)
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
        # if C is None:  # variable B
        #     C = x_dbl[:, -d_state:]  # (bl d)
        #     if C_proj_bias is not None:
        #         C = C + C_proj_bias.to(dtype=C.dtype)
        #     if not A.is_complex():
        #         C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        #     else:
        #         C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
        # y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
        return B

    

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)