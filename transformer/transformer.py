import torch
from einops import rearrange, einsum
import math

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        sig = math.sqrt(2 / (in_features + out_features))
        init_val = torch.nn.init.trunc_normal_(
            torch.empty(out_features, in_features, device=device, dtype=dtype),
            mean=0., std=sig, a=-3*sig, b=3*sig
        )
        self.W = torch.nn.Parameter(data=init_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... dim_in, dim_out dim_in -> ... dim_out") 
    

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        init_val = torch.nn.init.trunc_normal_(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
            mean=0., std=1.0, a=-3, b=3
        )
        self.embs = torch.nn.Parameter(data=init_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embs[x]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.gain = torch.nn.Parameter(data=torch.ones(size=(d_model,), device=device, dtype=dtype))
        self.eps = eps
        self.dtype = dtype
        self.d_model_float = float(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is ... d_model
        x = x.to(torch.float32)
        normalizer = 1./torch.sqrt(
            (einsum(x, x, "... d_model, ... d_model -> ...") + self.eps) / self.d_model_float
        )
        x = einsum(x, normalizer, "... d_model, ... -> ... d_model")
        x = einsum(x, self.gain, "... d_model, d_model -> ... d_model")
        x = x.to(self.dtype)
        return x


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_inp = self.w1(x)
        silu = gate_inp * torch.sigmoid(gate_inp)
        val = self.w3(x)
        return self.w2(silu * val)
        


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # we will pre-compute all of the values needed for the elementwise mul
        assert d_k % 2 == 0, "dimension of RoPE must be even."
        # θi = 10000**−2i/d
        i = torch.arange(start=0, end=d_k/2, step=1, device=device)
        thetas = theta ** (-2*i/d_k)
        # now, we need to precompte all the sins/cosines for t * \theta
        ts = torch.arange(start=0, end=max_seq_len, step=1, device=device)

        # (max_seq_len, d_k/2
        sines = torch.sin(einsum(thetas, ts, 'dk2, max -> max dk2'))
        cosines = torch.cos(einsum(thetas, ts, 'dk2, max -> max dk2'))

        # (max_seq_len, d_k/2)
        pos_embs = torch.complex(real=cosines, imag=sines)
        self.register_buffer('pos_embs', pos_embs, persistent=False)

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x is ... seq dk
        # pos is ... seq

        # this is (..., seq, dk/2)
        c_pos_embs = self.pos_embs[token_positions]
        
        # next, need to construct a (..., seq_len, d_k/2) complex tensor
        x_comp = torch.complex(real=x[..., ::2], imag=x[..., 1::2])

        # now elementwise to rotate
        rot = einsum(x_comp, c_pos_embs, "... seq dk2, ... seq dk2 -> ... seq dk2")
        # now we need to interleave them back
        return rearrange(torch.view_as_real(rot), "... seq dk2 two -> ... seq (dk2 two)")

