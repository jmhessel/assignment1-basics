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
        rot = x_comp * c_pos_embs

        # now we need to interleave them back
        return rearrange(torch.view_as_real(rot), "... seq dk2 two -> ... seq (dk2 two)")


class Softmax(torch.nn.Module):
    @staticmethod
    def _forward(in_features: torch.Tensor, dim: int):
        maxes = torch.max(input=in_features, dim=dim, keepdim=True).values
        in_features -= maxes
        exps = torch.exp(in_features)
        normalizers = torch.sum(input=exps, dim=dim, keepdim=True)
        return exps / normalizers
   
    def forward(self, in_features: torch.Tensor, dim: int):
        return Softmax._forward(in_features=in_features, dim=dim)
    
         

class SDPA(torch.nn.Module):
    @staticmethod
    def _forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
        dim = Q.size()[-1]
        logits = einsum(Q, K, '... queries d, ... keys d -> ... queries keys')
        logits /= math.sqrt(dim)

        mask_val = (~mask).float() * -999999999
        logits = logits + mask_val

        sms = Softmax._forward(logits, dim=-1)
        return einsum(sms, V, "... queries keys, ... keys d -> ... queries d")
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
        return SDPA._forward(Q=Q, K=K, V=V, mask=mask)
        

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_rope:bool=True, theta:float=10000, max_seq_len:int=4096, device=None, dtype=None):
        super().__init__()
        self.k_proj_weight = Linear(d_model, d_model, device=device, dtype=dtype)
        self.q_proj_weight = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj_weight = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj_weight = Linear(d_model, d_model, device=device, dtype=dtype)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE(theta=theta, d_k=d_model//num_heads, max_seq_len=max_seq_len, device=device)
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        
    def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor | None = None):
        shape_in = in_features.size()
        seq_len = shape_in[-2]
        d = shape_in[-1]
        
        k = self.k_proj_weight(in_features)
        q = self.q_proj_weight(in_features)
        v = self.v_proj_weight(in_features)

        # split by head
        k = rearrange(k, "... seq (h dh) -> ... h seq dh", h=self.num_heads)
        q = rearrange(q, "... seq (h dh) -> ... h seq dh", h=self.num_heads)
        v = rearrange(v, "... seq (h dh) -> ... h seq dh", h=self.num_heads)

        if self.use_rope:
            if token_positions is None:
                token_positions = rearrange(torch.arange(0, seq_len, dtype=torch.int), 'd -> () d')
            k = self.rope(x=k, token_positions=token_positions)
            q = self.rope(x=q, token_positions=token_positions)

        # get mask, then do sdpa
        mask = torch.tril(torch.ones(size=(seq_len, seq_len), device=self.device, dtype=torch.bool))

        # ... h seq dh
        result = SDPA._forward(Q=q, K=k, V=v, mask=mask)

        # concat then project
        result = rearrange(result, "... h seq dh -> ... seq (h dh)")
        
        result = self.o_proj_weight(result)

        return result


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, in_features: torch.Tensor):
        in_features = in_features + self.attn(self.ln1(in_features))
        return in_features + self.ffn(self.ln2(in_features))
