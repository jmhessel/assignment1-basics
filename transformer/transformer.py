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
