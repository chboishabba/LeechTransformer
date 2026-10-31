from dataclasses import dataclass

@dataclass
class LeechConfig:
    vocab_size: int
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 8
    block_size: int = 512
    dropout: float = 0.05
    bias: bool = False
    tie_weights: bool = True
    lambda_geo: float = 0.01
    resonance_threshold: float = 0.95
    moe_experts: int = 0
    moe_top_k: int = 2
    moe_expert_width: int | None = None
    moe_route_geometry: str = "raw"
    moe_transition_rank: int | None = None

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        head_dim = self.d_model // self.n_heads
        assert head_dim % 24 == 0, "head_dim должен быть кратен 24"