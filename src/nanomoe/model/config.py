"""Model configuration for scalable MoE architecture.

Based on Qwen3-30B-A3B architecture but configurable:
- Grouped Query Attention (GQA)
- RMSNorm
- SwiGLU FFN
- Mixture of Experts with top-k routing
- RoPE positional embeddings
"""

from dataclasses import dataclass


@dataclass
class MoEConfig:
    """Configuration for a scalable MoE transformer.

    Default values create a ~125M parameter model (similar to GPT-2 small).
    Scale up by increasing hidden_size, num_layers, num_experts.

    Qwen3-30B-A3B reference:
    - hidden_size: 4096
    - num_layers: 48
    - num_attention_heads: 32
    - num_key_value_heads: 4 (GQA)
    - intermediate_size: 6144 (per expert)
    - num_experts: 128
    - num_experts_per_tok: 8
    - vocab_size: 151936
    """

    # Model dimensions
    hidden_size: int = 768
    num_layers: int = 12
    vocab_size: int = 151936  # Qwen tokenizer vocab size

    # Attention
    num_attention_heads: int = 12
    num_key_value_heads: int = 4  # For GQA (set = num_attention_heads for MHA)
    head_dim: int | None = None  # If None, computed as hidden_size // num_attention_heads
    attention_type: str = "fsdp_attention"  # "fsdp_attention" or "flex_attention"
    max_position_embeddings: int = 8192
    rope_theta: float = 1000000.0  # RoPE base frequency

    # FFN / MoE
    intermediate_size: int = 2048  # Per-expert FFN hidden dim
    num_experts: int = 8  # Total experts
    num_experts_per_tok: int = 2  # Active experts per token (top-k)
    shared_expert: bool = False  # Whether to have a shared expert (always active)
    shared_expert_intermediate_size: int | None = None  # If None, uses intermediate_size
    shared_expert_scale: float = 1.0  # Scale factor for shared expert output (if used)

    # Expert routing
    router_type: str = "topk"  # Router type key from ROUTER_REGISTRY
    router_aux_loss_coef: float = 0.01  # Auxiliary load balancing loss coefficient
    router_jitter_noise: float = 0.0  # Jitter noise for router (training only)
    moe_kernel: str = "auto"  # "auto", "eager_mm", "grouped_mm", or "grouped_mm_fast"

    # Normalization & activation
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"  # Activation for FFN (silu = SwiGLU)

    # Regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Initialization
    initializer_range: float = 0.02
    depth_alpha: float = 0.0

    # Tie embeddings
    tie_word_embeddings: bool = False

    def __post_init__(self):
        valid_attention_types = {"fsdp_attention", "flex_attention"}
        if self.attention_type not in valid_attention_types:
            raise ValueError(
                f"Unsupported attention_type: {self.attention_type}. "
                f"Available: {sorted(valid_attention_types)}"
            )

        valid_moe_kernels = {"auto", "eager_mm", "grouped_mm", "grouped_mm_fast"}
        if self.moe_kernel not in valid_moe_kernels:
            raise ValueError(
                f"Unsupported moe_kernel: {self.moe_kernel}. "
                f"Available: {sorted(valid_moe_kernels)}"
            )

        if self.head_dim is None:
            object.__setattr__(self, "head_dim", self.hidden_size // self.num_attention_heads)

        if self.shared_expert and self.shared_expert_intermediate_size is None:
            object.__setattr__(self, "shared_expert_intermediate_size", self.intermediate_size)

    @property
    def num_active_params(self) -> int:
        """Estimate active parameters per forward pass."""
        # Embeddings (always active)
        embed_params = self.vocab_size * self.hidden_size
        head_dim = self.head_dim or (self.hidden_size // self.num_attention_heads)

        # Per layer
        # Attention: Q, K, V, O projections
        qkv_params = self.hidden_size * (
            self.num_attention_heads * head_dim  # Q
            + 2 * self.num_key_value_heads * head_dim  # K, V
        )
        o_params = self.num_attention_heads * head_dim * self.hidden_size
        attn_params = qkv_params + o_params

        # MoE FFN: only num_experts_per_tok experts active
        expert_params = 3 * self.hidden_size * self.intermediate_size  # gate, up, down
        active_expert_params = expert_params * self.num_experts_per_tok

        # Shared expert if present
        shared_params = 0
        if self.shared_expert:
            shared_params = 3 * self.hidden_size * (self.shared_expert_intermediate_size or self.intermediate_size)

        # Router
        router_params = self.hidden_size * self.num_experts

        # Norms
        norm_params = 2 * self.hidden_size  # attention norm + ffn norm

        layer_params = attn_params + active_expert_params + shared_params + router_params + norm_params

        # Final norm + LM head
        final_params = self.hidden_size + (0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size)

        return embed_params + self.num_layers * layer_params + final_params

    @property
    def num_total_params(self) -> int:
        """Estimate total parameters."""
        embed_params = self.vocab_size * self.hidden_size
        head_dim = self.head_dim or (self.hidden_size // self.num_attention_heads)

        qkv_params = self.hidden_size * (self.num_attention_heads * head_dim + 2 * self.num_key_value_heads * head_dim)
        o_params = self.num_attention_heads * head_dim * self.hidden_size
        attn_params = qkv_params + o_params

        # All experts
        expert_params = 3 * self.hidden_size * self.intermediate_size * self.num_experts

        shared_params = 0
        if self.shared_expert:
            shared_params = 3 * self.hidden_size * (self.shared_expert_intermediate_size or self.intermediate_size)

        router_params = self.hidden_size * self.num_experts
        norm_params = 2 * self.hidden_size

        layer_params = attn_params + expert_params + shared_params + router_params + norm_params

        final_params = self.hidden_size + (0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size)

        return embed_params + self.num_layers * layer_params + final_params

    @classmethod
    def tiny(cls) -> "MoEConfig":
        """Tiny model for testing (~10M params)."""
        return cls(
            hidden_size=256,
            num_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            num_experts=4,
            num_experts_per_tok=2,
        )

    @classmethod
    def small(cls) -> "MoEConfig":
        """Small model (~125M active, ~500M total)."""
        return cls(
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
            intermediate_size=2048,
            num_experts=8,
            num_experts_per_tok=2,
        )

    @classmethod
    def medium(cls) -> "MoEConfig":
        """Medium model (~350M active, ~1.5B total)."""
        return cls(
            hidden_size=1024,
            num_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            intermediate_size=2816,
            num_experts=16,
            num_experts_per_tok=2,
        )

    @classmethod
    def large(cls) -> "MoEConfig":
        """Large model (~1B active, ~7B total)."""
        return cls(
            hidden_size=2048,
            num_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            intermediate_size=5504,
            num_experts=32,
            num_experts_per_tok=4,
        )

    @classmethod
    def qwen3_30b_a3b_scaled(cls, scale: float = 0.1) -> "MoEConfig":
        """Scaled version of Qwen3-30B-A3B architecture.

        Args:
            scale: Scale factor (0.1 = 10% of original dims)
        """
        return cls(
            hidden_size=int(4096 * scale),
            num_layers=max(4, int(48 * scale)),
            num_attention_heads=max(4, int(32 * scale)),
            num_key_value_heads=max(2, int(4 * scale)),
            intermediate_size=int(6144 * scale),
            num_experts=max(4, int(128 * scale)),
            num_experts_per_tok=max(2, int(8 * scale)),
            vocab_size=151936,
            max_position_embeddings=8192,
            rope_theta=1000000.0,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "attention_type": self.attention_type,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "intermediate_size": self.intermediate_size,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "shared_expert": self.shared_expert,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "router_type": self.router_type,
            "router_aux_loss_coef": self.router_aux_loss_coef,
            "router_jitter_noise": self.router_jitter_noise,
            "moe_kernel": self.moe_kernel,
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_act": self.hidden_act,
            "attention_dropout": self.attention_dropout,
            "hidden_dropout": self.hidden_dropout,
            "initializer_range": self.initializer_range,
            "depth_alpha": self.depth_alpha,
            "tie_word_embeddings": self.tie_word_embeddings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MoEConfig":
        """Create from dictionary."""
        return cls(**d)
