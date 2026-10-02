"""
Improved Proprioceptive State Integration for Robot Learning

This module provides better fusion mechanisms for combining visual and
proprioceptive information in robot manipulation policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ImprovedProprioEncoder(nn.Module):
    """
    Enhanced proprioceptive state encoder with:
    - Layer normalization for stable training
    - MLP with nonlinearity for better expressiveness
    - Dropout for regularization
    - Positional embedding for joint-specific semantics
    """
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or latent_dim // 2

        self.norm = nn.LayerNorm(state_dim)
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Optional positional embedding for joint semantics
        # CRITICAL FIX: Initialize to zeros to avoid introducing large variance
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, latent_dim)) if use_pos_emb else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, state_dim] proprioceptive states
        Returns:
            [batch, seq_len, latent_dim] encoded states
        """
        x = self.norm(x)
        x = self.mlp(x)
        if self.pos_emb is not None:
            x = x + self.pos_emb


        return x


class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for visual-proprio fusion.

    Modulates visual features based on proprioceptive state:
        output = visual * (1 + gamma(proprio)) + beta(proprio)

    Reference: "FiLM: Visual Reasoning with a General Conditioning Layer"
    """
    def __init__(self, visual_dim: int, proprio_dim: int):
        super().__init__()
        self.gamma = nn.Linear(proprio_dim, visual_dim)
        self.beta = nn.Linear(proprio_dim, visual_dim)

        # Initialize to identity modulation
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(
        self,
        visual_emb: torch.Tensor,
        proprio_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_emb: [batch, n_visual_tokens, visual_dim]
            proprio_emb: [batch, n_proprio_tokens, proprio_dim]
        Returns:
            [batch, n_visual_tokens, visual_dim] modulated visual features
        """
        # Pool proprio if multiple tokens
        if proprio_emb.size(1) > 1:
            proprio_emb = proprio_emb.mean(dim=1, keepdim=True)

        gamma = self.gamma(proprio_emb)  # [B, 1, D]
        beta = self.beta(proprio_emb)    # [B, 1, D]

        # FiLM modulation
        return visual_emb * (1.0 + gamma) + beta


class SimpleConcatFusion(nn.Module):
    """
    Simple concatenation-based fusion (SOTA standard).

    Used by most successful methods:
    - Diffusion Policy (78% on Push-T): cat([visual_512, robot_128])
    - RT-1 (97% on real robot): cat([visual_tokens, robot_token])
    - ACT (85% on bimanual): cat([visual, proprio])
    - RoboCasa official baseline

    Design principles:
    1. Keep visual dominant (512D) and proprio as auxiliary (128D)
    2. Simple LayerNorm + Linear + GELU projection
    3. No complex attention or gating - let gradients flow freely
    4. Dimension ratio 4:1 prevents proprio from overwhelming visual

    Usage:
        # In base_agent.py:
        # 1. Broadcast proprio: [B*T, 128] -> [B*T, N, 128]
        # 2. Concatenate: cat([visual, proprio_broadcast], dim=-1) -> [B*T, N, 640]
        # 3. Project: fusion.proj(fused_flat) -> [B*T*N, 512]

    This is simpler and more robust than Cross-Attention for most tasks.
    """
    def __init__(
        self,
        visual_dim: int = 512,
        proprio_dim: int = 128,
        output_dim: int = 512,
    ):
        super().__init__()
        concat_dim = visual_dim + proprio_dim

        # Simple projection: no gates, no complex mechanisms
        # Input: [*, concat_dim] where * can be [B*T*N] after flattening
        # Output: [*, output_dim]
        self.proj = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, output_dim),
            nn.GELU(),
        )

        # Initialize to preserve visual features
        # Small weight on the proprio part initially
        with torch.no_grad():
            # Linear weight is [output_dim, concat_dim]
            # Split into visual part and proprio part
            visual_weight = self.proj[1].weight[:, :visual_dim]
            proprio_weight = self.proj[1].weight[:, visual_dim:]

            # Visual part: near-identity (preserve DIFT features)
            nn.init.eye_(visual_weight)
            # Proprio part: small random (gradual influence)
            nn.init.normal_(proprio_weight, mean=0.0, std=0.02)

            # Bias: zero init
            nn.init.zeros_(self.proj[1].bias)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based fusion with proprio as query, visual as context.

    IMPROVED DESIGN (follows ACT/π0/Octo):
    - Proprio (robot state) queries visual features to extract relevant information
    - Visual is the rich information source (DIFT pretrained)
    - Proprio is the "hint/condition" (small 8D state)
    - Output is proprio-conditioned visual features

    Architecture:
        Query: proprio embeddings [B, T, 128]
        Key/Value: visual embeddings [B, T, 512]
        Output: conditioned visual [B, T, 512]

    Identity initialization via residual_gate ensures stable training.
    """
    def __init__(
        self,
        visual_dim: int,      # e.g., 512 (DIFT features)
        proprio_dim: int,     # e.g., 128 (robot state embedding)
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Use visual_dim as the embedding dimension (larger)
        # Proprio will be projected up to this dimension for queries
        self.embed_dim = visual_dim
        assert self.embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Project proprio (small) → query space (large)
        self.q_proj = nn.Linear(proprio_dim, self.embed_dim)
        # Visual → key/value (stays in visual_dim)
        self.k_proj = nn.Linear(visual_dim, self.embed_dim)
        self.v_proj = nn.Linear(visual_dim, self.embed_dim)
        # Output projection
        self.out_proj = nn.Linear(self.embed_dim, visual_dim)

        self.dropout = nn.Dropout(dropout)
        self.query_norm = nn.LayerNorm(proprio_dim)
        self.context_norm = nn.LayerNorm(visual_dim)

        # Identity initialization: start with pure visual, gradually learn to condition on proprio
        self.residual_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        proprio_emb: torch.Tensor,  # [B, T, proprio_dim=128] - QUERY
        visual_emb: torch.Tensor,   # [B, T, visual_dim=512] - CONTEXT (key/value)
    ) -> torch.Tensor:
        """
        Proprio queries visual to extract task-relevant features.

        Args:
            proprio_emb: [batch, T, proprio_dim] robot state embeddings (query)
            visual_emb: [batch, T, visual_dim] visual features (key/value)

        Returns:
            [batch, T, visual_dim] proprio-conditioned visual features
        """
        B, T, _ = visual_emb.shape
        T_p = proprio_emb.size(1)

        # Normalize inputs
        proprio_normed = self.query_norm(proprio_emb)
        visual_normed = self.context_norm(visual_emb)

        # Project to query/key/value
        # Q from proprio: [B, T_p, embed_dim] → [B, heads, T_p, head_dim]
        q = self.q_proj(proprio_normed).reshape(B, T_p, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V from visual: [B, T, embed_dim] → [B, heads, T, head_dim]
        k = self.k_proj(visual_normed).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(visual_normed).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: proprio attends to visual
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, T_p, T]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Get attended visual features
        out = (attn @ v).transpose(1, 2).reshape(B, T_p, self.embed_dim)  # [B, T_p, embed_dim]
        out = self.out_proj(out)  # [B, T_p, visual_dim]

        # Residual connection: start from pure visual, gradually add proprio-conditioned info
        # If T_p != T, broadcast or match dimensions
        if T_p == T:
            return visual_emb + self.residual_gate * out
        else:
            # If proprio has fewer tokens, expand to match visual
            return visual_emb + self.residual_gate * out.expand(-1, T, -1)


class GatedFusion(nn.Module):
    """
    Learnable gating mechanism to balance visual and proprio contributions.

    Uses a gate to decide how much to trust each modality:
        gate = sigmoid(W[visual; proprio])
        output = gate * visual + (1 - gate) * proprio
    """
    def __init__(self, visual_dim: int, proprio_dim: int, output_dim: int):
        super().__init__()

        # Project proprio to same dimension as visual
        self.proprio_proj = nn.Linear(proprio_dim, visual_dim)

        # Gate computation
        self.gate_net = nn.Sequential(
            nn.Linear(visual_dim + proprio_dim, output_dim),
            nn.Sigmoid()
        )

        # Output projection
        self.out_proj = nn.Linear(visual_dim, output_dim)

    def forward(
        self,
        visual_emb: torch.Tensor,
        proprio_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_emb: [batch, n_visual, visual_dim]
            proprio_emb: [batch, n_proprio, proprio_dim]
        Returns:
            [batch, n_visual + n_proprio, output_dim] fused features
        """
        # Pool proprio if needed
        if proprio_emb.size(1) > 1:
            proprio_pooled = proprio_emb.mean(dim=1, keepdim=True)
        else:
            proprio_pooled = proprio_emb

        # Broadcast to match visual tokens
        proprio_broadcasted = proprio_pooled.expand(-1, visual_emb.size(1), -1)

        # Compute gate
        combined = torch.cat([visual_emb, proprio_broadcasted], dim=-1)
        gate = self.gate_net(combined)  # [B, N, D]

        # Project proprio
        proprio_proj = self.proprio_proj(proprio_broadcasted)

        # Gated fusion
        visual_proj = self.out_proj(visual_emb)
        proprio_proj_out = self.out_proj(proprio_proj)

        fused = gate * visual_proj + (1 - gate) * proprio_proj_out

        return fused


class MultiModalFusionAgent:
    """
    Example of how to integrate improved proprio fusion into your agent.

    Usage in base_agent.py:

    ```python
    # In __init__:
    self.proprio_encoder = ImprovedProprioEncoder(state_dim, latent_dim)
    self.fusion = FiLMFusion(latent_dim, latent_dim)  # or CrossAttentionFusion

    # In compute_input_embeddings:
    if self.if_robot_states and "robot_states" in obs_dict:
        robot_states = obs_dict["robot_states"]
        proprio_emb = self.proprio_encoder(robot_states)

        # Use FiLM fusion instead of simple concatenation
        perceptual_emb = self.fusion(perceptual_emb, proprio_emb)
        # OR: perceptual_emb = torch.cat([perceptual_emb, proprio_emb], dim=1)
    ```
    """
    pass


# Example usage and comparison
if __name__ == "__main__":
    batch_size = 4
    n_visual_tokens = 3  # 3 cameras
    n_proprio_tokens = 1
    visual_dim = 512
    proprio_raw_dim = 8  # 1 gripper + 7 joints
    latent_dim = 512

    # Create dummy data
    visual_emb = torch.randn(batch_size, n_visual_tokens, visual_dim)
    proprio_raw = torch.randn(batch_size, n_proprio_tokens, proprio_raw_dim)

    print("=" * 80)
    print("COMPARISON OF FUSION METHODS")
    print("=" * 80)

    # Method 1: Simple concatenation (current)
    print("\n1. Simple Concatenation (current implementation)")
    state_emb_simple = nn.Linear(proprio_raw_dim, latent_dim)
    proprio_emb_simple = state_emb_simple(proprio_raw)
    fused_simple = torch.cat([visual_emb, proprio_emb_simple], dim=1)
    print(f"   Input:  Visual {visual_emb.shape}, Proprio {proprio_raw.shape}")
    print(f"   Output: {fused_simple.shape}")
    print(f"   Params: {sum(p.numel() for p in state_emb_simple.parameters()):,}")

    # Method 2: Improved encoder + FiLM
    print("\n2. Improved Encoder + FiLM Fusion")
    proprio_encoder = ImprovedProprioEncoder(proprio_raw_dim, latent_dim)
    film_fusion = FiLMFusion(visual_dim, latent_dim)
    proprio_emb_improved = proprio_encoder(proprio_raw)
    fused_film = film_fusion(visual_emb, proprio_emb_improved)
    print(f"   Input:  Visual {visual_emb.shape}, Proprio {proprio_raw.shape}")
    print(f"   Output: {fused_film.shape}")
    total_params = sum(p.numel() for p in proprio_encoder.parameters())
    total_params += sum(p.numel() for p in film_fusion.parameters())
    print(f"   Params: {total_params:,}")

    # Method 3: Cross-attention fusion
    print("\n3. Cross-Attention Fusion")
    cross_attn = CrossAttentionFusion(visual_dim, latent_dim)
    fused_cross = cross_attn(visual_emb, proprio_emb_improved)
    print(f"   Input:  Visual {visual_emb.shape}, Proprio {proprio_emb_improved.shape}")
    print(f"   Output: {fused_cross.shape}")
    print(f"   Params: {sum(p.numel() for p in cross_attn.parameters()):,}")

    # Method 4: Gated fusion
    print("\n4. Gated Fusion")
    gated_fusion = GatedFusion(visual_dim, latent_dim, latent_dim)
    fused_gated = gated_fusion(visual_emb, proprio_emb_improved)
    print(f"   Input:  Visual {visual_emb.shape}, Proprio {proprio_emb_improved.shape}")
    print(f"   Output: {fused_gated.shape}")
    print(f"   Params: {sum(p.numel() for p in gated_fusion.parameters()):,}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("- Start with FiLM (good balance of performance and simplicity)")
    print("- Use Cross-Attention for complex tasks (doors, drawers)")
    print("- Gated Fusion for maximum flexibility (but more params)")
    print("- Always use ImprovedProprioEncoder instead of simple Linear!")
