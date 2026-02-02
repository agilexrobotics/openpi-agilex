"""深度图像编码器

这个模块提供了一个专门用于处理深度图像的编码器，可以替代或补充SigLIP来处理深度数据。
"""

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


class GELU(nnx.Module):
    """GELU activation function module."""
    
    def __call__(self, x: at.Array) -> at.Array:
        return jax.nn.gelu(x)


class DepthEncoderBlock(nnx.Module):
    """Transformer encoder block for depth images.
    
    每个block包含self-attention和MLP，用于提取深度图像的空间特征。
    """
    
    def __init__(
        self,
        output_dim: int,
        rngs: nnx.Rngs,
        num_heads: int = 12,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
        dtype: str = "bfloat16",
    ):
        self.output_dim = output_dim
        # Ensure num_heads divides output_dim; if not, pick the largest factor of output_dim
        if output_dim % num_heads != 0:
            # Try some common head counts that divide output_dim
            candidates = [h for h in (8, 16, 32, 64) if output_dim % h == 0]
            if not candidates:
                # Fallback: use 1 head
                num_heads = 1
            else:
                num_heads = max(candidates)
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.mlp_dim = mlp_dim or (output_dim * 4)
        self.dropout = dropout
        self.dtype = dtype
        
        # QKV projections for self-attention
        self.q_proj = nnx.Linear(output_dim, output_dim, rngs=rngs)
        self.k_proj = nnx.Linear(output_dim, output_dim, rngs=rngs)
        self.v_proj = nnx.Linear(output_dim, output_dim, rngs=rngs)
        self.out_proj = nnx.Linear(output_dim, output_dim, rngs=rngs)
        
        # Layer norms
        self.ln1 = nnx.LayerNorm(output_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(output_dim, rngs=rngs)
        
        # MLP block
        self.mlp_linear1 = nnx.Linear(output_dim, self.mlp_dim, rngs=rngs)
        self.mlp_gelu = GELU()
        self.mlp_linear2 = nnx.Linear(self.mlp_dim, output_dim, rngs=rngs)
        
        # Dropout (if needed)
        if dropout > 0:
            self.dropout_layer = nnx.Dropout(rate=dropout)
        else:
            self.dropout_layer = None
    
    def __call__(self, x: at.Float[at.Array, "b n d"], *, train: bool = False):
        """Apply transformer encoder block.
        
        Args:
            x: Input tokens, shape [B, N, D]
            train: Whether in training mode
            
        Returns:
            Output tokens, shape [B, N, D]
        """
        # Self-attention with residual
        x_norm = self.ln1(x)
        
        # Compute Q, K, V
        q = self.q_proj(x_norm)  # [B, N, D]
        k = self.k_proj(x_norm)  # [B, N, D]
        v = self.v_proj(x_norm)  # [B, N, D]
        
        # Reshape for multi-head attention: [B, N, D] -> [B, N, num_heads, head_dim]
        B, N, D = q.shape
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.num_heads, self.head_dim)
        v = v.reshape(B, N, self.num_heads, self.head_dim)
        
        # Transpose for attention: [B, num_heads, N, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn_probs = jax.nn.softmax(attn_scores.astype(jnp.float32), axis=-1).astype(q.dtype)
        attn_out = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, v)
        
        # Transpose back and reshape: [B, num_heads, N, head_dim] -> [B, N, D]
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
        attn_out = attn_out.reshape(B, N, D)
        
        # Output projection
        attn_out = self.out_proj(attn_out)
        
        if self.dropout_layer is not None and train:
            attn_out = self.dropout_layer(attn_out, rngs=nnx.Rngs())
        
        x = x + attn_out
        
        # MLP with residual
        x_norm = self.ln2(x)
        mlp_out = self.mlp_linear1(x_norm)
        mlp_out = self.mlp_gelu(mlp_out)
        mlp_out = self.mlp_linear2(mlp_out)
        if self.dropout_layer is not None and train:
            mlp_out = self.dropout_layer(mlp_out, rngs=nnx.Rngs())
        x = x + mlp_out
        
        return x


class DepthEncoder(nnx.Module):
    """用于处理深度图像的编码器。
    
    这个网络专门设计用于处理单通道深度图像，输出与SigLIP相同维度的特征。
    使用Transformer encoder blocks来提取层次化的空间特征。
    
    Args:
        output_dim: 输出特征维度，应该与SigLIP的输出维度匹配（通常是paligemma_config.width）
        depth: Transformer encoder的层数，默认12层（比SigLIP的27层少，因为深度图像信息密度较低）
        num_heads: Attention heads数量，默认12
        mlp_dim: MLP维度，默认4*output_dim
        dropout: Dropout率，默认0.0
        dtype: 数据类型，默认使用bfloat16以匹配模型其他部分
    """
    
    def __init__(
        self,
        output_dim: int,
        rngs: nnx.Rngs,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
        dtype: str = "bfloat16",
    ):
        self.output_dim = output_dim
        self.depth = depth
        self.num_heads = num_heads
        self.dtype = dtype
        
        # Patch size 16x16，与SigLIP保持一致
        patch_size = 16
        self.patch_size = patch_size
        
        # 计算patch数量 (224/16)^2 = 196
        num_patches = (224 // patch_size) ** 2
        
        # Patch embedding: 将16x16的patch投影到output_dim
        self.patch_embed = nnx.Linear(
            patch_size * patch_size,  # 16*16 = 256
            output_dim,
            rngs=rngs,
        )
        
        # 位置编码（可学习的）
        rng = rngs.params()
        pos_embed_init = jax.random.normal(
            rng,
            (1, num_patches, output_dim),
        )
        self.pos_embed = nnx.Param(pos_embed_init.astype(getattr(jnp, dtype)))
        
        # Transformer encoder blocks (use dict instead of list to avoid integer keys)
        self.encoder_blocks = {}
        for i in range(depth):
            block = DepthEncoderBlock(
                output_dim=output_dim,
                rngs=rngs,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                dtype=dtype,
            )
            self.encoder_blocks[str(i)] = block
        
        # Final layer norm
        self.layer_norm = nnx.LayerNorm(output_dim, rngs=rngs)
    
    def __call__(self, depth_image: at.Float[at.Array, "b h w c"], *, train: bool = False):
        """处理深度图像。
        
        Args:
            depth_image: 深度图像，形状为 [B, H, W, C]，值应该在 [-1, 1] 范围内
                        如果是单通道，C=1；如果是RGB格式的深度图，C=3
            train: 是否处于训练模式
        
        Returns:
            tokens: 图像token，形状为 [B, num_patches, output_dim]
            out: 可选的中间输出（用于调试）
        """
        batch_size = depth_image.shape[0]
        h, w = depth_image.shape[1:3]
        
        # 确保输入是单通道（如果是RGB，转换为灰度）
        if depth_image.shape[-1] == 3:
            # RGB转灰度: 使用标准权重
            depth_image = (
                0.299 * depth_image[..., 0:1] +
                0.587 * depth_image[..., 1:2] +
                0.114 * depth_image[..., 2:3]
            )
        elif depth_image.shape[-1] != 1:
            # 如果不是1或3通道，取第一个通道
            depth_image = depth_image[..., 0:1]
        
        # 提取patches: 使用einops更简洁
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Reshape to patches: [B, H, W, 1] -> [B, num_patches_h, num_patches_w, patch_size, patch_size, 1]
        patches = einops.rearrange(
            depth_image,
            "b (nh ph) (nw pw) c -> b nh nw (ph pw c)",
            nh=num_patches_h,
            nw=num_patches_w,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        
        # Flatten patches: [B, num_patches_h, num_patches_w, patch_size*patch_size] -> [B, num_patches, patch_size*patch_size]
        patches = patches.reshape(batch_size, num_patches, -1)
        
        # Project patches to output_dim
        tokens = self.patch_embed(patches)  # [B, num_patches, output_dim]
        
        # Add positional embedding
        tokens = tokens + self.pos_embed
        
        # Apply transformer encoder blocks
        for block in self.encoder_blocks.values():
            tokens = block(tokens, train=train)
        
        # Final layer norm
        tokens = self.layer_norm(tokens)
        
        return tokens, {}

