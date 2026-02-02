"""触觉数据编码器

用于将高维触觉数据（如50x32x6的六维力数据）压缩为低维特征向量。
"""

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


class TactileConvBlock(nnx.Module):
    """带残差连接的卷积块。
    
    用于构建更深的网络，同时保持训练稳定性。
    """
    
    def __init__(
        self,
        features: int,
        rngs: nnx.Rngs,
        in_features: int | None = None,
        kernel_size: tuple[int, int] = (3, 3),
        strides: tuple[int, int] = (1, 1),
        use_residual: bool = True,
        dtype: str = "bfloat16",
    ):
        self.features = features
        self.in_features = in_features
        self.use_residual = use_residual
        self.strides = strides
        
        # 主卷积层
        # nnx.Conv uses in_features and out_features, not features
        in_features_for_conv = in_features if in_features is not None else features
        self.conv = nnx.Conv(
            in_features=in_features_for_conv,
            out_features=features,
            kernel_size=kernel_size,
            strides=strides,
            padding="SAME",
            dtype=dtype,
            rngs=rngs,
        )
        
        # 残差连接：如果使用残差，需要处理通道数不匹配和下采样
        if use_residual:
            # 如果stride>1或通道数会变化，需要残差投影
            if strides != (1, 1) or (in_features is not None and in_features != features):
                self.residual_conv = nnx.Conv(
                    in_features=in_features if in_features is not None else features,
                    out_features=features,
                    kernel_size=(1, 1),
                    strides=strides,
                    padding="SAME",
                    dtype=dtype,
                    rngs=rngs,
                )
            else:
                self.residual_conv = None
        else:
            self.residual_conv = None
        
        # Layer norm
        self.layer_norm = nnx.LayerNorm(features, dtype=dtype, rngs=rngs)
    
    def __call__(self, x, *, train: bool = False):
        """应用卷积块。
        
        Args:
            x: 输入特征图，形状 [B, H, W, C]
            train: 是否处于训练模式
            
        Returns:
            输出特征图，形状 [B, H', W', features]
        """
        # 保存残差
        residual = x
        
        # 主路径
        out = self.conv(x)
        out = self.layer_norm(out)
        out = jax.nn.gelu(out)
        
        # 残差连接
        if self.use_residual:
            if self.residual_conv is not None:
                # 需要下采样或调整通道数
                residual = self.residual_conv(residual)
            # 如果维度仍然不匹配（不应该发生），跳过残差连接
            if residual.shape == out.shape:
                out = out + residual
        
        return out


class TactileEncoder(nnx.Module):
    """用于处理高维触觉数据的编码器。
    
    将空间排列的触觉数据（如50x32x6的六维力网格）压缩为低维特征向量。
    使用多层CNN和残差连接来提取层次化的空间特征。
    
    Args:
        output_dim: 输出特征维度（建议16-128之间）
        input_shape: 输入数据的空间形状，例如 (50, 32) 表示50x32的传感器网格
        force_dim: 每个传感器的力维度，例如 6 表示六维力（大小+方向）
        num_layers: 卷积层数，默认6层（比原来的3层更深）
        dtype: 数据类型
    """
    
    def __init__(
        self, 
        output_dim: int,
        rngs: nnx.Rngs,
        input_shape: tuple[int, int] = (50, 32),
        force_dim: int = 6,
        num_layers: int = 6,
        dtype: str = "bfloat16",
        adaptive_pool: bool = False,
        target_tokens: int | None = None,
    ):
        """初始化触觉编码器。
        
        Args:
            output_dim: 输出特征维度
            rngs: 随机数生成器
            input_shape: 输入数据的期望空间形状（仅用于参考，不强制）
            force_dim: 每个传感器的力维度
            num_layers: 卷积层数
            dtype: 数据类型
            adaptive_pool: 是否使用自适应池化统一 token 数量（推荐为 True）
            target_tokens: 目标 token 数量（仅在 adaptive_pool=True 时使用）
        """
        self.output_dim = output_dim
        self.input_shape = input_shape
        self.force_dim = force_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.adaptive_pool = adaptive_pool
        self.target_tokens = target_tokens
        
        # 使用CNN来提取空间特征
        # 输入: [B, H, W, C] = [B, 50, 32, 6]
        # 使用多层卷积来逐步降维和提取特征
        
        # 初始投影层：将输入通道投影到统一特征空间
        # nnx.Conv uses in_features and out_features
        # Input channels = force_dim (e.g., 3), output channels = 32
        self.input_proj = nnx.Conv(
            in_features=force_dim,
            out_features=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            dtype=dtype,
            rngs=rngs,
        )
        
        # 构建卷积块序列
        # 每2层进行一次下采样，逐步提取特征
        self.conv_blocks = {}
        
        # 第1-2层：提取局部特征，stride=2下采样
        self.conv_blocks[0] = TactileConvBlock(
            features=32,
            in_features=32,  # 输入是input_proj的输出
            kernel_size=(3, 3),
            strides=(2, 2),
            use_residual=True,
            dtype=dtype,
            rngs=rngs,
        )
        # 输出: [B, 25, 16, 32]
        
        self.conv_blocks[1] = TactileConvBlock(
            features=32,
            in_features=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            use_residual=True,
            dtype=dtype,
            rngs=rngs,
        )
        # 输出: [B, 25, 16, 32]
        
        # 第3-4层：进一步提取特征，stride=2下采样
        self.conv_blocks[2] = TactileConvBlock(
            features=64,
            in_features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            use_residual=True,
            dtype=dtype,
            rngs=rngs,
        )
        # 输出: [B, 13, 8, 64]
        
        self.conv_blocks[3] = TactileConvBlock(
            features=64,
            in_features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            use_residual=True,
            dtype=dtype,
            rngs=rngs,
        )
        # 输出: [B, 13, 8, 64]
        
        # 第5-6层：最终特征提取，stride=2下采样
        self.conv_blocks[4] = TactileConvBlock(
            features=128,
            in_features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            use_residual=True,
            dtype=dtype,
            rngs=rngs,
        )
        # 输出: [B, 7, 4, 128]
        
        self.conv_blocks[5] = TactileConvBlock(
            features=128,
            in_features=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            use_residual=True,
            dtype=dtype,
            rngs=rngs,
        )
        # 输出: [B, 7, 4, 128]
        
        # 投影到目标维度（处理空间特征图，输出tokens）
        # proj 会处理 [B, H*W, 128] 并输出 [B, H*W, output_dim]
        self.proj = nnx.Linear(128, output_dim, dtype=dtype, rngs=rngs)
        
        # Layer norm (处理每个token)
        self.layer_norm = nnx.LayerNorm(output_dim, dtype=dtype, rngs=rngs)
    
    def __call__(self, tactile_data, *, train: bool = False):
        """处理触觉数据。
        
        Args:
            tactile_data: 触觉数据，形状为 [B, H, W, C]
                        例如 [B, 50, 32, 6] 表示50x32的传感器网格，每个点有6维力信息
                        或 [B, 35, 20, 3] 表示35x20的传感器网格，每个点有3维力信息
                        或 [B, 224, 224, 1] 表示224x224的深度图
            train: 是否处于训练模式
        
        Returns:
            tokens: 触觉tokens，形状为 [B, num_spatial_tokens, output_dim]
                   num_spatial_tokens 会根据输入尺寸自适应变化
        """
        input_h, input_w, input_c = tactile_data.shape[1], tactile_data.shape[2], tactile_data.shape[3]
        
        # 通道数处理：调整到 force_dim
        if input_c != self.force_dim:
            if input_c > self.force_dim:
                # 裁剪多余通道
                tactile_data = tactile_data[..., :self.force_dim]
            else:
                # 零填充不足的通道
                pad_width = self.force_dim - input_c
                tactile_data = jnp.pad(
                    tactile_data,
                    ((0, 0), (0, 0), (0, 0), (0, pad_width)),
                    mode="constant",
                    constant_values=0,
                )
        
        # 初始投影
        x = self.input_proj(tactile_data)
        x = jax.nn.gelu(x)
        
        # 应用多层CNN特征提取（带残差连接）
        # CNN层会自动适应不同的输入尺寸，输出尺寸会根据输入和stride自动调整
        for conv_block in self.conv_blocks.values():
            x = conv_block(x, train=train)
        
        # 将空间特征图展平为tokens: [B, H', W', C'] -> [B, H'*W', C']
        # 这样触觉数据可以作为多个tokens输入，类似于图像patches
        B, H, W, C = x.shape
        
        # 可选：使用自适应池化统一 token 数量
        if self.adaptive_pool and self.target_tokens is not None:
            # 方案1: 使用自适应平均池化到目标 token 数量
            # 计算目标空间尺寸（近似正方形）
            target_h = int(jnp.sqrt(self.target_tokens))
            target_w = (self.target_tokens + target_h - 1) // target_h  # 向上取整
            
            # 使用 jax.image.resize 进行自适应池化
            # [B, H, W, C] -> [B, target_h, target_w, C]
            x = jax.image.resize(
                x,
                shape=(B, target_h, target_w, C),
                method="bilinear",
            )
            tokens = x.reshape(B, target_h * target_w, C)  # [B, target_tokens, C]
        else:
            # 标准路径：直接展平，token 数量根据输入尺寸自适应变化
            # num_spatial_tokens 会根据输入尺寸变化：
            #   - (35, 20) -> 约 15 tokens
            #   - (50, 32) -> 约 28 tokens
            #   - (224, 224) -> 约 784 tokens
            tokens = x.reshape(B, H * W, C)  # [B, num_spatial_tokens, C]
        
        # 投影到目标维度
        tokens = self.proj(tokens)  # [B, num_spatial_tokens, output_dim]
        
        # Layer norm
        tokens = self.layer_norm(tokens)
        
        return tokens


def create_tactile_encoder(
    output_dim: int,
    rngs: nnx.Rngs,
    input_shape: tuple[int, int] = (50, 32),
    force_dim: int = 6,
    num_layers: int = 6,
    dtype: str = "bfloat16",
    adaptive_pool: bool = False,
    target_tokens: int | None = None,
) -> TactileEncoder:
    """创建触觉编码器的工厂函数。
    
    Args:
        output_dim: 输出特征维度（建议16-128之间）
        rngs: 随机数生成器
        input_shape: 输入数据的期望空间形状（仅用于参考）
        force_dim: 每个传感器的力维度
        num_layers: 卷积层数，默认6层
        dtype: 数据类型
        adaptive_pool: 是否使用自适应池化统一 token 数量
        target_tokens: 目标 token 数量（仅在 adaptive_pool=True 时使用）
    
    Returns:
        配置好的TactileEncoder模块
    """
    return TactileEncoder(
        output_dim=output_dim,
        rngs=rngs,
        input_shape=input_shape,
        force_dim=force_dim,
        num_layers=num_layers,
        dtype=dtype,
        adaptive_pool=adaptive_pool,
        target_tokens=target_tokens,
    )

