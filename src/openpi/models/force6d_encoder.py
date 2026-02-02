"""Force6D Encoder: MLP-based encoder for 1D 6D force vectors."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.shared import array_typing as at


class Force6DEncoder(nnx.Module):
    """MLP encoder for 1D 6D force vectors.
    
    This encoder processes 1D force vectors (shape [6]) using a simple MLP
    to extract features that can be concatenated to state embedding.
    
    Args:
        output_dim: Output feature dimension (should match state embedding dim)
        hidden_dim: Hidden layer dimension (default: 32)
        num_layers: Number of MLP layers (default: 2)
        dtype: Data type
    """
    
    def __init__(
        self,
        output_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dtype: str = "bfloat16",
    ):
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dtype = dtype
        
        # Input dimension is 6 (6D force vector)
        input_dim = 6
        
        # Build MLP layers
        self.layers = {}
        if num_layers == 1:
            # Single layer: direct projection
            self.layers[0] = nnx.Linear(input_dim, output_dim, dtype=dtype, rngs=rngs)
        else:
            # First layer: input -> hidden
            self.layers[0] = nnx.Linear(input_dim, hidden_dim, dtype=dtype, rngs=rngs)
            # Middle layers: hidden -> hidden
            for i in range(1, num_layers - 1):
                self.layers[i] = nnx.Linear(hidden_dim, hidden_dim, dtype=dtype, rngs=rngs)
            # Last layer: hidden -> output
            self.layers[num_layers - 1] = nnx.Linear(hidden_dim, output_dim, dtype=dtype, rngs=rngs)
        
        # Layer norm for output
        self.layer_norm = nnx.LayerNorm(output_dim, dtype=dtype, rngs=rngs)
    
    def __call__(self, force6d: at.Float[at.Array, "*b 6"], *, train: bool = False) -> at.Float[at.Array, "*b output_dim"]:
        """Process 6D force vector.
        
        Args:
            force6d: 6D force vector, shape [*b, 6] where *b is batch dimensions
            train: Whether in training mode
        
        Returns:
            Encoded features, shape [*b, output_dim]
        """
        # Ensure input is float32 and has correct shape
        if force6d.shape[-1] != 6:
            raise ValueError(f"Expected force6d to have last dimension of 6, got shape {force6d.shape}")
        
        x = force6d.astype(jnp.float32)
        
        # Apply MLP layers
        for i, layer in enumerate(self.layers.values()):
            x = layer(x)
            # Apply activation (GELU) except for the last layer
            if i < len(self.layers) - 1:
                x = jax.nn.gelu(x)
        
        # Apply layer norm
        x = self.layer_norm(x)
        return x
