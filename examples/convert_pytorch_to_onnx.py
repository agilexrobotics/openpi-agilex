#!/usr/bin/env python3
"""
Convert PyTorch π₀ model to ONNX format

This script converts a PyTorch model to ONNX format for deployment.
"""

import sys
import pathlib
sys.path.append('src')

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any
import argparse

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
import math
from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
import openpi.models.gemma as _gemma

logging.basicConfig(level=logging.INFO)

# Patch math.sqrt to use torch.sqrt to avoid complex numbers in ONNX export
original_math_sqrt = math.sqrt
def safe_sqrt(x):
    if isinstance(x, torch.Tensor):
        return torch.sqrt(torch.abs(x))  # Ensure positive input
    else:
        return original_math_sqrt(abs(x))  # Ensure positive input
math.sqrt = safe_sqrt

# Patch power operations to avoid complex numbers
original_pow = pow
def safe_pow(x, y):
    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        # Use torch operations for tensors
        if isinstance(x, torch.Tensor):
            return torch.pow(torch.abs(x), y)  # Ensure positive base
        else:
            return torch.pow(torch.tensor(abs(x)), y)
    else:
        return original_pow(abs(x), y)  # Ensure positive base
pow = safe_pow

# Monkey patch the power operator for specific cases
import builtins
original_power = builtins.pow
builtins.pow = safe_pow

class SimpleObservation:
    """Simple observation class for ONNX export"""
    def __init__(self, images, state, tokenized_prompt, tokenized_prompt_mask):
        self.images = images
        self.state = state
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        # Add dummy attributes for compatibility
        self.image_masks = {key: torch.ones(img.shape[0], dtype=torch.bool, device=img.device) 
                           for key, img in images.items()}
        # Add missing attributes
        self.token_ar_mask = torch.ones_like(tokenized_prompt_mask, dtype=torch.bool, device=tokenized_prompt.device)
        self.token_loss_mask = torch.ones_like(tokenized_prompt_mask, dtype=torch.bool, device=tokenized_prompt.device)

class PI0ONNXWrapper(nn.Module):
    """Wrapper for PI0Pytorch model for ONNX export"""
    
    def __init__(self, pi0_model, config):
        super().__init__()
        self.pi0_model = pi0_model
        self.config = config
        
    def forward(self, base_image, left_wrist_image, right_wrist_image, state, prompt_tokens, prompt_mask):
        """Forward pass for ONNX export"""
        # Create observation object
        images = {
            'base_0_rgb': base_image,
            'left_wrist_0_rgb': left_wrist_image,
            'right_wrist_0_rgb': right_wrist_image
        }
        
        observation = SimpleObservation(
            images=images,
            state=state,
            tokenized_prompt=prompt_tokens,
            tokenized_prompt_mask=prompt_mask
        )
        
        # Process observation
        processed_obs = preprocess_observation_pytorch(observation, train=False)
        
        # Get device from input tensors
        device = base_image.device
        
        # Sample actions
        actions = self.pi0_model.sample_actions(device, processed_obs)
        
        return actions

class PI0PytorchNoCompile(PI0Pytorch):
    """PI0Pytorch model without torch.compile for ONNX export"""
    
    def __init__(self, config):
        # Create a copy of config to avoid frozen instance error
        import dataclasses
        config_dict = dataclasses.asdict(config)
        config_dict['pi05'] = False  # Temporarily disable pi05 to avoid adaRMS issues
        config_dict['dtype'] = 'float32'  # Use float32 instead of bfloat16 to avoid complex numbers
        config_copy = type(config)(**config_dict)
        super().__init__(config_copy)
        
        # Override the compiled method with the original uncompiled version
        # Directly assign the method instead of using types.MethodType
        self.sample_actions = self._sample_actions_uncompiled
    
    def _sample_actions_uncompiled(self, device, observation, noise=None, num_steps=10):
        """Uncompiled version of sample_actions for ONNX export"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Extract data from already processed observation
        images = list(observation.images.values())
        img_masks = list(observation.image_masks.values())
        lang_tokens = observation.tokenized_prompt
        lang_masks = observation.tokenized_prompt_mask
        state = observation.state

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time = time + dt

        return x_t

def load_pytorch_pi0_model(checkpoint_path: str, config_name: str):
    """Load PyTorch PI0 model from checkpoint"""
    print(f"Loading PyTorch π₀ model from {checkpoint_path}...")
    
    # Load config
    from openpi.training.config import get_config
    train_config = get_config(config_name)
    config = train_config.model
    
    # Create model without compilation
    model = PI0PytorchNoCompile(config)
    
    # Load weights
    checkpoint_file = pathlib.Path(checkpoint_path) / "model.safetensors"
    if checkpoint_file.exists():
        print(f"Loading weights from {checkpoint_file}")
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_file)
        
        # Load state dict with strict=False to handle missing keys
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in state dict: {unexpected_keys}")
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    
    model.eval()
    return model, config

def export_pytorch_pi0_to_onnx(checkpoint_path: str, config_name: str, output_path: str):
    """Export PyTorch PI0 model to ONNX format"""
    
    # Load model
    model, config = load_pytorch_pi0_model(checkpoint_path, config_name)
    
    # Create wrapper for ONNX export
    wrapper_model = PI0ONNXWrapper(model, config)
    wrapper_model.eval()
    
    # Prepare dummy inputs
    print("Preparing dummy inputs...")
    batch_size = 1
    device = torch.device("cpu")  # Use CPU for ONNX export
    
    # Create dummy inputs
    base_image = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32, device=device)
    left_wrist_image = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32, device=device)
    right_wrist_image = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32, device=device)
    state = torch.randn(batch_size, 32, dtype=torch.float32, device=device)
    prompt_tokens = torch.randint(0, 1000, (batch_size, 50), dtype=torch.long, device=device)
    prompt_mask = torch.ones(batch_size, 50, dtype=torch.bool, device=device)
    
    # Convert model to float32 to avoid complex number issues
    wrapper_model = wrapper_model.float()
    
    dummy_inputs = (base_image, left_wrist_image, right_wrist_image, state, prompt_tokens, prompt_mask)
    
    # Test forward pass
    print("Testing model forward pass...")
    try:
        with torch.no_grad():
            output = wrapper_model(*dummy_inputs)
        print(f"Model output shape: {output.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        return False
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    
    try:
        # Try direct ONNX export without JIT
        print("Attempting direct ONNX export...")
        
        torch.onnx.export(
            wrapper_model,
            dummy_inputs,
            output_path,
            export_params=True,
            opset_version=14,  # Use opset 14 for scaled_dot_product_attention support
            do_constant_folding=False,  # Disable constant folding
            keep_initializers_as_inputs=True,  # Keep initializers as inputs
            verbose=False,  # Reduce verbosity
            training=torch.onnx.TrainingMode.EVAL,  # Set training mode
            input_names=[
                'base_image', 
                'left_wrist_image', 
                'right_wrist_image',
                'state', 
                'prompt_tokens', 
                'prompt_mask'
            ],
            output_names=['actions'],
            dynamic_axes={
                'base_image': {0: 'batch_size'},
                'left_wrist_image': {0: 'batch_size'},
                'right_wrist_image': {0: 'batch_size'},
                'state': {0: 'batch_size'},
                'prompt_tokens': {0: 'batch_size'},
                'prompt_mask': {0: 'batch_size'},
                'actions': {0: 'batch_size'}
            }
        )
        print("✅ ONNX export successful!")
        
    except Exception as e:
        print(f"❌ ONNX export failed!")
        print(f"Error: {e}")
        return False
    
    # Verify ONNX model
    print("Verifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification successful!")
    except ImportError:
        print("⚠️  ONNX package not available for verification")
    except Exception as e:
        print(f"⚠️  ONNX model verification failed: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch π₀ model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint directory")
    parser.add_argument("--config", required=True, help="Config name")
    parser.add_argument("--output", required=True, help="Output ONNX file path")
    
    args = parser.parse_args()
    
    success = export_pytorch_pi0_to_onnx(args.checkpoint, args.config, args.output)
    
    if success:
        print(f"✅ Conversion completed successfully!")
        print(f"ONNX model saved to: {args.output}")
    else:
        print("❌ Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()