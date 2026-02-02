from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state_original": inputs["state_original"] if inputs.get("state_original") is not None else inputs["state"],
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_with_rtc_guidance(
        self, 
        obs: dict, 
        prev_action_chunk: np.ndarray,
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 1.0,
        delta_action_mask: np.ndarray | None = None,
        executed_steps: int = 0
    ) -> dict:
        """Infer with real-time chunking guidance.
        
        Note: Parameter order matches model.sample_actions_with_rtc_guidance exactly
        for consistent interface across websocket -> policy -> model layers.
        """
        if not self._has_rtc_support:
            raise ValueError("Model does not support RTC guidance")
            
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        
        # 重要：prev_action_chunk需要进行标准化处理
        # prev_action_chunk是上一次推理的输出（已经过output_transform反标准化）
        # 我们需要将其重新标准化到模型内部使用的空间
        
        # RTC数据预处理：这里只负责数据准备，具体约束逻辑在模型中
        prev_chunk_jax, delta_action_mask = self._preprocess_prev_action_for_rtc(
            prev_action_chunk, obs, executed_steps, delta_action_mask
        )

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        
        # 调试信息：检查输入参数
        print(f"RTC Debug: inference_delay={inference_delay}, prefix_attention_horizon={prefix_attention_horizon}")
        print(f"RTC Debug: executed_steps={executed_steps}, max_guidance_weight={max_guidance_weight}")
        print(f"RTC Debug: prev_chunk_shape={prev_chunk_jax.shape}")
        
        try:
            # 确保参数顺序与模型函数签名完全一致
            actions = self._sample_actions_with_rtc(
                sample_rng, 
                _model.Observation.from_dict(inputs),
                prev_chunk_jax,
                inference_delay,
                prefix_attention_horizon,
                prefix_attention_schedule,
                max_guidance_weight,
                jnp.asarray(delta_action_mask) if delta_action_mask is not None else None,
                executed_steps,
                **self._sample_kwargs
            )
            
            # 检查输出是否包含NaN
            if jnp.any(jnp.isnan(actions)):
                print("RTC Warning: Generated actions contain NaN values, using fallback")
                # 作为fallback，使用标准推理
                actions = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs)
            
            outputs = {
                "state_original": inputs["state_original"] if inputs.get("state_original") is not None else inputs["state"],
                "state": inputs["state"],
                "actions": actions,
            }
            
            print(f"RTC Debug: actions_shape={actions.shape}, contains_nan={jnp.any(jnp.isnan(actions))}")
            
        except Exception as e:
            print(f"RTC Error: {e}, falling back to standard inference")
            # Fallback to standard inference
            outputs = {
                "state_original": inputs["state_original"] if inputs.get("state_original") is not None else inputs["state"],
                "state": inputs["state"],
                "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
            }
        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _preprocess_prev_action_for_rtc(
        self, 
        prev_action_chunk: np.ndarray, 
        obs: dict, 
        executed_steps: int, 
        delta_action_mask: np.ndarray | None
    ) -> tuple[jax.Array, np.ndarray | None]:
        """
        预处理prev_action_chunk用于RTC约束：
        1. 根据executed_steps切片取未执行部分
        2. 维度匹配和填充
        3. Delta处理
        4. 标准化
        
        Args:
            prev_action_chunk: 原始前一次动作序列 [chunk_size, action_dim]
            obs: 当前观测
            executed_steps: 已执行的步数
            delta_action_mask: Delta处理的掩码
            
        Returns:
            tuple: (处理后的动作JAX数组 [1, remaining_steps, action_dim], 处理后的掩码)
        """
        chunk_size = prev_action_chunk.shape[0]
        remaining_steps = chunk_size - executed_steps
        
        print(f"RTC Preprocess: chunk_size={chunk_size}, executed_steps={executed_steps}, remaining_steps={remaining_steps}")
        
        # 1. 取未执行的部分：后面的 remaining_steps 对应新推理的前 remaining_steps
        if executed_steps > 0 and remaining_steps > 0:
            prev_action_unexecuted = prev_action_chunk[executed_steps:, :]  # [remaining_steps, action_dim]
            print(f"RTC Preprocess: Sliced prev_action to unexecuted part, shape: {prev_action_unexecuted.shape}")
        elif remaining_steps > 0:
            prev_action_unexecuted = prev_action_chunk  # 没有执行任何步骤
        else:
            # 所有步骤都已执行，返回空数组
            print("RTC Preprocess: All steps executed, returning empty constraint")
            return jnp.zeros((1, 0, self._model.action_dim)), delta_action_mask
        
        # 2. 维度匹配：从14维扩展到32维
        if prev_action_unexecuted.shape[-1] < self._model.action_dim:
            pad_width = self._model.action_dim - prev_action_unexecuted.shape[-1]
            prev_action_padded = np.pad(prev_action_unexecuted, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            print(f"RTC Preprocess: Padded action from {prev_action_unexecuted.shape[-1]} to {prev_action_padded.shape[-1]} dims")
            
            # 同时填充delta_action_mask
            if delta_action_mask is not None:
                if not isinstance(delta_action_mask, np.ndarray):
                    delta_action_mask = np.asarray(delta_action_mask)
                
                if delta_action_mask.shape[-1] < self._model.action_dim:
                    delta_mask_pad_width = self._model.action_dim - delta_action_mask.shape[-1]
                    delta_action_mask = np.pad(delta_action_mask, (0, delta_mask_pad_width), mode='constant', constant_values=False)
                    print(f"RTC Preprocess: Padded delta_mask from {delta_action_mask.shape[-1] - delta_mask_pad_width} to {delta_action_mask.shape[-1]} dims")
        else:
            prev_action_padded = prev_action_unexecuted
        
        # 3. Delta处理 + 标准化：使用原始状态进行正确的Delta计算
        try:
            # 使用原始状态，但需要确保维度匹配
            raw_state = obs["state"]  # 原始未处理的state
            
            print(f"RTC Delta Debug: === 开始Delta处理 ===")
            print(f"RTC Delta Debug: Raw state shape: {raw_state.shape}")
            print(f"RTC Delta Debug: Raw state values: {raw_state}")
            
            # 确保state也被填充到与actions相同的维度
            if raw_state.shape[-1] < self._model.action_dim:
                state_pad_width = self._model.action_dim - raw_state.shape[-1]
                raw_state_padded = np.pad(raw_state, (0, state_pad_width), mode='constant', constant_values=0)
                print(f"RTC Delta Debug: Padded state from {raw_state.shape[-1]} to {raw_state_padded.shape[-1]} dims")
            else:
                raw_state_padded = raw_state
            
            print(f"RTC Delta Debug: Padded state shape: {raw_state_padded.shape}")
            print(f"RTC Delta Debug: Prev action BEFORE delta shape: {prev_action_unexecuted.shape}")
            print(f"RTC Delta Debug: === All timesteps BEFORE delta ===")
            for i in range(min(prev_action_unexecuted.shape[0], 20)):  # 显示前20个时间步，或全部如果少于20
                print(f"  Step {i:2d}: {prev_action_unexecuted[i, :8]} ... (showing first 8 dims)")
            if prev_action_unexecuted.shape[0] > 20:
                print(f"  ... (showing first 20 of {prev_action_unexecuted.shape[0]} total steps)")
            
            # 手动展示Delta计算过程（用于调试）
            if delta_action_mask is not None:
                mask = np.asarray(delta_action_mask)
                print(f"RTC Delta Debug: Delta action mask (14-dim): {mask[:14]}")
                print(f"RTC Delta Debug: Delta action mask (full): {mask}")
                
                # 展示哪些维度会被Delta处理
                delta_dims = [i for i, m in enumerate(mask) if m and i < 14]  # 只看14维的真实维度
                non_delta_dims = [i for i, m in enumerate(mask) if not m and i < 14]
                print(f"RTC Delta Debug: Dimensions WITH delta: {delta_dims}")
                print(f"RTC Delta Debug: Dimensions WITHOUT delta: {non_delta_dims}")
                
                # 展示详细的Delta计算示例
                if len(delta_dims) > 0:
                    print(f"RTC Delta Debug: === Delta calculation examples ===")
                    print(f"RTC Delta Debug: Current state values: {raw_state}")
                    
                    for t in range(min(3, prev_action_unexecuted.shape[0])):  # 前3个时间步
                        print(f"  --- Timestep {t} ---")
                        print(f"    Original action: {prev_action_unexecuted[t, :14]}")
                        delta_results = []
                        for dim in range(14):
                            if dim in delta_dims:
                                original = prev_action_unexecuted[t, dim]
                                state_val = raw_state[dim]
                                delta_result = original - state_val
                                delta_results.append(delta_result)
                                if dim < 8:  # 只详细打印前8维
                                    print(f"    Dim {dim} (delta): {original:.6f} - {state_val:.6f} = {delta_result:.6f}")
                            else:
                                delta_results.append(prev_action_unexecuted[t, dim])
                        print(f"    Expected delta result: {np.array(delta_results)}")
            
            # 创建包含原始状态和原始动作的完整观测结构，用于delta+normalize处理
            # 注意：使用原始的14维state和14维actions，让变换管道自己处理维度匹配和填充
            temp_obs_with_actions = {
                "state": raw_state,  # 原始14维状态
                "images": obs.get("images", {}),
                "actions": prev_action_unexecuted,  # 原始14维动作（未填充）
                "prompt": obs.get("prompt", "null")
            }
            
            print(f"RTC Delta Debug: Using original 14-dim state and actions for transform pipeline")
            print(f"RTC Delta Debug: Original actions shape: {prev_action_unexecuted.shape}")
            print(f"RTC Delta Debug: Original state shape: {raw_state.shape}")
            
            # 应用完整的输入变换（DeltaActions + Normalize）
            # 这会先计算 actions - state (Delta)，然后进行标准化
            temp_transformed = self._input_transform(temp_obs_with_actions)
            processed_prev_chunk = jnp.asarray(temp_transformed["actions"])[np.newaxis, ...]  # [1, remaining_steps, action_dim]
            
            print(f"RTC Delta Debug: === AFTER delta+normalize ===")
            print(f"RTC Delta Debug: Final shape: {processed_prev_chunk.shape}")
            print(f"RTC Delta Debug: All timesteps AFTER delta+normalize (32-dim):")
            for i in range(min(processed_prev_chunk.shape[1], 20)):  # 显示前20个时间步
                print(f"  Step {i:2d}: {processed_prev_chunk[0, i, :8]} ... (showing first 8 of 32 dims)")
            if processed_prev_chunk.shape[1] > 20:
                print(f"  ... (showing first 20 of {processed_prev_chunk.shape[1]} total steps)")
            
            print(f"RTC Delta Debug: === Delta处理完成 ===")
            
            return processed_prev_chunk, delta_action_mask
            
        except Exception as e:
            print(f"RTC Preprocess: Transform failed: {e}")
            print(f"RTC Preprocess: Failing directly to debug the issue")
            # 直接重新抛出异常，不使用fallback
            raise e


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
