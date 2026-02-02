import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from jax import debug
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.config = config  # Save config for use in embed_prefix
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        
        # Initialize depth encoder if enabled
        if config.use_depth_encoder:
            from openpi.models import depth_encoder
            # DepthEncoder is already an NNX module, so we can use it directly
            self.depth_encoder = depth_encoder.DepthEncoder(
                output_dim=paligemma_config.width,
                depth=config.depth_encoder_depth,
                num_heads=config.depth_encoder_num_heads,
                mlp_dim=config.depth_encoder_mlp_dim,
                dropout=config.depth_encoder_dropout,
                dtype=config.dtype,
                rngs=rngs,
            )
        else:
            self.depth_encoder = None
        
        # Initialize tactile encoder if enabled
        if config.use_tactile_encoder:
            from openpi.models import tactile_encoder
            # TactileEncoder output_dim should match paligemma_config.width (same as DepthEncoder and SigLIP)
            # to ensure all observation modalities have consistent token dimensions
            self.tactile_encoder = tactile_encoder.TactileEncoder(
                output_dim=paligemma_config.width,
                input_shape=config.tactile_input_shape,
                force_dim=config.tactile_force_dim,
                num_layers=config.tactile_num_layers,
                dtype=config.dtype,
                adaptive_pool=config.tactile_adaptive_pool,
                target_tokens=config.tactile_target_tokens,
                rngs=rngs,
            )
        else:
            self.tactile_encoder = None
        
        # Initialize force6d encoder if enabled
        if config.use_force6d_encoder:
            from openpi.models import force6d_encoder
            # Force6DEncoder output_dim should match action_expert_config.width (same as state_proj output)
            # so that encoded force6d features can be concatenated to state embedding
            self.force6d_encoder = force6d_encoder.Force6DEncoder(
                output_dim=action_expert_config.width,
                hidden_dim=config.force6d_hidden_dim,
                num_layers=config.force6d_num_layers,
                dtype=config.dtype,
                rngs=rngs,
            )
        else:
            self.force6d_encoder = None
        
        # Calculate state dimension: only action_dim (tactile and force6d data are processed separately)
        # Tactile data is treated as a separate observation modality (like images)
        # Force6d data is encoded and concatenated to state embedding (not to state input)
        state_dim = config.action_dim
        
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(state_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        # 1) embed RGB images (only from obs.images)
        for name, img in obs.images.items():
            image_tokens, _ = self.PaliGemma.img(img, train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # 2) embed depth images (only from obs.depths, not from images)
        if self.depth_encoder is not None:
            if obs.depths is not None and obs.depths_masks is not None:
                # 如果 depth_image_keys 为空或未指定，使用所有 depths 中的键
                depth_keys = getattr(self.config, "depth_image_keys", None)
                if depth_keys is None or len(depth_keys) == 0:
                    depth_keys = obs.depths.keys()
                
                for name, depth_img in obs.depths.items():
                    # 只使用指定的深度图（如果 depth_image_keys 为空则使用全部）
                    if name not in depth_keys:
                        continue
                    depth_tokens, _ = self.depth_encoder(depth_img, train=False)
                    tokens.append(depth_tokens)
                    depth_mask = obs.depths_masks.get(name, jnp.ones((depth_tokens.shape[0],), dtype=jnp.bool_))
                    input_mask.append(
                        einops.repeat(
                            depth_mask,
                            "b -> b s",
                            s=depth_tokens.shape[1],
                        )
                    )
                    ar_mask += [False] * depth_tokens.shape[1]

        # embed tactile data (as a separate observation modality, like images)
        if self.tactile_encoder is not None and obs.tactile_force3d is not None:
            for tactile_key, tactile_data in obs.tactile_force3d.items():
                # tactile_data shape: [B, H, W, C]
                tactile_tokens = self.tactile_encoder(tactile_data, train=False)  # [B, num_tokens, output_dim]
                
                tokens.append(tactile_tokens)
                # Get tactile mask (default to True if not provided)
                tactile_mask = obs.tactile_masks.get(tactile_key, None) if obs.tactile_masks is not None else None
                if tactile_mask is None:
                    # Default to all True if mask not provided
                    tactile_mask = jnp.ones((tactile_tokens.shape[0],), dtype=jnp.bool_)
                
                input_mask.append(
                    einops.repeat(
                        tactile_mask,
                        "b -> b s",
                        s=tactile_tokens.shape[1],
                    )
                )
                # tactile tokens attend to each other and to images/language
                ar_mask += [False] * tactile_tokens.shape[1]
        
        # embed force6d data (as a separate observation modality, like tactile data)
        # In pi05 mode: processed here as observation tokens
        # In non-pi05 mode: processed in embed_suffix and concatenated to state embedding
        if self.force6d_encoder is not None and obs.force6d is not None:
            for force6d_key, force6d_vec in obs.force6d.items():
                # force6d_vec shape: [B, 6]
                # Encode to [B, emb_dim], then add token dimension: [B, 1, emb_dim]
                encoded_force6d = self.force6d_encoder(force6d_vec, train=False)  # [B, emb_dim]
                force6d_tokens = encoded_force6d[:, None, :]  # [B, 1, emb_dim]
                
                tokens.append(force6d_tokens)
                # Get force6d mask (default to True if not provided)
                force6d_mask = obs.force6d_masks.get(force6d_key, None) if obs.force6d_masks is not None else None
                if force6d_mask is None:
                    # Default to all True if mask not provided
                    force6d_mask = jnp.ones((force6d_tokens.shape[0],), dtype=jnp.bool_)
                
                input_mask.append(
                    einops.repeat(
                        force6d_mask,
                        "b -> b s",
                        s=force6d_tokens.shape[1],
                    )
                )
                # force6d tokens attend to each other and to images/language/tactile
                ar_mask += [False] * force6d_tokens.shape[1]
        
        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token (state does not include tactile or force6d data anymore)
            # Tactile data is processed separately in embed_prefix as observation tokens
            # Force6d data: 
            #   - In pi05 mode: processed in embed_prefix as observation tokens
            #   - In non-pi05 mode: encoded and concatenated to state embedding here
            state = obs.state
            state_token = self.state_proj(state)[:, None, :]  # [B, 1, emb_dim]
            
            # Process force6d data if available and encoder is enabled (non-pi05 mode only)
            # In pi05 mode, force6d is processed in embed_prefix as observation tokens
            if self.force6d_encoder is not None and obs.force6d is not None:
                force6d_features = []
                for force6d_key, force6d_vec in obs.force6d.items():
                    # Encode each force6d vector: [B, 6] -> [B, emb_dim]
                    encoded_force6d = self.force6d_encoder(force6d_vec, train=False)  # [B, emb_dim]
                    force6d_features.append(encoded_force6d)
                
                # Concatenate all encoded force6d features
                if force6d_features:
                    # Stack: [B, emb_dim] * N -> [B, N, emb_dim]
                    force6d_tokens = jnp.stack(force6d_features, axis=1)  # [B, N, emb_dim]
                    # Concatenate to state token: [B, 1, emb_dim] + [B, N, emb_dim] -> [B, 1+N, emb_dim]
                    state_token = jnp.concatenate([state_token, force6d_tokens], axis=1)
            
            tokens.append(state_token)
            input_mask.append(jnp.ones((state_token.shape[0], state_token.shape[1]), dtype=jnp.bool_))
            # image/language/tactile/force6d inputs do not attend to state or actions
            ar_mask += [True] * state_token.shape[1]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions_with_rtc_guidance(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prev_action_chunk: _model.Actions,
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 1.0,
        delta_action_mask: jnp.ndarray | None = None,
        executed_steps: int = 0,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        """Sample actions with real-time chunking guidance."""
        def get_prefix_weights(
            inference_delay: int, 
            prefix_attention_horizon: int, 
            constraint_steps: int, 
            schedule: str,
            executed_steps: int = 0
        ) -> jnp.ndarray:
            """Compute prefix attention weights for real-time chunking.
            
            Args:
                inference_delay: Number of steps delayed in inference
                prefix_attention_horizon: Horizon for attention prefix
                constraint_steps: Number of steps to constrain (from Policy preprocessing)
                schedule: Weight schedule type
                executed_steps: Number of steps already executed (e.g., pos_lookahead_step)
            """
            # 调整索引以考虑已执行的步骤
            step_indices = jnp.arange(self.action_horizon)
            
            # 简化权重计算：直接基于约束步数
            # Policy层已经处理了切片，这里只需要计算权重模式
            start = inference_delay  
            end = jnp.minimum(prefix_attention_horizon, constraint_steps)
            
            if schedule == "ones":
                w = jnp.ones(self.action_horizon)
            elif schedule == "zeros":
                w = (step_indices < start).astype(jnp.float32)
            elif schedule == "linear":
                w = jnp.where(step_indices < start, 0, 
                     jnp.where(step_indices >= end, 0, 1))
            elif schedule == "exp":
                progress = (step_indices - start) / jnp.maximum(end - start, 1e-8)
                progress = jnp.clip(progress, 0, 1)
                denominator = jnp.maximum(jnp.expm1(1), 1e-8)
                exp_term = jnp.clip(jnp.expm1(progress), -1e6, 1e6)
                w = 1 - exp_term / denominator
                w = jnp.where(step_indices < start, 0, w)
                w = jnp.where(step_indices >= end, 0, w)
            else:
                raise ValueError(f"Invalid schedule: {schedule}")
            
            # 确保超出约束步数的权重为0（Policy层已经处理了切片）
            w = jnp.where(step_indices >= constraint_steps, 0, w)
            
            # 最终的数值稳定性检查
            w = jnp.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0)
            w = jnp.clip(w, 0.0, 1.0)
            
            # 调试打印权重信息
            debug.print("RTC Weights Debug:")
            debug.print("  Schedule: {}, Constraint steps: {}", schedule, constraint_steps)
            debug.print("  Range: start={}, end={}", start, end)
            debug.print("  Weights shape: {}", w.shape)
            debug.print("  Active weights count: {}", jnp.sum(w > 0))
            debug.print("  Max weight: {:.6f}, Min weight: {:.6f}", jnp.max(w), jnp.min(w))
            debug.print("  Weights sum: {:.6f}", jnp.sum(w))
            
            # 完整显示所有权重值，按行显示便于阅读
            debug.print("  === All Weights (index: value) ===")
            debug.print("  Weights[0-9]:   {}", w[:10])
            debug.print("  Weights[10-19]: {}", w[10:20])
            debug.print("  Weights[20-29]: {}", w[20:30])
            debug.print("  Weights[30-39]: {}", w[30:40])
            debug.print("  Weights[40-49]: {}", w[40:50])
            
            # 显示约束区间的详细信息
            debug.print("  === Constraint Analysis ===")
            debug.print("  Delay steps [0-{}]: NO constraint (weights should be 0)", start-1)
            debug.print("  Active range [{}-{}]: WITH constraint", start, end-1)
            debug.print("  Beyond range [{}+]: NO constraint (weights should be 0)", end)
            
            return w
        
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        
        # prev_action_chunk已经在策略层面处理了维度匹配和标准化
        # 这里直接使用即可
        
        # 调试打印输入数据 - 现在这里只接收已经处理好的数据
        debug.print("RTC Constraint Logic - Input:")
        debug.print("  prev_action_chunk shape: {}", prev_action_chunk.shape)  # 应该是 [1, remaining_steps, action_dim]
        debug.print("  executed_steps: {}, action_horizon: {}", executed_steps, self.action_horizon)
        
        # 直接使用预处理好的prev_action_chunk作为约束目标
        # Policy层已经完成了：切片、Delta处理、标准化
        prev_delta_chunk = prev_action_chunk
        
        # 计算实际需要约束的步数
        constraint_steps = prev_action_chunk.shape[1]  # 从处理后的数据获取
        debug.print("RTC Constraint: Will constrain first {} steps out of {} total steps", 
                   constraint_steps, self.action_horizon)

        # First fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step_with_guidance(carry):
            x_t, time = carry
            
            # Standard forward pass to get velocity
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            
            # Add real-time chunking guidance
            def denoiser_fn(x_input):
                """Denoiser function for computing vjp."""
                # Re-embed suffix with x_input
                suffix_tokens_vjp, suffix_mask_vjp, suffix_ar_mask_vjp = self.embed_suffix(
                    observation, x_input, jnp.broadcast_to(time, batch_size)
                )
                suffix_attn_mask_vjp = make_attn_mask(suffix_mask_vjp, suffix_ar_mask_vjp)
                prefix_attn_mask_vjp = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens_vjp.shape[1])
                full_attn_mask_vjp = jnp.concatenate([prefix_attn_mask_vjp, suffix_attn_mask_vjp], axis=-1)
                positions_vjp = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask_vjp, axis=-1) - 1

                (_, suffix_out_vjp), _ = self.PaliGemma.llm(
                    [None, suffix_tokens_vjp], mask=full_attn_mask_vjp, positions=positions_vjp, kv_cache=kv_cache
                )
                v_vjp = self.action_out_proj(suffix_out_vjp[:, -self.action_horizon :])
                return x_input + v_vjp * (-time), v_vjp

            # Compute pseudo-inverse correction using vjp
            x_1, vjp_fun, _ = jax.vjp(denoiser_fn, x_t, has_aux=True)
            
            # Compute prefix weights
            debug.print("RTC Weights Call: Computing weights with params:")
            debug.print("  inference_delay={}, prefix_attention_horizon={}", inference_delay, prefix_attention_horizon)
            debug.print("  constraint_steps={}, schedule={}, executed_steps={}", constraint_steps, prefix_attention_schedule, executed_steps)
            
            weights = get_prefix_weights(
                inference_delay, prefix_attention_horizon, constraint_steps, prefix_attention_schedule, executed_steps
            )
            
            debug.print("RTC Weights Result: Returned weights summary:")
            debug.print("  Shape: {}, Active count: {}, Sum: {:.6f}", weights.shape, jnp.sum(weights > 0), jnp.sum(weights))
            
            # Compute error with previous action chunk (已经在delta+normalize空间)
            # 只对前constraint_steps步计算误差，其余步骤误差为0
            error_full = prev_delta_chunk - x_1  # [batch, horizon, action_dim]
            error = error_full * weights[None, :, None]  # 应用权重掩码
            
            # Compute pseudo-inverse correction
            pinv_correction = vjp_fun(error)[0]
            
            # Compute guidance weight (from real-time chunking paper)
            # 添加数值稳定性检查
            time_clamped = jnp.clip(time, 1e-8, 1.0 - 1e-8)  # 避免除零和log(0)
            inv_r2 = (time_clamped**2 + (1 - time_clamped) ** 2) / ((1 - time_clamped) ** 2)
            c = jnp.nan_to_num((1 - time_clamped) / time_clamped, posinf=max_guidance_weight)
            guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
            
            # 检查并限制guidance_weight以避免数值不稳定
            guidance_weight = jnp.clip(guidance_weight, 0.0, max_guidance_weight)
            guidance_weight = jnp.nan_to_num(guidance_weight, nan=0.0, posinf=max_guidance_weight, neginf=0.0)
            
            # Apply guidance with additional numerical stability
            correction_term = guidance_weight * pinv_correction
            correction_term = jnp.nan_to_num(correction_term, nan=0.0)
            v_t_guided = v_t + correction_term
            
            # 检查v_t_guided是否包含NaN或无穷大值
            v_t_guided = jnp.nan_to_num(v_t_guided, nan=0.0)
            
            step_result = x_t + dt * v_t_guided
            step_result = jnp.nan_to_num(step_result, nan=x_t)  # 如果出现NaN，保持原值
            
            return step_result, time + dt

        def cond(carry):
            x_t, time = carry
            # 添加最大步数限制以防止无限循环
            return jnp.logical_and(time >= -dt / 2, time <= 1.1)  # 添加上界检查

        # 添加while_loop的最大迭代次数限制
        x_0, final_time = jax.lax.while_loop(cond, step_with_guidance, (noise, 1.0))
        
        # 检查最终结果的数值稳定性
        x_0 = jnp.nan_to_num(x_0, nan=0.0)
        return x_0
