"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import json
import pathlib

import flax.traverse_util
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _build_key_mapping(repack_transform: transforms.RepackTransform | None) -> dict[str, str]:
    """Build reverse mapping from dataset keys to model keys.
    
    RepackTransform.structure defines the logical model keys, and these keys
    are preserved throughout the pipeline (AlohaInputsWithDepthAndTactile no
    longer renames them). The keys flow as:
    
        LeRobot dataset -> RepackTransform -> AlohaInputs -> Normalize
        
    Examples:
        - "observation.images.pikaDepthCamera" -> "images/cam_left_wrist"
        - "observation.depths.pikaDepthCamera_right_arm_tactile_left" -> "depths/tactile_depth_0"
        - "observation.force3d.right_arm_tactile_left_force" -> "tactile_force3d/tactile_force3d_0"
        - "observation.force6d.right_arm_tactile_left_forceresultant" -> "force6d/force6d_0"
    
    Returns:
        {dataset_key: model_key} mapping
    """
    key_mapping = {}
    
    if repack_transform is None:
        return key_mapping
    
    # Flatten the structure to get all mappings
    flat_structure = transforms.flatten_dict(repack_transform.structure)
    
    # Build reverse mapping: dataset_key -> model_key
    for model_key, dataset_key in flat_structure.items():
        if isinstance(dataset_key, str):
            # Keys are no longer renamed by AlohaInputs, so we use them as-is
            key_mapping[dataset_key] = model_key
    
    return key_mapping


def _load_lerobot_stats(directory: pathlib.Path, repack_transform: transforms.RepackTransform | None = None) -> dict[str, normalize.NormStats] | None:
    """Load stats from meta/stats.json (lerobot format) if it exists.
    
    Converts dataset keys to model keys using repack_transform mapping.
    
    Args:
        directory: Directory containing meta/stats.json
        repack_transform: RepackTransform containing key mappings (optional)
    
    Returns None if the file doesn't exist.
    """
    lerobot_stats_path = directory / "meta" / "stats.json"
    if not lerobot_stats_path.exists():
        return None
    
    lerobot_stats = json.loads(lerobot_stats_path.read_text())
    
    # Build key mapping from dataset keys to model keys
    key_mapping = _build_key_mapping(repack_transform)
    
    # Extract key types from repack_transform structure
    # Keys are preserved from RepackTransform, so we can directly infer types
    key_types = {}
    if repack_transform is not None:
        for top_level_key, value in repack_transform.structure.items():
            if isinstance(value, dict):
                # Nested structure: images, depths, tactile_force3d, force6d
                for sub_key in value.keys():
                    model_key = f"{top_level_key}/{sub_key}"
                    key_types[model_key] = top_level_key
            else:
                # Flat structure: state, actions
                key_types[top_level_key] = top_level_key
    
    # Convert lerobot format to openpi format
    # Lerobot format: {"key": {"mean": [...], "std": [...], "min": [...], "max": [...]}}
    # Openpi format: {"key": NormStats(mean=..., std=..., q01=..., q99=...)}
    result = {}
    
    # Process each key in lerobot_stats
    for dataset_key, stats_dict in lerobot_stats.items():
        if not isinstance(stats_dict, dict):
            continue
        
        # Skip non-statistic keys (count, etc.)
        if "mean" not in stats_dict:
            continue
        
        # Map dataset key to model key
        model_key = key_mapping.get(dataset_key, dataset_key)
        
        # Extract mean and std
        mean = np.array(stats_dict.get("mean", []))
        std = np.array(stats_dict.get("std", []))
        
        # For quantiles, use min/max if available, otherwise set to None
        q01 = None
        q99 = None
        if "min" in stats_dict and "max" in stats_dict:
            # Use min/max as approximate quantiles
            q01 = np.array(stats_dict["min"])
            q99 = np.array(stats_dict["max"])
        elif "q01" in stats_dict and "q99" in stats_dict:
            q01 = np.array(stats_dict["q01"])
            q99 = np.array(stats_dict["q99"])
        
        # Reshape statistics based on key type (inferred from repack_transform)
        # Normalize transform expects: stats[..., : x.shape[-1]]
        # Only reshape if the current shape doesn't match expectations
        
        key_type = key_types.get(model_key, None)
        
        # Handle different data types based on repack_transform structure
        if key_type == "images":
            # RGB images: stats should be (3,) for channel-wise normalization
            # Lerobot may store as (3, 1, 1), reshape to (3,) only if needed
            if mean.ndim == 3 and mean.shape[1] == 1 and mean.shape[2] == 1:
                mean = mean[:, 0, 0]  # (C, 1, 1) -> (C,)
                std = std[:, 0, 0]
                if q01 is not None:
                    q01 = q01[:, 0, 0]
                if q99 is not None:
                    q99 = q99[:, 0, 0]
            elif mean.ndim == 2 and mean.shape[1] == 1:
                mean = mean[:, 0]  # (C, 1) -> (C,)
                std = std[:, 0]
                if q01 is not None:
                    q01 = q01[:, 0]
                if q99 is not None:
                    q99 = q99[:, 0]
        
        elif key_type == "depths":
            # Depth images: stats should be (1,) for single-channel normalization
            # Lerobot may store as (3, 1, 1) or (1, 1, 1), reshape to (1,)
            if mean.ndim == 3 and mean.shape[1] == 1 and mean.shape[2] == 1:
                # Take first channel if stored as RGB
                mean = mean[0:1, 0, 0]  # (C, 1, 1) -> (1,)
                std = std[0:1, 0, 0]
                if q01 is not None:
                    q01 = q01[0:1, 0, 0]
                if q99 is not None:
                    q99 = q99[0:1, 0, 0]
            elif mean.ndim == 2 and mean.shape[1] == 1:
                mean = mean[0:1, 0]  # (C, 1) -> (1,)
                std = std[0:1, 0]
                if q01 is not None:
                    q01 = q01[0:1, 0]
                if q99 is not None:
                    q99 = q99[0:1, 0]
            elif mean.ndim == 1 and mean.shape[0] != 1:
                # Already 1D but wrong channel count, take first channel
                mean = mean[0:1]  # (C,) -> (1,)
                std = std[0:1]
                if q01 is not None:
                    q01 = q01[0:1]
                if q99 is not None:
                    q99 = q99[0:1]
        
        elif key_type == "tactile_force3d":
            # Tactile force3d: stats should be (3,) for channel-wise normalization
            # Lerobot may store as (H, W, 3), aggregate over spatial dimensions
            if mean.ndim == 3 and mean.shape[2] == 3:
                # (H, W, 3) -> (3,) by averaging over spatial dimensions
                mean = np.mean(mean, axis=(0, 1))  # (3,)
                std = np.mean(std, axis=(0, 1))  # (3,) - approximate
                if q01 is not None:
                    q01 = np.min(q01, axis=(0, 1))  # (3,) - take min across spatial dims
                if q99 is not None:
                    q99 = np.max(q99, axis=(0, 1))  # (3,) - take max across spatial dims
        
        # For force6d, state, actions: usually already correct shape (1D vector), no special handling needed
        
        # Ensure arrays are at least 1D
        if mean.ndim == 0:
            mean = mean[None]
        if std.ndim == 0:
            std = std[None]
        if q01 is not None and q01.ndim == 0:
            q01 = q01[None]
        if q99 is not None and q99.ndim == 0:
            q99 = q99[None]
        
        result[model_key] = normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)
    
    return result


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # Try to load existing stats from meta/stats.json
    output_path = config.assets_dirs / data_config.repo_id
    # Get RepackTransform from data_config to map dataset keys to model keys
    repack_transform = None
    if hasattr(data_config, 'repack_transforms') and data_config.repack_transforms:
        for transform in data_config.repack_transforms.inputs:
            if isinstance(transform, transforms.RepackTransform):
                repack_transform = transform
                break
    existing_stats = _load_lerobot_stats(output_path, repack_transform)
    
    if existing_stats is not None:
        print(f"Loaded existing stats from {output_path / 'meta' / 'stats.json'}")
        print(f"Found {len(existing_stats)} keys in existing stats")
    else:
        print(f"No existing stats found at {output_path / 'meta' / 'stats.json'}")
        existing_stats = {}

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    # Only compute stats for state and actions
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            if key in batch:
                stats[key].update(np.asarray(batch[key]))
    
    # Combine computed statistics
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    
    # Copy other keys from existing stats (excluding state and actions which must be recomputed)
    excluded_keys = {"state", "actions"}
    for key, value in existing_stats.items():
        if key not in excluded_keys:
            norm_stats[key] = value
            print(f"Copied stats for key: {key}")
    
    # Ensure state and actions are always recomputed (not copied from existing stats)
    if "state" in existing_stats:
        print("Note: 'state' stats were found in meta/stats.json but will be recomputed")
    if "actions" in existing_stats:
        print("Note: 'actions' stats were found in meta/stats.json but will be recomputed")

    print(f"Writing stats to: {output_path}")
    print(f"Total keys in output: {len(norm_stats)}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
