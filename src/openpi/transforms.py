from collections.abc import Callable, Mapping, Sequence
import dataclasses
import math
import re
from typing import ClassVar, Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import image_tools
from openpi.shared import image_tools as jax_image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


# 全局参数：6D pose 的格式
# True: 按行取旋转矩阵 [x, y, z, r00, r01, r10, r11, r20, r21]
# False: 按列取旋转矩阵 [x, y, z, r00, r10, r20, r01, r11, r21]
USE_ROW_POSE6D = False


T = TypeVar("T")
S = TypeVar("S")


def matrix_to_xyzrpy(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(-matrix[2, 0])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return np.array([x, y, z, roll, pitch, yaw])


def xyzrpy_to_matrix(xyzrpy):
    x = xyzrpy[0]
    y = xyzrpy[1]
    z = xyzrpy[2]
    roll = xyzrpy[3]
    pitch = xyzrpy[4]
    yaw = xyzrpy[5]
    transformation_matrix = np.eye(4)
    A = np.cos(yaw)
    B = np.sin(yaw)
    C = np.cos(pitch)
    D = np.sin(pitch)
    E = np.cos(roll)
    F = np.sin(roll)
    DE = D * E
    DF = D * F
    transformation_matrix[0, 0] = A * C
    transformation_matrix[0, 1] = A * DF - B * E
    transformation_matrix[0, 2] = B * F + A * DE
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 0] = B * C
    transformation_matrix[1, 1] = A * E + B * DF
    transformation_matrix[1, 2] = B * DE - A * F
    transformation_matrix[1, 3] = y
    transformation_matrix[2, 0] = -D
    transformation_matrix[2, 1] = C * F
    transformation_matrix[2, 2] = C * E
    transformation_matrix[2, 3] = z
    transformation_matrix[3, 0] = 0
    transformation_matrix[3, 1] = 0
    transformation_matrix[3, 2] = 0
    transformation_matrix[3, 3] = 1
    return transformation_matrix


def pose6d_to_matrix(pose: np.ndarray) -> np.ndarray:
    """
    将6D位姿转回4x4齐次变换矩阵
    
    Args:
        pose: [9,] 位姿向量 (3个平移 + 6个旋转)
              格式取决于 USE_ROW_POSE6D:
              - True:  [x, y, z, r00, r01, r10, r11, r20, r21] (按行)
              - False: [x, y, z, r00, r10, r20, r01, r11, r21] (按列)
        
    Returns:
        matrix: [4, 4] 齐次变换矩阵
    """
    assert pose.shape == (9,), "位姿必须是9维向量"
    
    # 1. 提取平移和旋转6D
    translation = pose[:3]  # [x, y, z]
    rot6d_flat = pose[3:]  # [6,]
    
    # 2. 根据格式重建旋转矩阵
    # 注意：无论哪种格式，都需要重建为两个3D列向量，然后叉乘得到第三列
    if USE_ROW_POSE6D:
        # 按行展平格式: [r00, r01, r10, r11, r20, r21]
        # 先重组为 (3, 2)，然后按列提取
        rot6d = rot6d_flat.reshape(3, 2)  # [[r00, r01], [r10, r11], [r20, r21]]
        col0 = rot6d[:, 0]  # [r00, r10, r20] - 第一列
        col1 = rot6d[:, 1]  # [r01, r11, r21] - 第二列
    else:
        # 按列展平格式: [r00, r10, r20, r01, r11, r21]
        # 先重组为 (2, 3)，然后转置再按列提取
        rot6d = rot6d_flat.reshape(2, 3)  # [[r00, r10, r20], [r01, r11, r21]]
        col0 = rot6d[0, :]  # [r00, r10, r20] - 第一列
        col1 = rot6d[1, :]  # [r01, r11, r21] - 第二列
    
    # 通过叉乘计算第三列（对两种格式都一样）
    col2 = np.cross(col0, col1)
    
    # 确保正交性（数值稳定性）
    col1 = np.cross(col2, col0)  # 重新正交化
    
    # 归一化
    col0 = col0 / np.linalg.norm(col0)
    col1 = col1 / np.linalg.norm(col1)
    col2 = col2 / np.linalg.norm(col2)
    
    # 组装旋转矩阵
    rotation_matrix = np.stack([col0, col1, col2], axis=1)  # [3, 3]
    
    # 3. 构建4x4矩阵
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation
    
    return matrix

def matrix_to_pose6d(matrix: np.ndarray) -> np.ndarray:
    """
    从4x4齐次变换矩阵提取6D位姿
    
    Args:
        matrix: [4, 4] 齐次变换矩阵
        
    Returns:
        pose: [9,] 6D位姿向量 (3个平移 + 6个旋转)
              格式取决于 USE_ROW_POSE6D:
              - True:  [x, y, z, r00, r01, r10, r11, r20, r21] (旋转矩阵前两行)
              - False: [x, y, z, r00, r10, r20, r01, r11, r21] (旋转矩阵前两列)
    """
    # 1. 提取平移
    translation = matrix[:3, 3]  # [x, y, z]
    
    # 2. 提取旋转矩阵并转换为6D表示
    rotation_matrix = matrix[:3, :3]
    
    if USE_ROW_POSE6D:
        # 按行取: 前两列按行展平
        # [[r00, r01, r02],     取前两列 →  [[r00, r01],
        #  [r10, r11, r12],                   [r10, r11],
        #  [r20, r21, r22]]                   [r20, r21]]
        # 按行展平 → [r00, r01, r10, r11, r20, r21]
        rotation_6d = rotation_matrix[:, :2].reshape(-1)  # [6,]
    else:
        # 按列取: 前两列按列展平
        # [[r00, r01, r02],     取前两列 →  [[r00, r01],
        #  [r10, r11, r12],                   [r10, r11],
        #  [r20, r21, r22]]                   [r20, r21]]
        # 转置后按行展平 → [r00, r10, r20, r01, r11, r21]
        rotation_6d = rotation_matrix[:, :2].T.reshape(-1)  # [6,]
    
    # 3. 组合
    pose_6d = np.concatenate([translation, rotation_6d])  # [9,]
    return pose_6d


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    
    Missing keys in the input data will raise a KeyError.
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        
        def get_value(k: str):
            if k in flat_item:
                return flat_item[k]
            # Raise error for missing keys instead of returning None
            raise KeyError(f"Missing key in input data: {k}")
        
        result = jax.tree.map(get_value, self.structure)
        return result


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will require all keys in the norm stats to be present in the data.
    # Set to False when unnormalizing outputs where only a subset of keys (e.g., actions) are present.
    strict: bool = True

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=self.strict,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class ResizeDepths(DataTransformFn):
    """Resize depth images to a fixed resolution, analogous to ResizeImages.

    Expects depth images to be stored under data["depths"] as a dict[name, img]
    with shape [..., h, w, c]. This does NOT mix depth into RGB images.
    
    Uses JAX-based resize_with_pad to support float32 depth images.
    """

    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        # If no depths field is present, this is a no-op.
        if "depths" not in data or data["depths"] is None:
            return data

        data["depths"] = {
            k: self._resize_depth(v, self.height, self.width)
            for k, v in data["depths"].items()
        }
        return data
    
    def _resize_depth(self, depth_img: np.ndarray, height: int, width: int) -> np.ndarray:
        """Resize depth image, handling both uint8 and float32 types."""
        # Convert numpy array to JAX array
        depth_jax = jnp.asarray(depth_img)
        # Use JAX-based resize_with_pad which supports float32
        resized_jax = jax_image_tools.resize_with_pad(depth_jax, height, width)
        # Convert back to numpy array
        return np.asarray(resized_jax)


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        # data["state_original"] = data["state"]
        # state = data["state"].copy()
        # state[6] = 0
        # data["state"] = state
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        # state = data["state_original"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class PoseInputs(DataTransformFn):
    """Converts absolute pose state/actions into xyz + 6D rotation + gripper per arm."""

    left_arm_start: int = 0
    left_arm_end: int = 7
    right_arm_start: int = 7
    right_arm_end: int = 14

    def __call__(self, data: DataDict) -> DataDict:
        # if "actions" not in data or "state" not in data:
        #     raise ValueError("Actions and state are required")
        def encode_arm(arm_array: np.ndarray) -> np.ndarray:
            pose6ds = []
            for row in arm_array:
                mat = xyzrpy_to_matrix(row[:6])
                pose6ds.append(matrix_to_pose6d(mat))
            pose6ds = np.stack(pose6ds, axis=0)
            gripper = arm_array[:, 6:7]
            return np.concatenate([pose6ds, gripper], axis=-1)
        if "state" in data:
            state = np.asarray(data["state"])[np.newaxis, :]
            left_state = state[:, self.left_arm_start : self.left_arm_end]
            right_state = state[:, self.right_arm_start : self.right_arm_end]
            left_state_features = encode_arm(left_state)
            right_state_features = encode_arm(right_state)
            new_state = np.concatenate([left_state_features, right_state_features], axis=-1).squeeze(0)
            data["state"] = new_state
        if "actions" in data:
            actions = np.asarray(data["actions"])
            left_actions = actions[:, self.left_arm_start : self.left_arm_end]
            right_actions = actions[:, self.right_arm_start : self.right_arm_end]
            left_action_features = encode_arm(left_actions)
            right_action_features = encode_arm(right_actions)
            new_actions = np.concatenate([left_action_features, right_action_features], axis=-1)
            data["actions"] = new_actions

        return data


@dataclasses.dataclass(frozen=True)
class PoseOutputs(DataTransformFn):
    """Converts absolute pose state/actions into xyz + 6D rotation + gripper per arm."""

    left_arm_start: int = 0
    left_arm_end: int = 10
    right_arm_start: int = 10
    right_arm_end: int = 20

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            raise ValueError("Actions are required")

        def encode_arm(arm_array: np.ndarray) -> np.ndarray:
            xyzrpy_features = []
            for row in arm_array:
                xyzrpy = matrix_to_xyzrpy(pose6d_to_matrix(row[:9]))
                xyzrpy_features.append(xyzrpy)
            xyzrpy = np.stack(xyzrpy_features, axis=0)
            gripper = arm_array[:, 9:10]
            return np.concatenate([xyzrpy, gripper], axis=-1)

        actions = np.asarray(data["actions"])
        left_actions = actions[:, self.left_arm_start : self.left_arm_end]
        right_actions = actions[:, self.right_arm_start : self.right_arm_end]
        left_action_features = encode_arm(left_actions)
        right_action_features = encode_arm(right_actions)
        new_actions = np.concatenate([left_action_features, right_action_features], axis=-1)
        data["actions"] = new_actions

        return data


@dataclasses.dataclass(frozen=True)
class PoseDeltaActions(DataTransformFn):
    """Repacks absolute pose actions into delta pose action space with 6D pose representation.
    
    Input format: left_arm (6D pose + gripper) + right_arm (6D pose + gripper) = 20 dims
    Output format: left_arm (delta_6d_pose + delta_gripper) + right_arm (delta_6d_pose + delta_gripper) = 20 dims
    
    The transform:
    1. Extracts 6D pose from actions and state for both arms
    2. Converts to transformation matrices
    3. Computes delta pose matrices (inv(state_matrix) * action_matrix)
    4. Converts delta rotation to 6D representation (rotation matrix first 2 columns)
    5. Outputs: delta_xyz (3) + delta_rotation_6d (6) + delta_gripper (1) for each arm
    """

    # Indices for left arm and right arm in actions/state
    # left_arm: [start, end) for 6D pose+gripper (10 dims)
    # right_arm: [start, end) for 6D pose+gripper (10 dims)
    mask: Sequence[bool] | None
    left_arm_start: int = 0
    left_arm_end: int = 10
    right_arm_start: int = 10
    right_arm_end: int = 20

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        actions = np.asarray(data["actions"])
        state = np.asarray(data["state_original"] if "state_original" in data else data["state"])

        # Extract left and right arm poses from actions
        left_arm_actions = actions[:, self.left_arm_start:self.left_arm_end]  # [N, 10]
        right_arm_actions = actions[:, self.right_arm_start:self.right_arm_end]  # [N, 10]

        # Extract left and right arm poses from state (state is 1D: [20])
        left_arm_state = state[self.left_arm_start:self.left_arm_end]  # [10]
        right_arm_state = state[self.right_arm_start:self.right_arm_end]  # [10]

        # Process each arm
        def process_arm(arm_action, arm_state, gripper_delta):
            """Process a single arm: convert 6D pose+gripper to delta_6d_pose + delta_gripper."""
            N = arm_action.shape[0]
            result = np.zeros((N, 10))  # xyz(3) + 6d_pose(6) + gripper(1)
            state_gripper = arm_state[9]
            state_matrix = pose6d_to_matrix(arm_state[:9])
            for i in range(N):
                action_gripper = arm_action[i, 9]
                action_matrix = pose6d_to_matrix(arm_action[i, :9])

                # Compute delta pose matrix: delta = inv(state) * action
                state_matrix_inv = np.linalg.inv(state_matrix)
                delta_matrix = np.dot(state_matrix_inv, action_matrix)
                result[i, 0:9] = matrix_to_pose6d(delta_matrix)
                if gripper_delta:
                    result[i, 9] = action_gripper - state_gripper  # Delta gripper
                else:
                    result[i, 9] = action_gripper

            return result

        # Process both arms
        left_arm_result = process_arm(left_arm_actions, left_arm_state, self.mask[self.left_arm_end - 1])
        right_arm_result = process_arm(right_arm_actions, right_arm_state, self.mask[self.right_arm_end - 1])

        # Concatenate results
        new_actions = np.concatenate([left_arm_result, right_arm_result], axis=-1)  # [N, 20]
        data["actions"] = new_actions
        return data


@dataclasses.dataclass(frozen=True)
class PoseAbsoluteActions(DataTransformFn):
    """Converts delta pose actions back into absolute 6D pose + gripper actions.

    Input format per arm: delta_xyz (3) + delta_rotation_6d (6) + delta_gripper (1) = 10 dims
    Output format per arm: absolute 6D pose (9) + gripper (1) = 10 dims
    """
    mask: Sequence[bool] | None
    left_arm_start: int = 0
    left_arm_end: int = 10
    right_arm_start: int = 10
    right_arm_end: int = 20

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or ("state" not in data and "state_original" not in data):
            raise ValueError("Actions and state are required")

        actions = np.asarray(data["actions"])
        state = np.asarray(data["state_original"] if "state_original" in data else data["state"])

        # Split delta actions per arm
        left_arm_delta = actions[:, self.left_arm_start : self.left_arm_end]
        right_arm_delta = actions[:, self.right_arm_start : self.right_arm_end]

        # Extract state slices (state is 1D: [20])
        left_arm_state = state[self.left_arm_start : self.left_arm_end]  # [10]
        right_arm_state = state[self.right_arm_start : self.right_arm_end]  # [10]

        def process_arm(delta_arm, arm_state, gripper_delta):
            """Convert delta pose back to absolute 6D pose + gripper."""
            N = delta_arm.shape[0]
            result = np.zeros((N, 10))  # 6D pose (9) + gripper (1)
            state_gripper = arm_state[9]
            state_matrix = pose6d_to_matrix(arm_state[:9])
            for i in range(N):
                delta_gripper = delta_arm[i, 9]
                delta_matrix = pose6d_to_matrix(delta_arm[i, :9])
                # Recover absolute pose: action = state @ delta
                action_matrix = np.dot(state_matrix, delta_matrix)
                result[i, :9] = matrix_to_pose6d(action_matrix)
                if gripper_delta:
                    result[i, 9] = state_gripper + delta_gripper
                else:
                    result[i, 9] = delta_gripper
            return result
        left_arm_abs = process_arm(left_arm_delta, left_arm_state, self.mask[self.left_arm_end - 1])
        right_arm_abs = process_arm(right_arm_delta, right_arm_state, self.mask[self.right_arm_end - 1])

        new_actions = np.concatenate([left_arm_abs, right_arm_abs], axis=-1)  # [N, 20]

        data["actions"] = new_actions
        return data


@dataclasses.dataclass(frozen=True)
class RelativePoseStateInputs(DataTransformFn):
    """Converts absolute pose state into xyz + 6D rotation + gripper per arm."""

    left_arm_start: int = 0
    left_arm_end: int = 10
    right_arm_start: int = 10
    right_arm_end: int = 20

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data
        data["state_original"] = data["state"]
        state = np.asarray(data["state"])
        
        # Extract state slices (state is 1D: [20])
        left = state[self.left_arm_start : self.left_arm_end]  # [10]
        right = state[self.right_arm_start : self.right_arm_end]  # [10]

        def encode_arm(arm_state: np.ndarray, rel) -> np.ndarray:
            # Convert 6D pose to matrix
            mat_rel = pose6d_to_matrix(rel[:9])
            mat_rel_inv = np.linalg.inv(mat_rel)
            mat = pose6d_to_matrix(arm_state[:9])
            mat = np.dot(mat_rel_inv, mat)
            pose6d = matrix_to_pose6d(mat)
            gripper = arm_state[9:10]
            return np.concatenate([pose6d, gripper], axis=-1)

        left_features = encode_arm(left, left)
        right_features = encode_arm(right, left)
        new_state = np.concatenate([left_features, right_features], axis=-1)
        data["state"] = new_state
        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
