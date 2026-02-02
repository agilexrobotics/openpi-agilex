import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


USE_ROW_POSE6D = False


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


from collections.abc import Callable, Mapping, Sequence
import dataclasses
import math
import re
from typing import ClassVar, Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


def matrix_to_xyzrpy(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    print(matrix)
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

xyzrpy = np.array([0.5, 0.1, 0.2, 0.2, 0.3, 0.1])
print("xyzrpy", xyzrpy)   
matrix = xyzrpy_to_matrix(xyzrpy)
print("matrix", matrix)
pose6d = matrix_to_pose6d(matrix)
print("pose6d", pose6d)
matrix = pose6d_to_matrix(pose6d)
print("matrix", matrix)
xyzrpy = matrix_to_xyzrpy(matrix)
print("xyzrpy", xyzrpy)