import json
import logging
import pathlib

import flax.traverse_util
import numpy as np
import numpydantic
import pydantic


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def _convert_lerobot_stats(lerobot_stats: dict) -> dict[str, NormStats]:
    """Convert lerobot stats.json format to openpi NormStats format.
    
    Lerobot format: {"key": {"mean": [...], "std": [...], "min": [...], "max": [...]}}
    Openpi format: {"key": NormStats(mean=..., std=..., q01=..., q99=...)}
    """
    result = {}
    flat_stats = flax.traverse_util.flatten_dict(lerobot_stats, sep="/")
    
    for key, stats_dict in flat_stats.items():
        if not isinstance(stats_dict, dict):
            continue
        
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
        
        # Ensure arrays are at least 1D
        if mean.ndim == 0:
            mean = mean[None]
        if std.ndim == 0:
            std = std[None]
        if q01 is not None and q01.ndim == 0:
            q01 = q01[None]
        if q99 is not None and q99.ndim == 0:
            q99 = q99[None]
        
        result[key] = NormStats(mean=mean, std=std, q01=q01, q99=q99)
    
    return result


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory.
    
    Tries to load in the following order:
    1. norm_stats.json (openpi format)
    2. meta/stats.json (lerobot format, will be converted)
    """
    directory = pathlib.Path(directory)
    
    # Try openpi format first
    norm_stats_path = directory / "norm_stats.json"
    if norm_stats_path.exists():
        return deserialize_json(norm_stats_path.read_text())
    
    # Try lerobot format
    lerobot_stats_path = directory / "meta" / "stats.json"
    if lerobot_stats_path.exists():
        lerobot_stats = json.loads(lerobot_stats_path.read_text())
        converted_stats = _convert_lerobot_stats(lerobot_stats)
        logging.info(f"Loaded and converted lerobot stats from {lerobot_stats_path}")
        return converted_stats
    
    raise FileNotFoundError(
        f"Norm stats file not found. Tried:\n"
        f"  - {norm_stats_path}\n"
        f"  - {lerobot_stats_path}"
    )
