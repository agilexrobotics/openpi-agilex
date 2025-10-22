# openpi-agilex

Openpi holds open-source models and packages for robotics, published by the [Physical Intelligence team](https://www.physicalintelligence.company/). This repository contains the AgileX Robotics distribution of openpi, tailored for integration with our hardware. If you need more technical details, please refer to the [original repo](https://github.com/Physical-Intelligence/openpi).

## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:agilexrobotics/openpi-agilex.git 

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

## Base Model Checkpoints

We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_{0.5}$    | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning    | `gs://openpi-assets/checkpoints/pi05_base`      |V

By default, checkpoints are automatically downloaded from `gs://openpi-assets` and are cached in `~/.cache/openpi` when needed. You can overwrite the download path by setting the `OPENPI_DATA_HOME` environment variable.

## Fine-Tuning Base Models on Your Own Data

We will fine-tune the $\pi_{0.5}$ model on the ALOHA dataset as a running example for how to fine-tune a base model on your own data. We will explain three steps:
1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs 
3. Running training
4. Spinning up a policy server 
5. Running inference

### 1. Convert your data to a LeRobot dataset

We provide a minimal example script for converting ALOHA data to a LeRobot dataset in [`examples/aloha_real/convert_aloha_data_to_lerobot.py`](examples/aloha_real/convert_aloha_data_to_lerobot.py). You can easily modify it to convert your own data! You can run the script with:

```bash
uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --data_dir /path/to/your/aloah/data
```

### 2. Defining training configs

To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for ALOHA below, which you can modify for your own dataset:
- [`TrainConfig`](src/openpi/training/config.py): Defines fine-tuning hyperparameters, data config, and weight loader.

We provide example fine-tuning configs for [π₀](src/openpi/training/config.py) and [π₀.₅](src/openpi/training/config.py) on ALOHA data.

Based on the `TrainConfig` settings (located in `src/openpi/training/config.py`), configure your fine-tuning job:

* Set `name` to your custom configuration identifier (e.g., `pi0_aloha_fold_shorts` and `pi05_aloha_fold_clothes`).
* Set `repo_id` to match your dataset's directory name.
* Define a `default_prompt` for single-task datasets.
* Adjust `batch_size` according to your available GPU memory.
* Modify `num_train_steps` to meet your training duration requirements.

**To enable LoRA fine-tuning (for $\pi_0$):**
* In the `model` field, pass `paligemma_variant="gemma_2b_lora"` and `action_expert_variant="gemma_300m_lora"` to `pi0_config.Pi0config()`.
* In the `TrainConfig()`, set `ema_decay=None` and pass the corresponding `freeze_filter`:
    ```python
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter()
    ```

### Example $\pi_{0.5}$ Model Training Configuration

To adapt a $\pi_0$ configuration for $\pi_{0.5}$ training, make the following modifications. All other configurations remain the same.

* **`model` field:** Set `pi05=True` in the `pi0_config.Pi0config()` call.
* **`assets` field:** Update the `AssetsConfig()` parameter `assets_dir` to point to the $\pi_{0.5}$ base assets:
    ```python
    assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets"
    ```
* **`weight_loader` field:** Update the `weight_loaders.CheckpointWeightLoader()` path to the $\pi_{0.5}$ base model parameters:
    ```python
    "gs://openpi-assets/checkpoints/pi05_base/params"
    ```

### 3. Running training
Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_aloha_fold_clothes
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_aloha_fold_clothes --exp-name=my_experiment --overwrite
```

The command will log training progress to the console and save checkpoints to the `checkpoints` directory. You can also monitor training progress on the Weights & Biases dashboard. For maximally using the GPU memory, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before running training -- this enables JAX to use up to 90% of the GPU memory (vs. the default of 75%).

**Note:** We provide functionality for *reloading* normalization statistics for state / action normalization from pre-training. This can be beneficial if you are fine-tuning to a new task on a robot that was part of our pre-training mixture. For more details on how to reload normalization statistics, see the [norm_stats.md](docs/norm_stats.md) file.

### 4. Spinning up a policy server

Once training is complete, we can run inference by spinning up a policy server. Launching a model server is easy (we use the checkpoint for iteration 20,000 for this example, modify as needed):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_aloha_fold_clothes --policy.dir=checkpoints/pi05_aloha_fold_clothes/my_experiment/20000
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it. We can then run an evaluation script (or robot runtime) that queries the server.

### 5. Running inference
If you want to embed a policy server call in your own robot runtime, we have a minimal example of how to do so in the [remote inference docs](docs/remote_inference.md).