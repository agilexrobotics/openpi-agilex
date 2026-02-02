import dataclasses
import enum
import logging
import os
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

# Import engine inference policy
import sys
from pathlib import Path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Jetson environments ship TensorRT Python bindings in /usr/lib/pythonX/dist-packages.
JETSON_DIST = Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")
if JETSON_DIST.exists() and str(JETSON_DIST) not in sys.path:
    sys.path.append(str(JETSON_DIST))

# from engine_inference import TensorRTEnginePolicy
# try:
#     from onnx_inference import ONNXPolicy
# except ImportError:
#     ONNXPolicy = None  # type: ignore
# try:
#     from engine_inference_split import TensorRTSplitPolicy
# except ImportError:
#     TensorRTSplitPolicy = None  # type: ignore
# try:
#     from onnx_inference_split import ONNXSplitPolicy
# except ImportError:
#     ONNXSplitPolicy = None  # type: ignore


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Engine:
    """Load a policy from a TensorRT engine file."""
    
    # Path to TensorRT engine file
    engine_path: str
    # Model config name (e.g., "pi05_aloha")
    config_name: str = "pi05_aloha"
    # Number of denoising steps
    num_steps: int = 10


@dataclasses.dataclass
class ONNX:
    """Load a policy from an ONNX model file."""
    
    # Path to ONNX model file
    onnx_path: str
    # Model config name (e.g., "pi05_aloha")
    config_name: str = "pi05_aloha"
    # Number of denoising steps
    num_steps: int = 10
    # Device to use ("cuda" or "cpu")
    device: str = "cuda"


@dataclasses.dataclass
class EngineSplit:
    """Load a policy composed of multiple TensorRT engines."""

    engine_dir: str
    config_name: str = "pi05_aloha"
    num_steps: int = 10
    device: str = "cuda"


@dataclasses.dataclass
class ONNXSplit:
    """Load a policy composed of multiple ONNX models."""

    onnx_dir: str
    config_name: str = "pi05_aloha"
    num_steps: int = 10
    device: str = "cuda"


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8010
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    # Can be Checkpoint, Default, Engine, ONNX, EngineSplit, or ONNXSplit
    policy: Checkpoint | Default | Engine | ONNX | EngineSplit | ONNXSplit = dataclasses.field(default_factory=Default)
    
    # RTC功能支持 (not supported with Engine/ONNX/Split modes)
    use_rtc_guidance: bool = False
    rtc_inference_delay: int = 4
    rtc_execute_horizon: int = 1
    rtc_prefix_attention_schedule: str = "exp"
    rtc_max_guidance_weight: float = 1.0


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha_lora",
        dir="gs://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)
        case Engine():
            # Try to infer checkpoint_dir from engine_path
            # Engine is usually in checkpoint_dir/engine/, so go up one level
            engine_path = args.policy.engine_path
            checkpoint_dir = None
            if os.path.exists(engine_path):
                # Try to find checkpoint directory
                engine_dir = os.path.dirname(os.path.abspath(engine_path))
                # Check if we're in an engine subdirectory
                if os.path.basename(engine_dir) == 'engine':
                    checkpoint_dir = os.path.dirname(engine_dir)
                else:
                    # Try going up to find checkpoint
                    parent = os.path.dirname(engine_dir)
                    if os.path.exists(os.path.join(parent, 'model.safetensors')):
                        checkpoint_dir = parent
            
            return TensorRTEnginePolicy(
                engine_path=engine_path,
                config_name=args.policy.config_name,
                num_steps=args.policy.num_steps,
                device="cuda",
                checkpoint_dir=checkpoint_dir,
            )
        case ONNX():
            # Try to infer checkpoint_dir from onnx_path
            # ONNX is usually in checkpoint_dir/onnx/, so go up one level
            onnx_path = args.policy.onnx_path
            checkpoint_dir = None
            if os.path.exists(onnx_path):
                # Try to find checkpoint directory
                onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
                # Check if we're in an onnx subdirectory
                if os.path.basename(onnx_dir) == 'onnx':
                    checkpoint_dir = os.path.dirname(onnx_dir)
                else:
                    # Try going up to find checkpoint
                    parent = os.path.dirname(onnx_dir)
                    if os.path.exists(os.path.join(parent, 'model.safetensors')):
                        checkpoint_dir = parent
            
            if ONNXPolicy is None:
                raise RuntimeError("onnx_inference.py not available in this environment")
            return ONNXPolicy(
                onnx_path=onnx_path,
                config_name=args.policy.config_name,
                num_steps=args.policy.num_steps,
                device=args.policy.device,
                checkpoint_dir=checkpoint_dir,
            )
        case EngineSplit():
            if TensorRTSplitPolicy is None:
                raise RuntimeError("engine_inference_split.py not available or TensorRT not installed")
            engine_dir = os.path.abspath(args.policy.engine_dir)
            checkpoint_dir = None
            parent = os.path.dirname(engine_dir)
            if os.path.exists(os.path.join(parent, "model.safetensors")):
                checkpoint_dir = parent
            return TensorRTSplitPolicy(
                engine_dir=engine_dir,
                config_name=args.policy.config_name,
                num_steps=args.policy.num_steps,
                device=args.policy.device,
                checkpoint_dir=checkpoint_dir,
            )
        case ONNXSplit():
            if ONNXSplitPolicy is None:
                raise RuntimeError("onnx_inference_split.py not available")
            onnx_dir = os.path.abspath(args.policy.onnx_dir)
            checkpoint_dir = None
            parent = os.path.dirname(onnx_dir)
            if os.path.exists(os.path.join(parent, "model.safetensors")):
                checkpoint_dir = parent
            return ONNXSplitPolicy(
                onnx_dir=onnx_dir,
                config_name=args.policy.config_name,
                num_steps=args.policy.num_steps,
                device=args.policy.device,
                checkpoint_dir=checkpoint_dir,
            )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # 准备RTC配置 (not supported with Engine or ONNX mode)
    if isinstance(args.policy, (Engine, ONNX, EngineSplit, ONNXSplit)):
        if args.use_rtc_guidance:
            logging.warning("RTC guidance is not supported with TensorRT Engine or ONNX mode. Disabling RTC.")
            args.use_rtc_guidance = False
    
    rtc_config = {
        "use_rtc_guidance": args.use_rtc_guidance,
        "rtc_inference_delay": args.rtc_inference_delay,
        "rtc_execute_horizon": args.rtc_execute_horizon,
        "rtc_prefix_attention_schedule": args.rtc_prefix_attention_schedule,
        "rtc_max_guidance_weight": args.rtc_max_guidance_weight,
    }

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        rtc_config=rtc_config,
    )
    
    if args.use_rtc_guidance:
        logging.info(f"RTC enabled: delay={args.rtc_inference_delay}, weight={args.rtc_max_guidance_weight}")
    
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
