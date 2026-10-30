#!/usr/bin/env python

"""Read SO-101 observations and optionally request one guarded ACT action."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.motors.feetech import OperatingMode, TorqueMode
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

MODEL_DEFAULT = Path(
    "outputs/train/act_so101_c270_pick_place_05_cpu_smoke_002/checkpoints/000020/pretrained_model"
)
DATASET_DEFAULT = Path("/datasets/so101_c270_pick_place_05")
IMAGE_KEY = "observation.images.top"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
EXPECTED_SHAPE = (3, 480, 640)
EXPECTED_JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
MOTOR_NAMES = tuple(name.removesuffix(".pos") for name in EXPECTED_JOINTS)
ACTION_CONFIRMATION = "SEND ONE ACTION"

logger = logging.getLogger("lerobot_so101_act_inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("observe", "infer", "step"), default="infer")
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--follower-port", default="/dev/ttyACM1")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--robot-id", required=True, help="ID whose existing SO-101 calibration file will be loaded")
    parser.add_argument(
        "--position-mode",
        choices=("degrees", "range_m100_100"),
        required=True,
        help="Must match the mode used when recording the training dataset; it is not stored in the checkpoint.",
    )
    parser.add_argument(
        "--confirm-single-step",
        action="store_true",
        help="Required acknowledgement for mode=step; an exact interactive confirmation is also required.",
    )
    parser.add_argument(
        "--position-mode-evidence",
        type=Path,
        help="Collection-time SO-101 config JSON containing robot.type and robot.use_degrees.",
    )
    parser.add_argument(
        "--max-relative-target",
        action="append",
        metavar="MOTOR=LIMIT",
        help="Per-motor movement cap in the selected position units; specify exactly once for all six motors.",
    )
    args = parser.parse_args()

    if args.mode == "step":
        if not args.confirm_single_step:
            parser.error("--mode step requires --confirm-single-step")
        if args.position_mode_evidence is None:
            parser.error("--mode step requires --position-mode-evidence from the actual collection configuration")
        try:
            args.relative_limits = parse_relative_limits(args.max_relative_target or [])
        except ValueError as exc:
            parser.error(str(exc))
    else:
        args.relative_limits = None
    return args


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def verify_position_mode_evidence(path: Path, requested_mode: str) -> str:
    """Require an explicit SO-101 collection config that agrees with inference."""
    evidence = _read_json(path)
    robot_config = evidence.get("robot")
    if not isinstance(robot_config, dict) or robot_config.get("type") != "so101_follower":
        raise ValueError("Position-mode evidence must contain robot.type='so101_follower'")
    use_degrees = robot_config.get("use_degrees")
    if not isinstance(use_degrees, bool):
        raise ValueError("Position-mode evidence must contain boolean robot.use_degrees")
    evidence_mode = "degrees" if use_degrees else "range_m100_100"
    if evidence_mode != requested_mode:
        raise ValueError(
            f"Training position mode is {evidence_mode!r}, but inference requested {requested_mode!r}; refusing to send"
        )
    return evidence_mode


def parse_relative_limits(items: list[str]) -> dict[str, float]:
    """Parse one finite, positive movement cap for each SO-101 motor."""
    limits: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid movement cap {item!r}; expected MOTOR=LIMIT")
        motor, raw_limit = item.split("=", 1)
        if motor not in MOTOR_NAMES:
            raise ValueError(f"Unexpected motor name {motor!r}; expected one of {MOTOR_NAMES}")
        if motor in limits:
            raise ValueError(f"Duplicate movement cap for {motor}")
        try:
            limit = float(raw_limit)
        except ValueError as exc:
            raise ValueError(f"Movement cap for {motor} must be numeric") from exc
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError(f"Movement cap for {motor} must be finite and greater than zero")
        limits[motor] = limit
    if set(limits) != set(MOTOR_NAMES):
        missing = sorted(set(MOTOR_NAMES) - set(limits))
        raise ValueError(f"A movement cap is required for every motor; missing {missing}")
    return limits


def calibrated_action_bounds(
    calibration: Mapping[str, Any],
    motors: Mapping[str, Any],
    resolution_table: Mapping[str, int],
    position_mode: str,
) -> dict[str, tuple[float, float]]:
    """Return valid action bounds in the SO-101 bus's selected user units."""
    if set(calibration) != set(MOTOR_NAMES) or set(motors) != set(MOTOR_NAMES):
        raise ValueError("Calibration and motor names must exactly match the six expected SO-101 joints")
    bounds: dict[str, tuple[float, float]] = {}
    for name in MOTOR_NAMES:
        cal = calibration[name]
        motor = motors[name]
        resolution = resolution_table.get(motor.model)
        if not isinstance(resolution, int) or resolution <= 1:
            raise ValueError(f"Invalid encoder resolution for {name}: {resolution!r}")
        low_tick, high_tick = cal.range_min, cal.range_max
        if (
            not isinstance(low_tick, int)
            or not isinstance(high_tick, int)
            or not 0 <= low_tick < high_tick < resolution
        ):
            raise ValueError(f"Invalid calibration tick range for {name}: [{low_tick}, {high_tick}]")

        if name == "gripper":
            bounds[f"{name}.pos"] = (0.0, 100.0)
        elif position_mode == "range_m100_100":
            bounds[f"{name}.pos"] = (-100.0, 100.0)
        elif position_mode == "degrees":
            max_res = resolution - 1
            midpoint = (low_tick + high_tick) / 2
            low = (low_tick - midpoint) * 360 / max_res
            high = (high_tick - midpoint) * 360 / max_res
            bounds[f"{name}.pos"] = (low, high)
        else:
            raise ValueError(f"Unsupported position mode {position_mode!r}")
    return bounds


def plan_limited_action(
    current: Mapping[str, float],
    predicted: Mapping[str, float],
    limits: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    """Validate action values and clamp each target by its cap around current state."""
    expected = set(EXPECTED_JOINTS)
    for label, values in (("current", current), ("predicted", predicted), ("bounds", bounds)):
        if set(values) != expected:
            raise ValueError(f"Unexpected {label} joint names: expected {sorted(expected)}, got {sorted(values)}")
    if set(limits) != set(MOTOR_NAMES):
        raise ValueError(f"Unexpected movement limit names: expected {sorted(MOTOR_NAMES)}, got {sorted(limits)}")

    safe_targets: dict[str, float] = {}
    for joint in EXPECTED_JOINTS:
        now = float(current[joint])
        goal = float(predicted[joint])
        limit = float(limits[joint.removesuffix(".pos")])
        low, high = (float(v) for v in bounds[joint])
        if not all(math.isfinite(v) for v in (now, goal, limit, low, high)):
            raise ValueError(f"Non-finite current, prediction, limit, or calibration bound for {joint}")
        if limit <= 0 or low >= high or limit > high - low:
            raise ValueError(f"Invalid movement limit or calibrated range for {joint}")
        if not low <= now <= high:
            raise ValueError(f"Current position for {joint} is outside calibrated range [{low}, {high}]")
        if not low <= goal <= high:
            raise ValueError(f"Predicted position for {joint} is outside calibrated range [{low}, {high}]")
        delta = min(max(goal - now, -limit), limit)
        safe_targets[joint] = now + delta
    return safe_targets


def _load_runtime(args: argparse.Namespace):
    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model_path}")

    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing required model/processor artifacts: {missing}")

    train_config_path = model_path / "train_config.json"
    if not train_config_path.is_file():
        raise FileNotFoundError(f"Training config is required to resolve the dataset: {train_config_path}")
    train_config = _read_json(train_config_path)
    dataset_config = train_config.get("dataset", {})
    dataset_root = args.dataset_root.expanduser()
    if args.dataset_root == DATASET_DEFAULT and dataset_config.get("root"):
        configured_root = Path(dataset_config["root"])
        if configured_root.exists():
            dataset_root = configured_root
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
    dataset_info = _read_json(info_path)
    features = dataset_info.get("features", {})
    state_names = tuple(features.get(STATE_KEY, {}).get("names") or ())
    action_names = tuple(features.get(ACTION_KEY, {}).get("names") or ())
    if state_names != EXPECTED_JOINTS or action_names != EXPECTED_JOINTS:
        raise ValueError(
            "Dataset joint names/order do not match this SO-101 policy. "
            f"state={state_names}, action={action_names}"
        )

    policy_config = PreTrainedConfig.from_pretrained(model_path, local_files_only=True)
    if policy_config.type != "act":
        raise ValueError(f"Expected ACT checkpoint, got policy type {policy_config.type!r}")
    if policy_config.device != "cpu":
        raise ValueError(f"Expected a CPU checkpoint, got device={policy_config.device!r}")
    _validate_policy_features(policy_config)

    policy_class = get_policy_class("act")
    policy = policy_class.from_pretrained(
        model_path,
        config=policy_config,
        local_files_only=True,
    )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_path),
    )

    print(f"Model: {model_path}")
    print(f"Device: {policy.config.device}")
    print(f"Follower serial port: {args.follower_port}")
    print(f"Camera device: {args.camera}")
    print(f"Input features: {policy.config.input_features}")
    print(f"Output features: {policy.config.output_features}")
    print(f"Dataset: {dataset_root}")
    print(f"Joint names/order: {', '.join(EXPECTED_JOINTS)}")
    print(f"Position mode supplied by operator: {args.position_mode}")
    body_unit = "degrees" if args.position_mode == "degrees" else "calibrated range [-100, 100]"
    print(f"Predicted action units: first 5 joints={body_unit}; gripper=calibrated range [0, 100]")
    return model_path, dataset_root, policy, preprocessor, postprocessor


def _validate_policy_features(config: PreTrainedConfig) -> None:
    expected = {STATE_KEY: (6,), IMAGE_KEY: EXPECTED_SHAPE}
    if set(config.input_features) != set(expected):
        raise ValueError(f"Unexpected policy input feature names: {list(config.input_features)}")
    for key, shape in expected.items():
        actual = tuple(config.input_features[key].shape)
        if actual != shape:
            raise ValueError(f"Policy input {key} has shape {actual}, expected {shape}")
    if set(config.output_features) != {ACTION_KEY}:
        raise ValueError(f"Unexpected policy output feature names: {list(config.output_features)}")
    if tuple(config.output_features[ACTION_KEY].shape) != (6,):
        raise ValueError(f"Policy action shape is {config.output_features[ACTION_KEY].shape}, expected (6,)")


def _make_robot(args: argparse.Namespace) -> SO101Follower:
    calibration_path = HF_LEROBOT_CALIBRATION / ROBOTS / "so101_follower" / f"{args.robot_id}.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Existing SO-101 calibration is required; refusing to calibrate or create one: {calibration_path}"
        )
    camera_config = OpenCVCameraConfig(
        index_or_path=args.camera,
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
    )
    config = SO101FollowerConfig(
        port=args.follower_port,
        id=args.robot_id,
        use_degrees=args.position_mode == "degrees",
        disable_torque_on_disconnect=False,
        max_relative_target=args.relative_limits if args.mode == "step" else None,
        cameras={IMAGE_KEY: camera_config},
    )
    return SO101Follower(config)


def _send_single_action(robot: SO101Follower, target: dict[str, float]) -> dict[str, float]:
    """Send at most one action after an exact interactive confirmation."""
    response = input(f"Type '{ACTION_CONFIRMATION}' to send this one action: ")
    if response.strip() != ACTION_CONFIRMATION:
        raise PermissionError("Single-action confirmation did not match; no motor goal was sent")
    return robot.send_action(target)


def _connect_read_only(robot: SO101Follower) -> None:
    """Open the bus and camera without calling SO101Follower.connect/configure."""
    robot.bus.connect(handshake=True)
    try:
        for camera in robot.cameras.values():
            camera.connect()
    except Exception:
        _disconnect_read_only(robot)
        raise


def _disconnect_read_only(robot: SO101Follower) -> None:
    for camera in robot.cameras.values():
        if camera.is_connected:
            try:
                camera.disconnect()
            except BaseException:
                logger.exception("Failed to release camera")
    if robot.bus.is_connected:
        try:
            # Avoid bus.disconnect's default Torque_Enable write.
            robot.bus.disconnect(disable_torque=False)
        except BaseException:
            logger.exception("Failed to close follower bus without changing torque")


def _validate_step_hardware(robot: SO101Follower) -> None:
    """Read back calibration, operating mode, and torque without configuring motors."""
    if not robot.is_calibrated:
        raise ValueError("Saved calibration does not match motor calibration registers; refusing to send")
    operating_modes = robot.bus.sync_read("Operating_Mode", normalize=False)
    if set(operating_modes) != set(MOTOR_NAMES) or any(
        int(mode) != OperatingMode.POSITION.value for mode in operating_modes.values()
    ):
        raise ValueError(f"All motors must already be in position mode; read {operating_modes}")
    torque_states = robot.bus.sync_read("Torque_Enable", normalize=False)
    if set(torque_states) != set(MOTOR_NAMES) or any(
        int(state) != TorqueMode.ENABLED.value for state in torque_states.values()
    ):
        raise ValueError(f"All motors must already have torque enabled; script will not change torque: {torque_states}")


def _validated_observation(robot: SO101Follower) -> tuple[np.ndarray, np.ndarray]:
    observation = robot.get_observation()
    missing = [key for key in (*EXPECTED_JOINTS, IMAGE_KEY) if key not in observation]
    if missing:
        raise ValueError(f"Robot observation is missing required values: {missing}")

    state = np.asarray([observation[name] for name in EXPECTED_JOINTS], dtype=np.float32)
    image = observation[IMAGE_KEY]
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError(f"Invalid robot state: shape={state.shape}, values={state}")
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (480, 640, 3):
        shape = getattr(image, "shape", None)
        dtype = getattr(image, "dtype", None)
        raise ValueError(f"Invalid RGB camera frame: shape={shape}, dtype={dtype}; expected (480, 640, 3) uint8")
    return state, image


def _policy_observation(state: np.ndarray, image_rgb: np.ndarray) -> dict[str, torch.Tensor]:
    if not np.isfinite(state).all():
        raise ValueError("State contains NaN or Inf")
    image_chw = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).to(torch.float32) / 255.0
    state_tensor = torch.from_numpy(state.copy())
    if tuple(image_chw.shape) != EXPECTED_SHAPE or state_tensor.shape != (6,):
        raise ValueError("Observation tensor shape mismatch; action generation is blocked")
    if not torch.isfinite(image_chw).all() or not torch.isfinite(state_tensor).all():
        raise ValueError("Observation tensor contains NaN or Inf; action generation is blocked")
    return {STATE_KEY: state_tensor, IMAGE_KEY: image_chw}


def _infer(
    observation: dict[str, torch.Tensor],
    policy: Any,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> torch.Tensor:
    model_input = preprocessor(observation)
    with torch.inference_mode():
        normalized_action = policy.select_action(model_input)
        action = postprocessor(normalized_action)
    if action.ndim == 2 and action.shape[0] == 1:
        action = action.squeeze(0)
    if action.shape != (6,) or not torch.isfinite(action).all():
        raise ValueError(f"Invalid postprocessed action: shape={tuple(action.shape)}, values={action}")
    return action.detach().cpu()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.mode == "step":
        verify_position_mode_evidence(args.position_mode_evidence, args.position_mode)
    _, dataset_root, policy, preprocessor, postprocessor = _load_runtime(args)
    if args.mode != "step":
        print("Motor goal writes: DISABLED in observe/infer modes")

    robot = _make_robot(args)
    try:
        # SO101Follower.connect() is deliberately avoided: it may calibrate and
        # configure motors; configure() temporarily disables then re-enables torque.
        _connect_read_only(robot)
        if args.mode == "step":
            _validate_step_hardware(robot)
        state, image_rgb = _validated_observation(robot)
        print(f"Observed state ({args.position_mode}; gripper range 0..100): {state.tolist()}")
        print(f"Observed image: RGB uint8 HWC {image_rgb.shape}; policy tensor shape {EXPECTED_SHAPE}")

        if args.mode in ("infer", "step"):
            raw_observation = _policy_observation(state, image_rgb)
            prediction = _infer(raw_observation, policy, preprocessor, postprocessor).numpy()
            if prediction.shape != (6,) or not np.isfinite(prediction).all():
                raise ValueError("Predicted action is invalid; no action may be sent")
            current_by_joint = dict(zip(EXPECTED_JOINTS, state.tolist(), strict=True))
            predicted_by_joint = dict(zip(EXPECTED_JOINTS, prediction.tolist(), strict=True))
            if args.mode == "infer":
                print("Postprocessed prediction (display only; no motor command sent):")
                for name in EXPECTED_JOINTS:
                    current = current_by_joint[name]
                    target = predicted_by_joint[name]
                    print(f"  {name}: current={current:.5f}, predicted={target:.5f}, delta={target-current:+.5f}")
            else:
                action_bounds = calibrated_action_bounds(
                    robot.bus.calibration,
                    robot.bus.motors,
                    robot.bus.model_resolution_table,
                    args.position_mode,
                )
                target_by_joint = plan_limited_action(
                    current_by_joint,
                    predicted_by_joint,
                    args.relative_limits,
                    action_bounds,
                )
                print("Single-action preview (no command has been sent yet):")
                for name in EXPECTED_JOINTS:
                    current = current_by_joint[name]
                    predicted = predicted_by_joint[name]
                    target = target_by_joint[name]
                    print(
                        f"  {name}: current={current:.5f}, predicted={predicted:.5f}, "
                        f"limited_target={target:.5f}, "
                        f"max_delta={args.relative_limits[name.removesuffix('.pos')]:.5f}"
                    )
                sent = _send_single_action(robot, target_by_joint)
                if set(sent) != set(EXPECTED_JOINTS) or not all(math.isfinite(float(v)) for v in sent.values()):
                    raise RuntimeError(f"Robot returned unexpected sent-action values: {sent}")
                print(f"One action sent through SO101Follower.send_action: {sent}")
        else:
            print("Observe mode complete. No model action was requested and no motor goal was written.")
    finally:
        _disconnect_read_only(robot)


if __name__ == "__main__":
    main()
