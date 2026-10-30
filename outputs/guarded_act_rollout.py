"""Bounded ACT episode with feedback and persistent result log."""
import json
import math
import time
import argparse
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import torch
from teleop_matching_monitor import checked_motor_status

from lerobot.scripts.lerobot_so101_act_inference import (
    MOTOR_NAMES, EXPECTED_JOINTS, MODEL_DEFAULT, DATASET_DEFAULT,
    _load_runtime, _make_robot, _connect_read_only, _disconnect_read_only,
    _validated_observation, _policy_observation, _infer, _validate_step_hardware,
    calibrated_action_bounds, plan_limited_action,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--output", type=Path, default=Path("outputs/act_rollout"))
    parser.add_argument("--elbow-relative-limit", type=float, choices=(1.0, 3.0, 5.0, 8.0), default=1.0)
    parser.add_argument("--relative-limit", type=float, choices=(1.0, 3.0, 5.0), default=1.0)
    parser.add_argument("--target-speed", type=float, choices=(2.0, 5.0, 10.0, 20.0, 30.0, 50.0), default=50.0)
    parser.add_argument("--position-envelope", type=float, choices=(25.0, 35.0, 50.0, 180.0), default=180.0)
    parser.add_argument("--reference-pose", type=Path)
    parser.add_argument("--seconds", type=int, choices=(20, 40, 90), default=20)
    cli = parser.parse_args()
    torch.set_num_threads(4)
    out = cli.output
    out.mkdir(exist_ok=True)
    args = Namespace(model=cli.model, dataset_root=DATASET_DEFAULT,
                     robot_id="follower_arm", follower_port="/dev/ttyACM1",
                     camera="/dev/video0", position_mode="range_m100_100",
                     mode="step", relative_limits=dict.fromkeys(MOTOR_NAMES, cli.relative_limit))
    args.relative_limits["elbow_flex"] = cli.elbow_relative_limit
    _, _, policy, pre, post = _load_runtime(args)
    policy.reset()
    robot = _make_robot(args)
    count = 0
    armed = False
    snapshot_pool = ThreadPoolExecutor(max_workers=1)
    snapshot_future = None
    def save_snapshot(frame):
        temporary = out / "latest.pending.jpg"
        if not cv2.imwrite(str(temporary), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("Failed to save monitoring image")
        temporary.replace(out / "latest.jpg")
    result = {"completed": False, "task_success": "unverified", "model": str(cli.model)}
    result["limits"] = dict(target_speed=cli.target_speed, position_envelope=cli.position_envelope,
                            relative_targets=args.relative_limits)
    result["requested_seconds"] = cli.seconds
    result["software_overload_stop"] = False
    result["software_temperature_stop"] = False
    result["motor_monitoring"] = []
    try:
        _connect_read_only(robot)
        _validate_step_hardware(robot)
        armed = True
        bounds = calibrated_action_bounds(robot.bus.calibration, robot.bus.motors,
                                         robot.bus.model_resolution_table, args.position_mode)
        state, image = _validated_observation(robot)
        if cli.reference_pose:
            ref = json.loads(cli.reference_pose.read_text())
            if ref.get("position_mode") != args.position_mode:
                raise RuntimeError("Reference units mismatch")
            reference = {f"{k}.pos": v for k, v in ref["target"].items()}
            origin = dict(zip(EXPECTED_JOINTS, state.tolist(), strict=True))
            if set(reference) != set(EXPECTED_JOINTS) or any(abs(reference[k]-origin[k]) > 5 for k in EXPECTED_JOINTS):
                raise RuntimeError("Reference pose requires more than five units of movement")
            previous = origin.copy()
            prep_start = previous_time = time.monotonic()
            preparation = []
            while time.monotonic()-prep_start < 8:
                current = {f"{k}.pos": float(v) for k, v in robot.bus.sync_read("Present_Position").items()}
                if any(abs(current[k]-origin[k]) > 6 or abs(current[k]-previous[k]) > 3.5 for k in EXPECTED_JOINTS):
                    raise RuntimeError("Reference preparation tracking error")
                now = time.monotonic()
                allowance = min(now-previous_time, 0.1)
                ramp = {k: previous[k]+max(-allowance,min(allowance,reference[k]-previous[k])) for k in EXPECTED_JOINTS}
                goal = plan_limited_action(current,ramp,args.relative_limits,bounds)
                previous = robot.send_action(goal)
                previous_time = time.monotonic()
                preparation.append(dict(t=previous_time-prep_start,current=current,sent=previous))
                if len(preparation) % 20 == 0:
                    _validate_step_hardware(robot)
                    checked_motor_status(robot.bus, result["motor_monitoring"])
                time.sleep(0.05)
            state,image = _validated_observation(robot)
            actual = dict(zip(EXPECTED_JOINTS,state.tolist(),strict=True))
            errors = {k: actual[k]-reference[k] for k in EXPECTED_JOINTS}
            (out/"pose_preparation.json").write_text(json.dumps(dict(reference=reference,actual=actual,errors=errors,trajectory=preparation),indent=2))
            print("REFERENCE_POSE_ERRORS "+json.dumps(errors),flush=True)
            if any(abs(v) > 2.5 for v in errors.values()):
                raise RuntimeError("Reference pose not reached within tolerance; no policy rollout")
        cv2.imwrite(str(out / "before.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        initial = dict(zip(EXPECTED_JOINTS, state.tolist(), strict=True))
        previous_target = initial.copy()
        start = last_send = last_check = time.monotonic()
        deadline = start + cli.seconds
        with (out / "actions.jsonl").open("w") as log:
            while time.monotonic() < deadline and count < cli.seconds * 30:
                iteration_start = time.monotonic()
                state, image = _validated_observation(robot)
                observation_seconds = time.monotonic()-iteration_start
                current = dict(zip(EXPECTED_JOINTS, state.tolist(), strict=True))
                if any(abs(current[k] - initial[k]) > cli.position_envelope or
                       abs(current[k] - previous_target[k]) > max(3.5, args.relative_limits[k.removesuffix('.pos')]+0.5) for k in EXPECTED_JOINTS):
                    raise RuntimeError("Position envelope or target tracking limit exceeded")
                queue_refresh = not policy._action_queue
                inference_start = time.monotonic()
                prediction = dict(zip(EXPECTED_JOINTS,
                    _infer(_policy_observation(state, image), policy, pre, post).tolist(), strict=True))
                inference_seconds = time.monotonic()-inference_start
                now = time.monotonic()
                if now >= deadline:
                    break
                monitoring_start = now
                if now - last_check >= 1:
                    # Static calibration was validated before arming. Avoid its
                    # repeated EEPROM reads in the timed control loop.
                    modes = robot.bus.sync_read("Operating_Mode", normalize=False)
                    torque = robot.bus.sync_read("Torque_Enable", normalize=False)
                    if any(v != 0 for v in modes.values()) or any(v != 1 for v in torque.values()):
                        raise RuntimeError(f"Unexpected mode or torque: modes={modes}, torque={torque}")
                    checked_motor_status(robot.bus, result["motor_monitoring"])
                    goal_readback=robot.bus.sync_read("Goal_Position")
                    if any(abs(float(goal_readback[k.removesuffix('.pos')])-previous_target[k]) > 0.15 for k in EXPECTED_JOINTS):
                        raise RuntimeError("Motor goal readback differs from sent action")
                    last_check = time.monotonic()
                    if snapshot_future is None or snapshot_future.done():
                        if snapshot_future is not None:
                            snapshot_future.result()
                        snapshot_future = snapshot_pool.submit(save_snapshot, image.copy())
                now = time.monotonic()
                monitoring_seconds = now-monitoring_start
                if now >= deadline:
                    break
                if now-last_send > 0.5:
                    raise RuntimeError(
                        f"Control update stalled for {now-last_send:.3f}s; "
                        f"observation={observation_seconds:.3f}s inference={inference_seconds:.3f}s "
                        f"monitoring={monitoring_seconds:.3f}s; stopping before next action"
                    )
                # Bound the goal velocity and cap the allowance after slow inference.
                allowance = min(now - last_send, 0.1) * cli.target_speed
                for k, v in prediction.items():
                    low, high = bounds[k]
                    if not math.isfinite(v) or not low <= v <= high:
                        raise RuntimeError(f"Invalid policy target {k}: {v}")
                ramped = {k: previous_target[k] + max(-allowance, min(allowance,
                          prediction[k] - previous_target[k])) for k in EXPECTED_JOINTS}
                target = plan_limited_action(current, ramped, args.relative_limits, bounds)
                if any(abs(target[k] - initial[k]) > cli.position_envelope for k in EXPECTED_JOINTS):
                    raise RuntimeError("Target exceeded initial-position envelope")
                rate_limited = {k: abs(prediction[k]-previous_target[k]) > allowance for k in EXPECTED_JOINTS}
                relative_limited = {k: abs(ramped[k]-target[k]) > 1e-6 for k in EXPECTED_JOINTS}
                command_start = time.monotonic()
                previous_target = robot.send_action(target)
                last_send = time.monotonic()
                command_seconds = last_send-command_start
                count += 1
                log.write(json.dumps(dict(t=last_send-start, current=current,
                                          prediction=prediction, sent=previous_target,
                                          ramped_target=ramped, planned_target=target,
                                          rate_limited=rate_limited, relative_limited=relative_limited,
                                          driver_limited={k:abs(previous_target[k]-target[k])>1e-6 for k in EXPECTED_JOINTS},
                                          allowance=allowance,queue_refresh=queue_refresh,
                                          inference_seconds=inference_seconds,
                                          observation_seconds=observation_seconds,
                                          monitoring_seconds=monitoring_seconds,
                                          command_seconds=command_seconds)) + "\n")
                log.flush()
                if count % 30 == 0:
                    print(f"PROGRESS seconds={last_send-start:.2f} actions={count}", flush=True)
                time.sleep(max(0, 1 / 30 - (time.monotonic() - iteration_start)))
        result.update(completed=True, elapsed=time.monotonic()-start, actions=count)
    except BaseException as exc:
        result.update(error=str(exc), actions=count)
        raise
    finally:
        try:
            if robot.bus.is_connected:
                # End the episode by holding measured position, preserving torque.
                if armed:
                    hold = robot.bus.sync_read("Present_Position", normalize=False)
                    robot.bus.sync_write("Goal_Position", hold, normalize=False)
                result["final_position"] = robot.bus.sync_read("Present_Position")
                result["torque"] = robot.bus.sync_read("Torque_Enable", normalize=False)
                result["status"] = robot.bus.sync_read("Status", normalize=False)
                if robot.is_connected:
                    _, image = _validated_observation(robot)
                    cv2.imwrite(str(out / "after.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        finally:
            (out / "result.json").write_text(json.dumps(result, indent=2))
            print("RESULT " + json.dumps(result), flush=True)
            snapshot_pool.shutdown(wait=True, cancel_futures=True)
            _disconnect_read_only(robot)


if __name__ == "__main__":
    main()
