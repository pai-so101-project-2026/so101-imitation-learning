"""Return along the reversed measured rollout path at reduced target speed."""
import json
import time
from argparse import Namespace, ArgumentParser
from pathlib import Path

import cv2
from teleop_matching_monitor import checked_motor_status

from lerobot.scripts.lerobot_so101_act_inference import (
    MOTOR_NAMES, EXPECTED_JOINTS, _make_robot, _connect_read_only, _disconnect_read_only,
    _validate_step_hardware, _validated_observation, calibrated_action_bounds, plan_limited_action,
)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/act_best5000_eval_speed20_001"))
    cli = parser.parse_args()
    source = cli.source
    rows = [json.loads(line) for line in (source / "actions.jsonl").read_text().splitlines()]
    destination = rows[0]["current"]
    output = source / "return_to_start"
    output.mkdir(exist_ok=True)
    limits = dict.fromkeys(MOTOR_NAMES, 5.0)
    limits["elbow_flex"] = 8.0
    args = Namespace(robot_id="follower_arm", camera="/dev/video0", follower_port="/dev/ttyACM1",
                     position_mode="range_m100_100", relative_limits=limits, mode="step")
    robot = _make_robot(args)
    armed = False
    result = {"completed": False, "destination": destination,
              "software_overload_stop": False, "software_temperature_stop": False,
              "motor_monitoring": []}
    started = time.monotonic()
    try:
        _connect_read_only(robot)
        _validate_step_hardware(robot)
        checked_motor_status(robot.bus, result["motor_monitoring"])
        state, frame = _validated_observation(robot)
        current = dict(zip(EXPECTED_JOINTS, state.tolist(), strict=True))
        recorded_end = rows[-1]["current"]
        saved_result = source / "result.json"
        if saved_result.is_file():
            final = json.loads(saved_result.read_text()).get("final_position", {})
            final = {k if k.endswith(".pos") else f"{k}.pos": v for k,v in final.items()}
            if set(final) == set(EXPECTED_JOINTS):
                if any(abs(final[k]-recorded_end[k]) > limits[k.removesuffix('.pos')]+0.5 for k in EXPECTED_JOINTS):
                    raise RuntimeError("Saved final pose differs excessively from the last recorded path point")
                recorded_end = final
        if any(abs(current[k]-recorded_end[k]) > 3 for k in EXPECTED_JOINTS):
            raise RuntimeError("Robot moved since recorded path ended")
        bounds = calibrated_action_bounds(robot.bus.calibration, robot.bus.motors,
                                         robot.bus.model_resolution_table, args.position_mode)
        previous = current.copy()
        last_send = last_check = time.monotonic()
        armed = True
        cv2.imwrite(str(output / "before.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        with (output / "actions.jsonl").open("w") as log:
            def follow(target, duration):
                nonlocal previous, last_send, last_check
                until = time.monotonic()+duration
                while time.monotonic() < until:
                    if time.monotonic()-started > 100:
                        raise RuntimeError("Return time limit exceeded")
                    positions = robot.bus.sync_read("Present_Position")
                    observed = {f"{k}.pos": float(v) for k, v in positions.items()}
                    if any(abs(observed[k]-previous[k]) > limits[k.removesuffix('.pos')]+0.5 for k in EXPECTED_JOINTS):
                        raise RuntimeError("Return tracking error exceeded limit")
                    now = time.monotonic()
                    if now-last_check >= 1:
                        # Calibration is checked once on connection. Keep the live
                        # path free of repeated EEPROM reads and image-file I/O.
                        modes = robot.bus.sync_read("Operating_Mode", normalize=False)
                        torque = robot.bus.sync_read("Torque_Enable", normalize=False)
                        if any(v != 0 for v in modes.values()) or any(v != 1 for v in torque.values()):
                            raise RuntimeError(f"Unexpected mode or torque: modes={modes}, torque={torque}")
                        checked_motor_status(robot.bus, result["motor_monitoring"])
                        last_check = time.monotonic()
                    elapsed = min(time.monotonic()-last_send, 0.1)
                    if time.monotonic()-last_send > 0.5:
                        raise RuntimeError("Return control update stalled for over 0.5 seconds")
                    ramped = {}
                    for k in EXPECTED_JOINTS:
                        speed = 5 if k == "gripper.pos" else 10
                        allowance = elapsed*speed
                        ramped[k] = previous[k]+max(-allowance,min(allowance,target[k]-previous[k]))
                    goal = plan_limited_action(observed, ramped, limits, bounds)
                    previous = robot.send_action(goal)
                    last_send = time.monotonic()
                    log.write(json.dumps(dict(t=last_send-started,current=observed,sent=previous))+"\n")
                    log.flush()
                    time.sleep(0.03)
            opened = min(30, max(r["current"]["gripper.pos"] for r in rows))
            target = current.copy()
            target["gripper.pos"] = opened
            follow(target, 6)
            previous_t = rows[-1]["t"]
            for i, row in enumerate(reversed(rows)):
                target = row["current"].copy()
                target["gripper.pos"] = opened
                duration = 1 if i == 0 else max(0.02, (previous_t-row["t"])*2)
                follow(target, duration)
                previous_t = row["t"]
                if i % 90 == 0:
                    print(f"RETURN progress={i}/{len(rows)} seconds={time.monotonic()-started:.1f}", flush=True)
            target = destination.copy()
            target["gripper.pos"] = opened
            follow(target, 3)
            positions = {f"{k}.pos": float(v) for k,v in robot.bus.sync_read("Present_Position").items()}
            if any(abs(positions[k]-destination[k]) > 3.5 for k in EXPECTED_JOINTS if k != "gripper.pos"):
                raise RuntimeError("Start pose not reached; gripper left open")
            follow(destination, 6)
            final = {f"{k}.pos": float(v) for k,v in robot.bus.sync_read("Present_Position").items()}
            errors = {k: final[k]-destination[k] for k in EXPECTED_JOINTS}
            result.update(completed=all(abs(v) <= 3.5 for v in errors.values()), final_position=final, errors=errors)
    except BaseException as exc:
        result["error"] = str(exc)
        raise
    finally:
        try:
            if armed and robot.bus.is_connected:
                hold = robot.bus.sync_read("Present_Position", normalize=False)
                robot.bus.sync_write("Goal_Position", hold, normalize=False)
                result["torque"] = robot.bus.sync_read("Torque_Enable", normalize=False)
                result["status"] = robot.bus.sync_read("Status", normalize=False)
                if robot.is_connected:
                    _, image = _validated_observation(robot)
                    cv2.imwrite(str(output / "after.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        finally:
            result["elapsed"] = time.monotonic()-started
            (output / "result.json").write_text(json.dumps(result,indent=2))
            print("RESULT "+json.dumps(result),flush=True)
            _disconnect_read_only(robot)


if __name__ == "__main__":
    main()
