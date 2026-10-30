"""Log thermal/overload telemetry without adding software stops for those faults."""
import time

# Protocol-0 status bits: overheating (bit 2), overload (bit 5).
TELEMETRY_ONLY_STATUS_BITS = 0x04 | 0x20


def checked_motor_status(bus, events):
    status = bus.sync_read("Status", normalize=False)
    temperatures = bus.sync_read("Present_Temperature", normalize=False)
    events.append(dict(time=time.monotonic(), status=status, temperatures=temperatures))
    other_faults = {k: int(v) & ~TELEMETRY_ONLY_STATUS_BITS for k, v in status.items()}
    if any(other_faults.values()):
        raise RuntimeError(f"Motor fault: {status}")
    return status
