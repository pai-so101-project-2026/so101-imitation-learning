"""Independent protocol-0 READ-only probe; never changes torque or motor targets."""
import json
import time
from pathlib import Path

import serial

OUT = Path(__file__).resolve().parent / "raw_temperature_probe"
OUT.mkdir(exist_ok=True)
records = []


def read(port, motor, address, length, mode):
    # Only instruction 2 (READ); no SDK, broadcast requests, or configuration writes.
    pending = port.read(port.in_waiting)
    payload = [motor, 4, 2, address, length]
    request = bytes([255, 255, *payload, (~sum(payload)) & 255])
    began = time.monotonic()
    port.write(request)
    response = port.read(length + 6)
    valid = (len(response) == length + 6 and response[:2] == b"\xff\xff"
             and response[2] == motor and response[3] == length + 2
             and sum(response[2:]) % 256 == 255)
    record = dict(t=began, mode=mode, motor=motor, address=address,
                  length=length, request=list(request), response=list(response),
                  pending=list(pending), valid=valid, elapsed=time.monotonic()-began)
    if valid:
        record["status"] = response[4]
        if address <= 63 < address + length:
            record["temperature"] = response[5 + 63-address]
    records.append(record)


try:
    with serial.Serial('/dev/ttyACM1',
                       baudrate=1_000_000, timeout=.08, exclusive=True) as port:
        for mode in ('temperature_only', 'contiguous_block', 'mixed_registers'):
            started = time.monotonic()
            while time.monotonic()-started < 15:
                for motor in range(1, 7):
                    if mode == 'contiguous_block':
                        read(port, motor, 56, 15, mode)
                    else:
                        read(port, motor, 63, 1, mode)
                        if mode == 'mixed_registers':
                            read(port, motor, 56, 2, mode)
                            read(port, motor, 63, 1, mode)
                            read(port, motor, 69, 2, mode)
                time.sleep(.02)
            print(json.dumps(dict(mode=mode, packets=len(records))), flush=True)
finally:
    (OUT/'packets.json').write_text(json.dumps(records))
    summary = dict(read_only=True, sdk_used=False, packets=len(records), modes={})
    for mode in ('temperature_only', 'contiguous_block', 'mixed_registers'):
        subset = [r for r in records if r['mode'] == mode]
        motors = {}
        for motor in range(1, 7):
            values = [r['temperature'] for r in subset if r['motor'] == motor and 'temperature' in r]
            motors[motor] = dict(n=len(values), minimum=min(values, default=None),
                                maximum=max(values, default=None), over_40=sum(v>=40 for v in values))
        summary['modes'][mode] = dict(packets=len(subset), invalid=sum(not r['valid'] for r in subset),
            pending_bytes=sum(len(r['pending']) for r in subset),
            status_errors=sum(r.get('status', 0)!=0 for r in subset), motors=motors)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
