"""Exercise installed SDK reply parsing using synthetic packets, without hardware."""
import json
from pathlib import Path

import scservo_sdk as sdk


class FakePort:
    def __init__(self,payload):
        # Protocol-0 response: header, ID, length, error, parameters, checksum.
        body=[6,len(payload)+2,0]+payload
        self.buffer=[255,255]+body+[(~sum(body))&255]
        self.is_using=False
        self.sent=None
    def clearPort(self):
        pass
    def writePort(self,packet):
        self.sent=list(packet)
        return len(packet)
    def readPort(self,length):
        data=self.buffer[:length]
        del self.buffer[:length]
        return data
    def setPacketTimeout(self,length):
        pass
    def isPacketTimeout(self):
        return not self.buffer


def run(payload):
    port=FakePort(payload)
    data,communication,error=sdk.PacketHandler(0).readTxRx(port,6,63,1)
    return dict(requested_register=63,requested_data_bytes=1,reply_data_bytes=len(payload),
        returned_data=data,communication_result=communication,servo_error=error,
        accepted=communication==sdk.COMM_SUCCESS)


results={"correct_length":run([33]),"wrong_length_same_id_valid_checksum":run([52,7]),
    "conclusion":"The installed SDK accepts a valid-checksum, same-ID reply with two payload bytes for a one-byte read and returns its first byte.",
    "historical_cause_proven":False,
    "limitations":["Synthetic payload 52 is illustrative and does not demonstrate that the historical 52 C sample was stale data.",
        "A one-byte reply from a different register cannot be distinguished by length; protocol-0 replies do not echo the requested register address.",
        "No live SDK, controller configuration, or motor was changed."]}
assert results["correct_length"]["returned_data"]==[33]
assert results["wrong_length_same_id_valid_checksum"]["accepted"]
assert results["wrong_length_same_id_valid_checksum"]["returned_data"]==[52]
Path("outputs/act_diagnosis/sdk_reply_validation.json").write_text(json.dumps(results,indent=2))
print(json.dumps(results),flush=True)
