"""Opt-in protocol-0 reply-length validation and bounded raw-packet capture."""
from collections import deque
from contextlib import contextmanager
import time


class PacketAudit:
    def __init__(self,handler,capacity=20000):
        self.handler=handler
        self.records=deque(maxlen=capacity)
        self.expected=None
        self.context=None
        self.original_txrx=handler.txRxPacket
        self.original_readrx=handler.readRx
        self.original_rx=handler.rxPacket
        handler.txRxPacket=self.txrx
        handler.readRx=self.readrx
        handler.rxPacket=self.receive

    @contextmanager
    def request(self,expected):
        previous=self.expected
        self.expected=expected
        try:
            yield
        finally:
            self.expected=previous

    def txrx(self,port,packet):
        # Protocol 0 READ instruction is 2. Other instructions keep SDK behavior.
        expected=dict(id=packet[2],length=packet[6],address=packet[5]) if packet[4]==2 and packet[2]!=254 else None
        with self.request(expected):
            return self.original_txrx(port,packet)

    def readrx(self,port,motor_id,length):
        with self.request(dict(id=motor_id,length=length,address=None)):
            return self.original_readrx(port,motor_id,length)

    def receive(self,port):
        began=time.monotonic()
        packet,result=self.original_rx(port)
        expected=self.expected.copy() if self.expected is not None else None
        record=dict(time=began,seconds=time.monotonic()-began,expected=expected,
                    context=self.context,packet=list(packet),communication_result=result)
        self.records.append(record)
        if result==0 and expected is not None and len(packet)>=6 and packet[2]==expected["id"] and packet[4]==0:
            actual=packet[3]-2
            if actual!=expected["length"] or len(packet)!=packet[3]+4:
                record["rejected"]=True
                raise ConnectionError(f"Unexpected servo reply length: expected={expected}, payload={actual}, packet={packet}")
        return packet,result

    def restore(self):
        self.handler.txRxPacket=self.original_txrx
        self.handler.readRx=self.original_readrx
        self.handler.rxPacket=self.original_rx
