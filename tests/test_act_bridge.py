"""Worker transport errors must fail clearly, without pickle or silent corruption."""
import socket
import struct
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from act_bridge import MAX_PACKET, receive_packet, send_packet

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = socket.socketpair()
        self.addCleanup(self.a.close)
        self.addCleanup(self.b.close)
        self.b.settimeout(1)
    def test_numpy_payload_preserves_dtype_and_shape(self):
        state=np.arange(6,dtype=np.float32)
        send_packet(self.a, command="action",state=state)
        data=receive_packet(self.b)
        self.assertEqual(data["command"].item(),"action")
        self.assertEqual(data["state"].dtype,np.float32)
        np.testing.assert_array_equal(data["state"],state)
    def test_worker_error_is_propagated(self):
        send_packet(self.a,error="checkpoint failed")
        with self.assertRaisesRegex(RuntimeError,"checkpoint failed"): receive_packet(self.b)
    def test_eof_fails_without_hanging(self):
        self.a.close()
        with self.assertRaises(ConnectionError): receive_packet(self.b)
    def test_invalid_size_is_rejected(self):
        self.a.sendall(struct.pack("!I",MAX_PACKET+1))
        with self.assertRaisesRegex(ValueError,"length"): receive_packet(self.b)

if __name__ == "__main__": unittest.main()
