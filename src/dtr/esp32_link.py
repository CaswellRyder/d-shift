"""Linux i2c-dev transport; no optional dependencies."""
import os
import fcntl
import time


class NullLink:
    def __init__(self, *args, **kwargs):
        self.sent, self.errors, self.last_error = 0, 0, None

    def send(self, packet):
        return True

    def close(self):
        pass


class I2CLink(NullLink):
    def __init__(self, bus=1, address=0x42, opener=None, writer=None,
                 ioctl=None, closer=None, clock=None, retry_s=1.0):
        super().__init__()
        self.bus, self.address = bus, address
        self.opener, self.writer = opener or os.open, writer or os.write
        self.ioctl, self.closer = ioctl or fcntl.ioctl, closer or os.close
        self.clock, self.retry_s = clock or time.monotonic, retry_s
        self.fd, self.next_open = None, 0

    def close(self):
        if self.fd is not None:
            try:
                self.closer(self.fd)
            except OSError:
                pass
            self.fd = None

    def send(self, packet):
        now = self.clock()
        if self.fd is None:
            if now < self.next_open:
                return False
            try:
                self.fd = self.opener(f"/dev/i2c-{self.bus}", os.O_RDWR)
                self.ioctl(self.fd, 0x0703, self.address)
            except OSError as exc:
                return self._fail(exc, now, reopen=True)
        try:
            if self.writer(self.fd, packet) != len(packet):
                raise OSError("Short I2C write")
        except OSError as exc:
            # A NACK or busy bus is per-transfer: keep the descriptor and try
            # again next frame. Backing off here would starve the ESP32
            # failsafe window (~300 ms) after a single glitch.
            return self._fail(exc, now, reopen=False)
        self.sent += 1
        return True

    def _fail(self, exc, now, reopen):
        self.errors += 1
        self.last_error = str(exc)
        if reopen:
            self.close()
            self.next_open = now + self.retry_s
        return False
