//go:build linux

package pibridge

import (
	"errors"
	"fmt"
	"os"
	"syscall"
)

const i2cSlave = 0x0703

func OpenMotorLink(bus int, addr uint8) (*MotorLink, error) {
	if bus < 0 || bus > 32 || addr < 0x08 || addr > 0x77 {
		return nil, errors.New("invalid I2C parameters")
	}
	f, err := os.OpenFile(fmt.Sprintf("/dev/i2c-%d", bus), os.O_RDWR, 0)
	if err != nil {
		return nil, err
	}
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, f.Fd(), uintptr(i2cSlave), uintptr(addr))
	if errno != 0 {
		f.Close()
		return nil, errno
	}
	link, err := newMotorLink(f)
	if err != nil {
		f.Close()
	}
	return link, err
}
