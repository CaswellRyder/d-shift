//go:build !linux

package pibridge

import "errors"

func OpenMotorLink(bus int, addr uint8) (*MotorLink, error) {
	return nil, errors.New("motor I2C is available only on Linux")
}
