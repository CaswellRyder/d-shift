package pibridge

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"time"
)

const motorPeriod = 50 * time.Millisecond
const motorInputLease = 100 * time.Millisecond

type MotorLink struct {
	device     io.ReadWriteCloser
	session    uint32
	seq        uint16
	mu         sync.Mutex
	hold       bool
	axes       [4]float64
	updated    time.Time
	faulted    bool
	lastStatus string
	lastOK     time.Time
	wake       chan struct{}
	closed     chan struct{}
	closeOnce  sync.Once
	closeError error
}

func newMotorLink(device io.ReadWriteCloser) (*MotorLink, error) {
	var nonce [4]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return nil, err
	}
	session := binary.LittleEndian.Uint32(nonce[:])
	if session == 0 {
		session = 1
	}
	return &MotorLink{device: device, session: session, lastStatus: "not_confirmed", wake: make(chan struct{}, 1), closed: make(chan struct{})}, nil
}

func (l *MotorLink) Close() error {
	l.closeOnce.Do(func() { close(l.closed); l.closeError = l.device.Close() })
	return l.closeError
}

// Update never waits for I2C. The worker keeps only the latest sampled intent.
func (l *MotorLink) Update(hold bool, axes [4]float64) {
	l.mu.Lock()
	released := l.hold && !hold
	l.hold, l.axes, l.updated = hold, axes, time.Now()
	if !hold {
		l.faulted = false
	}
	l.mu.Unlock()
	if released {
		select {
		case l.wake <- struct{}{}:
		default:
		}
	}
}

// Run owns bus I/O. A blocked adapter cannot block UDP safety handling; the ESP
// expires the last command independently. There is never an I/O goroutine queue.
func (l *MotorLink) Run(ctx context.Context) {
	ticker := time.NewTicker(motorPeriod)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-l.closed:
			return
		default:
		}
		l.mu.Lock()
		hold := l.hold && !l.faulted && time.Since(l.updated) < motorInputLease
		axes := l.axes
		l.mu.Unlock()
		if err := l.send(hold, axes); err != nil && hold {
			l.mu.Lock()
			l.faulted = true
			l.mu.Unlock()
		}
		select {
		case <-ctx.Done():
			return
		case <-l.closed:
			return
		case <-ticker.C:
		case <-l.wake:
		}
	}
}

func crc8b(b []byte) byte {
	var crc byte
	for _, v := range b {
		crc ^= v
		for i := 0; i < 8; i++ {
			if crc&0x80 != 0 {
				crc = (crc << 1) ^ 7
			} else {
				crc <<= 1
			}
		}
	}
	return crc
}
func milli(v float64) int16 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	return int16(math.Round(math.Max(-1, math.Min(1, v)) * 1000))
}
func motorPacket(session uint32, seq uint16, hold bool, axes [4]float64) [24]byte {
	var p [24]byte
	p[0], p[1], p[11] = 0xE1, 1, 20
	binary.LittleEndian.PutUint32(p[2:], session)
	binary.LittleEndian.PutUint16(p[6:], seq)
	if hold {
		p[10] = 3
		for i, a := range axes {
			binary.LittleEndian.PutUint16(p[12+2*i:], uint16(milli(a)))
		}
	}
	p[23] = crc8b(p[:23])
	return p
}

func motorAck(st []byte, session uint32, seq uint16) (string, error) {
	if len(st) != 24 || st[0] != 0xE3 || st[1] != 1 || st[23] != crc8b(st[:23]) ||
		binary.LittleEndian.Uint32(st[2:]) != session || binary.LittleEndian.Uint16(st[6:]) != seq || st[8] != 0xE1 {
		return "", errors.New("no matching E3 acknowledgement")
	}
	result, reason, flags := st[9], st[10], st[13]
	if st[11] != 0 || st[12] > 1 || flags & ^byte(7) != 0 || st[22] != 0 ||
		(result != 1 && result != 2 && result != 4) || (result == 4 && (reason == 0 || reason > 7)) ||
		(result != 4 && reason != 0) || (result == 2 && flags&7 != 5) {
		return "", errors.New("invalid E3 status")
	}
	status := fmt.Sprintf("esp_ack=%d reject=%d armed=%t failsafe=%t deadman=%t", result, reason, flags&1 != 0, flags&2 != 0, flags&4 != 0)
	if result == 4 {
		return status, fmt.Errorf("ESP32 rejected command: reason %d", reason)
	}
	return status, nil
}

func (l *MotorLink) setStatus(status string, confirmed bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.lastStatus = status
	if confirmed {
		l.lastOK = time.Now()
	}
}
func (l *MotorLink) send(hold bool, axes [4]float64) error {
	l.seq++
	p := motorPacket(l.session, l.seq, hold, axes)
	n, err := l.device.Write(p[:])
	if err == nil && n != len(p) {
		err = io.ErrShortWrite
	}
	if err != nil {
		l.setStatus("write_failed", false)
		return err
	}
	for attempt := 0; attempt < 4; attempt++ {
		var st [24]byte
		n, err = l.device.Read(st[:])
		if err == nil && n == 24 {
			status, ackErr := motorAck(st[:], l.session, l.seq)
			if status != "" {
				l.setStatus(status, true)
				return ackErr
			}
		}
		time.Sleep(2 * time.Millisecond)
	}
	l.setStatus("status_unavailable", false)
	return errors.New("no valid matching E3 acknowledgement")
}
func (l *MotorLink) Status() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	status := l.lastStatus
	if time.Since(l.lastOK) > 500*time.Millisecond {
		status = "stale: " + status
	}
	if l.faulted {
		status += " input_reset_required=true"
	}
	return status
}
