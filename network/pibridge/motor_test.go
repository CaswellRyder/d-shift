package pibridge

import (
	"context"
	"dtr.local/link/protocol"
	"encoding/binary"
	"encoding/hex"
	"io"
	"math"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestMotorPacketGoldenAndBounds(t *testing.T) {
	if crc8b([]byte("123456789")) != 0xF4 {
		t.Fatal("CRC vector")
	}
	p := motorPacket(1, 42, true, [4]float64{0, 0, .2, 0})
	if hex.EncodeToString(p[:]) != "e101010000002a000000031400000000c80000000000007e" {
		t.Fatalf("wire mismatch: %x", p)
	}
	for _, tc := range []struct {
		v    float64
		want int16
	}{{.2, 200}, {-.2, -200}, {10, 1000}, {-10, -1000}, {math.NaN(), 0}, {math.Inf(1), 0}} {
		if milli(tc.v) != tc.want {
			t.Fatal("axis scaling", tc)
		}
	}
	p = motorPacket(1, 43, false, [4]float64{1, 1, 1, 1})
	for _, b := range p[10:11] {
		if b != 0 {
			t.Fatal("release deadman")
		}
	}
	for _, b := range p[12:23] {
		if b != 0 {
			t.Fatal("release carried axes")
		}
	}
}
func acknowledgement(p []byte) [24]byte {
	var s [24]byte
	s[0], s[1], s[8], s[9] = 0xE3, 1, 0xE1, 1
	copy(s[2:8], p[2:8])
	s[23] = crc8b(s[:23])
	return s
}
func TestMotorAckValidation(t *testing.T) {
	p := motorPacket(1, 42, true, [4]float64{})
	for _, tc := range []struct {
		name      string
		change    func(*[24]byte)
		wantError bool
	}{
		{"received", func(s *[24]byte) {}, false},
		{"applied", func(s *[24]byte) { s[9] = 2; s[12] = 1; s[13] = 5 }, false},
		{"rejected", func(s *[24]byte) { s[9] = 4; s[10] = 7 }, true},
		{"stale_sequence", func(s *[24]byte) { binary.LittleEndian.PutUint16(s[6:], 41) }, true},
		{"other_session", func(s *[24]byte) { s[2] = 2 }, true},
		{"reserved", func(s *[24]byte) { s[22] = 1 }, true},
		{"applied_rejection", func(s *[24]byte) { s[9] = 2; s[10] = 5; s[13] = 5 }, true},
		{"applied_disarmed", func(s *[24]byte) { s[9] = 2 }, true},
		{"unknown_result", func(s *[24]byte) { s[9] = 255 }, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := acknowledgement(p[:])
			tc.change(&s)
			s[23] = crc8b(s[:23])
			_, err := motorAck(s[:], 1, 42)
			if (err != nil) != tc.wantError {
				t.Fatal(err)
			}
		})
	}
	s := acknowledgement(p[:])
	s[23] ^= 1
	if _, err := motorAck(s[:], 1, 42); err == nil {
		t.Fatal("bad CRC accepted")
	}
}

type motorDevice struct {
	mu     sync.Mutex
	last   [24]byte
	writes chan [24]byte
	closed chan struct{}
	once   sync.Once
	block  bool
	reject atomic.Bool
}

func newMotorDevice(block bool) *motorDevice {
	return &motorDevice{writes: make(chan [24]byte, 100), closed: make(chan struct{}), block: block}
}
func (d *motorDevice) Write(p []byte) (int, error) {
	d.mu.Lock()
	copy(d.last[:], p)
	frame := d.last
	d.mu.Unlock()
	select {
	case d.writes <- frame:
	case <-d.closed:
		return 0, io.EOF
	}
	return len(p), nil
}
func (d *motorDevice) Read(p []byte) (int, error) {
	if d.block {
		<-d.closed
		return 0, io.EOF
	}
	d.mu.Lock()
	s := acknowledgement(d.last[:])
	if d.reject.Load() {
		s[9] = 4
		s[10] = 7
		s[23] = crc8b(s[:23])
	}
	d.mu.Unlock()
	return copy(p, s[:]), nil
}
func (d *motorDevice) Close() error { d.once.Do(func() { close(d.closed) }); return nil }
func TestMotorWorkerExpiresUnrefreshedInput(t *testing.T) {
	d := newMotorDevice(false)
	l, err := newMotorLink(d)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	l.Update(true, [4]float64{0, 0, .2, 0})
	go l.Run(ctx)
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	seenHold := false
	for {
		select {
		case p := <-d.writes:
			if p[10] == 3 {
				seenHold = true
			} else if seenHold {
				return
			}
		case <-timer.C:
			t.Fatal("motor input did not expire")
		}
	}
}
func TestBlockedMotorCannotBlockUDPOrServerShutdown(t *testing.T) {
	d := newMotorDevice(true)
	l, err := newMotorLink(d)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	conn, err := Listen("127.0.0.1", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	server := NewServer(&recording{ok: true}, nil)
	server.Motor = l
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	key := make([]byte, 32)
	go func() { done <- server.Serve(ctx, conn, key) }()
	client, err := net.DialUDP("udp", nil, conn.LocalAddr().(*net.UDPAddr))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	exchange := func(p protocol.Packet) protocol.Packet {
		t.Helper()
		wire, err := protocol.Marshal(key, p)
		if err != nil {
			t.Fatal(err)
		}
		client.SetDeadline(time.Now().Add(time.Second))
		if _, err = client.Write(wire); err != nil {
			t.Fatal(err)
		}
		b := make([]byte, protocol.MaxDatagram+1)
		n, err := client.Read(b)
		if err != nil {
			t.Fatal(err)
		}
		out, err := protocol.Unmarshal(key, b[:n])
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	select {
	case <-d.writes:
	case <-time.After(time.Second):
		t.Fatal("motor worker not started")
	}
	w := exchange(packet(protocol.Hello, protocol.SessionID{}, 0, 0, protocol.HelloPayload{Nonce: "01010101010101010101010101010101"}))
	p := exchange(packet(protocol.Poll, w.Session, 1, w.TickMS, struct{}{}))
	ack := exchange(packet(protocol.Safety, w.Session, 1, p.TickMS, protocol.SafetyPayload{Action: "estop"}))
	var receipt protocol.Receipt
	if protocol.DecodeJSON(ack.Payload, &receipt) != nil || !receipt.BenchEstop {
		t.Fatal("blocked I2C prevented E-stop", receipt)
	}
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("blocked I2C prevented shutdown")
	}
}

func TestMotorFaultRequiresReleaseBeforeHeldInputCanResume(t *testing.T) {
	d := newMotorDevice(false)
	d.reject.Store(true)
	l, err := newMotorLink(d)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	l.Update(true, [4]float64{0, 0, .2, 0})
	go l.Run(ctx)
	deadline := time.Now().Add(time.Second)
	for !strings.Contains(l.Status(), "input_reset_required=true") {
		if time.Now().After(deadline) {
			t.Fatal("rejection did not latch motor inhibition")
		}
		time.Sleep(time.Millisecond)
	}
	d.reject.Store(false)
	// Discard the initial held frame that triggered rejection.
	<-d.writes
	l.Update(true, [4]float64{0, 0, .2, 0})
	select {
	case p := <-d.writes:
		if p[10] != 0 {
			t.Fatal("held input automatically resumed after fault")
		}
	case <-time.After(time.Second):
		t.Fatal("no inhibited frame")
	}
	l.Update(false, [4]float64{})
	l.Update(true, [4]float64{0, 0, .2, 0})
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	for {
		select {
		case p := <-d.writes:
			if p[10] == 3 {
				return
			}
		case <-timer.C:
			t.Fatal("fresh input after release did not resume")
		}
	}
}
