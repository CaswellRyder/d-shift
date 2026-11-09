// Package laptop owns the authenticated bench recording client.
package laptop

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"net"
	"net/netip"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"dtr.local/link/protocol"
)

var ErrUnknown = errors.New("mission outcome unknown; explicit reset required")
var ErrLink = errors.New("authenticated bench link unavailable")

const launchUnavailableReason = "ESP32 flight controller and launch command are not integrated; no controller acknowledgement is available."

// LaunchCapability describes controller execution, independently of bench-link health.
type LaunchCapability struct {
	Available              bool   `json:"available"`
	ControllerAcknowledged bool   `json:"controller_acknowledged"`
	Reason                 string `json:"reason"`
}

func unavailableLaunch() LaunchCapability {
	return LaunchCapability{Reason: launchUnavailableReason}
}

type State struct {
	PauseRequested   bool                `json:"pause_requested"`
	PauseStatus      string              `json:"pause_status"`
	InputGeneration  string              `json:"input_generation"`
	EstopRequested   bool                `json:"estop_requested"`
	EstopStatus      string              `json:"estop_status"`
	InputInhibited   bool                `json:"input_inhibited"`
	Connected        bool                `json:"connected"`
	Telemetry        *protocol.Telemetry `json:"telemetry"`
	AgeMS            *int64              `json:"age_ms"`
	MissionUnknown   bool                `json:"mission_unknown"`
	BenchOnly        bool                `json:"bench_only"`
	Applied          bool                `json:"applied"`
	ControllerActual any                 `json:"controller_actual"`
	ESPStatus        any                 `json:"esp_status"`
	LocalJournalOK   *bool               `json:"local_journal_ok"`
	Launch           LaunchCapability    `json:"launch"`
}
type Client struct {
	controlMu     sync.Mutex
	paused        atomic.Bool
	pauseStatus   string
	safetyPending atomic.Int64
	inhibited     atomic.Bool
	generation    atomic.Uint64
	epoch         uint64
	estopStatus   string
	mu            sync.Mutex
	conn          *net.UDPConn
	key           []byte
	session       protocol.SessionID
	seq           map[protocol.Kind]uint64
	tick          uint64
	anchor        time.Time
	telemetry     *protocol.Telemetry
	received      time.Time
	telemetryAt   time.Time
	unknown       bool
	journal       *Journal
}

func New(peer string, key []byte, journal *Journal) (*Client, error) {
	return NewOnInterface(peer, "", key, journal)
}

// NewOnInterface binds the UDP source to an IPv4 address on the selected
// interface's directly connected Pi subnet. It never falls back to Ethernet.
func NewOnInterface(peer, interfaceName string, key []byte, journal *Journal) (*Client, error) {
	endpoint, err := netip.ParseAddrPort(peer)
	if err != nil || endpoint.Port() == 0 || endpoint.Addr().IsUnspecified() || endpoint.Addr().IsMulticast() || len(key) != 32 {
		return nil, ErrLink
	}
	addr := net.UDPAddrFromAddrPort(endpoint)
	var local *net.UDPAddr
	if interfaceName != "" {
		iface, e := net.InterfaceByName(interfaceName)
		if e != nil || iface.Flags&net.FlagUp == 0 || !endpoint.Addr().Is4() {
			return nil, ErrLink
		}
		addresses, e := iface.Addrs()
		if e != nil {
			return nil, ErrLink
		}
		for _, address := range addresses {
			subnet, ok := address.(*net.IPNet)
			if ok && subnet.IP.To4() != nil && subnet.Contains(addr.IP) {
				if local != nil {
					return nil, ErrLink
				}
				local = &net.UDPAddr{IP: subnet.IP.To4()}
			}
		}
		if local == nil {
			return nil, ErrLink
		}
	}
	conn, err := net.DialUDP("udp", local, addr)
	if err != nil {
		return nil, ErrLink
	}
	c := &Client{conn: conn, key: append([]byte(nil), key...), seq: make(map[protocol.Kind]uint64), journal: journal}
	var seed [8]byte
	if _, err := rand.Read(seed[:]); err != nil {
		conn.Close()
		return nil, ErrLink
	}
	var generation uint64
	for _, b := range seed {
		generation = generation<<8 | uint64(b)
	}
	c.generation.Store(generation & protocol.MaxSequence)
	return c, nil
}
func (c *Client) Close() error { return c.conn.Close() }

// InterfaceCurrent reports whether this socket's source still belongs to the
// requested interface. The station supervisor restarts after a DHCP change.
func (c *Client) InterfaceCurrent(name string) bool {
	iface, err := net.InterfaceByName(name)
	if err != nil || iface.Flags&net.FlagUp == 0 {
		return false
	}
	addresses, err := iface.Addrs()
	if err != nil {
		return false
	}
	local := c.conn.LocalAddr().(*net.UDPAddr).IP
	for _, address := range addresses {
		if subnet, ok := address.(*net.IPNet); ok && subnet.IP.Equal(local) {
			return true
		}
	}
	return false
}

func (c *Client) State() State {
	c.mu.Lock()
	defer c.mu.Unlock()
	s := State{BenchOnly: true, MissionUnknown: c.unknown, EstopRequested: c.inhibited.Load(), InputInhibited: c.inhibited.Load() || c.paused.Load(), PauseRequested: c.paused.Load(), PauseStatus: c.pauseStatus, EstopStatus: c.estopStatus, Launch: unavailableLaunch()}
	if s.PauseStatus == "" {
		s.PauseStatus = "clear"
	}
	s.InputGeneration = strconv.FormatUint(c.generation.Load(), 10)
	if s.EstopStatus == "" {
		s.EstopStatus = "clear"
	}
	if c.journal != nil {
		ok := c.journal.OK()
		s.LocalJournalOK = &ok
	}
	if !c.received.IsZero() {
		age := time.Since(c.received).Milliseconds()
		s.AgeMS = &age
		if age <= 750 {
			s.Connected = c.session != (protocol.SessionID{})
			t := *c.telemetry
			if t.Vision.AgeMS != nil {
				// Include the full request/response interval conservatively, then
				// advance the source age while this snapshot remains cached.
				x := *t.Vision.AgeMS + float64(time.Since(c.telemetryAt).Milliseconds())
				t.Vision.AgeMS = &x
			}
			if t.Vision.AgeMS == nil || *t.Vision.AgeMS < 0 || *t.Vision.AgeMS > 2000 || !t.Vision.Ready {
				t.Vision.Ready = false
				t.Vision.Target, t.Vision.Avoid = nil, nil
				if t.Vision.Error == "" {
					t.Vision.Error = "stale"
				}
			}
			s.Telemetry = &t
		}
	}
	return s
}
func (c *Client) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.unknown = false
	c.record("operator_reset", nil)
}
func (c *Client) record(event string, receipt *protocol.Receipt) {
	if c.journal != nil {
		c.journal.Record(event, receipt)
	}
}
func (c *Client) exchange(ctx context.Context, p protocol.Packet, want protocol.Kind, budget time.Duration, retry bool) (protocol.Packet, time.Time, error) {
	generation := c.generation.Load()
	if expected, ok := ctx.Value(exchangeGenerationKey{}).(uint64); ok {
		generation = expected
	}
	start := time.Now()
	deadline := start.Add(budget)
	if d, ok := ctx.Deadline(); ok && d.Before(deadline) {
		deadline = d
	}
	wire, err := protocol.Marshal(c.key, p)
	if err != nil {
		return protocol.Packet{}, start, ErrLink
	}
	next := start
	for time.Now().Before(deadline) {
		if generation != c.generation.Load() || ((p.Kind == protocol.Manual || p.Kind == protocol.Mission) && c.safetyPending.Load() > 0) {
			return protocol.Packet{}, start, ErrLink
		}
		if ctx.Err() != nil {
			return protocol.Packet{}, start, ctx.Err()
		}
		if !time.Now().Before(next) {
			if !time.Now().Before(deadline) || ctx.Err() != nil {
				break
			}
			if _, err = c.conn.Write(wire); err != nil {
				return protocol.Packet{}, start, ErrLink
			}
			if retry {
				next = time.Now().Add(100 * time.Millisecond)
			} else {
				next = deadline
			}
		}
		readUntil := deadline
		if next.Before(readUntil) {
			readUntil = next
		}
		c.conn.SetReadDeadline(readUntil)
		var b [protocol.MaxDatagram + 1]byte
		n, err := c.conn.Read(b[:])
		if err != nil {
			if ne, ok := err.(net.Error); ok && ne.Timeout() {
				continue
			}
			return protocol.Packet{}, start, ErrLink
		}
		r, err := protocol.Unmarshal(c.key, b[:n])
		if err != nil || r.Kind != want || r.Sequence != p.Sequence {
			continue
		}
		if want != protocol.Welcome && r.Session != p.Session {
			continue
		}
		if want == protocol.MissionAck || want == protocol.ManualAck || want == protocol.SafetyAck {
			var ack protocol.Receipt
			if protocol.DecodeJSON(r.Payload, &ack) != nil || ack.Digest != protocol.Digest(p.Payload) || !ack.BenchOnly || ack.Applied || (ack.Status != protocol.StatusQueued && ack.Status != protocol.StatusBusy && ack.Status != protocol.StatusRejected) {
				continue
			}
		}
		if !time.Now().Before(deadline) {
			break
		}
		if generation != c.generation.Load() {
			return protocol.Packet{}, start, ErrLink
		}
		return r, start, nil
	}
	return protocol.Packet{}, start, ErrLink
}
func (c *Client) hello(ctx context.Context) error {
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return ErrLink
	}
	ns := hex.EncodeToString(nonce[:])
	raw, _ := protocol.JSON(protocol.HelloPayload{Nonce: ns})
	r, sent, err := c.exchange(ctx, protocol.Packet{Kind: protocol.Hello, Payload: raw}, protocol.Welcome, 200*time.Millisecond, false)
	if err != nil {
		return err
	}
	var w protocol.WelcomePayload
	if protocol.DecodeJSON(r.Payload, &w) != nil || w.Nonce != ns || !w.BenchOnly || w.Applied || r.Session == (protocol.SessionID{}) {
		return ErrLink
	}
	c.session = r.Session
	c.seq = make(map[protocol.Kind]uint64)
	c.tick = r.TickMS
	c.anchor = sent
	return nil
}
func (c *Client) packet(kind protocol.Kind, v any) (protocol.Packet, error) {
	if c.seq[kind] >= protocol.MaxSequence {
		return protocol.Packet{}, ErrLink
	}
	c.seq[kind]++
	raw, err := protocol.JSON(v)
	return protocol.Packet{Kind: kind, Sequence: c.seq[kind], Session: c.session, TickMS: c.tick, Payload: raw}, err
}
func (c *Client) poll(ctx context.Context) error {
	p, err := c.packet(protocol.Poll, struct{}{})
	if err != nil {
		return err
	}
	r, sent, err := c.exchange(ctx, p, protocol.TelemetryReply, 200*time.Millisecond, false)
	if err != nil {
		return err
	}
	var t protocol.Telemetry
	if protocol.DecodeJSON(r.Payload, &t) != nil || !t.BenchOnly || t.Applied {
		return ErrLink
	}
	c.tick = r.TickMS
	c.anchor = sent
	c.epoch = t.ControlEpoch
	if t.BenchPaused {
		c.paused.Store(true)
		c.pauseStatus = "confirmed"
	}
	if t.BenchEstop {
		c.inhibited.Store(true)
		c.estopStatus = "confirmed"
	}
	c.telemetry = &t
	c.telemetryAt = sent
	c.received = time.Now()
	return nil
}
func (c *Client) connect(ctx context.Context) error {
	generation := c.generation.Load()
	deadline := time.Now().Add(3 * time.Second)
	if d, ok := ctx.Deadline(); ok && d.Before(deadline) {
		deadline = d
	}
	ctx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	for ctx.Err() == nil {
		if generation != c.generation.Load() {
			return ErrLink
		}
		if c.hello(ctx) == nil && c.poll(ctx) == nil {
			return nil
		}
		select {
		case <-ctx.Done():
			return ErrLink
		case <-time.After(50 * time.Millisecond):
		}
	}
	return ErrLink
}
func (c *Client) Poll(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.session == (protocol.SessionID{}) || time.Since(c.anchor) > 300*time.Millisecond {
		return c.connect(ctx)
	}
	if err := c.poll(ctx); err != nil {
		c.session = protocol.SessionID{}
		return err
	}
	return nil
}
func (c *Client) Run(ctx context.Context) {
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		_ = c.Poll(ctx)
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

type manualGenerationKey struct{}
type exchangeGenerationKey struct{}

// WithControlGeneration binds an explicit operator action to the state it observed.
func WithControlGeneration(ctx context.Context, generation string) (context.Context, error) {
	value, err := strconv.ParseUint(generation, 10, 64)
	if err != nil {
		return nil, errors.New("fresh input generation required")
	}
	return context.WithValue(ctx, manualGenerationKey{}, value), nil
}

func (c *Client) command(ctx context.Context, kind protocol.Kind, v any) (protocol.Receipt, error) {
	generation := c.generation.Load()
	ctx = context.WithValue(ctx, exchangeGenerationKey{}, generation)
	if expected, ok := ctx.Value(manualGenerationKey{}).(uint64); ok && expected != generation {
		return protocol.Receipt{}, errors.New("old manual input generation")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if generation != c.generation.Load() || (c.inhibited.Load() || c.paused.Load()) || c.safetyPending.Load() > 0 {
		return protocol.Receipt{}, errors.New("bench input inhibited")
	}
	if ctx.Err() != nil {
		return protocol.Receipt{}, ctx.Err()
	}
	if kind == protocol.Mission && c.unknown {
		return protocol.Receipt{}, ErrUnknown
	}
	if c.journal != nil && !c.journal.OK() {
		return protocol.Receipt{}, errors.New("journal unavailable")
	}
	if c.session == (protocol.SessionID{}) || time.Since(c.anchor) > 100*time.Millisecond {
		// A manual sample belongs to this operator instant. Never hold it through
		// reconnect; polling may reconnect separately and the operator must sample again.
		if kind == protocol.Manual {
			if c.session == (protocol.SessionID{}) {
				return protocol.Receipt{}, ErrLink
			}
			if err := c.poll(ctx); err != nil {
				c.session = protocol.SessionID{}
				return protocol.Receipt{}, err
			}
		} else if err := c.connectOrPoll(ctx); err != nil {
			return protocol.Receipt{}, err
		}
	}
	if generation != c.generation.Load() || (c.inhibited.Load() || c.paused.Load()) || c.safetyPending.Load() > 0 {
		return protocol.Receipt{}, errors.New("bench input inhibited")
	}
	switch value := v.(type) {
	case protocol.ManualPayload:
		value.Epoch = c.epoch
		v = value
	case protocol.MissionPayload:
		value.Epoch = c.epoch
		v = value
	}
	p, err := c.packet(kind, v)
	if err != nil {
		return protocol.Receipt{}, err
	}
	want := protocol.ManualAck
	if kind == protocol.Mission {
		want = protocol.MissionAck
	}
	r, sent, err := c.exchange(ctx, p, want, 200*time.Millisecond, kind == protocol.Mission)
	if err == nil {
		var receipt protocol.Receipt
		err = protocol.DecodeJSON(r.Payload, &receipt)
		if err == nil && receipt.Digest == protocol.Digest(p.Payload) && receipt.BenchOnly && !receipt.Applied && (receipt.Status == protocol.StatusQueued || receipt.Status == protocol.StatusBusy || receipt.Status == protocol.StatusRejected) {
			c.tick = r.TickMS
			c.anchor = sent
			c.record("receipt", &receipt)
			return receipt, nil
		}
		err = ErrLink
	}
	if kind == protocol.Mission {
		c.unknown = true
		c.record("mission_unknown", nil)
		return protocol.Receipt{}, ErrUnknown
	}
	return protocol.Receipt{}, err
}
func (c *Client) connectOrPoll(ctx context.Context) error {
	if c.session == (protocol.SessionID{}) {
		return c.connect(ctx)
	}
	if err := c.poll(ctx); err != nil {
		c.session = protocol.SessionID{}
		return c.connect(ctx)
	}
	return nil
}
func (c *Client) Manual(ctx context.Context, p protocol.ManualPayload) (protocol.Receipt, error) {
	ctx, cancel := context.WithTimeout(ctx, 250*time.Millisecond)
	defer cancel()
	if !ValidManual(p) {
		return protocol.Receipt{}, errors.New("invalid manual sample")
	}
	return c.command(ctx, protocol.Manual, p)
}
func (c *Client) Mission(ctx context.Context, p protocol.MissionPayload) (protocol.Receipt, error) {
	if len(p.Args) > 96 {
		return protocol.Receipt{}, errors.New("invalid mission")
	}
	if _, err := hex.DecodeString(p.Args); err != nil {
		return protocol.Receipt{}, errors.New("invalid mission")
	}
	return c.command(ctx, protocol.Mission, p)
}
func ValidManual(p protocol.ManualPayload) bool {
	for _, v := range []float64{p.Forward, p.Right, p.Up, p.Yaw} {
		if !(v >= -1 && v <= 1) || (!p.Hold && v != 0) {
			return false
		}
	}
	return true
}

// Safety interrupts pending exchanges before acquiring the transport lock.
// Both local latches survive transport failures and mission outcome resets.
func (c *Client) Safety(ctx context.Context, action string) (protocol.Receipt, error) {
	if action == "stop" {
		action = "pause"
	}
	if action != "pause" && action != "resume" && action != "estop" && action != "reset" {
		return protocol.Receipt{}, errors.New("invalid safety action")
	}
	clearing := action == "reset" || action == "resume"
	c.controlMu.Lock()
	if action == "resume" && c.inhibited.Load() {
		c.controlMu.Unlock()
		return protocol.Receipt{}, errors.New("E-stop must be reset before resume")
	}
	var generation uint64
	if clearing {
		expected, ok := ctx.Value(manualGenerationKey{}).(uint64)
		if !ok || !c.generation.CompareAndSwap(expected, expected+1) {
			c.controlMu.Unlock()
			return protocol.Receipt{}, errors.New("fresh safety generation required")
		}
		generation = expected + 1
	} else {
		generation = c.generation.Add(1)
	}
	c.safetyPending.Add(1)
	if action == "estop" {
		c.inhibited.Store(true)
		c.paused.Store(true)
	}
	if action == "pause" {
		c.paused.Store(true)
	}
	c.controlMu.Unlock()
	ctx = context.WithValue(ctx, exchangeGenerationKey{}, generation)
	defer c.safetyPending.Add(-1)
	c.conn.SetReadDeadline(time.Now())
	c.mu.Lock()
	defer c.mu.Unlock()
	if generation != c.generation.Load() {
		return protocol.Receipt{}, ErrLink
	}
	if action == "estop" {
		c.estopStatus = "requested"
	}
	if action == "pause" {
		c.pauseStatus = "requested"
	}
	c.record("safety_"+action, nil)
	fail := func() {
		if action == "estop" || action == "reset" {
			c.inhibited.Store(true)
			c.estopStatus = "unknown"
		}
		if action == "pause" || action == "resume" {
			c.paused.Store(true)
			c.pauseStatus = "unknown"
		}
	}
	if err := c.connectOrPoll(ctx); err != nil {
		fail()
		return protocol.Receipt{}, err
	}
	if action == "resume" && c.inhibited.Load() {
		return protocol.Receipt{}, errors.New("E-stop must be reset before resume")
	}
	if clearing && c.journal != nil && !c.journal.OK() {
		fail()
		return protocol.Receipt{}, errors.New("journal unavailable")
	}
	// An emergency request may have superseded a Pause before either reached
	// the receiver. Establish the requested Pause before any emergency clear,
	// including recovery from an unknown/lost E-stop outcome.
	if action == "reset" && c.paused.Load() && (c.telemetry == nil || !c.telemetry.BenchPaused) {
		barrier, err := c.packet(protocol.Safety, protocol.SafetyPayload{Action: "pause", Epoch: c.epoch})
		if err != nil {
			fail()
			return protocol.Receipt{}, err
		}
		reply, sent, err := c.exchange(ctx, barrier, protocol.SafetyAck, 200*time.Millisecond, true)
		if err != nil {
			fail()
			return protocol.Receipt{}, err
		}
		var ack protocol.Receipt
		if protocol.DecodeJSON(reply.Payload, &ack) != nil || !ack.BenchPaused {
			fail()
			return protocol.Receipt{}, ErrLink
		}
		c.epoch, c.tick, c.anchor = ack.Epoch, reply.TickMS, sent
		c.pauseStatus = "confirmed"
		c.record("safety_receipt", &ack)
	}
	// A newer Pause may supersede an E-stop exchange before it reached the Pi.
	// Propagate that stronger local latch first; Pause must never suppress it.
	if action == "pause" && c.inhibited.Load() && (c.telemetry == nil || !c.telemetry.BenchEstop) {
		priority, err := c.packet(protocol.Safety, protocol.SafetyPayload{Action: "estop", Epoch: c.epoch})
		if err != nil {
			fail()
			return protocol.Receipt{}, err
		}
		reply, sent, err := c.exchange(ctx, priority, protocol.SafetyAck, 200*time.Millisecond, true)
		if err != nil {
			c.estopStatus = "unknown"
			fail()
			return protocol.Receipt{}, err
		}
		var ack protocol.Receipt
		if protocol.DecodeJSON(reply.Payload, &ack) != nil || !ack.BenchEstop {
			c.estopStatus = "unknown"
			fail()
			return protocol.Receipt{}, ErrLink
		}
		c.epoch = ack.Epoch
		c.tick = reply.TickMS
		c.anchor = sent
		c.estopStatus = "confirmed"
		c.record("safety_receipt", &ack)
	}
	p, err := c.packet(protocol.Safety, protocol.SafetyPayload{Action: action, Epoch: c.epoch})
	if err != nil {
		fail()
		return protocol.Receipt{}, err
	}
	r, sent, err := c.exchange(ctx, p, protocol.SafetyAck, 200*time.Millisecond, true)
	if err != nil {
		fail()
		return protocol.Receipt{}, err
	}
	var receipt protocol.Receipt
	if protocol.DecodeJSON(r.Payload, &receipt) != nil {
		fail()
		return protocol.Receipt{}, ErrLink
	}
	c.tick = r.TickMS
	c.anchor = sent
	c.epoch = receipt.Epoch
	c.controlMu.Lock()
	defer c.controlMu.Unlock()
	if receipt.BenchEstop {
		c.inhibited.Store(true)
		c.estopStatus = "confirmed"
	} else if action == "reset" && receipt.Status == protocol.StatusQueued && generation == c.generation.Load() {
		c.inhibited.Store(false)
		c.estopStatus = "clear"
	} else if action == "estop" {
		c.estopStatus = "unknown"
	}
	if receipt.BenchPaused {
		c.paused.Store(true)
		c.pauseStatus = "confirmed"
	} else if action == "resume" && receipt.Status == protocol.StatusQueued && generation == c.generation.Load() {
		c.paused.Store(false)
		c.pauseStatus = "clear"
	} else if action == "pause" {
		c.pauseStatus = "unknown"
	}
	c.record("safety_receipt", &receipt)
	return receipt, nil
}
