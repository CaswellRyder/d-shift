package pibridge

import (
	"context"
	"dtr.local/link/protocol"
	"encoding/hex"
	"errors"
	"math"
	"net"
	"net/netip"
	"os"
	"strconv"
	"strings"
	"time"
)

type challenge struct {
	source  netip.AddrPort
	nonce   string
	session protocol.SessionID
	created uint64
}
type Server struct {
	estop          bool
	paused         bool
	epoch          uint64
	safetyDigest   string
	safetyReceipt  protocol.Receipt
	Recorder       Recorder
	Source         *Source
	Motor          *MotorLink
	start          time.Time
	pending        []challenge
	rateStart      uint64
	rate           int
	owner          netip.AddrPort
	session        protocol.SessionID
	last           uint64
	seq            map[protocol.Kind]uint64
	manual         protocol.ManualState
	manualAt       uint64
	mission        *protocol.MissionState
	missionDigest  string
	missionReceipt protocol.Receipt
}

func NewServer(r Recorder, s *Source) *Server {
	return &Server{Recorder: r, Source: s, start: time.Now(), seq: make(map[protocol.Kind]uint64)}
}

// NewPausedServer initializes bench inhibition before any packet can be handled.
// A new session does not clear it; an explicit Resume is required. This does not
// establish physical position hold or any ESP32 controller state.
func NewPausedServer(r Recorder, source *Source) *Server {
	s := NewServer(r, source)
	s.paused = true
	s.epoch = 1
	s.manual.Reason = "startup_pause"
	return s
}
func (s *Server) neutral(now uint64, reason string) {
	if s.manual.Hold {
		s.Recorder.Manual(Event{Kind: "neutral", TickMS: now, Reason: reason, BenchOnly: true})
		s.manual.Hold = false
		s.manual.Axes = nil
		s.manual.Reason = reason
	}
}
func (s *Server) Advance(now uint64) {
	if s.manual.Hold && now-s.manualAt >= 300 {
		s.neutral(now, "manual_expired")
	}
	if s.session != (protocol.SessionID{}) && now-s.last >= 2000 {
		s.neutral(now, "session_expired")
		s.session = protocol.SessionID{}
		s.seq = make(map[protocol.Kind]uint64)
		s.mission = nil
	}
}
func response(kind protocol.Kind, p protocol.Packet, now uint64, value any) *protocol.Packet {
	raw, e := protocol.JSON(value)
	if e != nil {
		return nil
	}
	return &protocol.Packet{Kind: kind, Sequence: p.Sequence, Session: p.Session, TickMS: now, Payload: raw}
}
func (s *Server) Handle(p protocol.Packet, source netip.AddrPort, now uint64) *protocol.Packet {
	s.Advance(now)
	if p.Kind == protocol.Hello {
		var h protocol.HelloPayload
		if protocol.DecodeJSON(p.Payload, &h) != nil || p.Sequence != 0 || p.TickMS != 0 || p.Session != (protocol.SessionID{}) {
			return nil
		}
		nonce, e := hex.DecodeString(h.Nonce)
		if e != nil || len(nonce) != 16 {
			return nil
		}
		kept := s.pending[:0]
		for _, c := range s.pending {
			if now-c.created < 1000 {
				kept = append(kept, c)
			}
		}
		s.pending = kept
		for _, c := range s.pending {
			if c.source == source && c.nonce == h.Nonce {
				p.Session = c.session
				return response(protocol.Welcome, p, now, protocol.WelcomePayload{Nonce: h.Nonce, BenchOnly: true})
			}
		}
		if now-s.rateStart >= 1000 {
			s.rateStart = now
			s.rate = 0
		}
		if len(s.pending) >= 4 || s.rate >= 8 {
			return nil
		}
		id, e := protocol.NewSession()
		if e != nil {
			return nil
		}
		s.rate++
		s.pending = append(s.pending, challenge{source, h.Nonce, id, now})
		p.Session = id
		return response(protocol.Welcome, p, now, protocol.WelcomePayload{Nonce: h.Nonce, BenchOnly: true})
	}
	if p.Sequence == 0 || p.Sequence > protocol.MaxSequence || p.TickMS > now || now-p.TickMS > 300 {
		return nil
	}
	if p.Kind != protocol.Poll && p.Kind != protocol.Manual && p.Kind != protocol.Mission && p.Kind != protocol.Safety {
		return nil
	}
	if s.session == (protocol.SessionID{}) {
		if p.Kind != protocol.Poll {
			return nil
		}
		var empty struct{}
		if protocol.DecodeJSON(p.Payload, &empty) != nil {
			return nil
		}
		valid := false
		for _, c := range s.pending {
			if c.session == p.Session && c.source == source && now-c.created < 1000 {
				valid = true
			}
		}
		if !valid {
			return nil
		}
		s.session = p.Session
		s.owner = source
		s.pending = nil
		s.manual = protocol.ManualState{Reason: "new_session"}
		s.seq = make(map[protocol.Kind]uint64)
	}
	if p.Session != s.session || source != s.owner {
		return nil
	}
	digest := protocol.Digest(p.Payload)
	if p.Kind == protocol.Safety && p.Sequence == s.seq[protocol.Safety] {
		if digest != s.safetyDigest {
			return nil
		}
		return response(protocol.SafetyAck, p, now, s.safetyReceipt)
	}
	if p.Kind == protocol.Mission && p.Sequence == s.seq[protocol.Mission] {
		if digest != s.missionDigest {
			return nil
		}
		return response(protocol.MissionAck, p, now, s.missionReceipt)
	}
	if p.Sequence <= s.seq[p.Kind] {
		return nil
	}
	switch p.Kind {
	case protocol.Safety:
		var v protocol.SafetyPayload
		if protocol.DecodeJSON(p.Payload, &v) != nil || (v.Action != "stop" && v.Action != "pause" && v.Action != "resume" && v.Action != "estop" && v.Action != "reset") {
			return nil
		}
		if v.Action == "stop" {
			v.Action = "pause"
		}
		r := protocol.Receipt{Status: protocol.StatusRejected, Code: "epoch", Digest: digest, BenchOnly: true}
		if v.Epoch == s.epoch && s.epoch < protocol.MaxSequence && !(v.Action == "resume" && s.estop) {
			// Inhibit first even if the journal is unhealthy/full. Reset only after recording.
			if v.Action == "pause" || v.Action == "estop" {
				s.neutral(now, v.Action)
				s.manual.Hold = false
				s.manual.Axes = nil
				s.manual.Reason = v.Action
				if v.Action == "estop" {
					s.estop = true
					// Clearing emergency cutoff must never resume mission/input.
					s.paused = true
				} else {
					s.paused = true
				}
				s.epoch++
			}
			e := Event{Kind: "safety", TickMS: now, Sequence: p.Sequence, Value: v, BenchOnly: true}
			queued := false
			if q, ok := s.Recorder.(interface{ Safety(Event) bool }); ok {
				queued = q.Safety(e)
			} else {
				queued = s.Recorder.Mission(e)
			}
			r.Status = protocol.StatusBusy
			r.Code = "journal"
			if queued {
				r.Status = protocol.StatusQueued
				r.Code = ""
				if v.Action == "reset" {
					s.estop = false
					s.epoch++
				}
				if v.Action == "resume" {
					s.paused = false
					s.epoch++
				}
			}
		}
		r.Epoch = s.epoch
		r.BenchEstop = s.estop
		r.BenchPaused = s.paused
		s.seq[p.Kind] = p.Sequence
		s.last = now
		s.safetyDigest = digest
		s.safetyReceipt = r
		return response(protocol.SafetyAck, p, now, r)
	case protocol.Poll:
		var empty struct{}
		if protocol.DecodeJSON(p.Payload, &empty) != nil {
			return nil
		}
		s.seq[p.Kind] = p.Sequence
		s.last = now
		t := s.telemetry(now)
		out := response(protocol.TelemetryReply, p, now, t)
		if out == nil || len(out.Payload) > protocol.MaxPayload {
			t.Vision = protocol.Vision{Source: "unavailable", Error: "size_limit"}
			out = response(protocol.TelemetryReply, p, now, t)
		}
		return out
	case protocol.Manual:
		var m protocol.ManualPayload
		if protocol.DecodeJSON(p.Payload, &m) != nil {
			return nil
		}
		axes := [4]float64{m.Forward, m.Right, m.Up, m.Yaw}
		for _, x := range axes {
			if math.IsNaN(x) || math.IsInf(x, 0) || x < -1 || x > 1 || (!m.Hold && x != 0) {
				return nil
			}
		}
		r := protocol.Receipt{Status: protocol.StatusBusy, Digest: digest, BenchOnly: true, Epoch: s.epoch, BenchEstop: s.estop, BenchPaused: s.paused}
		if s.estop || s.paused || m.Epoch != s.epoch {
			r.Status = protocol.StatusRejected
			r.Code = "inhibited"
		} else if s.Recorder.Healthy() && s.Recorder.Manual(Event{Kind: "manual", TickMS: now, Sequence: p.Sequence, Value: m, BenchOnly: true}) {
			r.Status = protocol.StatusQueued
			s.manual = protocol.ManualState{Received: true, Sequence: p.Sequence, Hold: m.Hold, Axes: &axes}
			s.manualAt = now
		}
		s.seq[p.Kind] = p.Sequence
		s.last = now
		return response(protocol.ManualAck, p, now, r)
	case protocol.Mission:
		var m protocol.MissionPayload
		if protocol.DecodeJSON(p.Payload, &m) != nil {
			return nil
		}
		args, e := hex.DecodeString(m.Args)
		if e != nil || len(args) > 48 {
			return nil
		}
		r := protocol.Receipt{Status: protocol.StatusBusy, Digest: digest, BenchOnly: true, Epoch: s.epoch, BenchEstop: s.estop, BenchPaused: s.paused}
		if s.estop || s.paused || m.Epoch != s.epoch {
			r.Status = protocol.StatusRejected
			r.Code = "inhibited"
		} else if s.Recorder.Healthy() && s.Recorder.Mission(Event{Kind: "mission", TickMS: now, Sequence: p.Sequence, Value: m, BenchOnly: true}) {
			r.Status = protocol.StatusQueued
		}
		s.seq[p.Kind] = p.Sequence
		s.last = now
		s.missionDigest = digest
		s.missionReceipt = r
		s.mission = &protocol.MissionState{Sequence: p.Sequence, Status: r.Status}
		return response(protocol.MissionAck, p, now, r)
	}
	return nil
}
func (s *Server) telemetry(now uint64) protocol.Telemetry {
	v := protocol.Vision{Source: "unavailable", Error: "unavailable"}
	if s.Source != nil {
		v = s.Source.Snapshot(time.Now())
	}
	m := s.manual
	if m.Received {
		age := now - s.manualAt
		m.AgeMS = &age
		remaining := uint64(0)
		if m.Hold && age < 300 {
			remaining = 300 - age
		}
		m.ExpiresInMS = &remaining
	}
	return protocol.Telemetry{BenchEstop: s.estop, BenchPaused: s.paused, ControlEpoch: s.epoch, BenchOnly: true, Vision: v, Manual: m, Mission: s.mission, JournalOK: s.Recorder.Healthy(), ControllerStatus: s.controllerStatus(), CPUTempC: s.temperature()}
}
func cpuTemp() *float64 {
	b, e := os.ReadFile("/sys/class/thermal/thermal_zone0/temp")
	if e != nil {
		return nil
	}
	x, e := strconv.ParseFloat(strings.TrimSpace(string(b)), 64)
	if e != nil || math.IsNaN(x) || math.IsInf(x, 0) {
		return nil
	}
	x /= 1000
	return &x
}
func Listen(bind string, port uint16) (*net.UDPConn, error) {
	a, e := netip.ParseAddr(bind)
	if e != nil || a.IsUnspecified() || a.IsMulticast() {
		return nil, errors.New("bind must be explicit literal unicast IP")
	}
	return net.ListenUDP("udp", net.UDPAddrFromAddrPort(netip.AddrPortFrom(a, port)))
}
func (s *Server) Serve(ctx context.Context, conn *net.UDPConn, key []byte) error {
	motorCtx, stopMotor := context.WithCancel(ctx)
	defer stopMotor()
	if s.Motor != nil {
		go s.Motor.Run(motorCtx)
	}
	defer func() {
		s.neutral(uint64(time.Since(s.start).Milliseconds()), "shutdown")
		if s.Motor != nil {
			s.Motor.Update(false, [4]float64{})
		}
	}()
	buf := make([]byte, protocol.MaxDatagram+1)
	for {
		if ctx.Err() != nil {
			return nil
		}
		now := uint64(time.Since(s.start).Milliseconds())
		s.Advance(now)
		if s.Motor != nil {
			var axes [4]float64
			hold := !s.estop && !s.paused && s.session != (protocol.SessionID{}) && s.manual.Hold && now >= s.manualAt && now-s.manualAt < 300 && s.manual.Axes != nil
			if hold {
				axes = *s.manual.Axes
			}
			s.Motor.Update(hold, axes)
		}
		conn.SetReadDeadline(time.Now().Add(20 * time.Millisecond))
		n, addr, e := conn.ReadFromUDPAddrPort(buf)
		if e != nil {
			if ne, ok := e.(net.Error); ok && ne.Timeout() {
				continue
			}
			return e
		}
		p, e := protocol.Unmarshal(key, buf[:n])
		if e != nil {
			continue
		}
		reply := s.Handle(p, addr, uint64(time.Since(s.start).Milliseconds()))
		if reply != nil {
			wire, e := protocol.Marshal(key, *reply)
			if e == nil {
				conn.WriteToUDPAddrPort(wire, addr)
			}
		}
	}
}

func (s *Server) temperature() *float64 {
	if s.Source == nil {
		return nil
	}
	s.Source.mu.RLock()
	defer s.Source.mu.RUnlock()
	return s.Source.temp
}

func (s *Server) controllerStatus() string {
	if s.Motor == nil {
		return "unavailable"
	}
	return s.Motor.Status()
}
