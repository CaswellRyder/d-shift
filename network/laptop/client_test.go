package laptop

import (
	"context"
	"dtr.local/link/protocol"
	"encoding/hex"
	"net"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakePi struct {
	ackSafety  bool
	conn       *net.UDPConn
	key        []byte
	session    protocol.SessionID
	mu         sync.Mutex
	missions   int
	manuals    int
	drop       bool
	wrong      bool
	nonceWrong bool
	delay      time.Duration
	requests   []protocol.Packet
}

func fake(t *testing.T) *fakePi {
	t.Helper()
	conn, e := net.ListenUDP("udp", &net.UDPAddr{IP: net.ParseIP("127.0.0.1")})
	if e != nil {
		t.Fatal(e)
	}
	s, _ := protocol.NewSession()
	p := &fakePi{conn: conn, key: make([]byte, 32), session: s}
	t.Cleanup(func() { conn.Close() })
	go p.serve()
	return p
}
func (f *fakePi) serve() {
	var estop, paused bool
	var epoch uint64
	var b [1500]byte
	for {
		n, a, e := f.conn.ReadFromUDP(b[:])
		if e != nil {
			return
		}
		p, e := protocol.Unmarshal(f.key, b[:n])
		if e != nil {
			continue
		}
		f.mu.Lock()
		f.requests = append(f.requests, p)
		drop, wrong, nonceWrong, delay := f.drop, f.wrong, f.nonceWrong, f.delay
		ackSafety := f.ackSafety
		if p.Kind == protocol.Mission {
			f.missions++
		}
		if p.Kind == protocol.Manual {
			f.manuals++
		}
		f.mu.Unlock()
		r := protocol.Packet{Session: f.session, Sequence: p.Sequence, TickMS: 100}
		switch p.Kind {
		case protocol.Hello:
			var h protocol.HelloPayload
			protocol.DecodeJSON(p.Payload, &h)
			if nonceWrong {
				h.Nonce = hex.EncodeToString(make([]byte, 16))
			}
			r.Kind = protocol.Welcome
			r.Payload, _ = protocol.JSON(protocol.WelcomePayload{Nonce: h.Nonce, BenchOnly: true})
		case protocol.Poll:
			r.Kind = protocol.TelemetryReply
			r.Payload, _ = protocol.JSON(protocol.Telemetry{BenchOnly: true, JournalOK: true, BenchEstop: estop, BenchPaused: paused, ControlEpoch: epoch})
		case protocol.Safety:
			if !ackSafety {
				continue
			}
			var safety protocol.SafetyPayload
			protocol.DecodeJSON(p.Payload, &safety)
			if safety.Action == "estop" {
				estop = true
				paused = true
			}
			if safety.Action == "pause" {
				paused = true
			}
			if safety.Action == "reset" {
				estop = false
			}
			if safety.Action == "resume" {
				paused = false
			}
			epoch++
			r.Kind = protocol.SafetyAck
			r.Payload, _ = protocol.JSON(protocol.Receipt{Status: protocol.StatusQueued, Digest: protocol.Digest(p.Payload), BenchOnly: true, BenchEstop: estop, BenchPaused: paused, Epoch: epoch})
		case protocol.Manual, protocol.Mission:
			if drop {
				continue
			}
			r.Kind = protocol.ManualAck
			if p.Kind == protocol.Mission {
				r.Kind = protocol.MissionAck
			}
			r.Payload, _ = protocol.JSON(protocol.Receipt{Status: protocol.StatusQueued, Digest: protocol.Digest(p.Payload), BenchOnly: true})
			if wrong {
				bad := r
				bad.Sequence++
				w, _ := protocol.Marshal(f.key, bad)
				f.conn.WriteToUDP(w, a)
				bad = r
				bad.Payload, _ = protocol.JSON(protocol.Receipt{Status: protocol.StatusQueued, Digest: "wrong", BenchOnly: true})
				w, _ = protocol.Marshal(f.key, bad)
				f.conn.WriteToUDP(w, a)
			}
		}
		time.Sleep(delay)
		w, _ := protocol.Marshal(f.key, r)
		f.conn.WriteToUDP(w, a)
	}
}
func client(t *testing.T, f *fakePi) *Client {
	t.Helper()
	c, e := New(f.conn.LocalAddr().String(), f.key, nil)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { c.Close() })
	return c
}
func TestHandshakeAndFreshState(t *testing.T) {
	f := fake(t)
	c := client(t, f)
	if e := c.Poll(context.Background()); e != nil {
		t.Fatal(e)
	}
	if s := c.State(); !s.Connected || s.Telemetry == nil || s.ESPStatus != nil {
		t.Fatal(s)
	}
	c.mu.Lock()
	c.received = time.Now().Add(-751 * time.Millisecond)
	c.mu.Unlock()
	if s := c.State(); s.Telemetry != nil || s.Connected {
		t.Fatal("stale telemetry exposed")
	}
}
func TestMatchedReceiptAndSingleManual(t *testing.T) {
	f := fake(t)
	f.mu.Lock()
	f.wrong = true
	f.mu.Unlock()
	c := client(t, f)
	if e := c.Poll(context.Background()); e != nil {
		t.Fatal(e)
	}
	if _, e := c.Manual(context.Background(), protocol.ManualPayload{Forward: 1, Hold: true}); e != nil {
		t.Fatal(e)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.manuals != 1 {
		t.Fatal("manual retried")
	}
}
func TestUnknownMissionNoReplay(t *testing.T) {
	f := fake(t)
	f.mu.Lock()
	f.drop = true
	f.mu.Unlock()
	c := client(t, f)
	start := time.Now()
	_, e := c.Mission(context.Background(), protocol.MissionPayload{Op: 9})
	if e != ErrUnknown {
		t.Fatal(e)
	}
	if time.Since(start) > 300*time.Millisecond {
		t.Fatal("mission deadline extended")
	}
	f.mu.Lock()
	count := f.missions
	var identity protocol.Packet
	for _, p := range f.requests {
		if p.Kind == protocol.Mission {
			if identity.Kind == 0 {
				identity = p
			} else if p.Sequence != identity.Sequence || string(p.Payload) != string(identity.Payload) {
				t.Fatal("retry changed identity")
			}
		}
	}
	f.mu.Unlock()
	if count != 2 {
		t.Fatalf("want two sends got %d", count)
	}
	if e = c.Poll(context.Background()); e != nil {
		t.Fatal(e)
	}
	if _, e = c.Mission(context.Background(), protocol.MissionPayload{}); e != ErrUnknown {
		t.Fatal(e)
	}
	f.mu.Lock()
	if f.missions != count {
		t.Fatal("automatic mission replay")
	}
	f.drop = false
	f.mu.Unlock()
	c.Reset()
	if _, e = c.Mission(context.Background(), protocol.MissionPayload{}); e != nil {
		t.Fatal(e)
	}
}
func TestNonceAndLateReply(t *testing.T) {
	for _, mode := range []string{"nonce", "late"} {
		t.Run(mode, func(t *testing.T) {
			f := fake(t)
			f.mu.Lock()
			if mode == "nonce" {
				f.nonceWrong = true
			} else {
				f.delay = 260 * time.Millisecond
			}
			f.mu.Unlock()
			c := client(t, f)
			ctx, cancel := context.WithTimeout(context.Background(), 240*time.Millisecond)
			defer cancel()
			if c.Poll(ctx) == nil {
				t.Fatal("invalid handshake accepted")
			}
			if c.State().Connected {
				t.Fatal("connected")
			}
		})
	}
}
func TestTickAnchorOriginalSend(t *testing.T) {
	f := fake(t)
	c := client(t, f)
	if c.Poll(context.Background()) != nil {
		t.Fatal("poll")
	}
	f.mu.Lock()
	f.delay = 100 * time.Millisecond
	f.mu.Unlock()
	before := time.Now()
	c.mu.Lock()
	e := c.poll(context.Background())
	anchor := c.anchor
	c.mu.Unlock()
	if e != nil {
		t.Fatal(e)
	}
	if anchor.Sub(before) > 50*time.Millisecond || time.Since(anchor) < 90*time.Millisecond {
		t.Fatal("tick anchored at receipt")
	}
}
func TestAPIProtection(t *testing.T) {
	f := fake(t)
	c := client(t, f)
	h := Handler(c, "127.0.0.1:8782")
	cases := []struct {
		method, path, host, origin, body, header, ctype string
		want                                            int
	}{{"POST", "/api/estop", "127.0.0.1:8782", "", "{}", "", "application/json", 403}, {"POST", "/api/stop", "127.0.0.1:8782", "", "{\"hold\":true}", "1", "application/json", 400}, {"POST", "/api/estop/reset", "127.0.0.1:8782", "", "null", "1", "application/json", 400}, {"GET", "/api/state", "evil", "", "", "", "", 403}, {"GET", "/api/state", "127.0.0.1:8782", "http://evil", "", "", "", 403}, {"GET", "/api/manual", "127.0.0.1:8782", "", "", "", "", 405}, {"POST", "/api/reset", "127.0.0.1:8782", "", "{}", "", "application/json", 403}, {"POST", "/api/reset", "127.0.0.1:8782", "", "{}", "1", "text/plain", 403}, {"POST", "/api/reset", "127.0.0.1:8782", "", "{\"unknown\":1}", "1", "application/json", 400}, {"POST", "/api/reset", "127.0.0.1:8782", "", "{\"a\":1,\"a\":2}", "1", "application/json", 400}, {"POST", "/api/manual", "127.0.0.1:8782", "", strings.Repeat("x", 2049), "1", "application/json", 400}, {"POST", "/api/manual", "127.0.0.1:8782", "", "{\"forward\":1,\"right\":0,\"up\":0,\"yaw\":0,\"hold\":false}", "1", "application/json", 409}, {"GET", "/api/state", "127.0.0.1:8782", "", "", "", "", 200}, {"POST", "/api/reset", "127.0.0.1:8782", "", "{}", "1", "application/json", 200}}
	for _, tc := range cases {
		r := httptest.NewRequest(tc.method, "http://"+tc.host+tc.path, strings.NewReader(tc.body))
		r.Header.Set("Origin", tc.origin)
		r.Header.Set("X-DTR-Request", tc.header)
		r.Header.Set("Content-Type", tc.ctype)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != tc.want {
			t.Errorf("%s %s: got %d want %d", tc.method, tc.path, w.Code, tc.want)
		}
	}
}
func TestJournalExclusive(t *testing.T) {
	p := t.TempDir() + "/journal"
	j, e := NewJournal(p)
	if e != nil {
		t.Fatal(e)
	}
	j.Record("receipt", &protocol.Receipt{Status: protocol.StatusQueued, BenchOnly: true})
	j.Close()
	if _, e = NewJournal(p); e == nil {
		t.Fatal("overwrote journal")
	}
}

func TestManualNeverReconnectsHeldSample(t *testing.T) {
	f := fake(t)
	c := client(t, f)
	if _, err := c.Manual(context.Background(), protocol.ManualPayload{Forward: 1, Hold: true}); err == nil {
		t.Fatal("held sample reconnected")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.requests) != 0 {
		t.Fatal("manual started reconnect")
	}
}

func TestManualExpiresWhileWaitingForOwner(t *testing.T) {
	f := fake(t)
	c := client(t, f)
	if err := c.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	c.mu.Lock()
	done := make(chan error, 1)
	go func() {
		_, err := c.Manual(context.Background(), protocol.ManualPayload{Forward: 1, Hold: true})
		done <- err
	}()
	time.Sleep(270 * time.Millisecond)
	c.mu.Unlock()
	if err := <-done; err == nil {
		t.Fatal("expired queued sample sent")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.manuals != 0 {
		t.Fatal("expired sample reached Pi")
	}
}

func TestCachedGeometryExpiresFromTelemetryRequestTime(t *testing.T) {
	f := fake(t)
	c := client(t, f)
	if err := c.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	age := 1980.0
	c.mu.Lock()
	c.telemetry.Vision = protocol.Vision{Source: "camera", Ready: true, AgeMS: &age,
		Target: &protocol.Target{Label: "red"}, Avoid: &protocol.Target{Label: "blue"}}
	c.telemetryAt = time.Now().Add(-81 * time.Millisecond)
	c.received = time.Now()
	// A command acknowledgement can refresh the tick anchor, but cannot make
	// an older telemetry snapshot younger.
	c.anchor = time.Now()
	c.mu.Unlock()
	s := c.State()
	if !s.Connected || s.Telemetry == nil {
		t.Fatal("fresh transport snapshot disappeared")
	}
	v := s.Telemetry.Vision
	if v.Ready || v.Target != nil || v.Avoid != nil || v.AgeMS == nil || *v.AgeMS < 2061 || v.Error != "stale" {
		t.Fatalf("expired geometry exposed: %+v", v)
	}
	if *c.telemetry.Vision.AgeMS != 1980 || !c.telemetry.Vision.Ready || c.telemetry.Vision.Target == nil {
		t.Fatal("state projection mutated the stored snapshot")
	}
}

func TestLaptopJournalFailureIsExposedAndReturned(t *testing.T) {
	f := fake(t)
	c := client(t, f)
	if err := c.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	j, err := NewJournal(t.TempDir() + "/journal")
	if err != nil {
		t.Fatal(err)
	}
	c.journal = j
	if err := j.f.Close(); err != nil {
		t.Fatal(err)
	}
	j.Record("write_failure_probe", nil)
	deadline := time.Now().Add(time.Second)
	for j.OK() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	s := c.State()
	if s.LocalJournalOK == nil || *s.LocalJournalOK || s.Telemetry == nil || !s.Telemetry.JournalOK {
		t.Fatal("local and Pi journal health were not distinguished", s)
	}
	if _, err := c.Mission(context.Background(), protocol.MissionPayload{Op: 255}); err == nil {
		t.Fatal("command sent despite failed local journal")
	}
	if err := j.Close(); err == nil {
		t.Fatal("journal failure disappeared at shutdown")
	}
}

func TestEstopImmediatelyInhibitsAndUnknownResetIsSeparate(t *testing.T) {
	f := fake(t)
	c, e := New(f.conn.LocalAddr().String(), f.key, nil)
	if e != nil {
		t.Fatal(e)
	}
	defer c.Close()
	if e = c.Poll(context.Background()); e != nil {
		t.Fatal(e)
	}
	f.mu.Lock()
	f.drop = true
	f.mu.Unlock()
	done := make(chan error, 1)
	go func() { _, e := c.Manual(context.Background(), protocol.ManualPayload{Hold: true}); done <- e }()
	time.Sleep(20 * time.Millisecond)
	safety := make(chan error, 1)
	go func() { _, e := c.Safety(context.Background(), "estop"); safety <- e }()
	deadline := time.Now().Add(time.Second)
	for !c.inhibited.Load() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if !c.inhibited.Load() {
		t.Fatal("local latch delayed")
	}
	if _, e = c.Manual(context.Background(), protocol.ManualPayload{Hold: true}); e == nil {
		t.Fatal("held sample accepted")
	}
	<-done
	<-safety
	c.Reset()
	state := c.State()
	if !state.InputInhibited || !state.EstopRequested || state.EstopStatus != "unknown" {
		t.Fatal("mission reset cleared bench latch", state)
	}
	if e = c.Poll(context.Background()); e != nil {
		t.Fatal(e)
	}
	if !c.State().InputInhibited {
		t.Fatal("poll cleared latch")
	}
}

func TestSafetyInterruptsReconnectRatherThanWaitingForItsBudget(t *testing.T) {
	// Bound localhost socket deliberately does not answer authenticated requests.
	conn, e := net.ListenUDP("udp", &net.UDPAddr{IP: net.ParseIP("127.0.0.1")})
	if e != nil {
		t.Fatal(e)
	}
	defer conn.Close()
	c, e := New(conn.LocalAddr().String(), make([]byte, 32), nil)
	if e != nil {
		t.Fatal(e)
	}
	defer c.Close()
	poll := make(chan error, 1)
	go func() { poll <- c.Poll(context.Background()) }()
	time.Sleep(20 * time.Millisecond)
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	safety := make(chan error, 1)
	go func() { _, e := c.Safety(ctx, "estop"); safety <- e }()
	select {
	case <-poll:
	case <-time.After(250 * time.Millisecond):
		t.Fatal("E-stop queued behind reconnect budget")
	}
	<-safety
	if !c.inhibited.Load() {
		t.Fatal("failed link cleared immediate inhibition")
	}
}

func TestPauseImmediateLatchGenerationAndIndependentReset(t *testing.T) {
	f := fake(t)
	c := client(t, f)
	if err := c.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	old := c.State().InputGeneration
	ctx, err := WithControlGeneration(context.Background(), old)
	if err != nil {
		t.Fatal(err)
	}
	// This fake peer does not acknowledge Safety: local pause must survive that.
	timeout, cancel := context.WithTimeout(context.Background(), 80*time.Millisecond)
	defer cancel()
	done := make(chan error, 1)
	go func() { _, err := c.Safety(timeout, "stop"); done <- err }()
	deadline := time.Now().Add(time.Second)
	for !c.paused.Load() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if !c.paused.Load() {
		t.Fatal("pause latch delayed")
	}
	if err := <-done; err == nil {
		t.Fatal("fake acknowledged pause")
	}
	state := c.State()
	if !state.PauseRequested || !state.InputInhibited || state.EstopRequested || state.PauseStatus != "unknown" {
		t.Fatal(state)
	}
	if _, err := c.Safety(ctx, "resume"); err == nil {
		t.Fatal("stale resume accepted")
	}
	if _, err := c.Safety(context.Background(), "resume"); err == nil {
		t.Fatal("tokenless resume accepted")
	}
	if _, err := c.Manual(ctx, protocol.ManualPayload{Hold: true, Forward: 1}); err == nil {
		t.Fatal("pre-pause held input accepted")
	}
	if _, err := c.Mission(context.Background(), protocol.MissionPayload{Op: 1}); err == nil {
		t.Fatal("paused mission accepted")
	}
	c.Reset()
	if err := c.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !c.State().PauseRequested || !c.State().InputInhibited {
		t.Fatal("poll/reset cleared pause")
	}
}

func TestResumeHTTPRequiresFreshGenerationAndCannotUndoNewerPause(t *testing.T) {
	f := fake(t)
	c := client(t, f)
	handler := Handler(c, "127.0.0.1:8782")
	old := c.State().InputGeneration
	timeout, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	c.Safety(timeout, "pause")
	for _, generation := range []string{"", old} {
		req := httptest.NewRequest("POST", "http://127.0.0.1:8782/api/resume", strings.NewReader(`{}`))
		req.Header.Set("X-DTR-Request", "1")
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-DTR-Control-Generation", generation)
		out := httptest.NewRecorder()
		handler.ServeHTTP(out, req)
		want := 409
		if generation == "" {
			want = 400
		}
		if out.Code != want || !c.State().PauseRequested {
			t.Fatal("stale/missing resume", out.Code, c.State())
		}
	}
	// A queued clearing action cannot clear a pause which superseded it while
	// it was waiting for ownership of the transport mutex.
	ctx, _ := WithControlGeneration(context.Background(), c.State().InputGeneration)
	c.mu.Lock()
	before := c.generation.Load()
	done := make(chan error, 1)
	go func() { _, err := c.Safety(ctx, "resume"); done <- err }()
	deadline := time.Now().Add(time.Second)
	for c.generation.Load() == before && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if c.generation.Load() == before {
		c.mu.Unlock()
		t.Fatal("resume did not acquire generation")
	}
	paused := make(chan error, 1)
	go func() { _, err := c.Safety(timeout, "pause"); paused <- err }()
	deadline = time.Now().Add(time.Second)
	for c.generation.Load() == before+1 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	c.mu.Unlock()
	if err := <-done; err == nil {
		t.Fatal("superseded resume accepted")
	}
	<-paused
	if !c.State().PauseRequested {
		t.Fatal("newer pause cleared")
	}
}

func TestPausePropagatesStrongerPendingEstopFirst(t *testing.T) {
	f := fake(t)
	f.mu.Lock()
	f.ackSafety = true
	f.mu.Unlock()
	c := client(t, f)
	if err := c.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	// Model an E-stop which latched locally before its exchange was superseded.
	c.inhibited.Store(true)
	receipt, err := c.Safety(context.Background(), "pause")
	if err != nil || !receipt.BenchEstop || !receipt.BenchPaused {
		t.Fatal("pause suppressed estop", receipt, err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	var actions []string
	for _, p := range f.requests {
		if p.Kind == protocol.Safety {
			var v protocol.SafetyPayload
			protocol.DecodeJSON(p.Payload, &v)
			actions = append(actions, v.Action)
		}
	}
	if strings.Join(actions, ",") != "estop,pause" {
		t.Fatal("priority order", actions)
	}
}

func TestResumeCannotSupersedePendingEstop(t *testing.T) {
	f := fake(t)
	f.mu.Lock()
	f.ackSafety = true
	f.mu.Unlock()
	c := client(t, f)
	if err := c.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	c.mu.Lock()
	done := make(chan error, 1)
	go func() { _, err := c.Safety(context.Background(), "estop"); done <- err }()
	deadline := time.Now().Add(time.Second)
	for !c.inhibited.Load() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if !c.inhibited.Load() {
		c.mu.Unlock()
		t.Fatal("E-stop did not latch")
	}
	before := c.generation.Load()
	ctx, _ := WithControlGeneration(context.Background(), strconv.FormatUint(before, 10))
	if _, err := c.Safety(ctx, "resume"); err == nil {
		c.mu.Unlock()
		t.Fatal("resume accepted during E-stop")
	}
	if c.generation.Load() != before {
		c.mu.Unlock()
		t.Fatal("resume superseded emergency generation")
	}
	c.mu.Unlock()
	if err := <-done; err != nil {
		t.Fatal("E-stop propagation cancelled", err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	found := false
	for _, p := range f.requests {
		if p.Kind == protocol.Safety {
			var v protocol.SafetyPayload
			protocol.DecodeJSON(p.Payload, &v)
			found = found || v.Action == "estop"
		}
	}
	if !found {
		t.Fatal("E-stop never reached receiver")
	}
}

func TestEstopSupersedingPausePreservesPauseAfterReset(t *testing.T) {
	f := fake(t)
	f.mu.Lock()
	f.ackSafety = true
	f.mu.Unlock()
	c := client(t, f)
	if err := c.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	c.mu.Lock()
	pauseDone := make(chan error, 1)
	go func() { _, err := c.Safety(context.Background(), "pause"); pauseDone <- err }()
	deadline := time.Now().Add(time.Second)
	for !c.paused.Load() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if !c.paused.Load() {
		c.mu.Unlock()
		t.Fatal("pause did not latch")
	}
	estopDone := make(chan error, 1)
	go func() { _, err := c.Safety(context.Background(), "estop"); estopDone <- err }()
	deadline = time.Now().Add(time.Second)
	for !c.inhibited.Load() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if !c.inhibited.Load() {
		c.mu.Unlock()
		t.Fatal("E-stop did not latch")
	}
	c.mu.Unlock()
	<-pauseDone
	if err := <-estopDone; err != nil {
		t.Fatal(err)
	}
	ctx, _ := WithControlGeneration(context.Background(), c.State().InputGeneration)
	ack, err := c.Safety(ctx, "reset")
	if err != nil || ack.BenchEstop || !ack.BenchPaused {
		t.Fatal("reset lost pending pause", ack, err)
	}
	if err := c.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	state := c.State()
	if !state.Telemetry.BenchPaused || state.Telemetry.BenchEstop || !state.InputInhibited {
		t.Fatal("receiver did not preserve pause", state)
	}
}
