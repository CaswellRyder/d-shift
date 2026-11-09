package pibridge

import (
	"context"
	"dtr.local/link/protocol"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type recording struct {
	ok     bool
	busy   bool
	events []Event
}

func (r *recording) Healthy() bool { return r.ok }
func (r *recording) Mission(e Event) bool {
	if r.busy {
		return false
	}
	r.events = append(r.events, e)
	return r.ok
}
func (r *recording) Manual(e Event) bool { r.events = append(r.events, e); return r.ok }
func packet(k protocol.Kind, id protocol.SessionID, seq, tick uint64, v any) protocol.Packet {
	raw, _ := protocol.JSON(v)
	return protocol.Packet{Kind: k, Session: id, Sequence: seq, TickMS: tick, Payload: raw}
}

var addr = netip.MustParseAddrPort("127.0.0.1:12000")

func activate(t *testing.T, s *Server, now uint64) protocol.SessionID {
	t.Helper()
	w := s.Handle(packet(protocol.Hello, protocol.SessionID{}, 0, 0, protocol.HelloPayload{Nonce: strings.Repeat("ab", 16)}), addr, now)
	if w == nil {
		t.Fatal("welcome")
	}
	if s.Handle(packet(protocol.Poll, w.Session, 1, now, struct{}{}), addr, now) == nil {
		t.Fatal("activate")
	}
	return w.Session
}
func TestFSMFreshnessExpiryAndBusy(t *testing.T) {
	r := &recording{ok: true}
	s := NewServer(r, nil)
	id := activate(t, s, 10)
	m := packet(protocol.Manual, id, 1, 10, protocol.ManualPayload{Hold: true, Forward: .5})
	if s.Handle(m, addr, 11) == nil {
		t.Fatal("manual")
	}
	if s.Handle(packet(protocol.Poll, id, 2, 11, struct{}{}), netip.MustParseAddrPort("127.0.0.1:12001"), 12) != nil {
		t.Fatal("source binding")
	}
	if s.Handle(packet(protocol.Poll, id, 2, 1000, struct{}{}), addr, 12) != nil {
		t.Fatal("future tick")
	}
	s.Handle(packet(protocol.Poll, id, 2, 100, struct{}{}), addr, 300)
	s.Advance(311)
	if s.manual.Hold || len(r.events) != 2 || r.events[1].Reason != "manual_expired" {
		t.Fatal("poll renewed manual", r.events)
	}
	r.busy = true
	p := packet(protocol.Mission, id, 1, 311, protocol.MissionPayload{Op: 255, Args: "ab"})
	ack := s.Handle(p, addr, 312)
	if ack == nil {
		t.Fatal("mission")
	}
	r.busy = false
	retry := s.Handle(p, addr, 313)
	if string(ack.Payload) != string(retry.Payload) || len(r.events) != 2 {
		t.Fatal("BUSY must be cached")
	}
	p.Payload, _ = protocol.JSON(protocol.MissionPayload{Op: 1})
	if s.Handle(p, addr, 314) != nil {
		t.Fatal("conflicting sequence")
	}
	if s.Handle(packet(protocol.Poll, id, 3, 10, struct{}{}), addr, 400) != nil {
		t.Fatal("stale tick")
	}
}
func TestHelloCannotTakeover(t *testing.T) {
	s := NewServer(&recording{ok: true}, nil)
	id := activate(t, s, 0)
	w := s.Handle(packet(protocol.Hello, protocol.SessionID{}, 0, 0, protocol.HelloPayload{Nonce: strings.Repeat("cd", 16)}), addr, 100)
	if w == nil {
		t.Fatal("challenge")
	}
	if s.Handle(packet(protocol.Poll, w.Session, 2, 100, struct{}{}), addr, 101) != nil || s.session != id {
		t.Fatal("takeover")
	}
	s.Advance(2000)
	if s.Handle(packet(protocol.Poll, w.Session, 2, 2000, struct{}{}), addr, 2000) != nil {
		t.Fatal("expired challenge")
	}
}
func TestJournalExclusiveAndFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events")
	j, e := NewJournal(path)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = NewJournal(path); e == nil {
		t.Fatal("overwrite")
	}
	info, _ := os.Stat(path)
	if info.Mode().Perm() != 0600 {
		t.Fatal(info.Mode())
	}
	j.Mission(Event{Kind: "mission", BenchOnly: true})
	if e = j.Close(); e != nil {
		t.Fatal(e)
	}
	b, _ := os.ReadFile(path)
	if !strings.Contains(string(b), "mission") {
		t.Fatal("not drained")
	}
	r := &recording{ok: false}
	s := NewServer(r, nil)
	id := activate(t, s, 0)
	a := s.Handle(packet(protocol.Manual, id, 1, 0, protocol.ManualPayload{Hold: true}), addr, 1)
	var receipt protocol.Receipt
	protocol.DecodeJSON(a.Payload, &receipt)
	if receipt.Status != "BUSY" || s.manual.Hold {
		t.Fatal("failed writer accepted")
	}
}
func TestSourceNormalizationAndHTTP(t *testing.T) {
	fixture := `{"source":"camera","ready":true,"frame_age_ms":10,"phase":"search","goal_color":"red","metrics":{"fps":12},"flight_telemetry":{"coordinates":{"target":{"label":"red","score":0.9,"box_px":[1,2,3,4],"body_xyz_m":[1,2,3],"yaw_err_deg":2,"age_ms":10},"avoid":{"label":"blue","score":0.8,"box_px":[4,5,6,7],"body_xyz_m":[3,2,1]}},"mapping_status":{"lens_calibrated":true,"mount_calibrated":false},"outgoing":{"draft_packet":{"seq":10,"intent":"TRACK"}}}}`
	v := Normalize([]byte(fixture))
	if !v.Ready || v.Target == nil || v.Target.Box == nil || v.Avoid == nil || v.FrameSequence == nil {
		t.Fatalf("bad normalization %+v", v)
	}
	s := NewServer(&recording{ok: true}, nil)
	s.Source = &Source{vision: v, fetched: time.Now()}
	raw, _ := json.Marshal(s.telemetry(0))
	if len(raw) > protocol.MaxPayload {
		t.Fatal("normal full snapshot exceeds payload", len(raw))
	}
	s.Source.fetched = time.Now().Add(-3 * time.Second)
	stale := s.Source.Snapshot(time.Now())
	if stale.Ready || stale.Target != nil || stale.Avoid != nil {
		t.Fatal("stale geometry")
	}
	for _, u := range []string{"http://example.com/state", "http://0.0.0.0/state", "https://127.0.0.1/state"} {
		if _, e := NewSource(u); e == nil {
			t.Fatal("unsafe URL")
		}
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(fixture)) }))
	defer srv.Close()
	src, e := NewSource(srv.URL)
	if e != nil {
		t.Fatal(e)
	}
	src.Fetch(context.Background())
	if !src.Snapshot(time.Now()).Ready {
		t.Fatal("fetch")
	}
	oversize := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(strings.Repeat("x", 2*1024*1024+1))) }))
	defer oversize.Close()
	src, _ = NewSource(oversize.URL)
	src.Fetch(context.Background())
	if v := src.Snapshot(time.Now()); v.Error != "size_limit" || v.Target != nil {
		t.Fatal("oversize", v)
	}
	if v := Normalize([]byte(`{"frame_seq":7,"age_ms":12}`)); v.Ready || v.Target != nil || v.FrameSequence == nil {
		t.Fatal("preview fabricated perception")
	}
}
func TestUDPLocalhost(t *testing.T) {
	conn, e := Listen("127.0.0.1", 0)
	if e != nil {
		t.Fatal(e)
	}
	defer conn.Close()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	key := make([]byte, 32)
	go func() { done <- NewServer(&recording{ok: true}, nil).Serve(ctx, conn, key) }()
	client, e := net.DialUDP("udp", nil, conn.LocalAddr().(*net.UDPAddr))
	if e != nil {
		t.Fatal(e)
	}
	defer client.Close()
	exchange := func(p protocol.Packet) protocol.Packet {
		t.Helper()
		wire, e := protocol.Marshal(key, p)
		if e != nil {
			t.Fatal(e)
		}
		client.SetDeadline(time.Now().Add(time.Second))
		client.Write(wire)
		b := make([]byte, 1200)
		n, e := client.Read(b)
		if e != nil {
			t.Fatal(e)
		}
		out, e := protocol.Unmarshal(key, b[:n])
		if e != nil {
			t.Fatal(e)
		}
		return out
	}
	w := exchange(packet(protocol.Hello, protocol.SessionID{}, 0, 0, protocol.HelloPayload{Nonce: strings.Repeat("01", 16)}))
	reply := exchange(packet(protocol.Poll, w.Session, 1, w.TickMS, struct{}{}))
	if reply.Kind != protocol.TelemetryReply {
		t.Fatal("telemetry")
	}
	cancel()
	if e := <-done; e != nil {
		t.Fatal(e)
	}
}

func TestBoundedQueuesAndWriterFailure(t *testing.T) {
	j := &Journal{missions: make(chan Event, 16), manual: make(chan Event, 1)}
	j.healthy.Store(true)
	for i := 0; i < 16; i++ {
		if !j.Mission(Event{Sequence: uint64(i)}) {
			t.Fatal("early full")
		}
	}
	if j.Mission(Event{}) {
		t.Fatal("mission queue unbounded")
	}
	j.Manual(Event{Sequence: 1})
	j.Manual(Event{Sequence: 2})
	if len(j.manual) != 1 || (<-j.manual).Sequence != 2 {
		t.Fatal("manual must keep latest only")
	}
	disk, e := NewJournal(filepath.Join(t.TempDir(), "failed"))
	if e != nil {
		t.Fatal(e)
	}
	disk.file.Close()
	disk.Mission(Event{Kind: "test"})
	deadline := time.Now().Add(time.Second)
	for disk.Healthy() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if disk.Healthy() || disk.Mission(Event{}) || disk.Manual(Event{}) {
		t.Fatal("writer failure must reject commands")
	}
	if disk.Close() == nil {
		t.Fatal("writer failure lost")
	}
}
func TestSourceFailureClearsPriorGeometry(t *testing.T) {
	fail := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if fail {
			http.Error(w, "error", 500)
			return
		}
		w.Write([]byte(`{"ready":true,"frame_age_ms":1,"flight_telemetry":{"coordinates":{"target":{"label":"red"}}}}`))
	}))
	defer srv.Close()
	s, _ := NewSource(srv.URL)
	s.Fetch(context.Background())
	if !s.Snapshot(time.Now()).Ready {
		t.Fatal("initial")
	}
	fail = true
	s.Fetch(context.Background())
	v := s.Snapshot(time.Now())
	if v.Ready || v.Target != nil || v.Error != "source_error" {
		t.Fatal("source failure retained geometry")
	}
}

func TestSafetyEpochLatchSessionAndReplay(t *testing.T) {
	r := &recording{ok: true}
	s := NewServer(r, nil)
	id := activate(t, s, 10)
	safety := func(seq, tick uint64, action string, epoch uint64) *protocol.Packet {
		return s.Handle(packet(protocol.Safety, id, seq, tick, protocol.SafetyPayload{Action: action, Epoch: epoch}), addr, tick)
	}
	s.Handle(packet(protocol.Manual, id, 1, 10, protocol.ManualPayload{Hold: true, Forward: .5}), addr, 10)
	stop := safety(1, 11, "stop", 0)
	if stop == nil || s.manual.Hold || s.epoch != 1 {
		t.Fatal("stop failed")
	}
	// Higher manual sequence cannot defeat the epoch barrier.
	ack := s.Handle(packet(protocol.Manual, id, 100, 11, protocol.ManualPayload{Hold: true, Forward: .8}), addr, 12)
	var receipt protocol.Receipt
	protocol.DecodeJSON(ack.Payload, &receipt)
	if receipt.Status != protocol.StatusRejected || s.manual.Hold {
		t.Fatal("pre-stop hold restored")
	}
	estop := safety(2, 13, "estop", 1)
	if estop == nil || !s.estop || s.epoch != 2 {
		t.Fatal("estop failed")
	}
	mission := s.Handle(packet(protocol.Mission, id, 1, 13, protocol.MissionPayload{Epoch: 2, Op: 1}), addr, 14)
	protocol.DecodeJSON(mission.Payload, &receipt)
	if receipt.Status != protocol.StatusRejected {
		t.Fatal("latched mission accepted")
	}
	reset := packet(protocol.Safety, id, 3, 15, protocol.SafetyPayload{Action: "reset", Epoch: 2})
	s.Handle(reset, addr, 15)
	if s.estop || s.manual.Hold || s.epoch != 3 {
		t.Fatal("reset resumed hold")
	}
	safety(4, 16, "estop", 3)
	s.Handle(reset, addr, 17)
	if !s.estop || s.epoch != 4 {
		t.Fatal("delayed reset cleared latch")
	}
	s.Advance(2016)
	id = activate(t, s, 2017)
	if !s.estop || s.epoch != 4 {
		t.Fatal("session churn cleared latch")
	}
	// Fresh-sequence reset with an old epoch also cannot clear a newer latch.
	safety(1, 2018, "reset", 2)
	if !s.estop {
		t.Fatal("old epoch reset")
	}
}

func TestSafetyInhibitsWithFailedJournal(t *testing.T) {
	r := &recording{ok: true}
	s := NewServer(r, nil)
	id := activate(t, s, 10)
	s.Handle(packet(protocol.Manual, id, 1, 10, protocol.ManualPayload{Hold: true}), addr, 10)
	r.ok = false
	p := packet(protocol.Safety, id, 1, 11, protocol.SafetyPayload{Action: "estop"})
	out := s.Handle(p, addr, 11)
	var receipt protocol.Receipt
	protocol.DecodeJSON(out.Payload, &receipt)
	if !s.estop || s.manual.Hold || receipt.Status != protocol.StatusBusy {
		t.Fatal("journal failure prevented inhibition")
	}
	r.ok = true
	replay := s.Handle(p, addr, 12)
	if string(replay.Payload) != string(out.Payload) {
		t.Fatal("safety BUSY not terminal")
	}
	r.ok = false
	s.Handle(packet(protocol.Safety, id, 2, 13, protocol.SafetyPayload{Action: "reset", Epoch: 1}), addr, 13)
	if !s.estop {
		t.Fatal("unrecorded reset cleared latch")
	}
}

func TestPauseLatchRequiresRecordedResumeAndSurvivesSessions(t *testing.T) {
	r := &recording{ok: true}
	s := NewServer(r, nil)
	id := activate(t, s, 10)
	send := func(seq, now uint64, action string, epoch uint64) protocol.Receipt {
		t.Helper()
		out := s.Handle(packet(protocol.Safety, id, seq, now, protocol.SafetyPayload{Action: action, Epoch: epoch}), addr, now)
		var receipt protocol.Receipt
		if out == nil || protocol.DecodeJSON(out.Payload, &receipt) != nil {
			t.Fatal("missing receipt")
		}
		return receipt
	}
	r.ok = false
	pause := send(1, 11, "pause", 0)
	if !pause.BenchPaused || pause.Status != protocol.StatusBusy || s.epoch != 1 {
		t.Fatal("failed journal prevented pause", pause)
	}
	r.ok = true
	// Matching current epoch still cannot drive through the persistent latch.
	out := s.Handle(packet(protocol.Manual, id, 1, 12, protocol.ManualPayload{Epoch: 1, Hold: true, Up: 1}), addr, 12)
	var ack protocol.Receipt
	protocol.DecodeJSON(out.Payload, &ack)
	if ack.Status != protocol.StatusRejected || s.manual.Hold {
		t.Fatal("paused manual accepted", ack)
	}
	s.Advance(2012)
	id = activate(t, s, 2013)
	if !s.telemetry(2013).BenchPaused {
		t.Fatal("session cleared pause")
	}
	if got := send(1, 2014, "resume", 0); got.Status != protocol.StatusRejected || !got.BenchPaused {
		t.Fatal("old epoch resumed", got)
	}
	r.ok = false
	if got := send(2, 2015, "resume", 1); got.Status != protocol.StatusBusy || !got.BenchPaused {
		t.Fatal("unrecorded resume", got)
	}
	r.ok = true
	if got := send(3, 2016, "estop", 1); !got.BenchEstop || !got.BenchPaused {
		t.Fatal(got)
	}
	if got := send(4, 2017, "resume", 2); got.Status != protocol.StatusRejected || !got.BenchEstop || !got.BenchPaused {
		t.Fatal("resume bypassed estop", got)
	}
	if got := send(5, 2018, "reset", 2); got.BenchEstop || !got.BenchPaused {
		t.Fatal("reset cleared pause", got)
	}
	if got := send(6, 2019, "resume", 3); got.Status != protocol.StatusQueued || got.BenchPaused || s.epoch != 4 || s.manual.Hold {
		t.Fatal("fresh resume", got)
	}
	// Delayed pre-pause held input remains invalid after Resume.
	out = s.Handle(packet(protocol.Manual, id, 2, 2020, protocol.ManualPayload{Epoch: 0, Hold: true, Forward: 1}), addr, 2020)
	protocol.DecodeJSON(out.Payload, &ack)
	if ack.Status != protocol.StatusRejected || s.manual.Hold {
		t.Fatal("old held input replayed")
	}
	send(7, 2021, "pause", 4)
	if got := send(8, 2022, "resume", 4); got.Status != protocol.StatusRejected || !got.BenchPaused {
		t.Fatal("old resume undid newer pause", got)
	}
}

func TestEmergencyCutoffAlsoLatchesPauseUntilSeparateResume(t *testing.T) {
	s := NewServer(&recording{ok: true}, nil)
	id := activate(t, s, 10)
	out := s.Handle(packet(protocol.Safety, id, 1, 10, protocol.SafetyPayload{Action: "estop", Epoch: 0}), addr, 11)
	var ack protocol.Receipt
	if out == nil || protocol.DecodeJSON(out.Payload, &ack) != nil || !ack.BenchEstop || !ack.BenchPaused {
		t.Fatal("emergency did not suspend input", ack)
	}
	out = s.Handle(packet(protocol.Safety, id, 2, 11, protocol.SafetyPayload{Action: "reset", Epoch: 1}), addr, 12)
	protocol.DecodeJSON(out.Payload, &ack)
	if ack.BenchEstop || !ack.BenchPaused {
		t.Fatal("reset resumed input", ack)
	}
}
