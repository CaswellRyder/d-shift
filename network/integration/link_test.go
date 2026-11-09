package integration_test

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"dtr.local/link/laptop"
	"dtr.local/link/pibridge"
	"dtr.local/link/protocol"
)

// Use the actual HTTP normalizer, Pi receiver, codec and laptop client together.
// Preview identity deliberately differs from the current telemetry identity.
func TestTelemetryIdentityAndSelectedVectorAcrossPeers(t *testing.T) {
	for _, tc := range []struct {
		name, intent, slot string
		missingSequence    bool
	}{
		{"track", "TRACK", "target", false},
		{"avoid", "AVOID", "avoid", false},
		{"missing_identity", "TRACK", "target", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			draft := map[string]any{"intent": tc.intent, "target_vector": map[string]any{"yaw_deg": 11, "elev_m": 2, "travel_m": 3}}
			if !tc.missingSequence {
				draft["seq"] = 99
			}
			body, err := json.Marshal(map[string]any{
				"source": "camera", "ready": true, "frame_age_ms": 10,
				"frame": map[string]any{"sequence": 8},
				"flight_telemetry": map[string]any{
					"coordinates": map[string]any{tc.slot: map[string]any{"label": "red", "box_px": []float64{1, 2, 3, 4}, "body_xyz_m": []float64{3, 1, 2}}},
					"outgoing":    map[string]any{"draft_packet": draft},
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			httpSource := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write(body) }))
			defer httpSource.Close()
			source, err := pibridge.NewSource(httpSource.URL)
			if err != nil {
				t.Fatal(err)
			}
			source.Fetch(context.Background())
			conn, err := pibridge.Listen("127.0.0.1", 0)
			if err != nil {
				t.Fatal(err)
			}
			journal, err := pibridge.NewJournal(t.TempDir() + "/pi.jsonl")
			if err != nil {
				conn.Close()
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			done := make(chan error, 1)
			key := make([]byte, 32) // Synthetic test key, never a bench credential.
			go func() { done <- pibridge.NewServer(journal, source).Serve(ctx, conn, key) }()
			defer func() {
				cancel()
				if err := <-done; err != nil {
					t.Error(err)
				}
				conn.Close()
				if err := journal.Close(); err != nil {
					t.Error(err)
				}
			}()
			client, err := laptop.New(conn.LocalAddr().String(), key, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer client.Close()
			if err := client.Poll(context.Background()); err != nil {
				t.Fatal(err)
			}
			state := client.State()
			if !state.Connected || state.Telemetry == nil || !state.Telemetry.Vision.Ready {
				t.Fatal("missing live telemetry", state)
			}
			v := state.Telemetry.Vision
			if tc.missingSequence {
				if v.FrameSequence != nil {
					t.Fatal("preview identity used for current geometry", *v.FrameSequence)
				}
			} else if v.FrameSequence == nil || *v.FrameSequence != 99 {
				t.Fatal("telemetry identity mismatch", v.FrameSequence)
			}
			selected := v.Target
			if tc.slot == "avoid" {
				selected = v.Avoid
				if v.Target != nil {
					t.Fatal("avoid observation became a target")
				}
			}
			if selected == nil || selected.YawDeg == nil || *selected.YawDeg != 11 || selected.ElevM == nil || *selected.ElevM != 2 || selected.TravelM == nil || *selected.TravelM != 3 {
				t.Fatalf("selected vector lost: %+v", selected)
			}
		})
	}
}

func TestBenchSafetyAcrossAuthenticatedPeers(t *testing.T) {
	conn, err := pibridge.Listen("127.0.0.1", 0)
	if err != nil {
		t.Fatal(err)
	}
	journal, err := pibridge.NewJournal(t.TempDir() + "/bench.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	key := make([]byte, 32)
	go func() { done <- pibridge.NewServer(journal, nil).Serve(ctx, conn, key) }()
	defer func() {
		cancel()
		if e := <-done; e != nil {
			t.Error(e)
		}
		conn.Close()
		if e := journal.Close(); e != nil {
			t.Error(e)
		}
	}()
	client, err := laptop.New(conn.LocalAddr().String(), key, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if err = client.Poll(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err = client.Manual(ctx, protocol.ManualPayload{Hold: true, Forward: .5}); err != nil {
		t.Fatal(err)
	}
	receipt, err := client.Safety(ctx, "stop")
	if err != nil || receipt.Status != protocol.StatusQueued || receipt.Applied {
		t.Fatal("stop", receipt, err)
	}
	receipt, err = client.Safety(ctx, "estop")
	if err != nil || !receipt.BenchEstop || receipt.Status != protocol.StatusQueued {
		t.Fatal("estop", receipt, err)
	}
	if _, err = client.Mission(ctx, protocol.MissionPayload{Op: 1}); err == nil {
		t.Fatal("latched mission accepted")
	}
	if err = client.Poll(ctx); err != nil {
		t.Fatal(err)
	}
	state := client.State()
	if !state.InputInhibited || state.EstopStatus != "confirmed" || state.Telemetry == nil || !state.Telemetry.BenchEstop || state.Telemetry.Manual.Hold {
		t.Fatal("unconfirmed bench latch", state)
	}
	client.Reset()
	if !client.State().InputInhibited {
		t.Fatal("mission reset cleared latch")
	}
	resetCtx, err := laptop.WithControlGeneration(ctx, client.State().InputGeneration)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err = client.Safety(resetCtx, "reset")
	if err != nil || receipt.BenchEstop || receipt.Status != protocol.StatusQueued {
		t.Fatal("reset", receipt, err)
	}
	if err = client.Poll(ctx); err != nil {
		t.Fatal(err)
	}
	state = client.State()
	if !state.InputInhibited || !state.PauseRequested || state.Telemetry.Manual.Hold {
		t.Fatal("E-stop reset cleared pause", state)
	}
	resumeCtx, err := laptop.WithControlGeneration(ctx, client.State().InputGeneration)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err = client.Safety(resumeCtx, "resume")
	if err != nil || receipt.BenchPaused || client.State().InputInhibited {
		t.Fatal("explicit resume", receipt, err, client.State())
	}
	if err = client.Poll(ctx); err != nil {
		t.Fatal(err)
	}
	if client.State().Telemetry.Manual.Hold {
		t.Fatal("resume replayed held input")
	}

}

func TestWaitingConsoleAuthenticatesWithoutAutomaticallyResuming(t *testing.T) {
	interfaces, err := net.Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	name := ""
	for _, iface := range interfaces {
		addresses, _ := iface.Addrs()
		for _, a := range addresses {
			if subnet, ok := a.(*net.IPNet); ok && subnet.IP.Equal(net.ParseIP("127.0.0.1")) {
				name = iface.Name
			}
		}
	}
	if name == "" {
		t.Skip("no loopback")
	}
	conn, err := pibridge.Listen("127.0.0.1", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	journal, err := pibridge.NewJournal(t.TempDir() + "/pi.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer journal.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	key := make([]byte, 32)
	go func() { done <- pibridge.NewServer(journal, nil).Serve(ctx, conn, key) }()
	defer func() {
		cancel()
		if err := <-done; err != nil {
			t.Error(err)
		}
	}()
	client, err := laptop.NewWaitingOnInterface(conn.LocalAddr().String(), name, key, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if err = client.Poll(ctx); err != nil {
		t.Fatal(err)
	}
	state := client.State()
	if !state.Connected || !state.PauseRequested || !state.InputInhibited {
		t.Fatal("connection resumed controls", state)
	}
	fresh, err := laptop.WithControlGeneration(ctx, state.InputGeneration)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := client.Safety(fresh, "resume")
	if err != nil || receipt.Status != protocol.StatusQueued || client.State().PauseRequested {
		t.Fatal("fresh Resume failed", receipt, err)
	}
}
