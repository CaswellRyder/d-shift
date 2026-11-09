package pibridge

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"dtr.local/link/protocol"
)

func TestPortalNavigationAndNoCommandSurface(t *testing.T) {
	s := httptest.NewUnstartedServer(nil)
	s.Config.Handler = Portal(s.Listener.Addr().String())
	s.Start()
	defer s.Close()
	for _, tc := range []struct {
		method, path string
		status       int
	}{
		{http.MethodGet, "/", http.StatusOK},
		{http.MethodHead, "/", http.StatusOK},
		{http.MethodGet, "/healthz", http.StatusOK},
		{http.MethodGet, "/api/state", http.StatusNotFound},
		{http.MethodGet, "/api/launch", http.StatusNotFound},
		{http.MethodPost, "/api/launch", http.StatusMethodNotAllowed},
		{http.MethodPost, "/api/manual", http.StatusMethodNotAllowed},
		{http.MethodPost, "/api/estop", http.StatusMethodNotAllowed},
		{http.MethodPost, "/", http.StatusMethodNotAllowed},
	} {
		t.Run(tc.method+tc.path, func(t *testing.T) {
			r, err := http.NewRequest(tc.method, s.URL+tc.path, nil)
			if err != nil {
				t.Fatal(err)
			}
			resp, err := s.Client().Do(r)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			b, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != tc.status || resp.Header.Get("Cache-Control") != "no-store" {
				t.Fatalf("status=%d cache=%q body=%s", resp.StatusCode, resp.Header.Get("Cache-Control"), b)
			}
			if tc.method == http.MethodHead && len(b) != 0 {
				t.Fatal("HEAD returned a body")
			}
			if tc.method == http.MethodGet && tc.path == "/" {
				for _, want := range []string{`href="http://127.0.0.1:8782/"`, "DTR-Drone-Bench", "laptop station must be running", "Launch is unavailable", "Bench controls only"} {
					if !strings.Contains(string(b), want) {
						t.Fatalf("missing navigation information: %s", want)
					}
				}
				if strings.Contains(string(b), "<script") || !strings.Contains(resp.Header.Get("Content-Security-Policy"), "default-src 'none'") {
					t.Fatal("portal gained an active script surface")
				}
			}
		})
	}
}

func TestPortalRejectsForeignHostAndOrigin(t *testing.T) {
	const host = "192.168.50.1:8782"
	for _, tc := range []struct{ host, origin string }{
		{"other.invalid:8782", ""},
		{host, "http://other.invalid"},
		{host, "null"},
		{host, "http://192.168.50.1:8782.attacker.invalid"},
	} {
		r := httptest.NewRequest(http.MethodGet, "http://"+tc.host+"/", nil)
		r.Header.Set("Origin", tc.origin)
		w := httptest.NewRecorder()
		Portal(host).ServeHTTP(w, r)
		if w.Code != http.StatusForbidden {
			t.Fatalf("accepted host=%q origin=%q", tc.host, tc.origin)
		}
	}
	r := httptest.NewRequest(http.MethodGet, "http://"+host+"/", nil)
	r.Header.Set("Origin", "http://"+host)
	w := httptest.NewRecorder()
	Portal(host).ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatal("rejected exact origin")
	}
}

func TestPortalListenerRejectsWildcardNamesAndZeroPort(t *testing.T) {
	for _, addr := range []string{"", ":8782", "0.0.0.0:8782", "[::]:8782", "localhost:8782", "192.168.50.1:0", "224.0.0.1:8782", "192.168.50.1", "192.168.50.1:65536"} {
		listener, err := ListenPortal(addr)
		if listener != nil {
			listener.Close()
		}
		if err == nil {
			t.Fatalf("accepted %q", addr)
		}
	}
}

func TestStartupPauseSurvivesSessionAndRequiresResume(t *testing.T) {
	r := &recording{ok: true}
	s := NewPausedServer(r, nil)
	if !s.paused || s.estop || s.epoch != 1 {
		t.Fatal("startup did not establish Pause")
	}
	id := activate(t, s, 10)
	manual := protocol.ManualPayload{Hold: true, Forward: .2, Epoch: 1}
	ack := s.Handle(packet(protocol.Manual, id, 1, 11, manual), addr, 11)
	var receipt protocol.Receipt
	if ack == nil || protocol.DecodeJSON(ack.Payload, &receipt) != nil || receipt.Code != "inhibited" || !receipt.BenchPaused || len(r.events) != 0 {
		t.Fatal("startup Pause accepted manual input", ack)
	}
	s.Advance(2011)
	id = activate(t, s, 2012)
	if !s.paused {
		t.Fatal("new session cleared startup Pause")
	}
	ack = s.Handle(packet(protocol.Safety, id, 1, 2013, protocol.SafetyPayload{Action: "resume", Epoch: 1}), addr, 2013)
	if ack == nil || protocol.DecodeJSON(ack.Payload, &receipt) != nil || receipt.Status != protocol.StatusQueued || receipt.BenchPaused {
		t.Fatal("explicit Resume failed", ack)
	}
	manual.Epoch = receipt.Epoch
	ack = s.Handle(packet(protocol.Manual, id, 1, 2014, manual), addr, 2014)
	if ack == nil || protocol.DecodeJSON(ack.Payload, &receipt) != nil || receipt.Status != protocol.StatusQueued || !s.manual.Hold {
		t.Fatal("fresh manual input after Resume failed", ack)
	}
}
