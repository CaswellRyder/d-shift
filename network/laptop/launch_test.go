package laptop

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func TestLaunchRemainsUnavailableWithFreshBenchLink(t *testing.T) {
	f := fake(t)
	c := client(t, f)
	journalPath := t.TempDir() + "/journal.jsonl"
	j, err := NewJournal(journalPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { j.Close() })
	c.journal = j
	if err := c.Poll(context.Background()); err != nil {
		t.Fatal(err)
	}
	h := Handler(c, "127.0.0.1:8782")
	r := httptest.NewRequest("GET", "http://127.0.0.1:8782/api/state", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	var state State
	if err := json.Unmarshal(w.Body.Bytes(), &state); err != nil {
		t.Fatal(err)
	}
	if w.Code != http.StatusOK || !state.Connected || state.Launch.Available || state.Launch.ControllerAcknowledged || state.Launch.Reason == "" {
		t.Fatalf("fresh bench link incorrectly implies launch readiness: %+v", state)
	}
	f.mu.Lock()
	before := len(f.requests)
	f.mu.Unlock()
	r = httptest.NewRequest("POST", "http://127.0.0.1:8782/api/launch", strings.NewReader("{}"))
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("X-DTR-Request", "1")
	w = httptest.NewRecorder()
	h.ServeHTTP(w, r)
	var response struct {
		Error  string           `json:"error"`
		Launch LaunchCapability `json:"launch"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if w.Code != http.StatusConflict || response.Error != "launch_unavailable" || response.Launch.Available || response.Launch.ControllerAcknowledged || response.Launch.Reason != state.Launch.Reason {
		t.Fatalf("unexpected blocked launch response: %d %+v", w.Code, response)
	}
	f.mu.Lock()
	if len(f.requests) != before || f.missions != 0 || f.manuals != 0 {
		t.Error("blocked launch sent a command")
	}
	f.mu.Unlock()
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(journalPath)
	if err != nil || len(data) != 0 {
		t.Fatalf("blocked launch wrote a command journal entry: %q, %v", data, err)
	}
}

func TestLaunchRequestBoundaries(t *testing.T) {
	c := client(t, fake(t))
	h := Handler(c, "127.0.0.1:8782")
	for _, tc := range []struct {
		name, method, host, origin, body, header, contentType string
		want                                                  int
	}{
		{"get", "GET", "127.0.0.1:8782", "", "", "", "", 405},
		{"foreign host", "POST", "evil.invalid", "", "{}", "1", "application/json", 403},
		{"foreign origin", "POST", "127.0.0.1:8782", "http://evil.invalid", "{}", "1", "application/json", 403},
		{"missing request header", "POST", "127.0.0.1:8782", "", "{}", "", "application/json", 403},
		{"form", "POST", "127.0.0.1:8782", "", "{}", "1", "text/plain", 403},
		{"unknown field", "POST", "127.0.0.1:8782", "", `{"op":1}`, "1", "application/json", 400},
		{"null", "POST", "127.0.0.1:8782", "", "null", "1", "application/json", 400},
		{"trailing value", "POST", "127.0.0.1:8782", "", "{}{}", "1", "application/json", 400},
		{"oversized", "POST", "127.0.0.1:8782", "", strings.Repeat(" ", 2049), "1", "application/json", 400},
		{"offline", "POST", "127.0.0.1:8782", "", "{}", "1", "application/json", 409},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(tc.method, "http://"+tc.host+"/api/launch", strings.NewReader(tc.body))
			r.Header.Set("Origin", tc.origin)
			r.Header.Set("X-DTR-Request", tc.header)
			r.Header.Set("Content-Type", tc.contentType)
			w := httptest.NewRecorder()
			h.ServeHTTP(w, r)
			if w.Code != tc.want {
				t.Fatalf("got %d, want %d: %s", w.Code, tc.want, w.Body.String())
			}
		})
	}
}
