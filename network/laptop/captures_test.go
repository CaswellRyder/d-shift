package laptop

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"dtr.local/link/protocol"
)

func TestCaptureTransferIntegrityAndLocalReview(t *testing.T) {
	for _, bad := range []bool{false, true} {
		t.Run(strconv.FormatBool(bad), func(t *testing.T) {
			id := "20261109T120000-abcdef12"
			raw := []byte("synthetic raw archive bytes")
			sum := sha256.Sum256(raw)
			digest := hex.EncodeToString(sum[:])
			advertised := digest
			if bad {
				advertised = strings.Repeat("0", 64)
			}
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/raw-captures":
					json.NewEncoder(w).Encode(map[string]any{"captures": []RemoteCapture{{ID: id, Bytes: int64(len(raw)), SHA256: advertised, Frames: 1, Source: "replay"}}})
				case "/raw-captures/" + id + "/capture.tar":
					w.Write(raw)
				default:
					http.NotFound(w, r)
				}
			}))
			defer upstream.Close()
			directory := t.TempDir()
			processor := filepath.Join(directory, "processor.sh")
			script := fmt.Sprintf("mkdir -p \"$4/%s\"\nprintf '%%s' '%s' > \"$4/%s/summary.json\"\nprintf '%%s' '%s' > \"$4/%s/raw-provenance.json\"\nprintf 'review' > \"$4/%s/review.html\"\n", id, `{"id":"`+id+`","frames":1,"source":"replay","complete":false,"export_complete":true,"capture_error":"fixture incomplete"}`, id, `{"archive_sha256":"`+digest+`"}`, id, id)
			if e := os.WriteFile(processor, []byte(script), 0600); e != nil {
				t.Fatal(e)
			}
			c, e := NewCaptures(upstream.URL, filepath.Join(directory, "captures"), "/bin/sh", processor, "127.0.0.1:8782")
			if e != nil {
				t.Fatal(e)
			}
			defer c.Close()
			e = c.collectOne(id)
			if bad {
				if e == nil || !strings.Contains(e.Error(), "integrity") {
					t.Fatal("bad hash accepted", e)
				}
				if len(c.State().Local) != 0 {
					t.Fatal("invalid transfer published")
				}
				return
			}
			if e != nil {
				t.Fatal(e)
			}
			state := c.State()
			if len(state.Local) != 1 || state.Local[0].CaptureComplete || state.Local[0].ArchiveSHA256 != digest {
				t.Fatal(state)
			}
			req := httptest.NewRequest("GET", "http://127.0.0.1:8782"+state.Local[0].ReviewURL, nil)
			w := httptest.NewRecorder()
			c.ServeHTTP(w, req)
			if w.Code != 200 || w.Body.String() != "review" {
				t.Fatal(w.Code, w.Body.String())
			}
			req = httptest.NewRequest("GET", "http://foreign.invalid/api/captures", nil)
			w = httptest.NewRecorder()
			c.ServeHTTP(w, req)
			if w.Code != 403 {
				t.Fatal("foreign host accepted")
			}
			req = httptest.NewRequest("GET", "http://127.0.0.1:8782/captures/"+id+"/../../processor.sh", nil)
			w = httptest.NewRecorder()
			c.ServeHTTP(w, req)
			if w.Code != 404 {
				t.Fatal("traversal allowed", w.Code)
			}
		})
	}
}

func TestManualHTTPGenerationRejectsDelayedPreStopHold(t *testing.T) {
	f := fake(t)
	c := client(t, f)
	if e := c.Poll(context.Background()); e != nil {
		t.Fatal(e)
	}
	old := c.State().InputGeneration
	c.generation.Add(1)
	raw, _ := protocol.JSON(protocol.ManualPayload{Forward: .2, Hold: true})
	request := httptest.NewRequest("POST", "http://127.0.0.1:8782/api/manual", strings.NewReader(string(raw)))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-DTR-Request", "1")
	request.Header.Set("X-DTR-Control-Generation", old)
	response := httptest.NewRecorder()
	Handler(c, "127.0.0.1:8782").ServeHTTP(response, request)
	if response.Code != 409 {
		t.Fatal("old HTTP hold accepted", response.Code, response.Body.String())
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.manuals != 0 {
		t.Fatal("old HTTP hold transmitted")
	}
}

func TestDelayedHTTPResetCannotClearNewerEstop(t *testing.T) {
	f := fake(t)
	c := client(t, f)
	old := c.State().InputGeneration
	c.generation.Add(1)
	c.inhibited.Store(true)
	request := httptest.NewRequest("POST", "http://127.0.0.1:8782/api/estop/reset", strings.NewReader("{}"))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-DTR-Request", "1")
	request.Header.Set("X-DTR-Control-Generation", old)
	response := httptest.NewRecorder()
	Handler(c, "127.0.0.1:8782").ServeHTTP(response, request)
	if response.Code != 409 || !c.inhibited.Load() {
		t.Fatal("stale reset cleared new latch", response.Code)
	}
}
func TestCaptureStartAcceptsUpstream202(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(202)
		w.Write([]byte(`{"id":"20261109T120000-abcdef12","state":"capturing"}`))
	}))
	defer upstream.Close()
	c := &Captures{base: upstream.URL, client: upstream.Client()}
	if _, err := c.forward(context.Background(), "/api/capture", []byte(`{}`)); err != nil {
		t.Fatal("successful asynchronous start rejected", err)
	}
}
