package laptop

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"dtr.local/link/protocol"
)

const maxCaptureBytes int64 = 64 << 20

var captureID = regexp.MustCompile(`^\d{8}T\d{6}-[0-9a-f]{8}$`)
var captureImage = regexp.MustCompile(`^(raw|masks|overlay|tracker-masks)/\d{4}\.png$`)

type RemoteCapture struct {
	ID        string `json:"id"`
	Bytes     int64  `json:"bytes"`
	SHA256    string `json:"sha256"`
	Frames    int    `json:"frames"`
	CreatedAt string `json:"created_at"`
	Source    string `json:"source"`
}
type LocalCapture struct {
	ID              string `json:"id"`
	Status          string `json:"status"`
	ReviewURL       string `json:"review_url"`
	SummaryURL      string `json:"summary_url"`
	ArchiveSHA256   string `json:"archive_sha256"`
	Source          string `json:"source"`
	Frames          int    `json:"frames"`
	CaptureComplete bool   `json:"capture_complete"`
	Error           string `json:"error,omitempty"`
}
type CaptureState struct {
	Configured      bool            `json:"configured"`
	Busy            bool            `json:"busy"`
	Status          string          `json:"status"`
	Error           string          `json:"error"`
	ID              string          `json:"id"`
	Remote          []RemoteCapture `json:"remote"`
	Local           []LocalCapture  `json:"local"`
	Lab             map[string]any  `json:"lab"`
	UpdatedAt       time.Time       `json:"updated_at"`
	BytesDownloaded int64           `json:"bytes_downloaded"`
	TotalBytes      int64           `json:"total_bytes"`
}
type Captures struct {
	mu                                       sync.RWMutex
	state                                    CaptureState
	base, directory, python, processor, host string
	client                                   *http.Client
	ctx                                      context.Context
	cancel                                   context.CancelFunc
}

func NewCaptures(base, directory, python, processor, host string) (*Captures, error) {
	u, e := url.Parse(base)
	if e != nil {
		return nil, errors.New("invalid lab URL")
	}
	addr, e := netip.ParseAddr(u.Hostname())
	if e != nil || !addr.IsLoopback() || u.Scheme != "http" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || (u.Path != "" && u.Path != "/") {
		return nil, errors.New("lab URL must be literal loopback HTTP")
	}
	for _, p := range []string{directory, python, processor} {
		if !filepath.IsAbs(p) {
			return nil, errors.New("capture paths must be absolute")
		}
	}
	if !ListenAddress(host) {
		return nil, errors.New("capture listener must be loopback")
	}
	for _, p := range []string{python, processor} {
		s, e := os.Stat(p)
		if e != nil || !s.Mode().IsRegular() {
			return nil, errors.New("capture processor is unavailable")
		}
	}
	for _, p := range []string{directory, filepath.Join(directory, "raw"), filepath.Join(directory, "review")} {
		if e := os.MkdirAll(p, 0700); e != nil {
			return nil, e
		}
	}
	directory, e = filepath.EvalSymlinks(directory)
	if e != nil {
		return nil, e
	}
	ctx, cancel := context.WithCancel(context.Background())
	c := &Captures{base: strings.TrimRight(base, "/"), directory: directory, python: python, processor: processor, host: host, ctx: ctx, cancel: cancel, client: &http.Client{Timeout: 3 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return errors.New("redirect forbidden") }}, state: CaptureState{Configured: true, Status: "idle", Remote: []RemoteCapture{}, Local: []LocalCapture{}}}
	c.loadLocal()
	return c, nil
}
func (c *Captures) Close() { c.cancel() }
func (c *Captures) State() CaptureState {
	c.mu.RLock()
	defer c.mu.RUnlock()
	s := c.state
	s.Remote = append([]RemoteCapture{}, s.Remote...)
	s.Local = append([]LocalCapture{}, s.Local...)
	return s
}
func (c *Captures) get(ctx context.Context, path string, dst any) error {
	req, e := http.NewRequestWithContext(ctx, "GET", c.base+path, nil)
	if e != nil {
		return e
	}
	r, e := c.client.Do(req)
	if e != nil {
		return errors.New("Pi lab unavailable")
	}
	defer r.Body.Close()
	if r.StatusCode != 200 {
		return fmt.Errorf("Pi lab status %d", r.StatusCode)
	}
	raw, e := io.ReadAll(io.LimitReader(r.Body, (1<<20)+1))
	if e != nil || len(raw) > 1<<20 {
		return errors.New("Pi metadata exceeds limit")
	}
	return json.Unmarshal(raw, dst)
}
func (c *Captures) refresh() {
	var lab map[string]any
	var index struct {
		Captures []RemoteCapture `json:"captures"`
	}
	e := c.get(c.ctx, "/api/telemetry", &lab)
	if e == nil {
		e = c.get(c.ctx, "/api/raw-captures", &index)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if e != nil {
		c.state.Lab = map[string]any{"ready": false, "error": "Pi capture service unavailable"}
		return
	}
	valid := make([]RemoteCapture, 0, len(index.Captures))
	for _, r := range index.Captures {
		if validRemote(r) {
			valid = append(valid, r)
		}
	}
	c.state.Lab = lab
	c.state.Remote = valid
	c.state.UpdatedAt = time.Now().UTC()
}
func (c *Captures) Run(ctx context.Context) {
	defer c.cancel()
	c.refresh()
	t := time.NewTicker(time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-c.ctx.Done():
			return
		case <-t.C:
			c.refresh()
		}
	}
}
func validRemote(r RemoteCapture) bool {
	b, e := hex.DecodeString(r.SHA256)
	return captureID.MatchString(r.ID) && r.Bytes > 0 && r.Bytes <= maxCaptureBytes && r.Frames > 0 && r.Frames <= 140 && e == nil && len(b) == 32 && (r.Source == "camera" || r.Source == "replay")
}
func (c *Captures) update(status string, e error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.state.Status = status
	if e != nil {
		c.state.Error = e.Error()
	} else {
		c.state.Error = ""
	}
}
func (c *Captures) forward(ctx context.Context, path string, raw []byte) (json.RawMessage, error) {
	req, e := http.NewRequestWithContext(ctx, "POST", c.base+path, strings.NewReader(string(raw)))
	if e != nil {
		return nil, e
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Lab-Request", "1")
	r, e := c.client.Do(req)
	if e != nil {
		return nil, errors.New("Pi lab unavailable")
	}
	defer r.Body.Close()
	b, e := io.ReadAll(io.LimitReader(r.Body, 65537))
	if e != nil || len(b) > 65536 {
		return nil, errors.New("invalid Pi response")
	}
	if r.StatusCode < 200 || r.StatusCode >= 300 {
		return nil, fmt.Errorf("Pi rejected request (status %d)", r.StatusCode)
	}
	var response map[string]json.RawMessage
	if protocol.DecodeJSON(b, &response) != nil {
		return nil, errors.New("invalid Pi response")
	}
	return b, nil
}
func (c *Captures) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	if !validOperatorRequest(r, c.host) {
		http.Error(w, "invalid host or origin", 403)
		return
	}
	if strings.HasPrefix(r.URL.Path, "/captures/") {
		c.serveFile(w, r)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if r.URL.Path == "/api/captures" && r.Method == "GET" {
		json.NewEncoder(w).Encode(c.State())
		return
	}
	if r.Method != "POST" {
		http.Error(w, "method not allowed", 405)
		return
	}
	if r.Header.Get("X-DTR-Request") != "1" || strings.ToLower(strings.TrimSpace(strings.Split(r.Header.Get("Content-Type"), ";")[0])) != "application/json" {
		http.Error(w, "JSON request header required", 403)
		return
	}
	raw, e := io.ReadAll(http.MaxBytesReader(w, r.Body, 4096))
	if e != nil {
		http.Error(w, "invalid request", 400)
		return
	}
	var body map[string]json.RawMessage
	if protocol.DecodeJSON(raw, &body) != nil {
		http.Error(w, "invalid JSON object", 400)
		return
	}
	switch r.URL.Path {
	case "/api/captures/start", "/api/captures/config":
		allowed := map[string]bool{"diagnostic": true, "distance_m": true, "target_label": true, "expected_count": true, "scene": true, "notes": true, "data_purpose": true, "session_group": true}
		path := "/api/capture"
		if strings.HasSuffix(r.URL.Path, "/config") {
			allowed = map[string]bool{"phase": true, "goal_color": true}
			path = "/api/config"
		}
		for k := range body {
			if !allowed[k] {
				http.Error(w, "unknown field", 400)
				return
			}
		}
		c.mu.RLock()
		copying := c.state.Busy && (c.state.Status == "copying" || c.state.Status == "queued")
		c.mu.RUnlock()
		if copying {
			http.Error(w, "wait for transfer to finish", 409)
			return
		}
		response, e := c.forward(r.Context(), path, raw)
		if e != nil {
			http.Error(w, e.Error(), 409)
			return
		}
		w.Write(response)
	case "/api/captures/collect":
		var request struct {
			ID string `json:"id"`
		}
		if protocol.DecodeJSON(raw, &request) != nil || !captureID.MatchString(request.ID) {
			http.Error(w, "invalid capture ID", 400)
			return
		}
		c.mu.Lock()
		if c.state.Busy || c.ctx.Err() != nil {
			c.mu.Unlock()
			http.Error(w, "capture worker unavailable or busy", 409)
			return
		}
		c.state.Busy = true
		c.state.Status = "queued"
		c.state.ID = request.ID
		c.state.Error = ""
		c.state.BytesDownloaded = 0
		c.state.TotalBytes = 0
		c.mu.Unlock()
		go c.collect(request.ID)
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(c.State())
	default:
		http.NotFound(w, r)
	}
}
func (c *Captures) collect(id string) {
	defer func() { c.mu.Lock(); c.state.Busy = false; c.mu.Unlock() }()
	e := c.collectOne(id)
	if e != nil {
		c.update("failed", e)
	} else {
		c.update("ready", nil)
	}
}
func (c *Captures) collectOne(id string) error {
	var index struct {
		Captures []RemoteCapture `json:"captures"`
	}
	if e := c.get(c.ctx, "/api/raw-captures", &index); e != nil {
		return e
	}
	var remote RemoteCapture
	for _, r := range index.Captures {
		if r.ID == id && validRemote(r) {
			remote = r
			break
		}
	}
	if remote.ID == "" {
		return errors.New("finished raw capture is unavailable")
	}
	// Pi server independently rejects downloads while capturing/spooling, and
	// prevents another take starting while this transfer owns its response.
	c.update("copying", nil)
	c.mu.Lock()
	c.state.TotalBytes = remote.Bytes
	c.mu.Unlock()
	ctx, cancel := context.WithTimeout(c.ctx, 3*time.Minute)
	defer cancel()
	req, e := http.NewRequestWithContext(ctx, "GET", c.base+"/raw-captures/"+id+"/capture.tar", nil)
	if e != nil {
		return e
	}
	client := http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return errors.New("redirect forbidden") }}
	response, e := client.Do(req)
	if e != nil {
		return errors.New("capture download failed")
	}
	defer response.Body.Close()
	if response.StatusCode != 200 {
		return fmt.Errorf("Pi rejected transfer (status %d)", response.StatusCode)
	}
	if response.ContentLength >= 0 && response.ContentLength != remote.Bytes {
		return errors.New("archive length mismatch")
	}
	file, e := os.CreateTemp(filepath.Join(c.directory, "raw"), ".partial-"+id+"-")
	if e != nil {
		return e
	}
	partial := file.Name()
	h := sha256.New()
	progress := &captureProgress{c: c}
	n, copyErr := io.Copy(io.MultiWriter(file, h, progress), io.LimitReader(response.Body, remote.Bytes+1))
	syncErr := file.Sync()
	closeErr := file.Close()
	if copyErr != nil || syncErr != nil || closeErr != nil {
		return errors.New("archive copy failed; partial retained")
	}
	digest := hex.EncodeToString(h.Sum(nil))
	if n != remote.Bytes || digest != strings.ToLower(remote.SHA256) {
		return errors.New("archive integrity check failed; partial retained")
	}
	archive := filepath.Join(c.directory, "raw", id+".tar")
	if existing, e := os.ReadFile(archive); e == nil {
		sum := sha256.Sum256(existing)
		if hex.EncodeToString(sum[:]) != digest {
			return errors.New("existing archive identity conflicts")
		}
		os.Remove(partial)
	} else if !errors.Is(e, os.ErrNotExist) {
		return e
	} else if e = os.Rename(partial, archive); e != nil {
		return e
	}
	c.update("verified", nil)
	c.mu.RLock()
	already := false
	for _, x := range c.state.Local {
		if x.ID == id && x.ArchiveSHA256 == digest {
			already = true
		}
	}
	c.mu.RUnlock()
	if already {
		return nil
	}
	c.update("processing", nil)
	processCtx, stop := context.WithTimeout(c.ctx, 10*time.Minute)
	defer stop()
	cmd := exec.CommandContext(processCtx, c.python, c.processor, "--input", archive, "--output", filepath.Join(c.directory, "review"))
	log, e := os.OpenFile(filepath.Join(c.directory, "raw", id+"-processing.log"), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0600)
	if e != nil {
		return e
	}
	cmd.Stdout = log
	cmd.Stderr = log
	processErr := cmd.Run()
	log.Close()
	if processErr != nil {
		return errors.New("laptop processing failed; archive and processing log retained")
	}
	local, e := c.localCapture(id)
	if e != nil {
		return e
	}
	if local.ArchiveSHA256 != digest {
		return errors.New("processed provenance does not match verified archive")
	}
	c.mu.Lock()
	c.state.Local = append(c.state.Local, local)
	c.mu.Unlock()
	return nil
}

type captureProgress struct{ c *Captures }

func (p *captureProgress) Write(b []byte) (int, error) {
	p.c.mu.Lock()
	p.c.state.BytesDownloaded += int64(len(b))
	p.c.mu.Unlock()
	return len(b), nil
}
func (c *Captures) localCapture(id string) (LocalCapture, error) {
	root := filepath.Join(c.directory, "review", id)
	var s struct {
		ID             string  `json:"id"`
		Frames         int     `json:"frames"`
		Source         string  `json:"source"`
		Complete       bool    `json:"complete"`
		ExportComplete bool    `json:"export_complete"`
		CaptureError   *string `json:"capture_error"`
	}
	b, e := os.ReadFile(filepath.Join(root, "summary.json"))
	if e != nil {
		return LocalCapture{}, e
	}
	if json.Unmarshal(b, &s) != nil || s.ID != id || !s.ExportComplete || s.Frames < 1 {
		return LocalCapture{}, errors.New("processed capture is incomplete")
	}
	var provenance map[string]any
	b, e = os.ReadFile(filepath.Join(root, "raw-provenance.json"))
	if e != nil {
		return LocalCapture{}, e
	}
	if json.Unmarshal(b, &provenance) != nil {
		return LocalCapture{}, errors.New("invalid processed provenance")
	}
	digest, _ := provenance["archive_sha256"].(string)
	x := LocalCapture{ID: id, Status: "ready", ReviewURL: "/captures/" + id + "/review.html", SummaryURL: "/captures/" + id + "/summary.json", ArchiveSHA256: digest, Source: s.Source, Frames: s.Frames, CaptureComplete: s.Complete}
	if s.CaptureError != nil {
		x.Error = *s.CaptureError
	}
	return x, nil
}
func (c *Captures) loadLocal() {
	entries, _ := os.ReadDir(filepath.Join(c.directory, "review"))
	for _, entry := range entries {
		if entry.IsDir() && captureID.MatchString(entry.Name()) {
			if x, e := c.localCapture(entry.Name()); e == nil {
				c.state.Local = append(c.state.Local, x)
			}
		}
	}
}
func (c *Captures) serveFile(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" && r.Method != "HEAD" {
		http.Error(w, "method not allowed", 405)
		return
	}
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/captures/"), "/")
	if len(parts) < 2 || !captureID.MatchString(parts[0]) {
		http.NotFound(w, r)
		return
	}
	id := parts[0]
	rel := strings.Join(parts[1:], "/")
	allowed := map[string]bool{"review.html": true, "summary.json": true, "scene.json": true, "frames.jsonl": true, "clip.avi": true, "download.zip": true, "mapping.json": true, "wire-v1.jsonl": true, "wire-v2.jsonl": true, "esp32-draft.jsonl": true, "events.jsonl": true, "rehearsal.jsonl": true, "raw-manifest.json": true, "raw-provenance.json": true}
	if !allowed[rel] && !captureImage.MatchString(rel) {
		http.NotFound(w, r)
		return
	}
	root := filepath.Join(c.directory, "review", id)
	path, e := filepath.EvalSymlinks(filepath.Join(root, filepath.FromSlash(rel)))
	if e != nil {
		http.NotFound(w, r)
		return
	}
	absoluteRoot, e := filepath.EvalSymlinks(root)
	if e != nil || !strings.HasPrefix(path, absoluteRoot+string(os.PathSeparator)) {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
	http.ServeFile(w, r, path)
}
