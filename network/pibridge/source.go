package pibridge

import (
	"context"
	"dtr.local/link/protocol"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/netip"
	"net/url"
	"sync"
	"time"
)

type Source struct {
	temp    *float64
	mu      sync.RWMutex
	vision  protocol.Vision
	fetched time.Time
	client  *http.Client
	URL     string
}

func NewSource(raw string) (*Source, error) {
	u, e := url.Parse(raw)
	if e != nil {
		return nil, e
	}
	a, e := netip.ParseAddr(u.Hostname())
	if e != nil || !a.IsLoopback() || u.Scheme != "http" || u.User != nil || u.Fragment != "" {
		return nil, errors.New("source must be literal loopback HTTP")
	}
	return &Source{URL: raw, client: &http.Client{Timeout: 400 * time.Millisecond, CheckRedirect: func(*http.Request, []*http.Request) error { return errors.New("redirect forbidden") }}, vision: protocol.Vision{Source: "unavailable", Error: "unavailable"}}, nil
}
func (s *Source) Run(ctx context.Context) {
	s.Fetch(ctx)
	t := time.NewTicker(500 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.Fetch(ctx)
		}
	}
}
func (s *Source) Fetch(ctx context.Context) {
	start := time.Now()
	v := protocol.Vision{Source: "unavailable", Error: "source_error"}
	req, e := http.NewRequestWithContext(ctx, "GET", s.URL, nil)
	if e == nil {
		var r *http.Response
		r, e = s.client.Do(req)
		if e == nil {
			defer r.Body.Close()
			var b []byte
			b, e = io.ReadAll(io.LimitReader(r.Body, 2*1024*1024+1))
			if len(b) > 2*1024*1024 {
				v.Error = "size_limit"
			} else if e == nil && r.StatusCode == 200 {
				v = Normalize(b)
				if v.AgeMS != nil {
					*v.AgeMS += float64(time.Since(start).Milliseconds())
				}
			}
		}
	}
	s.mu.Lock()
	s.vision = v
	s.temp = cpuTemp()
	s.fetched = time.Now()
	s.mu.Unlock()
}
func (s *Source) Snapshot(now time.Time) protocol.Vision {
	s.mu.RLock()
	v := s.vision
	f := s.fetched
	s.mu.RUnlock()
	if v.AgeMS != nil {
		x := *v.AgeMS + float64(now.Sub(f).Milliseconds())
		v.AgeMS = &x
	}
	if v.AgeMS == nil || *v.AgeMS > 2000 || !v.Ready {
		v.Ready = false
		v.Target = nil
		v.Avoid = nil
		if v.Error == "" {
			v.Error = "stale"
		}
	}
	return v
}
func obj(x any) map[string]any { m, _ := x.(map[string]any); return m }
func str(x any) *string {
	s, ok := x.(string)
	if !ok {
		return nil
	}
	return &s
}
func number(x any) *float64 {
	n, ok := x.(float64)
	if !ok || math.IsNaN(n) || math.IsInf(n, 0) {
		return nil
	}
	return &n
}
func boolean(x any) *bool {
	b, ok := x.(bool)
	if !ok {
		return nil
	}
	return &b
}
func target(x any) *protocol.Target {
	m := obj(x)
	if m == nil {
		return nil
	}
	label, _ := m["label"].(string)
	t := &protocol.Target{Label: label, Score: number(m["score"]), YawDeg: number(m["yaw_deg"]), ElevM: number(m["elev_m"]), TravelM: number(m["travel_m"]), AgeMS: number(m["age_ms"])}

	if a, ok := m["box_px"].([]any); ok && len(a) == 4 {
		var b [4]float64
		valid := true
		for i, x := range a {
			n := number(x)
			if n == nil {
				valid = false
				break
			}
			b[i] = *n
		}
		if valid {
			t.Box = &b
		}
	}
	if a, ok := m["body_xyz_m"].([]any); ok && len(a) == 3 {
		var b [3]float64
		valid := true
		for i, x := range a {
			n := number(x)
			if n == nil {
				valid = false
				break
			}
			b[i] = *n
		}
		if valid {
			t.BodyM = &b
		}
	}
	return t
}
func Normalize(raw []byte) protocol.Vision {
	v := protocol.Vision{Source: "unavailable", Error: "source_error"}
	var m map[string]any
	if json.Unmarshal(raw, &m) != nil {
		return v
	}
	if s := str(m["source"]); s != nil {
		v.Source = *s
	}
	v.AgeMS = number(m["frame_age_ms"])
	if v.AgeMS == nil {
		v.AgeMS = number(m["age_ms"])
	}
	v.Phase = str(m["phase"])
	v.GoalColor = str(m["goal_color"])
	if e := str(m["error"]); e != nil && *e != "" {
		v.Error = "source_error"
		return v
	}
	v.Error = ""
	if n := number(m["frame_seq"]); n != nil && *n >= 0 && math.Trunc(*n) == *n && *n < 1<<63 {
		x := uint64(*n)
		v.FrameSequence = &x
	}
	if n := number(obj(m["frame"])["sequence"]); n != nil && *n >= 0 && math.Trunc(*n) == *n && *n < 1<<63 {
		x := uint64(*n)
		v.FrameSequence = &x
	}
	ft := obj(m["flight_telemetry"])
	coords := obj(ft["coordinates"])
	mapping := obj(ft["mapping_status"])
	draft := obj(obj(ft["outgoing"])["draft_packet"])
	if ft != nil {
		// The preview refreshes more slowly than flight telemetry. Its frame
		// sequence must never label geometry from a newer telemetry row.
		v.FrameSequence = nil
		if n := number(draft["seq"]); n != nil && *n >= 0 && math.Trunc(*n) == *n && *n < 1<<63 {
			x := uint64(*n)
			v.FrameSequence = &x
		}
	}
	v.Intent = str(draft["intent"])
	v.Target = target(coords["target"])
	v.Avoid = target(coords["avoid"])
	selected := v.Target
	if v.Intent != nil && *v.Intent == "AVOID" {
		selected = v.Avoid
	}
	if selected != nil {
		vector := obj(draft["target_vector"])
		selected.YawDeg = number(vector["yaw_deg"])
		selected.ElevM = number(vector["elev_m"])
		selected.TravelM = number(vector["travel_m"])
	}
	v.LensCalibrated = boolean(mapping["lens_calibrated"])
	v.MountCalibrated = boolean(mapping["mount_calibrated"])
	v.FPS = number(obj(m["metrics"])["fps"])
	ready := boolean(m["ready"])
	v.Ready = ft != nil && ready != nil && *ready && v.AgeMS != nil && *v.AgeMS >= 0 && *v.AgeMS <= 2000
	if !v.Ready {
		v.Target = nil
		v.Avoid = nil
		v.Error = "unavailable"
	}
	return v
}
