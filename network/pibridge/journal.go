package pibridge

import (
	"encoding/json"
	"errors"
	"os"
	"sync"
	"sync/atomic"
)

type Event struct {
	Kind      string `json:"kind"`
	TickMS    uint64 `json:"tick_ms"`
	Sequence  uint64 `json:"seq"`
	Value     any    `json:"value"`
	Reason    string `json:"reason,omitempty"`
	BenchOnly bool   `json:"bench_only"`
	Applied   bool   `json:"applied"`
}
type Recorder interface {
	Healthy() bool
	Mission(Event) bool
	Manual(Event) bool
}
type Journal struct {
	mu       sync.Mutex
	file     *os.File
	safety   chan Event
	missions chan Event
	manual   chan Event
	done     chan struct{}
	healthy  atomic.Bool
	closed   bool
}

func NewJournal(path string) (*Journal, error) {
	f, e := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if e != nil {
		return nil, e
	}
	j := &Journal{file: f, safety: make(chan Event, 16), missions: make(chan Event, 16), manual: make(chan Event, 1), done: make(chan struct{})}
	j.healthy.Store(true)
	go j.run()
	return j, nil
}
func (j *Journal) Healthy() bool { return j.healthy.Load() }
func (j *Journal) Mission(e Event) bool {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.closed || !j.Healthy() {
		return false
	}
	select {
	case j.missions <- e:
		return true
	default:
		return false
	}
}
func (j *Journal) Safety(e Event) bool {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.closed || !j.Healthy() {
		return false
	}
	select {
	case j.safety <- e:
		return true
	default:
		return false
	}
}
func (j *Journal) Manual(e Event) bool {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.closed || !j.Healthy() {
		return false
	}
	select {
	case old := <-j.manual:
		// Neutral events must reach the writer before a new held sample can replace them.
		if old.Kind == "neutral" && e.Kind != "neutral" {
			j.manual <- old
			return false
		}
	default:
	}
	j.manual <- e
	return true
}
func (j *Journal) run() {
	defer close(j.done)
	enc := json.NewEncoder(j.file)
	safety, missions, manual := j.safety, j.missions, j.manual
	for safety != nil || missions != nil || manual != nil {
		var e Event
		var ok bool
		select {
		case e, ok = <-safety:
			if !ok {
				safety = nil
				continue
			}
		case e, ok = <-missions:
			if !ok {
				missions = nil
				continue
			}
		case e, ok = <-manual:
			if !ok {
				manual = nil
				continue
			}
		}
		if enc.Encode(e) != nil {
			j.healthy.Store(false)
		}
	}
	if j.file.Sync() != nil {
		j.healthy.Store(false)
	}
	if j.file.Close() != nil {
		j.healthy.Store(false)
	}
}
func (j *Journal) Close() error {
	j.mu.Lock()
	if !j.closed {
		j.closed = true
		close(j.safety)
		close(j.missions)
		close(j.manual)
	}
	j.mu.Unlock()
	<-j.done
	if !j.Healthy() {
		return errors.New("journal writer failed")
	}
	return nil
}
