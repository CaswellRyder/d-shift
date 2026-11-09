package laptop

import (
	"dtr.local/link/protocol"
	"encoding/json"
	"errors"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

type Journal struct {
	f      *os.File
	q      chan any
	ok     atomic.Bool
	mu     sync.Mutex
	closed bool
	done   chan struct{}
}

func NewJournal(path string) (*Journal, error) {
	f, e := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return nil, e
	}
	j := &Journal{f: f, q: make(chan any, 128), done: make(chan struct{})}
	j.ok.Store(true)
	go func() {
		defer close(j.done)
		enc := json.NewEncoder(f)
		for x := range j.q {
			if enc.Encode(x) != nil {
				j.ok.Store(false)
			}
		}
		if f.Sync() != nil {
			j.ok.Store(false)
		}
		if f.Close() != nil {
			j.ok.Store(false)
		}
	}()
	return j, nil
}
func (j *Journal) OK() bool { return j.ok.Load() }
func (j *Journal) Record(event string, r *protocol.Receipt) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.closed {
		return
	}
	select {
	case j.q <- struct {
		Time    time.Time         `json:"time"`
		Event   string            `json:"event"`
		Receipt *protocol.Receipt `json:"receipt,omitempty"`
	}{time.Now().UTC(), event, r}:
	default:
		j.ok.Store(false)
	}
}
func (j *Journal) Close() error {
	j.mu.Lock()
	if !j.closed {
		j.closed = true
		close(j.q)
	}
	j.mu.Unlock()
	<-j.done
	if !j.OK() {
		return errors.New("laptop journal writer failed")
	}
	return nil
}
