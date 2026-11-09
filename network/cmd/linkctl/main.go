package main

import (
	"bytes"
	"context"
	"dtr.local/link/laptop"
	"dtr.local/link/protocol"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
func request(ctx context.Context, base, path string, body any) error {
	method := "GET"
	var b io.Reader
	if body != nil {
		method = "POST"
		raw, e := json.Marshal(body)
		if e != nil {
			return e
		}
		b = bytes.NewReader(raw)
	}
	r, e := http.NewRequestWithContext(ctx, method, strings.TrimRight(base, "/")+"/api/"+path, b)
	if e != nil {
		return e
	}
	if body != nil {
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("X-DTR-Request", "1")
	}
	client := http.Client{Timeout: 4 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	if sample, ok := body.(protocol.ManualPayload); ok && sample.Hold {
		stateRequest, err := http.NewRequestWithContext(ctx, "GET", strings.TrimRight(base, "/")+"/api/state", nil)
		if err != nil {
			return err
		}
		stateResponse, err := client.Do(stateRequest)
		if err != nil {
			return err
		}
		var state struct {
			InputGeneration string `json:"input_generation"`
		}
		err = json.NewDecoder(io.LimitReader(stateResponse.Body, 16384)).Decode(&state)
		stateResponse.Body.Close()
		if err != nil || stateResponse.StatusCode != 200 || state.InputGeneration == "" {
			return fmt.Errorf("fresh control generation unavailable")
		}
		r.Header.Set("X-DTR-Control-Generation", state.InputGeneration)
	}
	res, e := client.Do(r)
	if e != nil {
		return e
	}
	defer res.Body.Close()
	raw, e := io.ReadAll(io.LimitReader(res.Body, 16384))
	if e != nil {
		return e
	}
	if res.StatusCode != 200 {
		return fmt.Errorf("operator API status %d", res.StatusCode)
	}
	fmt.Println(string(raw))
	return nil
}
func run() error {
	global := flag.NewFlagSet("linkctl", flag.ContinueOnError)
	base := global.String("url", "http://127.0.0.1:8782", "operator API")
	if e := global.Parse(os.Args[1:]); e != nil {
		return e
	}
	if !laptop.OperatorURL(*base) {
		return fmt.Errorf("operator URL must be literal loopback HTTP")
	}
	args := global.Args()
	cmd := "state"
	if len(args) > 0 {
		cmd = args[0]
		args = args[1:]
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	switch cmd {
	case "state":
		if len(args) != 0 {
			return fmt.Errorf("unexpected arguments")
		}
		return request(ctx, *base, "state", nil)
	case "reset":
		if len(args) != 0 {
			return fmt.Errorf("unexpected arguments")
		}
		return request(ctx, *base, "reset", struct{}{})
	case "mission":
		f := flag.NewFlagSet(cmd, flag.ContinueOnError)
		op := f.Uint("op", 0, "opaque operation byte")
		a := f.String("args", "", "hex arguments")
		if e := f.Parse(args); e != nil {
			return e
		}
		if *op > 255 || f.NArg() != 0 {
			return fmt.Errorf("invalid mission arguments")
		}
		return request(ctx, *base, "mission", protocol.MissionPayload{Op: uint8(*op), Args: *a})
	case "manual":
		f := flag.NewFlagSet(cmd, flag.ContinueOnError)
		p := protocol.ManualPayload{}
		f.Float64Var(&p.Forward, "forward", 0, "normalized axis")
		f.Float64Var(&p.Right, "right", 0, "normalized axis")
		f.Float64Var(&p.Up, "up", 0, "normalized axis")
		f.Float64Var(&p.Yaw, "yaw", 0, "normalized axis")
		f.BoolVar(&p.Hold, "hold", false, "hold sample")
		seconds := f.Float64("seconds", 0, "explicit bounded demo duration, at most 60 seconds")
		if e := f.Parse(args); e != nil {
			return e
		}
		if f.NArg() != 0 || !laptop.ValidManual(p) || !(*seconds >= 0 && *seconds <= 60) {
			return fmt.Errorf("invalid manual sample or duration")
		}
		if *seconds == 0 {
			return request(ctx, *base, "manual", p)
		}
		defer func() {
			release, cancel := context.WithTimeout(context.Background(), 4*time.Second)
			defer cancel()
			_ = request(release, *base, "manual", protocol.ManualPayload{})
		}()
		deadline := time.NewTimer(time.Duration(*seconds * float64(time.Second)))
		defer deadline.Stop()
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		for {
			if e := request(ctx, *base, "manual", p); e != nil {
				return e
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-deadline.C:
				return nil
			case <-ticker.C:
			}
		}
	default:
		return fmt.Errorf("expected state, manual, mission or reset")
	}
}
