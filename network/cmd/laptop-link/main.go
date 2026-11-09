package main

import (
	"context"
	"dtr.local/link/laptop"
	"dtr.local/link/protocol"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() (result error) {
	peer := flag.String("peer", "192.168.50.1:47110", "literal Pi UDP endpoint")
	wifiInterface := flag.String("interface", "", "optional Wi-Fi interface on the directly connected Pi subnet")
	keyPath := flag.String("key-file", "", "private key path")
	listen := flag.String("listen", "127.0.0.1:8782", "loopback operator API")
	journalPath := flag.String("journal", "", "new private JSONL journal")
	seconds := flag.Int("seconds", 0, "optional duration, at most 3600 seconds")
	labURL := flag.String("lab-url", "http://127.0.0.1:8771", "loopback Pi lab tunnel")
	captureDir := flag.String("capture-dir", "", "absolute laptop capture destination (enables capture service)")
	capturePython := flag.String("capture-python", "", "absolute laptop Python executable")
	captureProcessor := flag.String("capture-processor", "", "absolute laptop raw capture processor script")
	flag.Parse()
	if !laptop.ListenAddress(*listen) || *seconds < 0 || *seconds > 3600 || *journalPath == "" {
		return fmt.Errorf("invalid listener, duration or journal")
	}
	key, e := protocol.LoadKey(*keyPath)
	if e != nil {
		return e
	}
	j, e := laptop.NewJournal(*journalPath)
	if e != nil {
		return e
	}
	defer func() { result = errors.Join(result, j.Close()) }()
	c, e := laptop.NewWaitingOnInterface(*peer, *wifiInterface, key, j)
	if e != nil {
		return e
	}
	defer c.Close()
	listener, e := net.Listen("tcp", *listen)
	if e != nil {
		return e
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if *seconds > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(*seconds)*time.Second)
		defer cancel()
	}
	host := listener.Addr().String()
	var captures *laptop.Captures
	var captureHandler http.Handler
	if *captureDir != "" {
		captures, e = laptop.NewCaptures(*labURL, *captureDir, *capturePython, *captureProcessor, host)
		if e != nil {
			listener.Close()
			return e
		}
		defer captures.Close()
		captureHandler = captures
		go captures.Run(ctx)
	} else {
		captureHandler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "GET" && r.URL.Path == "/api/captures" {
				w.Header().Set("Content-Type", "application/json")
				w.Write([]byte(`{"configured":false,"busy":false,"status":"not configured","remote":[],"local":[]}`))
				return
			}
			http.Error(w, "capture service is not configured", 503)
		})
	}
	server := &http.Server{Handler: laptop.Dashboard(host, laptop.Handler(c, host), captureHandler), ReadHeaderTimeout: 2 * time.Second, ReadTimeout: 4 * time.Second, WriteTimeout: 10 * time.Second, IdleTimeout: 10 * time.Second, MaxHeaderBytes: 8192}
	done := make(chan error, 1)
	go func() { done <- server.Serve(listener) }()
	go c.Run(ctx)

	select {
	case e = <-done:
		if e != http.ErrServerClosed {
			return e
		}
		e = nil
	case <-ctx.Done():
	}
	shutdown, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	return errors.Join(e, server.Shutdown(shutdown))
}
