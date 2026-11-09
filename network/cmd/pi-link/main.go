package main

import (
	"context"
	"dtr.local/link/pibridge"
	"dtr.local/link/protocol"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func run() (result error) {
	bind := flag.String("bind", "", "explicit literal Pi bind address")
	port := flag.Uint("port", 47110, "UDP port")
	keyPath := flag.String("key-file", "", "private key file")
	journalPath := flag.String("journal", "", "new exclusive JSONL journal")
	sourceURL := flag.String("source-url", "http://127.0.0.1:8771/api/telemetry", "loopback HTTP source")
	seconds := flag.Uint("seconds", 0, "optional runtime, at most 3600 seconds")
	portalListen := flag.String("portal-listen", "", "optional explicit Pi navigation portal, e.g. 192.168.50.1:8782")
	motorBus := flag.Int("motor-i2c-bus", -1, "enable ESP32 motor bridge on this Linux I2C bus; default disabled")
	motorAddr := flag.Uint("motor-i2c-address", 0x42, "ESP32 slave address")
	startPaused := flag.Bool("start-paused", true, "start with bench Pause latched; explicit Resume required")
	flag.Parse()
	if *bind == "" || *keyPath == "" || *journalPath == "" || *port == 0 || *port > 65535 || *seconds > 3600 || flag.NArg() != 0 {
		return fmt.Errorf("require --bind --key-file --journal; valid port and seconds <=3600")
	}
	if *motorBus < -1 || *motorBus > 32 {
		return fmt.Errorf("invalid motor I2C bus")
	}
	source, e := pibridge.NewSource(*sourceURL)
	if e != nil {
		return e
	}
	key, e := protocol.LoadKey(*keyPath)
	if e != nil {
		return e
	}
	conn, e := pibridge.Listen(*bind, uint16(*port))
	if e != nil {
		return e
	}
	defer conn.Close()
	journal, e := pibridge.NewJournal(*journalPath)
	if e != nil {
		return e
	}
	defer func() { result = errors.Join(result, journal.Close()) }()
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if *seconds > 0 {
		var c context.CancelFunc
		ctx, c = context.WithTimeout(ctx, time.Duration(*seconds)*time.Second)
		defer c()
	}
	var portal *http.Server
	var portalDone chan error
	if *portalListen != "" {
		listener, err := pibridge.ListenPortal(*portalListen)
		if err != nil {
			return err
		}
		defer listener.Close()
		portal = &http.Server{Handler: pibridge.Portal(listener.Addr().String()), ReadHeaderTimeout: 2 * time.Second, ReadTimeout: 4 * time.Second, WriteTimeout: 4 * time.Second, IdleTimeout: 10 * time.Second, MaxHeaderBytes: 8192}
		portalDone = make(chan error, 1)
		go func() { portalDone <- portal.Serve(listener) }()
	}
	go source.Run(ctx)
	server := pibridge.NewServer(journal, source)
	if *startPaused {
		server = pibridge.NewPausedServer(journal, source)
	}
	if *motorBus >= 0 {
		if *motorAddr < 0x08 || *motorAddr > 0x77 {
			return fmt.Errorf("invalid motor I2C address")
		}
		motor, err := pibridge.OpenMotorLink(*motorBus, uint8(*motorAddr))
		if err != nil {
			return fmt.Errorf("motor I2C: %w", err)
		}
		defer motor.Close()
		server.Motor = motor
	}
	udpDone := make(chan error, 1)
	go func() { udpDone <- server.Serve(ctx, conn, key) }()
	udpFinished := false
	select {
	case e = <-udpDone:
		udpFinished = true
	case e = <-portalDone:
		if errors.Is(e, http.ErrServerClosed) {
			e = nil
		}
	case <-ctx.Done():
	}
	cancel()
	if !udpFinished {
		e = errors.Join(e, <-udpDone)
	}
	if portal != nil {
		shutdown, stop := context.WithTimeout(context.Background(), time.Second)
		defer stop()
		e = errors.Join(e, portal.Shutdown(shutdown))
	}
	return e
}
func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, "pi-link:", e)
		os.Exit(1)
	}
}
