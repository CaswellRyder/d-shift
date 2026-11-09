package laptop

import (
	"context"
	"dtr.local/link/protocol"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// Handler accepts the loopback listener and its same-origin localhost alias.
func Handler(c *Client, host string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		if !validOperatorRequest(r, host) {
			http.Error(w, "invalid host or origin", http.StatusForbidden)
			return
		}
		if r.URL.Path == "/api/state" && r.Method == http.MethodGet {
			json.NewEncoder(w).Encode(c.State())
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", 405)
			return
		}
		if r.Header.Get("X-DTR-Request") != "1" || strings.ToLower(strings.TrimSpace(strings.Split(r.Header.Get("Content-Type"), ";")[0])) != "application/json" {
			http.Error(w, "JSON request header required", 403)
			return
		}
		raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 2048))
		if err != nil {
			http.Error(w, "invalid request", 400)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 3500*time.Millisecond)
		defer cancel()
		var receipt protocol.Receipt
		switch r.URL.Path {
		case "/api/launch":
			var p struct{}
			if protocol.DecodeJSON(raw, &p) != nil {
				http.Error(w, "invalid launch request", http.StatusBadRequest)
				return
			}
			w.WriteHeader(http.StatusConflict)
			json.NewEncoder(w).Encode(struct {
				Error  string           `json:"error"`
				Launch LaunchCapability `json:"launch"`
			}{Error: "launch_unavailable", Launch: unavailableLaunch()})
			return
		case "/api/manual":
			var p protocol.ManualPayload
			if protocol.DecodeJSON(raw, &p) != nil {
				http.Error(w, "invalid manual", 400)
				return
			}
			if p.Hold {
				generation, parseErr := strconv.ParseUint(r.Header.Get("X-DTR-Control-Generation"), 10, 64)
				if parseErr != nil {
					http.Error(w, "fresh input generation required", 400)
					return
				}
				ctx = context.WithValue(ctx, manualGenerationKey{}, generation)
			}
			receipt, err = c.Manual(ctx, p)
		case "/api/mission":
			var p protocol.MissionPayload
			if protocol.DecodeJSON(raw, &p) != nil {
				http.Error(w, "invalid mission", 400)
				return
			}
			receipt, err = c.Mission(ctx, p)
		case "/api/stop", "/api/pause", "/api/resume", "/api/estop", "/api/estop/reset":
			var p struct{}
			if protocol.DecodeJSON(raw, &p) != nil {
				http.Error(w, "invalid safety request", 400)
				return
			}
			action := "pause"
			if r.URL.Path == "/api/estop" {
				action = "estop"
			}
			if r.URL.Path == "/api/estop/reset" || r.URL.Path == "/api/resume" {
				action = "reset"
				if r.URL.Path == "/api/resume" {
					action = "resume"
				}
				generation, parseErr := strconv.ParseUint(r.Header.Get("X-DTR-Control-Generation"), 10, 64)
				if parseErr != nil {
					http.Error(w, "fresh reset generation required", 400)
					return
				}
				ctx = context.WithValue(ctx, manualGenerationKey{}, generation)
			}
			receipt, err = c.Safety(ctx, action)
		case "/api/reset":
			var p struct{}
			if protocol.DecodeJSON(raw, &p) != nil {
				http.Error(w, "invalid reset", 400)
				return
			}
			c.Reset()
			json.NewEncoder(w).Encode(c.State())
			return
		default:
			http.NotFound(w, r)
			return
		}
		if err != nil {
			http.Error(w, err.Error(), 409)
			return
		}
		json.NewEncoder(w).Encode(struct {
			Receipt          protocol.Receipt `json:"receipt"`
			ControllerActual any              `json:"controller_actual"`
			ESPStatus        any              `json:"esp_status"`
		}{Receipt: receipt})
	})
}
func ListenAddress(address string) bool {
	h, p, e := net.SplitHostPort(address)
	return e == nil && h == "127.0.0.1" && p != ""
}
func OperatorURL(raw string) bool {
	u, e := url.Parse(raw)
	return e == nil && u.Scheme == "http" && u.User == nil && u.RawQuery == "" && u.Fragment == "" && (u.Path == "" || u.Path == "/") && ListenAddress(u.Host)
}

func validOperatorRequest(r *http.Request, host string) bool {
	if r.Host != host {
		_, port, err := net.SplitHostPort(host)
		if err != nil || !ListenAddress(host) || r.Host != "localhost:"+port {
			return false
		}
	}
	origin := r.Header.Get("Origin")
	return origin == "" || origin == "http://"+r.Host
}
