package pibridge

import (
	"errors"
	"net"
	"net/http"
	"net/netip"
)

// ListenPortal requires an explicit literal unicast address. The portal exposes
// navigation only; commands remain on the laptop's loopback operator service.
func ListenPortal(address string) (net.Listener, error) {
	a, err := netip.ParseAddrPort(address)
	if err != nil || a.Port() == 0 || a.Addr().IsUnspecified() || a.Addr().IsMulticast() {
		return nil, errors.New("portal listener must be an explicit literal unicast IP and nonzero port")
	}
	return net.Listen("tcp", address)
}

// Portal never forwards traffic or executes commands. Its console link opens a
// service on the device viewing the page, rather than exposing that service over
// Wi-Fi. A laptop station must already be running on that device.
func Portal(host string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
		if r.Host != host || (r.Header.Get("Origin") != "" && r.Header.Get("Origin") != "http://"+host) {
			http.Error(w, "invalid host or origin", http.StatusForbidden)
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			http.Error(w, "navigation only; commands are available in the laptop console", http.StatusMethodNotAllowed)
			return
		}
		switch r.URL.Path {
		case "/":
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			if r.Method == http.MethodGet {
				w.Write([]byte(portalHTML))
			}
		case "/healthz":
			w.Header().Set("Content-Type", "application/json")
			if r.Method == http.MethodGet {
				w.Write([]byte(`{"role":"navigation_portal","commands":false}`))
			}
		default:
			http.NotFound(w, r)
		}
	})
}

const portalHTML = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DTR Wi-Fi station</title>
<style>body{margin:0;background:#f2f5f8;color:#182a3a;font:18px/1.55 system-ui,sans-serif}main{max-width:640px;margin:8vh auto;padding:28px}h1{font-size:2rem;line-height:1.2}a{display:inline-block;padding:14px 22px;background:#14558c;color:white;border-radius:8px;text-decoration:none;font-weight:650}a:focus-visible{outline:3px solid #dd7800;outline-offset:4px}.notice{border-left:4px solid #dd7800;padding:4px 18px;margin-top:28px}small{display:block;margin-top:24px;color:#465d70}code{overflow-wrap:anywhere}</style></head>
<body><main><p>DTR blimp</p><h1>Wi-Fi station</h1><p>Join <strong>DTR-Drone-Bench</strong>, then open the console on this laptop.</p><p><a href="http://127.0.0.1:8782/">Open laptop console</a></p><p>The laptop station must be running. If the console does not open, start the laptop station and try again.</p><div class="notice"><strong>Bench controls only</strong><p>Commands currently record input on the Pi. No ESP32 flight controller or motor acknowledgement is connected. Launch is unavailable.</p></div><small>This page opens <code>http://127.0.0.1:8782/</code> on the device you are using. Use the configured laptop.</small></main></body></html>`
