package laptop

import (
	"embed"
	"io/fs"
	"net/http"
	"strings"
)

//go:embed web/index.html web/style.css web/app.js web/manual.js
var dashboardFiles embed.FS

// Dashboard serves only local operator assets and delegates authenticated APIs.
func Dashboard(host string, api http.Handler, captures http.Handler) http.Handler {
	assets, _ := fs.Sub(dashboardFiles, "web")
	files := http.FileServer(http.FS(assets))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
		if r.Host != host || (r.Header.Get("Origin") != "" && r.Header.Get("Origin") != "http://"+host) {
			http.Error(w, "invalid host or origin", 403)
			return
		}
		p := r.URL.Path
		if p == "/api/captures" || strings.HasPrefix(p, "/api/captures/") || strings.HasPrefix(p, "/captures/") {
			if captures == nil {
				http.Error(w, "capture service unavailable", 503)
				return
			}
			captures.ServeHTTP(w, r)
			return
		}
		if strings.HasPrefix(p, "/api/") {
			api.ServeHTTP(w, r)
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "method not allowed", 405)
			return
		}
		switch p {
		case "/", "/style.css", "/app.js", "/manual.js":
			files.ServeHTTP(w, r)
		default:
			http.NotFound(w, r)
		}
	})
}
