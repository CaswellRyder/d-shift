package laptop

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDashboardRoutesAndBoundaries(t *testing.T) {
	handler := func(name string) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(name)) })
	}
	d := Dashboard("127.0.0.1:8782", handler("api"), handler("capture"))
	for _, tc := range []struct {
		path, host, origin string
		status             int
		contains           string
	}{
		{"/", "127.0.0.1:8782", "", 200, "Bench console"}, {"/app.js", "127.0.0.1:8782", "", 200, "ManualStream"},
		{"/api/state", "127.0.0.1:8782", "", 200, "api"}, {"/api/captures", "127.0.0.1:8782", "", 200, "capture"}, {"/captures/abc/review.html", "127.0.0.1:8782", "", 200, "capture"},
		{"/", "evil.invalid", "", 403, "invalid"}, {"/", "127.0.0.1:8782", "http://evil.invalid", 403, "invalid"}, {"/manual.test.mjs", "127.0.0.1:8782", "", 404, ""},
	} {
		r := httptest.NewRequest("GET", tc.path, nil)
		r.Host = tc.host
		r.Header.Set("Origin", tc.origin)
		w := httptest.NewRecorder()
		d.ServeHTTP(w, r)
		if w.Code != tc.status || !strings.Contains(w.Body.String(), tc.contains) {
			t.Errorf("%s: %d %s", tc.path, w.Code, w.Body.String())
		}
		if w.Header().Get("Content-Security-Policy") == "" {
			t.Error("missing CSP")
		}
	}
}
