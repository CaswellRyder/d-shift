package laptop

import (
	"context"
	"dtr.local/link/protocol"
	"encoding/json"
	"net"
	"net/http/httptest"
	"testing"
	"time"
)

func TestInterfaceBindingSelectsSourceAndRejectsMissingSubnet(t *testing.T) {
	interfaces, err := net.Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	name := ""
	for _, iface := range interfaces {
		addresses, _ := iface.Addrs()
		for _, address := range addresses {
			if subnet, ok := address.(*net.IPNet); ok && subnet.IP.Equal(net.ParseIP("127.0.0.1")) {
				name = iface.Name
			}
		}
	}
	if name == "" {
		t.Skip("no IPv4 loopback interface")
	}
	key := make([]byte, 32)
	c, err := NewOnInterface("127.0.0.1:47110", name, key, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if !c.InterfaceCurrent(name) || c.InterfaceCurrent("dtr-missing-interface") {
		t.Fatal("interface ownership check did not match the bound source")
	}
	if !c.conn.LocalAddr().(*net.UDPAddr).IP.Equal(net.ParseIP("127.0.0.1")) {
		t.Fatalf("wrong source: %s", c.conn.LocalAddr())
	}
	for _, tc := range []struct{ peer, iface string }{
		{"192.0.2.1:47110", name},
		{"127.0.0.1:47110", "dtr-missing-interface"},
		{"[::1]:47110", name},
	} {
		other, e := NewOnInterface(tc.peer, tc.iface, key, nil)
		if other != nil {
			other.Close()
		}
		if e == nil {
			t.Fatalf("unexpected fallback for %+v", tc)
		}
	}
}

func loopbackInterface(t *testing.T) string {
	t.Helper()
	interfaces, err := net.Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	for _, iface := range interfaces {
		addresses, _ := iface.Addrs()
		for _, address := range addresses {
			if subnet, ok := address.(*net.IPNet); ok && subnet.IP.Equal(net.ParseIP("127.0.0.1")) {
				return iface.Name
			}
		}
	}
	t.Skip("no IPv4 loopback")
	return ""
}
func TestWaitingClientServesOfflineStateWithoutWrongNetworkFallback(t *testing.T) {
	c, err := NewWaitingOnInterface("192.0.2.1:47110", loopbackInterface(t), make([]byte, 32), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if err = c.Poll(context.Background()); err == nil {
		t.Fatal("off-subnet poll succeeded")
	}
	if c.conn != nil {
		t.Fatal("opened transport through wrong interface")
	}
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "http://localhost:8782/api/state", nil)
	Handler(c, "127.0.0.1:8782").ServeHTTP(w, r)
	var state State
	if w.Code != 200 || json.Unmarshal(w.Body.Bytes(), &state) != nil || state.Connected || !state.InputInhibited {
		t.Fatal(w.Code, w.Body.String())
	}
	if _, err = c.Manual(context.Background(), protocol.ManualPayload{Hold: true, Forward: .2}); err == nil {
		t.Fatal("offline manual accepted")
	}
	if _, err = c.Safety(context.Background(), "estop"); err == nil {
		t.Fatal("offline E-stop claimed receipt")
	}
	if !c.State().EstopRequested {
		t.Fatal("offline E-stop did not latch locally")
	}
}
func TestWaitingClientBindsOnPollAndRebindInvalidatesInput(t *testing.T) {
	c, err := NewWaitingOnInterface("127.0.0.1:47110", loopbackInterface(t), make([]byte, 32), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	initial := c.generation.Load()
	c.mu.Lock()
	err = c.ensureInterface()
	c.mu.Unlock()
	if err != nil || !c.InterfaceCurrent(loopbackInterface(t)) {
		t.Fatal("no direct socket", err)
	}
	if c.generation.Load() == initial {
		t.Fatal("first connection reused input generation")
	}
	// Simulate an invalidated source and an existing held-input generation.
	c.waitInterface = "dtr-missing"
	c.received = time.Now()
	c.telemetry = &protocol.Telemetry{}
	previous := c.generation.Load()
	c.mu.Lock()
	err = c.ensureInterface()
	c.mu.Unlock()
	if err == nil || c.conn != nil || c.generation.Load() == previous || c.State().Connected || !c.State().InputInhibited {
		t.Fatal("lost interface retained old control state")
	}
	c.waitInterface = loopbackInterface(t)
	c.mu.Lock()
	err = c.ensureInterface()
	c.mu.Unlock()
	if err != nil || !c.InterfaceCurrent(c.waitInterface) || !c.State().PauseRequested {
		t.Fatal("rebind failed or cleared Pause", err)
	}
}
