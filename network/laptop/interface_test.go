package laptop

import (
	"net"
	"testing"
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
