package protocol

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

func TestWireGolden(t *testing.T) {
	key := bytes.Repeat([]byte{0x42}, 32)
	p := Packet{Kind: Poll, Sequence: 0x0102030405060708, TickMS: 0x1112131415161718, Payload: json.RawMessage(`{}`)}
	for i := range p.Session {
		p.Session[i] = byte(i)
	}
	raw, err := Marshal(key, p)
	if err != nil {
		t.Fatal(err)
	}
	want := "4454524704030807060504030201000102030405060708090a0b0c0d0e0f181716151413121102007b7d"
	if hex.EncodeToString(raw[:42]) != want {
		t.Fatalf("header/payload golden mismatch: %x", raw[:42])
	}
	mac := hmac.New(sha256.New, key)
	header, _ := hex.DecodeString(want)
	mac.Write(header)
	if hex.EncodeToString(raw[42:]) != "e98e4ca21b634984501791097b31bbb4b737e428945305791810550fac481857" {
		t.Fatal("HMAC golden mismatch")
	}
	if !bytes.Equal(mac.Sum(nil), raw[42:]) {
		t.Fatal("tag mismatch")
	}
	got, err := Unmarshal(key, raw)
	if err != nil || got.Kind != p.Kind || got.Session != p.Session || got.Sequence != p.Sequence || got.TickMS != p.TickMS || !bytes.Equal(got.Payload, p.Payload) {
		t.Fatal("round trip mismatch", err)
	}
	raw[40] = 'x'
	if string(got.Payload) != "{}" {
		t.Fatal("payload aliases input")
	}
}
func TestPacketRejections(t *testing.T) {
	key := make([]byte, 32)
	p := Packet{Kind: Poll, Payload: json.RawMessage(`{}`)}
	good, _ := Marshal(key, p)
	for i := range good {
		bad := bytes.Clone(good)
		bad[i] ^= 1
		if _, err := Unmarshal(key, bad); err == nil {
			t.Fatalf("tamper byte %d accepted", i)
		}
	}
	for _, raw := range [][]byte{nil, good[:71], append(bytes.Clone(good), 0), make([]byte, 1201)} {
		if _, err := Unmarshal(key, raw); err == nil {
			t.Fatal("invalid length accepted")
		}
	}
	for _, k := range [][]byte{nil, make([]byte, 31), make([]byte, 33)} {
		if _, err := Marshal(k, p); err == nil {
			t.Fatal("bad key accepted")
		}
		if _, err := Unmarshal(k, good); err == nil {
			t.Fatal("bad key accepted")
		}
	}
	for _, kind := range []Kind{0, 11, 255} {
		p.Kind = kind
		if _, err := Marshal(key, p); err == nil {
			t.Fatal("bad kind accepted")
		}
	}
	p.Kind = Poll
	p.Sequence = MaxSequence + 1
	if _, err := Marshal(key, p); err == nil {
		t.Fatal("sequence overflow accepted")
	}
	p.Sequence = MaxSequence
	p.Payload = json.RawMessage(`{"x":"` + strings.Repeat("a", MaxPayload-8) + `"}`)
	if len(p.Payload) != MaxPayload {
		t.Fatal("fixture")
	}
	if _, err := Marshal(key, p); err != nil {
		t.Fatal(err)
	}
	p.Payload = append(p.Payload, ' ')
	if _, err := Marshal(key, p); err == nil {
		t.Fatal("oversize accepted")
	}
}
func TestStrictJSON(t *testing.T) {
	for _, raw := range []string{``, `null`, `[]`, `1`, `{"x":1,"x":2}`, `{"x":1,"\u0078":2}`, `{"a":{"b":1,"b":2}}`, `{} {}`, `{"a":[NaN]}`, string([]byte{'{', '"', 'x', '"', ':', '"', 255, '"', '}'}), `{"a":` + strings.Repeat("[", 16) + `0` + strings.Repeat("]", 16) + `}`} {
		if _, err := JSON(json.RawMessage(raw)); err == nil {
			t.Fatalf("invalid JSON accepted %q", raw)
		}
	}
	for _, raw := range []string{`{}`, `{"x":[{"y":true},null,1e30]}`, `{"a":` + strings.Repeat("[", 15) + `0` + strings.Repeat("]", 15) + `}`} {
		if _, err := JSON(json.RawMessage(raw)); err != nil {
			t.Fatalf("valid JSON rejected: %v", err)
		}
	}
	var dst HelloPayload
	if err := DecodeJSON([]byte(`{"nonce":"abc","extra":1}`), &dst); err == nil {
		t.Fatal("unknown accepted")
	}
	if err := DecodeJSON([]byte(`{"nonce":1}`), &dst); err == nil {
		t.Fatal("wrong type accepted")
	}
	if err := DecodeJSON([]byte(`{"nonce":"abc"}`), &dst); err != nil || dst.Nonce != "abc" {
		t.Fatal("decode failed")
	}
}
func TestSessionAndDigest(t *testing.T) {
	a, err := NewSession()
	if err != nil || a == (SessionID{}) || len(a.String()) != 32 {
		t.Fatal("invalid session", err)
	}
	b, err := NewSession()
	if err != nil || a == b {
		t.Fatal("session repeated")
	}
	if Digest([]byte("abc")) != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" {
		t.Fatal("digest mismatch")
	}
}
func TestPrivateKeys(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "key")
	if err := GenerateKey(path); err != nil {
		t.Fatal(err)
	}
	key, err := LoadKey(path)
	if err != nil || len(key) != 32 {
		t.Fatal(err)
	}
	if err := GenerateKey(path); err == nil {
		t.Fatal("overwrote key")
	}
	again, _ := LoadKey(path)
	if !bytes.Equal(key, again) {
		t.Fatal("existing key changed")
	}
	for _, mode := range []os.FileMode{0644, 0640, 0610, 0601} {
		if err := os.Chmod(path, mode); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadKey(path); err == nil {
			t.Fatal("permissive key accepted")
		}
	}
	if err := os.Chmod(path, 0400); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadKey(path); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(path, link); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadKey(link); err == nil {
		t.Fatal("symlink accepted")
	}
	if _, err := LoadKey(dir); err == nil {
		t.Fatal("directory accepted")
	}
	fifo := filepath.Join(dir, "fifo")
	if err := syscall.Mkfifo(fifo, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadKey(fifo); err == nil {
		t.Fatal("fifo accepted")
	}
	for _, size := range []int{0, 31, 33} {
		p := filepath.Join(dir, strings.Repeat("x", size+1))
		if err := os.WriteFile(p, make([]byte, size), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadKey(p); err == nil {
			t.Fatal("bad size accepted")
		}
	}
}
func FuzzUnmarshal(f *testing.F) {
	key := make([]byte, 32)
	raw, _ := Marshal(key, Packet{Kind: Poll, Payload: json.RawMessage(`{}`)})
	f.Add(raw)
	f.Add([]byte{})
	f.Add(bytes.Repeat([]byte{255}, 1201))
	f.Fuzz(func(t *testing.T, raw []byte) {
		p, err := Unmarshal(key, raw)
		if err != nil {
			return
		}
		encoded, err := Marshal(key, p)
		if err != nil || !bytes.Equal(encoded, raw) {
			t.Fatal("accepted packet cannot reencode")
		}
	})
}
func FuzzJSON(f *testing.F) {
	for _, s := range []string{`{}`, `{"x":1,"x":2}`, `{"x":[1,true,null]}`, `{} []`} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		raw, err := JSON(json.RawMessage(s))
		if err == nil && !json.Valid(raw) {
			t.Fatal("invalid result")
		}
	})
}

func TestTypedPresenceAndShape(t *testing.T) {
	for _, raw := range []string{`{}`, `{"forward":0,"right":0,"up":0,"yaw":0}`, `{"forward":null,"right":0,"up":0,"yaw":0,"hold":false}`, `{"forward":0,"right":0,"up":0,"yaw":0,"hold":null}`, `{"Forward":0,"right":0,"up":0,"yaw":0,"hold":false}`} {
		var dst ManualPayload
		if DecodeJSON([]byte(raw), &dst) == nil {
			t.Fatalf("invalid manual accepted %s", raw)
		}
	}
	var manual ManualPayload
	if DecodeJSON([]byte(`{"forward":0,"right":0,"up":0,"yaw":0,"hold":false}`), &manual) != nil {
		t.Fatal("complete manual rejected")
	}
	type shape struct {
		Axes     *[4]float64 `json:"axes"`
		Optional string      `json:"optional,omitempty"`
	}
	for _, raw := range []string{`{}`, `{"axes":[1,2,3]}`, `{"axes":[1,2,3,4,5]}`, `{"axes":[1,2,null,4]}`} {
		var dst shape
		if DecodeJSON([]byte(raw), &dst) == nil {
			t.Fatal("invalid array accepted")
		}
	}
	for _, raw := range []string{`{"axes":null}`, `{"axes":[1,2,3,4]}`} {
		var dst shape
		if DecodeJSON([]byte(raw), &dst) != nil {
			t.Fatal("valid nullable shape rejected")
		}
	}
	var empty struct{}
	if DecodeJSON([]byte(`{}`), &empty) != nil {
		t.Fatal("empty poll rejected")
	}
}

func TestV4RequiresPauseFieldsAndRejectsV3(t *testing.T) {
	for _, value := range []any{Receipt{BenchOnly: true}, Telemetry{BenchOnly: true}} {
		raw, err := JSON(value)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]json.RawMessage
		json.Unmarshal(raw, &fields)
		delete(fields, "bench_paused")
		missing, _ := json.Marshal(fields)
		var dst any
		switch value.(type) {
		case Receipt:
			dst = &Receipt{}
		case Telemetry:
			dst = &Telemetry{}
		}
		if DecodeJSON(missing, dst) == nil {
			t.Fatal("missing required bench_paused accepted")
		}
		fields["bench_paused"] = json.RawMessage("null")
		missing, _ = json.Marshal(fields)
		if DecodeJSON(missing, dst) == nil {
			t.Fatal("null bench_paused accepted")
		}
	}
	key := bytes.Repeat([]byte{42}, 32)
	wire, _ := Marshal(key, Packet{Kind: Poll, Payload: json.RawMessage(`{}`)})
	wire[4] = 3
	mac := hmac.New(sha256.New, key)
	mac.Write(wire[:len(wire)-TagSize])
	copy(wire[len(wire)-TagSize:], mac.Sum(nil))
	if _, err := Unmarshal(key, wire); err == nil {
		t.Fatal("authenticated v3 peer accepted")
	}
}
