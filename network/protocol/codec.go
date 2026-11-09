package protocol

import (
	"bytes"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
	"unicode/utf8"
)

var errPacket = errors.New("invalid protocol packet")
var errJSON = errors.New("invalid protocol JSON object")

func Marshal(key []byte, p Packet) ([]byte, error) {
	if len(key) != 32 || p.Kind < Hello || p.Kind > SafetyAck || p.Sequence > MaxSequence || len(p.Payload) > MaxPayload {
		return nil, errPacket
	}
	if err := validateJSON(p.Payload); err != nil {
		return nil, err
	}
	raw := make([]byte, HeaderSize+len(p.Payload)+TagSize)
	copy(raw, "DTRG")
	raw[4] = Version
	raw[5] = byte(p.Kind)
	binary.LittleEndian.PutUint64(raw[6:14], p.Sequence)
	copy(raw[14:30], p.Session[:])
	binary.LittleEndian.PutUint64(raw[30:38], p.TickMS)
	binary.LittleEndian.PutUint16(raw[38:40], uint16(len(p.Payload)))
	copy(raw[HeaderSize:], p.Payload)
	mac := hmac.New(sha256.New, key)
	mac.Write(raw[:len(raw)-TagSize])
	copy(raw[len(raw)-TagSize:], mac.Sum(nil))
	return raw, nil
}

func Unmarshal(key, raw []byte) (Packet, error) {
	var p Packet
	if len(key) != 32 || len(raw) < HeaderSize+TagSize || len(raw) > MaxDatagram {
		return p, errPacket
	}
	n := int(binary.LittleEndian.Uint16(raw[38:40]))
	if string(raw[:4]) != "DTRG" || raw[4] != Version || Kind(raw[5]) < Hello || Kind(raw[5]) > SafetyAck || n != len(raw)-HeaderSize-TagSize || binary.LittleEndian.Uint64(raw[6:14]) > MaxSequence {
		return p, errPacket
	}
	mac := hmac.New(sha256.New, key)
	mac.Write(raw[:len(raw)-TagSize])
	if !hmac.Equal(mac.Sum(nil), raw[len(raw)-TagSize:]) {
		return p, errPacket
	}
	payload := raw[HeaderSize : len(raw)-TagSize]
	if err := validateJSON(payload); err != nil {
		return p, err
	}
	p.Kind = Kind(raw[5])
	p.Sequence = binary.LittleEndian.Uint64(raw[6:14])
	copy(p.Session[:], raw[14:30])
	p.TickMS = binary.LittleEndian.Uint64(raw[30:38])
	p.Payload = append(json.RawMessage(nil), payload...)
	return p, nil
}

func JSON(v any) (json.RawMessage, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, errJSON
	}
	if err = validateJSON(raw); err != nil {
		return nil, err
	}
	return raw, nil
}

func DecodeJSON(raw []byte, dst any) error {
	if err := validateJSON(raw); err != nil {
		return err
	}
	typ := reflect.TypeOf(dst)
	if typ == nil || typ.Kind() != reflect.Pointer || reflect.ValueOf(dst).IsNil() {
		return errJSON
	}
	if validateShape(raw, typ.Elem()) != nil {
		return errJSON
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if d.Decode(dst) != nil {
		return errJSON
	}
	return nil
}

func validateJSON(raw []byte) error {
	if !utf8.Valid(raw) {
		return errJSON
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	tok, err := d.Token()
	if err != nil || tok != json.Delim('{') {
		return errJSON
	}
	if consumeObject(d, 1) != nil {
		return errJSON
	}
	if _, err = d.Token(); err != io.EOF {
		return errJSON
	}
	return nil
}
func consumeObject(d *json.Decoder, depth int) error {
	if depth > 16 {
		return errJSON
	}
	seen := map[string]bool{}
	for d.More() {
		tok, err := d.Token()
		if err != nil {
			return errJSON
		}
		k, ok := tok.(string)
		if !ok || seen[k] {
			return errJSON
		}
		seen[k] = true
		if consumeValue(d, depth) != nil {
			return errJSON
		}
	}
	t, err := d.Token()
	if err != nil || t != json.Delim('}') {
		return errJSON
	}
	return nil
}
func consumeValue(d *json.Decoder, depth int) error {
	tok, err := d.Token()
	if err != nil {
		return errJSON
	}
	switch tok {
	case json.Delim('{'):
		return consumeObject(d, depth+1)
	case json.Delim('['):
		if depth+1 > 16 {
			return errJSON
		}
		for d.More() {
			if consumeValue(d, depth+1) != nil {
				return errJSON
			}
		}
		t, err := d.Token()
		if err != nil || t != json.Delim(']') {
			return errJSON
		}
	}
	return nil
}

func Digest(raw []byte) string     { sum := sha256.Sum256(raw); return hex.EncodeToString(sum[:]) }
func (s SessionID) String() string { return hex.EncodeToString(s[:]) }
func NewSession() (SessionID, error) {
	for {
		var s SessionID
		if _, err := rand.Read(s[:]); err != nil {
			return s, errors.New("session randomness unavailable")
		}
		if s != (SessionID{}) {
			return s, nil
		}
	}
}

// validateShape preserves the distinction between omitted/null fields and zero.
func validateShape(raw []byte, typ reflect.Type) error {
	if typ == reflect.TypeOf(json.RawMessage{}) {
		return nil
	}
	if typ.Kind() == reflect.Pointer {
		if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return nil
		}
		return validateShape(raw, typ.Elem())
	}
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return errJSON
	}
	switch typ.Kind() {
	case reflect.Struct:
		var fields map[string]json.RawMessage
		if json.Unmarshal(raw, &fields) != nil || fields == nil {
			return errJSON
		}
		for i := 0; i < typ.NumField(); i++ {
			field := typ.Field(i)
			if field.PkgPath != "" {
				continue
			}
			tag := strings.Split(field.Tag.Get("json"), ",")
			name := tag[0]
			if name == "-" {
				continue
			}
			if name == "" {
				name = field.Name
			}
			value, ok := fields[name]
			if !ok {
				optional := false
				for _, opt := range tag[1:] {
					if opt == "omitempty" {
						optional = true
					}
				}
				if !optional {
					return errJSON
				}
				continue
			}
			if validateShape(value, field.Type) != nil {
				return errJSON
			}
			delete(fields, name)
		}
		if len(fields) != 0 {
			return errJSON
		}
	case reflect.Array, reflect.Slice:
		var values []json.RawMessage
		if json.Unmarshal(raw, &values) != nil {
			return errJSON
		}
		if typ.Kind() == reflect.Array && len(values) != typ.Len() {
			return errJSON
		}
		for _, value := range values {
			if validateShape(value, typ.Elem()) != nil {
				return errJSON
			}
		}
	case reflect.Map:
		var values map[string]json.RawMessage
		if json.Unmarshal(raw, &values) != nil {
			return errJSON
		}
		for _, value := range values {
			if validateShape(value, typ.Elem()) != nil {
				return errJSON
			}
		}
	}
	return nil
}
