#include <array>
#include <cassert>
#include <iostream>
#include "../manual_controller.h"
using namespace dtr_manual;
using Frame = std::array<uint8_t, 24>;

Frame command(uint16_t seq, bool hold = true, int up = 0, uint32_t session = 1, int ttl = 20) {
  Frame p{};
  p[0] = kOperatorMagic; p[1] = kVersion;
  write32(p.data() + 2, session); write16(p.data() + 6, seq);
  p[10] = hold ? 3 : 0; p[11] = uint8_t(ttl);
  write16(p.data() + 16, uint16_t(up)); p[23] = crc8(p.data());
  return p;
}
Frame status(const Controller& c, uint32_t now) {
  Frame s{}; c.status(s.data(), now);
  assert(s[23] == crc8(s.data())); return s;
}
void receive(Controller& c, Frame p, uint32_t now) { c.receive(p.data(), now); }
void arm(Controller& c, uint32_t start = 0, uint32_t session = 1) {
  c.update(start, true);
  for (uint16_t i = 0; i <= 40; ++i) receive(c, command(i + 1, true, 0, session), start + i * 50);
  assert(c.armed());
}
int main() {
  Controller c;
  auto s = status(c, 0);
  assert(s[9] == uint8_t(AckResult::NONE) && s[8] == 0 && read16(s.data() + 14) == kUnknownAge);
  c.update(0, true);
  for (uint16_t i = 0; i < 50; ++i) receive(c, command(i + 1, true, 200), i * 50);
  assert(!c.armed());

  c = Controller{}; arm(c);
  receive(c, command(42, true, 200), 2050);
  assert(c.armed() && c.output(0) == 200 && c.output(1) == 200);
  receive(c, command(43), 2100);
  assert(c.armed() && c.output(0) == 0);
  receive(c, command(44, true, 200), 2150);
  assert(c.output(0) == 200);
  receive(c, command(45, false), 2200);
  assert(!c.armed() && c.output(0) == 0);
  receive(c, command(46, true, 200), 2250);
  assert(!c.armed());

  c = Controller{}; arm(c);
  receive(c, command(40), 2050);
  c.update(2052, true);
  s = status(c, 2052);
  assert(s[9] == uint8_t(AckResult::REJECTED) && s[10] == uint8_t(RejectReason::STALE));
  assert(read16(s.data() + 6) == 40 && read16(s.data() + 18) == 41);
  auto bad = command(42); bad[23] ^= 1;
  receive(c, bad, 2060); c.update(2062, true); s = status(c, 2062);
  assert(s[9] == uint8_t(AckResult::REJECTED) && read32(s.data() + 2) == 0 && s[8] == 0);
  c.update(2200, true); assert(!c.armed());

  c = Controller{}; arm(c);
  for (uint16_t i = 1; i <= 50; ++i) receive(c, command(i, true, 0, 2), 2000 + i * 50);
  assert(!c.armed() && c.needsCycle());
  c.update(4501, false); c.update(4502, true);
  for (uint16_t i = 51; i <= 91; ++i) receive(c, command(i, true, 0, 2), 4503 + (i - 51) * 50);
  assert(c.armed() && !c.needsCycle());

  c = Controller{}; c.update(0, true);
  for (uint16_t i = 0; i < 50; ++i) receive(c, command(i, true, 0, 1, 1), i * 50);
  assert(!c.armed());
  c = Controller{}; arm(c); receive(c, command(42, true, 200, 1, 1), 2050);
  c.update(2060, true); assert(!c.armed());
  c = Controller{}; arm(c);
  receive(c, command(42, true, 200), 2300);
  assert(!c.armed()); // Late valid traffic cannot renew an expired armed lease.

  c = Controller{}; arm(c, UINT32_MAX - 1000);
  c.update(1000, false); assert(!c.armed());
  c = Controller{}; c.update(0, true);
  receive(c, command(65535), 0); receive(c, command(0), 50); receive(c, command(0), 100);
  assert(status(c, 100)[10] == uint8_t(RejectReason::STALE));
  receive(c, command(65535), 110);
  assert(status(c, 110)[10] == uint8_t(RejectReason::STALE));
  receive(c, command(1), 120);
  assert(status(c, 120)[9] == uint8_t(AckResult::RECEIVED));

  c = Controller{}; arm(c); c.receiveFault(); assert(!c.armed());
  c = Controller{}; arm(c); c.update(2001, true, false); assert(!c.armed());
  for (int i = 0; i < 4; ++i) assert(c.output(i) == 0);

  for (int offset : {8, 9, 10, 11, 20, 21, 22}) {
    c = Controller{}; c.update(0, true);
    bad = command(1); bad[offset] = offset == 11 ? 0 : 255; bad[23] = crc8(bad.data());
    receive(c, bad, 0); assert(status(c, 0)[9] == uint8_t(AckResult::REJECTED));
  }
  // Shared golden E1: session 1, seq 42, held Up 0.2, TTL 200 ms.
  const auto golden = command(42, true, 200);
  assert(golden[23] == 0x7E);
  std::cout << "PASS: arming handoff, release, expiry, rejection identity, switch cycle, sequence/clock wrap, driver faults\n";
}
