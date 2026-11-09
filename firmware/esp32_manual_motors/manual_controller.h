#ifndef DTR_MANUAL_CONTROLLER_H
#define DTR_MANUAL_CONTROLLER_H

#include <algorithm>
#include <string.h>
#include "command_contract.h"

namespace dtr_manual {
using namespace dtr_command_contract;
inline uint16_t read16(const uint8_t* p) { return uint16_t(p[0]) | uint16_t(p[1]) << 8; }
inline uint32_t read32(const uint8_t* p) { return uint32_t(read16(p)) | uint32_t(read16(p + 2)) << 16; }
inline void write16(uint8_t* p, uint16_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); }
inline void write32(uint8_t* p, uint32_t v) { write16(p, uint16_t(v)); write16(p + 2, uint16_t(v >> 16)); }

class Controller {
 public:
  static constexpr uint32_t kTimeoutMS = 200;
  static constexpr uint32_t kArmMS = 2000;

  void update(uint32_t now, bool physical, bool healthy = true) {
    physical_ = physical;
    healthy_ = healthy;
    if (needs_cycle_) {
      if (!physical) disabled_seen_ = true;
      if (physical && disabled_seen_) {
        needs_cycle_ = false;
        neutral_tracking_ = false;
      }
    }
    expire(now);
    failsafe_ = !healthy_ || !physical_ || needs_cycle_ || !has_session_ || !hold_ || expired(now);
    if (failsafe_) disarm();
    else if (!armed_ && neutral_tracking_ && now - neutral_since_ >= kArmMS) armed_ = true;
    if (armed_ && accepted_ack_) result_ = AckResult::APPLIED;
  }

  void receive(const uint8_t* p, uint32_t now) {
    expire(now);
    accepted_ack_ = false;
    ack_session_ = 0;
    ack_seq_ = 0;
    ack_magic_ = 0;
    if (p[0] != kOperatorMagic || p[1] != kVersion || p[23] != crc8(p)) {
      reject(RejectReason::BAD_FRAME);
      return;
    }
    const uint32_t next_session = read32(p + 2);
    const uint16_t seq = read16(p + 6);
    ack_session_ = next_session;
    ack_seq_ = seq;
    ack_magic_ = kOperatorMagic;
    if (!next_session || p[9] || (p[10] != 0 && p[10] != 3) || !p[11] || p[11] > 30 || p[20] || p[21] || p[22]) {
      reject(RejectReason::BAD_FRAME);
      return;
    }
    if (p[8] != uint8_t(OperatorOp::HEARTBEAT)) {
      reject(RejectReason::UNKNOWN_OP);
      return;
    }
    const bool hold = p[10] == 3;
    int16_t next_axes[4];
    bool neutral = true;
    for (int i = 0; i < 4; ++i) {
      next_axes[i] = int16_t(read16(p + 12 + i * 2));
      if (next_axes[i] < -1000 || next_axes[i] > 1000 || (!hold && next_axes[i])) {
        reject(RejectReason::BAD_FRAME);
        return;
      }
      if (next_axes[i]) neutral = false;
    }
    if (has_session_ && next_session != session_) {
      disarm();
      needs_cycle_ = true;
      disabled_seen_ = !physical_;
      has_seq_ = false;
    }
    if (has_session_ && next_session == session_ && has_seq_) {
      const uint16_t delta = uint16_t(seq - last_seq_);
      if (!delta || delta >= 0x8000) {
        reject(RejectReason::STALE);
        return;
      }
    }
    session_ = next_session;
    has_session_ = has_seq_ = true;
    last_seq_ = seq;
    last_valid_ = now;
    lease_ms_ = std::min(uint32_t(p[11]) * 10, uint32_t(kTimeoutMS));
    hold_ = hold;
    memcpy(axes_, next_axes, sizeof(axes_));
    if (!hold || !neutral || !physical_ || !healthy_ || needs_cycle_) neutral_tracking_ = false;
    else if (!neutral_tracking_) {
      neutral_tracking_ = true;
      neutral_since_ = now;
    }
    if (!hold) disarm();
    accepted_ack_ = true;
    result_ = AckResult::RECEIVED;
    reason_ = RejectReason::NONE;
    update(now, physical_, healthy_);
  }

  void receiveFault() {
    hold_ = false;
    failsafe_ = true;
    disarm();
    ack_session_ = 0;
    ack_seq_ = ack_magic_ = 0;
    accepted_ack_ = false;
    reject(RejectReason::BAD_FRAME);
  }

  void status(uint8_t* p, uint32_t now) const {
    memset(p, 0, kPacketSize);
    p[0] = kStatusMagic;
    p[1] = kVersion;
    write32(p + 2, ack_session_);
    write16(p + 6, ack_seq_);
    p[8] = ack_magic_;
    p[9] = uint8_t(result_);
    p[10] = uint8_t(reason_);
    p[12] = armed_ ? uint8_t(Owner::PILOT) : uint8_t(Owner::NONE);
    p[13] = (armed_ ? ARMED : 0) | (failsafe_ ? FAILSAFE : 0) | (hold_ ? STATUS_DEADMAN : 0);
    write16(p + 14, has_session_ ? uint16_t(std::min(now - last_valid_, uint32_t(kMaximumKnownAge))) : kUnknownAge);
    write16(p + 16, kUnknownAge);
    write16(p + 18, last_seq_);
    p[23] = crc8(p);
  }

  int16_t output(int motor) const {
    if (!armed_) return 0;
    const int up = std::max(0, int(axes_[2]));
    const int value = motor == 0 ? up + axes_[1] : motor == 1 ? up - axes_[1] : motor == 2 ? axes_[0] : axes_[3];
    return int16_t(std::max(0, std::min(value, 1000)));
  }
  bool armed() const { return armed_; }
  bool needsCycle() const { return needs_cycle_; }

 private:
  bool expired(uint32_t now) const { return has_session_ && now - last_valid_ >= lease_ms_; }
  void disarm() { armed_ = false; neutral_tracking_ = false; }
  void expire(uint32_t now) {
    if (expired(now)) {
      hold_ = false;
      failsafe_ = true;
      disarm();
    }
  }
  void reject(RejectReason reason) { result_ = AckResult::REJECTED; reason_ = reason; }

  uint32_t session_ = 0, last_valid_ = 0, lease_ms_ = kTimeoutMS, neutral_since_ = 0;
  uint32_t ack_session_ = 0;
  uint16_t last_seq_ = 0, ack_seq_ = 0;
  uint8_t ack_magic_ = 0;
  int16_t axes_[4] = {};
  bool has_session_ = false, has_seq_ = false, hold_ = false, armed_ = false;
  bool physical_ = false, healthy_ = false, failsafe_ = true, neutral_tracking_ = false;
  bool needs_cycle_ = false, disabled_seen_ = false, accepted_ack_ = false;
  AckResult result_ = AckResult::NONE;
  RejectReason reason_ = RejectReason::NONE;
};
}  // namespace dtr_manual
#endif
