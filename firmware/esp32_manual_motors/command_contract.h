#ifndef DTR_ESP32_COMMAND_CONTRACT_H
#define DTR_ESP32_COMMAND_CONTRACT_H

#include <stdint.h>
#include <stddef.h>

// PROPOSED wire contract only. No transport, Arduino dependency, or motor code.
// All multibyte wire fields are LITTLE ENDIAN. Packed structs describe offsets;
// big-endian hosts must explicitly read/write LE bytes instead of memcpy casts.
// Legacy D7 v2 remains 20 bytes and has no SIM/source tag (replay is possible).
namespace dtr_command_contract {

constexpr size_t kPacketSize = 24;
constexpr size_t kCrcOffset = 23;
constexpr uint8_t kVersion = 1;
constexpr uint8_t kOperatorMagic = 0xE1;
constexpr uint8_t kSimMagic = 0xE2;
constexpr uint8_t kStatusMagic = 0xE3;
constexpr uint8_t kSimSource = 0x53;  // ASCII S; simulation, never perception.
constexpr uint8_t kNoLabel = 0xFF;
constexpr int16_t kMissingSignedGeometry = INT16_MIN;
constexpr uint16_t kMissingTravel = 0xFFFF;
constexpr uint16_t kUnknownAge = 0xFFFF;
constexpr uint16_t kMaximumKnownAge = 0xFFFE;
constexpr uint8_t kNotLost = 0xFF;
constexpr uint8_t kMaximumLost100ms = 0xFE;
constexpr uint8_t kDefaultTtl10ms = 30;  // Proposal: 300 ms, not hardware validated.

enum class OperatorOp : uint8_t {
  HEARTBEAT = 0, MODE_SHADOW = 1, MODE_SCRIPT = 2, SIM_START = 3,
  SIM_CANCEL = 4, MODE_OFF = 5
};
enum class SimMode : uint8_t { OFF = 0, SHADOW = 1, SCRIPT = 2 };
enum class Command : uint8_t { SEARCH = 0, TRACK = 1, LOST = 2, FAN_ON = 3, ENTER = 4, AVOID = 5 };
enum class Phase : uint8_t { BALLOON = 0, CAPTURE = 1, GOAL = 2 };
enum class Label : uint8_t {
  GREEN_BALLOON = 0, PURPLE_BALLOON = 1, ORANGE_CIRCLE = 2, ORANGE_SQUARE = 3,
  ORANGE_TRIANGLE = 4, YELLOW_CIRCLE = 5, YELLOW_SQUARE = 6, YELLOW_TRIANGLE = 7, NONE = 255
};
enum class AckResult : uint8_t {
  NONE = 0, RECEIVED = 1, APPLIED = 2, SHADOW_LOGGED = 3, REJECTED = 4
};
enum class RejectReason : uint8_t {
  NONE = 0, BAD_FRAME = 1, UNKNOWN_OP = 2, DEADMAN = 3, PILOT_OVERRIDE = 4,
  STALE = 5, SESSION = 6, CONTROLLER_GATE = 7
};
enum class Owner : uint8_t { NONE = 0, PILOT = 1, SCRIPT = 2 };
enum OperatorFlag : uint8_t { DEADMAN = 1u << 0, MANUAL_AXES_VALID = 1u << 1 };
enum SimFlag : uint8_t {
  YAW_VALID = 1u << 0, ELEV_VALID = 1u << 1, TRAVEL_VALID = 1u << 2,
  FAN_REQUESTED = 1u << 3, FOLLOW_FRESH = 1u << 4, END_OF_SCRIPT = 1u << 5,
  SEARCH_RIGHT = 1u << 6, SEARCH_LEFT = 1u << 7
};
enum StatusFlag : uint8_t {
  ARMED = 1u << 0, FAILSAFE = 1u << 1, STATUS_DEADMAN = 1u << 2,
  PILOT_OVERRIDE = 1u << 3, SIM_ACTIVE = 1u << 4
};

#pragma pack(push, 1)
struct OperatorPacket {
  uint8_t magic;                  // 0: E1
  uint8_t version;                // 1: 1
  uint32_t session;               // 2: nonzero; not authentication
  uint16_t seq;                   // 6
  uint8_t op;                     // 8: OperatorOp
  uint8_t requested_sim_mode;     // 9: SimMode; request is not authorization
  uint8_t flags;                  // 10: OperatorFlag; other bits zero
  uint8_t ttl_10ms;               // 11: proposed lease duration
  int16_t forward_milli;          // 12: normalized intent [-1000,1000]
  int16_t right_milli;            // 14
  int16_t up_milli;               // 16
  int16_t yaw_milli;              // 18: +right; never physical PWM
  uint8_t arg;                    // 20: SIM_START scenario 1; otherwise zero
  uint8_t reserved[2];            // 21..22: zero
  uint8_t crc;                    // 23
};

struct SimPacket {
  uint8_t magic;                  // 0: E2
  uint8_t version;                // 1: 1
  uint32_t session;               // 2: must match active operator session
  uint16_t seq;                   // 6: E2 stream sequence
  uint8_t source;                 // 8: 53, ASCII S
  uint8_t cmd;                    // 9: Command
  uint8_t phase;                  // 10: Phase
  uint8_t label;                  // 11: Label or FF
  uint8_t flags;                  // 12: SimFlag
  uint8_t reserved;               // 13: zero
  int16_t yaw_cd;                 // 14: +right, INT16_MIN absent
  int16_t elev_cm;                // 16: +up, INT16_MIN absent
  uint16_t travel_cm;             // 18: FFFF absent
  uint16_t age_ms;                // 20: FFFF unknown; known saturates FFFE
  uint8_t lost_100ms;             // 22: FF not lost; floor ms/100, saturate FE
  uint8_t crc;                    // 23
};

struct StatusPacket {
  uint8_t magic;                  // 0: E3
  uint8_t version;                // 1: 1
  uint32_t session;               // 2
  uint16_t ack_seq;               // 6: binds result to this stream sequence
  uint8_t ack_magic;              // 8: E1/E2, zero when AckResult::NONE
  uint8_t result;                 // 9: AckResult
  uint8_t reject_reason;          // 10: RejectReason
  uint8_t actual_sim_mode;        // 11: SimMode, actual controller state
  uint8_t owner;                  // 12: Owner
  uint8_t flags;                  // 13: StatusFlag; other bits zero
  uint16_t operator_age_ms;       // 14: FFFF unknown; known saturates FFFE
  uint16_t sim_age_ms;            // 16: FFFF unknown; known saturates FFFE
  uint16_t last_operator_seq;     // 18
  uint16_t last_sim_seq;          // 20
  uint8_t reserved;               // 22: zero
  uint8_t crc;                    // 23
};
#pragma pack(pop)

static_assert(sizeof(OperatorPacket) == kPacketSize, "E1 must be 24 bytes");
static_assert(sizeof(SimPacket) == kPacketSize, "E2 must be 24 bytes");
static_assert(sizeof(StatusPacket) == kPacketSize, "E3 must be 24 bytes");
static_assert(offsetof(OperatorPacket, session) == 2, "E1 session offset");
static_assert(offsetof(OperatorPacket, seq) == 6, "E1 sequence offset");
static_assert(offsetof(OperatorPacket, flags) == 10, "E1 flags offset");
static_assert(offsetof(OperatorPacket, forward_milli) == 12, "E1 axes offset");
static_assert(offsetof(OperatorPacket, arg) == 20, "E1 arg offset");
static_assert(offsetof(OperatorPacket, crc) == kCrcOffset, "E1 CRC offset");
static_assert(offsetof(SimPacket, session) == 2, "E2 session offset");
static_assert(offsetof(SimPacket, seq) == 6, "E2 sequence offset");
static_assert(offsetof(SimPacket, source) == 8, "E2 source offset");
static_assert(offsetof(SimPacket, flags) == 12, "E2 flags offset");
static_assert(offsetof(SimPacket, yaw_cd) == 14, "E2 geometry offset");
static_assert(offsetof(SimPacket, age_ms) == 20, "E2 age offset");
static_assert(offsetof(SimPacket, lost_100ms) == 22, "E2 loss offset");
static_assert(offsetof(SimPacket, crc) == kCrcOffset, "E2 CRC offset");
static_assert(offsetof(StatusPacket, ack_seq) == 6, "E3 ack sequence offset");
static_assert(offsetof(StatusPacket, ack_magic) == 8, "E3 ack stream offset");
static_assert(offsetof(StatusPacket, flags) == 13, "E3 flags offset");
static_assert(offsetof(StatusPacket, operator_age_ms) == 14, "E3 ages offset");
static_assert(offsetof(StatusPacket, last_sim_seq) == 20, "E3 last SIM sequence offset");
static_assert(offsetof(StatusPacket, crc) == kCrcOffset, "E3 CRC offset");

inline uint8_t crc8(const uint8_t* data, size_t length = kCrcOffset) {
  uint8_t crc = 0;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = static_cast<uint8_t>((crc << 1) ^ ((crc & 0x80) ? 0x07 : 0));
    }
  }
  return crc;
}

// Receiver requirements: reject malformed/reserved/CRC/session/stale frames.
// A missing geometry field uses BOTH a cleared validity flag and its sentinel.
// Process END_OF_SCRIPT BEFORE cmd: cancel SCRIPT, return SHADOW; repeated END
// is idempotent. Never begin SEARCH from an END frame. LOST clears FOLLOW_FRESH.
// FAN_REQUESTED is intent only; it never confirms capture. No auto launch/shoot.
// SHADOW guidance is SHADOW_LOGGED, never APPLIED. APPLIED may acknowledge manual
// intent or SCRIPT, and ack_magic + ack_seq identify which frame was applied.
// Manual intent uses HEARTBEAT + MANUAL_AXES_VALID, independently of sim mode;
// downstream deadman, pilot override, leases, arming, and controller gates apply.

}  // namespace dtr_command_contract

#endif  // DTR_ESP32_COMMAND_CONTRACT_H
