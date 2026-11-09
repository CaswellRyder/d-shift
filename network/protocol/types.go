// Package protocol defines the isolated Go version-4 recording link.
package protocol

import "encoding/json"

const (
	Version                 = 4
	HeaderSize              = 40
	TagSize                 = 32
	MaxDatagram             = 1200
	MaxPayload              = MaxDatagram - HeaderSize - TagSize
	MaxSequence      uint64 = 1<<63 - 1
	FreshMS          uint64 = 300
	ManualTimeoutMS  uint64 = 300
	SessionTimeoutMS uint64 = 2000
)

type Kind byte

const (
	Hello Kind = iota + 1
	Welcome
	Poll
	TelemetryReply
	Manual
	ManualAck
	Mission
	MissionAck
	Safety
	SafetyAck
)

type SessionID [16]byte

const (
	StatusQueued   = "QUEUED_BENCH"
	StatusBusy     = "BUSY"
	StatusRejected = "REJECTED"
)

type Packet struct {
	Kind     Kind
	Sequence uint64
	Session  SessionID
	TickMS   uint64
	Payload  json.RawMessage
}

type HelloPayload struct {
	Nonce string `json:"nonce"`
}
type WelcomePayload struct {
	Nonce     string `json:"nonce"`
	BenchOnly bool   `json:"bench_only"`
	Applied   bool   `json:"applied"`
}
type SafetyPayload struct {
	Action string `json:"action"`
	Epoch  uint64 `json:"epoch"`
}
type ManualPayload struct {
	Epoch   uint64  `json:"epoch,omitempty"`
	Forward float64 `json:"forward"`
	Right   float64 `json:"right"`
	Up      float64 `json:"up"`
	Yaw     float64 `json:"yaw"`
	Hold    bool    `json:"hold"`
}
type MissionPayload struct {
	Epoch uint64 `json:"epoch,omitempty"`
	Op    uint8  `json:"op"`
	Args  string `json:"args"`
}
type Receipt struct {
	BenchPaused bool   `json:"bench_paused"`
	Epoch       uint64 `json:"epoch"`
	BenchEstop  bool   `json:"bench_estop"`
	Status      string `json:"status"`
	Code        string `json:"code,omitempty"`
	Digest      string `json:"digest"`
	BenchOnly   bool   `json:"bench_only"`
	Applied     bool   `json:"applied"`
}
type ManualState struct {
	Received    bool        `json:"received"`
	Sequence    uint64      `json:"seq"`
	Hold        bool        `json:"hold"`
	Axes        *[4]float64 `json:"axes"`
	AgeMS       *uint64     `json:"age_ms"`
	ExpiresInMS *uint64     `json:"expires_in_ms"`
	Reason      string      `json:"reason,omitempty"`
}
type MissionState struct {
	Sequence uint64 `json:"seq"`
	Status   string `json:"status"`
}
type Target struct {
	Label   string      `json:"label"`
	Score   *float64    `json:"score"`
	Box     *[4]float64 `json:"box"`
	BodyM   *[3]float64 `json:"body_m"`
	YawDeg  *float64    `json:"yaw_deg"`
	ElevM   *float64    `json:"elev_m"`
	TravelM *float64    `json:"travel_m"`
	AgeMS   *float64    `json:"age_ms"`
}
type Vision struct {
	Source          string   `json:"source"`
	Ready           bool     `json:"ready"`
	AgeMS           *float64 `json:"age_ms"`
	FrameSequence   *uint64  `json:"frame"`
	Phase           *string  `json:"phase"`
	GoalColor       *string  `json:"goal"`
	Intent          *string  `json:"intent"`
	Target          *Target  `json:"target"`
	Avoid           *Target  `json:"avoid"`
	FPS             *float64 `json:"fps"`
	LensCalibrated  *bool    `json:"lens_calibrated"`
	MountCalibrated *bool    `json:"mount_calibrated"`
	RangeQualified  bool     `json:"range_qualified"`
	Error           string   `json:"error,omitempty"`
}
type Telemetry struct {
	BenchPaused      bool          `json:"bench_paused"`
	BenchEstop       bool          `json:"bench_estop"`
	ControlEpoch     uint64        `json:"control_epoch"`
	BenchOnly        bool          `json:"bench_only"`
	Applied          bool          `json:"applied"`
	Vision           Vision        `json:"vision"`
	Manual           ManualState   `json:"manual"`
	Mission          *MissionState `json:"mission"`
	JournalOK        bool          `json:"journal_ok"`
	CPUTempC         *float64      `json:"cpu_temp_c"`
	ControllerStatus string        `json:"controller_status"`
	BatteryV         *float64      `json:"battery_v"`
	IMU              *[3]float64   `json:"imu"`
}
