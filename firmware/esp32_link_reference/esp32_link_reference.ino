// Reference/bench sketch only. NOT flight control code.
#include <Wire.h>
struct __attribute__((packed)) Packet {
  uint8_t magic, version;
  uint16_t seq;
  uint8_t cmd, phase, label, flags;
  int16_t yaw_cd, elev_cm;
  uint16_t travel_cm, age_ms, lost_ms;
  int8_t search_dir;
  uint8_t crc;
};
static_assert(sizeof(Packet) == 20);
portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
uint8_t pending[20];
bool available = false;
size_t received = 0;
bool failsafe = true;
uint32_t lastValid = 0;
void receivePacket(int count) {
  uint8_t local[20]; size_t n = 0;
  while (Wire.available()) {
    int b = Wire.read();
    if (n < 20) local[n] = b;
    ++n;
  }
  portENTER_CRITICAL(&mux);
  received = n;
  if (n == 20) memcpy(pending, local, 20);
  available = true;
  portEXIT_CRITICAL(&mux);
}
uint8_t crc8(const uint8_t *data) {
  uint8_t crc = 0;
  for (int i=0; i<19; ++i) {
    crc ^= data[i];
    for (int j=0; j<8; ++j) crc = (crc << 1) ^ ((crc & 128) ? 7 : 0);
  }
  return crc;
}
void setup() {
  Serial.begin(115200);
  Wire.onReceive(receivePacket);
  // Seeed XIAO ESP32: use the board's own I2C pads (D4 = SDA, D5 = SCL).
  // On the XIAO ESP32-S3 these are GPIO5/GPIO6, as wired to the Pi.
  delay(1500);  // Let the USB serial monitor attach before printing.
  Serial.printf("I2C slave 0x42 on SDA=GPIO%d SCL=GPIO%d\n", SDA, SCL);
  if (!Wire.begin((uint8_t)0x42, SDA, SCL, 100000)) {
    Serial.println("I2C slave init FAILED (check board and pins)");
  }
}
void loop() {
  Packet p; bool got; size_t n;
  portENTER_CRITICAL(&mux);
  got = available; n = received;
  if (got && n == 20) memcpy(&p, pending, 20);
  available = false;
  portEXIT_CRITICAL(&mux);
  if (got && n == 20 && p.magic == 0xD7 && p.version == 2 &&
      p.crc == crc8((uint8_t*)&p) && p.cmd < 6 && p.phase < 3 &&
      (p.label < 8 || p.label == 255)) {
    lastValid = millis(); failsafe = false;
    Serial.printf("seq=%u cmd=%u phase=%u label=%u flags=%u yaw_cd=%d elev_cm=%d travel_cm=%u age=%u lost=%u dir=%d\n",
      p.seq,p.cmd,p.phase,p.label,p.flags,p.yaw_cd,p.elev_cm,p.travel_cm,p.age_ms,p.lost_ms,p.search_dir);
  }
  bool expired = millis() - lastValid >= 300;
  if (expired && !failsafe) Serial.println("FAILSAFE");
  failsafe |= expired;
  delay(1);
}
