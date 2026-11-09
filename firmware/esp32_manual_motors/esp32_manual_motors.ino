// XIAO ESP32-S3, Arduino-ESP32 3.3.12. Manual bench control only.
#include <Arduino.h>
#include <Wire.h>
#include "manual_controller.h"

constexpr uint8_t I2C_ADDRESS = 0x42;
constexpr int SDA_PIN = 5, SCL_PIN = 6;
constexpr int ESC_PIN[4] = {1, 2, 4, 7};  // D0, D1, D3, D8
constexpr int ENABLE_PIN = 8;            // D9, normally open switch to GND
constexpr uint16_t PWM_MIN_US = 1000, PWM_MAX_TEST_US = 1250;
constexpr uint8_t PWM_BITS = 14;

dtr_manual::Controller controller;
bool outputsReady = false;
bool attached[4] = {};
portMUX_TYPE rxMux = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE statusMux = portMUX_INITIALIZER_UNLOCKED;
uint8_t incoming[24], latestStatus[24];
bool pending = false, receiveFault = false;

void onReceive(int count) {
  uint8_t frame[24];
  size_t n = 0;
  while (Wire.available()) {
    const int value = Wire.read();
    if (n < sizeof(frame)) frame[n] = uint8_t(value);
    ++n;
  }
  portENTER_CRITICAL(&rxMux);
  if (count == 24 && n == 24 && !pending) {
    memcpy(incoming, frame, 24);
    pending = true;
  } else receiveFault = true;
  portEXIT_CRITICAL(&rxMux);
}

void onRequest() {
  uint8_t frame[24];
  portENTER_CRITICAL(&statusMux);
  memcpy(frame, latestStatus, 24);
  portEXIT_CRITICAL(&statusMux);
  Wire.write(frame, 24);
}

void publish() {
  uint8_t frame[24];
  controller.status(frame, millis());
  portENTER_CRITICAL(&statusMux);
  memcpy(latestStatus, frame, 24);
  portEXIT_CRITICAL(&statusMux);
}

bool writeOutputs() {
  bool ok = true;
  for (int i = 0; i < 4; ++i) {
    if (!attached[i]) continue;
    const uint32_t pulse = PWM_MIN_US + uint32_t(controller.output(i)) * (PWM_MAX_TEST_US - PWM_MIN_US) / 1000;
    if (!ledcWrite(ESC_PIN[i], pulse * (1u << PWM_BITS) / 20000u)) ok = false;
  }
  return ok;
}

void setup() {
  Serial.begin(115200);
  pinMode(ENABLE_PIN, INPUT_PULLUP);
  outputsReady = true;
  for (int i = 0; i < 4; ++i) {
    attached[i] = ledcAttach(ESC_PIN[i], 50, PWM_BITS);
    if (!attached[i]) {
      outputsReady = false;
      Serial.printf("PWM attach failed on GPIO%d; arming inhibited\n", ESC_PIN[i]);
    }
  }
  if (!writeOutputs()) outputsReady = false;
  controller.update(millis(), false, outputsReady);
  publish();
  Wire.onReceive(onReceive);
  Wire.onRequest(onRequest);
  if (!Wire.begin(I2C_ADDRESS, SDA_PIN, SCL_PIN, 100000)) {
    outputsReady = false;
    Serial.println("I2C init failed; arming inhibited");
  }
  Serial.println("D-Shift manual E1/E3 bench controller; enable D9->GND, hold Space/deadman to arm");
}

void loop() {
  uint8_t frame[24];
  bool got = false, bad;
  portENTER_CRITICAL(&rxMux);
  if (pending) {
    memcpy(frame, incoming, 24);
    pending = false;
    got = true;
  }
  bad = receiveFault;
  receiveFault = false;
  portEXIT_CRITICAL(&rxMux);

  const bool physical = digitalRead(ENABLE_PIN) == LOW;
  controller.update(millis(), physical, outputsReady);
  if (bad) controller.receiveFault();
  else if (got) controller.receive(frame, millis());
  controller.update(millis(), physical, outputsReady);
  if (!writeOutputs()) {
    outputsReady = false;
    controller.update(millis(), physical, false);
    writeOutputs();
  }
  publish();
  static uint32_t lastReport = 0;
  if (millis() - lastReport >= 2000 && Serial && Serial.availableForWrite() >= 64) {
    lastReport = millis();
    Serial.printf("D-SHIFT ready=%d armed=%d enable=%d cycle=%d\n", outputsReady, controller.armed(), physical, controller.needsCycle());
  }
  delay(2);
}
