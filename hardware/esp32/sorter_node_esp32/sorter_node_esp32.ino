// sorter_node_esp32.ino - esp32 firmware for the prototype: dht22 (humidity), mq-135 (gas), tilt servo, optional leds and buzzer, boot button as trigger
//
// Same one-line serial protocol as the Arduino version (115200 baud):
//   P      ping                      -> PONG
//   S      read sensors              -> {"m":<humidity x10, 0-1000>,"g":<gas 0-1023>,"t":<temp C>,"h":<humidity %>}
//   R      auto-sorted: tray tilts left
//   H      hazardous: long beep, tray tilts right
//   V      manual review: tray stays level, 3 short beeps
//   C      contamination reject: tray stays level, 2 long beeps
//   A<deg> move the tray to an angle for tuning, e.g. A60
// The BOOT button (GPIO0) sends the line BTN when pressed after start-up.
// "m" is the DHT22 relative humidity times 10 (it plays the role of the moisture channel); m = 0 means the DHT22 gave no reading,
// which makes the host fall back to the gas-only model by itself.
//
// WIRING (ESP32 DevKit, 30/38 pin):
//   DHT22   VCC -> 3V3,  GND -> GND,  DATA -> GPIO4   (bare 4-pin DHT22: add 10k between DATA and 3V3; 3-pin module already has it)
//   MQ-135  VCC -> VIN (5V pin, board powered over USB),  GND -> GND,
//           AO -> 10k resistor -> GPIO34, and 10k resistor from GPIO34 to GND   (voltage divider: AO is up to 5V, the ESP32 pin only takes 3.3V)
//   Servo   red -> VIN (5V),  brown/black -> GND,  orange/yellow signal -> GPIO18
//   Optional green LED GPIO25, red LED GPIO26, yellow LED GPIO27 (each through 220 ohm to GND), active buzzer GPIO14 (+) to GND (-)
//
// ARDUINO IDE: board "ESP32 Dev Module"; libraries "ESP32Servo", "DHT sensor library" (Adafruit) and "Adafruit Unified Sensor".

#include <ESP32Servo.h>
#include <DHT.h>

const int PIN_DHT = 4;
const int PIN_GAS = 34;
const int PIN_SERVO = 18;
const int PIN_BTN = 0;
const int LED_G = 25;
const int LED_R = 26;
const int LED_Y = 27;
const int PIN_BUZZ = 14;
const int LED_ONBOARD = 2;

const float GAS_DIVIDER = 2.0;     // two equal resistors (10k + 10k) halve the voltage; use 1.5 for 10k on top and 20k at the bottom

// tune these on the bench so the tray tilts about 25-30 degrees each way
int ANGLE_CENTER = 90;
int ANGLE_LEFT = 60;
int ANGLE_RIGHT = 120;

DHT dht(PIN_DHT, DHT22);
Servo tray;
String line = "";
float rh = NAN, tc = NAN;
unsigned long lastDht = 0;
bool btnWas = false;
unsigned long btnAt = 0;

void leds(bool g, bool r, bool y) {
  digitalWrite(LED_G, g);
  digitalWrite(LED_R, r);
  digitalWrite(LED_Y, y);
}

void beep(int ms) {
  digitalWrite(PIN_BUZZ, HIGH);
  digitalWrite(LED_ONBOARD, HIGH);
  delay(ms);
  digitalWrite(PIN_BUZZ, LOW);
  digitalWrite(LED_ONBOARD, LOW);
}

// the DHT22 can only be read about every 2 seconds, so keep a smoothed value up to date in the background
void updateDht(bool force) {
  if (!force && millis() - lastDht < 2500) return;
  lastDht = millis();
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  if (!isnan(h)) rh = isnan(rh) ? h : 0.5f * rh + 0.5f * h;
  if (!isnan(t)) tc = t;
}

// MQ-135 output on a 0-1023 scale of the 5 V supply (the same scale the host software expects)
int readGas() {
  long sum = 0;
  for (int i = 0; i < 32; i++) {
    sum += analogReadMilliVolts(PIN_GAS);
    delay(2);
  }
  float pin_mv = sum / 32.0f;
  float out_mv = pin_mv * GAS_DIVIDER;
  int g = (int)(out_mv / 5000.0f * 1023.0f);
  return constrain(g, 0, 1023);
}

void tilt(int angle) {
  tray.write(angle);
  delay(700);
}

void level() {
  tray.write(ANGLE_CENTER);
  delay(400);
}

void act(char c) {
  switch (c) {
    case 'R':
      leds(true, false, false);
      tilt(ANGLE_LEFT);
      delay(500);
      level();
      break;
    case 'H':
      leds(false, true, false);
      beep(600);
      tilt(ANGLE_RIGHT);
      delay(500);
      level();
      break;
    case 'V':
      leds(false, false, true);
      for (int i = 0; i < 3; i++) { beep(120); delay(120); }
      break;
    case 'C':
      leds(false, false, true);
      for (int i = 0; i < 2; i++) { beep(400); delay(200); }
      break;
    default:
      return;
  }
  Serial.println("OK");
}

void handle(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  char c = cmd.charAt(0);
  if (c == 'P') {
    Serial.println("PONG");
  } else if (c == 'S') {
    if (isnan(rh)) updateDht(true);
    int g = readGas();
    int m = isnan(rh) ? 0 : (int)(rh * 10.0f);
    Serial.print("{\"m\":");
    Serial.print(m);
    Serial.print(",\"g\":");
    Serial.print(g);
    Serial.print(",\"t\":");
    Serial.print(isnan(tc) ? 0.0f : tc, 1);
    Serial.print(",\"h\":");
    Serial.print(isnan(rh) ? 0.0f : rh, 1);
    Serial.println("}");
  } else if (c == 'A') {
    int a = constrain(cmd.substring(1).toInt(), 20, 160);
    tray.write(a);
    Serial.println("OK");
  } else {
    act(c);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_G, OUTPUT);
  pinMode(LED_R, OUTPUT);
  pinMode(LED_Y, OUTPUT);
  pinMode(PIN_BUZZ, OUTPUT);
  pinMode(LED_ONBOARD, OUTPUT);
  pinMode(PIN_BTN, INPUT_PULLUP);
  analogSetPinAttenuation(PIN_GAS, ADC_11db);
  dht.begin();
  tray.setPeriodHertz(50);
  tray.attach(PIN_SERVO, 500, 2400);
  tray.write(ANGLE_CENTER);
  leds(true, true, true);
  digitalWrite(LED_ONBOARD, HIGH);
  delay(500);
  leds(false, false, false);
  digitalWrite(LED_ONBOARD, LOW);
  Serial.println("READY");
}

void loop() {
  updateDht(false);
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n') {
      handle(line);
      line = "";
    } else if (ch != '\r') {
      line += ch;
    }
  }
  bool pressed = digitalRead(PIN_BTN) == LOW;
  if (pressed && !btnWas && millis() - btnAt > 300 && millis() > 3000) {
    Serial.println("BTN");
    btnAt = millis();
  }
  btnWas = pressed;
}
