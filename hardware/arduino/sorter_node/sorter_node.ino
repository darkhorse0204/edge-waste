// sorter_node.ino - arduino firmware for the 2-day prototype: reads moisture and gas sensors, tilts the tray, lights leds, beeps
//
// Talks to the laptop over USB serial at 115200 baud, one command per line:
//   P   ping                       -> PONG
//   S   read both sensors          -> {"m":<0-1023>,"g":<0-1023>}   (each is the average of 16 readings)
//   R   recycle: green led, tilt tray left  (item slides into the left bin)
//   H   hazard:  red led, tilt tray right, long beep (item slides into the hazardous bin)
//   V   review:  yellow led, tray stays level, 3 short beeps (a person must look at the item)
//   C   contaminated reject: yellow led, tray stays level, 2 long beeps
//   A<deg>  move the tray to an angle for tuning, e.g. A60
// The push button on D2 sends the line BTN when pressed (capture trigger).
//
// Wiring (Arduino Uno / Nano):
//   MQ-135 gas module   AO -> A0, VCC -> 5V, GND -> GND
//   capacitive moisture AOUT -> A1, VCC -> 5V, GND -> GND
//   SG90 servo          signal -> D9, VCC -> 5V (add 470 uF across 5V and GND near the servo), GND -> GND
//   LEDs via 220 ohm    green D4, red D5, yellow D6 (long leg to the pin, short leg to GND)
//   buzzer              D7 -> buzzer + , buzzer - -> GND
//   push button         D2 -> button -> GND (internal pull-up is used)

#include <Servo.h>

const int PIN_GAS = A0;
const int PIN_MOIST = A1;
const int PIN_SERVO = 9;
const int PIN_BTN = 2;
const int LED_G = 4;
const int LED_R = 5;
const int LED_Y = 6;
const int PIN_BUZZ = 7;

// tune these three angles on the bench so the tray tilts about 25-30 degrees each way
int ANGLE_CENTER = 90;
int ANGLE_LEFT = 60;
int ANGLE_RIGHT = 120;

Servo tray;
String line = "";
bool btnWas = false;
unsigned long btnAt = 0;

void leds(bool g, bool r, bool y) {
  digitalWrite(LED_G, g);
  digitalWrite(LED_R, r);
  digitalWrite(LED_Y, y);
}

void beep(int ms) {
  digitalWrite(PIN_BUZZ, HIGH);
  delay(ms);
  digitalWrite(PIN_BUZZ, LOW);
}

int readAvg(int pin) {
  long sum = 0;
  for (int i = 0; i < 16; i++) {
    sum += analogRead(pin);
    delay(3);
  }
  return (int)(sum / 16);
}

void tilt(int angle) {
  tray.write(angle);
  delay(700);        // let the item slide off
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
    int g = readAvg(PIN_GAS);
    int m = readAvg(PIN_MOIST);
    Serial.print("{\"m\":");
    Serial.print(m);
    Serial.print(",\"g\":");
    Serial.print(g);
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
  pinMode(PIN_BTN, INPUT_PULLUP);
  tray.attach(PIN_SERVO);
  tray.write(ANGLE_CENTER);
  leds(true, true, true);      // all leds on for a moment: power-up check
  delay(500);
  leds(false, false, false);
  Serial.println("READY");
}

void loop() {
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
  if (pressed && !btnWas && millis() - btnAt > 250) {
    Serial.println("BTN");
    btnAt = millis();
  }
  btnWas = pressed;
}
