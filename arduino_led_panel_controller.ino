/*
=============================================================================
Project: Multi-Robot Communication System with Vision-Based Tracking
Module : Arduino LED Panel Controller
Author : Amunugama H.M.K.D. (E/21/029)
Version: 1.0 (Serial LED Panel GPIO Controller)
Date   : May 30, 2026

Purpose:
Receives simple serial LED commands from the PC navigation controller and
drives a physical 2x2 LED panel.

Hardware Architecture:
  PC (Python/OpenCV) -> Serial USB -> Arduino -> 2x2 LED Panel

Why Arduino Handles GPIO:
  The PC is responsible for OpenCV camera processing, ArUco tracking, occupancy
  mapping, A* planning, and NavigationCommand generation. Those tasks need much
  more processing power than an Arduino provides.

  The Arduino is responsible only for deterministic GPIO control. It receives
  small serial packets from the PC and updates the LED pins immediately. Python
  should not directly drive Arduino GPIO pins because the Arduino owns those
  physical pins and can update them reliably in real time.

Supported Serial Formats:
  1. Single command byte from Python:
       0x0A means 1010, 0x03 means 0011, etc.

  2. Human-readable debug packet:
       R<robot_id>:ABCD

Examples:
  byte 0x0A
  R1:1010
  R1:1100
  R1:1111

The single-byte format is compact for the PC control program. The text format
is kept because it is beginner friendly and easy to test in Serial Monitor.

Panel Layout:
  [ A ][ B ]
  [ C ][ D ]

LED Bit Meaning:
  1 = LED ON
  0 = LED OFF
=============================================================================
*/

// ================================================================
//  PIN CONFIG
// ================================================================
const int A_LED_PIN = 2;
const int B_LED_PIN = 3;
const int C_LED_PIN = 4;
const int D_LED_PIN = 5;

// Maximum packet length is small, but this buffer leaves room for line endings.
const int PACKET_BUFFER_SIZE = 24;

char packetBuffer[PACKET_BUFFER_SIZE];
int packetIndex = 0;

// ================================================================
//  PANEL HELPERS
// ================================================================
void clearPanel() {
  digitalWrite(A_LED_PIN, LOW);
  digitalWrite(B_LED_PIN, LOW);
  digitalWrite(C_LED_PIN, LOW);
  digitalWrite(D_LED_PIN, LOW);
}

void applyPattern(const int pattern[4]) {
  digitalWrite(A_LED_PIN, pattern[0] == 1 ? HIGH : LOW);
  digitalWrite(B_LED_PIN, pattern[1] == 1 ? HIGH : LOW);
  digitalWrite(C_LED_PIN, pattern[2] == 1 ? HIGH : LOW);
  digitalWrite(D_LED_PIN, pattern[3] == 1 ? HIGH : LOW);
}

bool parsePacket(const char *packet, int pattern[4]) {
  /*
    Expected format:
      R1:1010

    The parser accepts any numeric robot ID length:
      R2:0011
      R12:1111

    Only the four bits after ':' are used by this physical panel.
  */
  if (packet[0] != 'R') {
    return false;
  }

  int colonIndex = -1;
  for (int i = 1; packet[i] != '\0'; i++) {
    if (packet[i] == ':') {
      colonIndex = i;
      break;
    }
  }

  if (colonIndex < 0) {
    return false;
  }

  for (int i = 0; i < 4; i++) {
    char bit = packet[colonIndex + 1 + i];
    if (bit == '1') {
      pattern[i] = 1;
    } else if (bit == '0') {
      pattern[i] = 0;
    } else {
      return false;
    }
  }

  // Reject extra characters after the four LED bits.
  if (packet[colonIndex + 5] != '\0') {
    return false;
  }

  return true;
}

void patternFromByte(byte commandByte, int pattern[4]) {
  /*
    Byte format:
      bit 3 -> A
      bit 2 -> B
      bit 1 -> C
      bit 0 -> D

    Example:
      0x0A = binary 1010 -> A=1, B=0, C=1, D=0
  */
  pattern[0] = (commandByte & 0x08) ? 1 : 0;
  pattern[1] = (commandByte & 0x04) ? 1 : 0;
  pattern[2] = (commandByte & 0x02) ? 1 : 0;
  pattern[3] = (commandByte & 0x01) ? 1 : 0;
}

void handleCommandByte(byte commandByte) {
  int pattern[4] = {0, 0, 0, 0};
  patternFromByte(commandByte, pattern);
  applyPattern(pattern);

  Serial.print("OK byte 0x");
  if (commandByte < 16) {
    Serial.print("0");
  }
  Serial.print(commandByte, HEX);
  Serial.print(" -> A:");
  Serial.print(pattern[0]);
  Serial.print(" B:");
  Serial.print(pattern[1]);
  Serial.print(" C:");
  Serial.print(pattern[2]);
  Serial.print(" D:");
  Serial.println(pattern[3]);
}

void handlePacket(const char *packet) {
  int pattern[4] = {0, 0, 0, 0};

  if (parsePacket(packet, pattern)) {
    applyPattern(pattern);

    Serial.print("OK ");
    Serial.print(packet);
    Serial.print(" -> A:");
    Serial.print(pattern[0]);
    Serial.print(" B:");
    Serial.print(pattern[1]);
    Serial.print(" C:");
    Serial.print(pattern[2]);
    Serial.print(" D:");
    Serial.println(pattern[3]);
  } else {
    Serial.print("ERR invalid packet: ");
    Serial.println(packet);
  }
}

// ================================================================
//  ARDUINO SETUP
// ================================================================
void setup() {
  pinMode(A_LED_PIN, OUTPUT);
  pinMode(B_LED_PIN, OUTPUT);
  pinMode(C_LED_PIN, OUTPUT);
  pinMode(D_LED_PIN, OUTPUT);

  clearPanel();

  Serial.begin(115200);
  Serial.println("Arduino LED Panel Controller ready.");
  Serial.println("Waiting for command bytes like 0x0A or packets like R1:1010");
  Serial.println("Panel layout: [A][B] / [C][D]");
}

// ================================================================
//  MAIN LOOP
// ================================================================
void loop() {
  while (Serial.available() > 0) {
    char ch = Serial.read();

    // Python sends compact command bytes in the range 0x00 to 0x0F.
    // Text packets begin with 'R', so non-text low values can be applied
    // immediately as LED bit patterns.
    if (packetIndex == 0 && (byte)ch <= 0x0F) {
      handleCommandByte((byte)ch);
      continue;
    }

    if (ch == '\n' || ch == '\r') {
      if (packetIndex > 0) {
        packetBuffer[packetIndex] = '\0';
        handlePacket(packetBuffer);
        packetIndex = 0;
      }
    } else {
      if (packetIndex < PACKET_BUFFER_SIZE - 1) {
        packetBuffer[packetIndex] = ch;
        packetIndex++;
      } else {
        packetBuffer[PACKET_BUFFER_SIZE - 1] = '\0';
        Serial.print("ERR packet too long: ");
        Serial.println(packetBuffer);
        packetIndex = 0;
        clearPanel();
      }
    }
  }
}
