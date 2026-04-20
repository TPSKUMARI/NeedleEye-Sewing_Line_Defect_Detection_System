#define LED_PIN 3
unsigned long offTime = 0;
bool ledActive = false;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);  // Off (Active Low)
}

void loop() {
  if (Serial.available() > 0) {
    // Read one byte at a time
    byte incomingByte = Serial.read();

    // Check if the byte is 0x01 (Hex for 1)
    if (incomingByte == 0x01) {
      digitalWrite(LED_PIN, HIGH);  // Turn LED ON
      offTime = millis() + 2000;
      ledActive = true;
    }
  }

  // Non-blocking 2s timer
  if (ledActive && (millis() >= offTime)) {
    digitalWrite(LED_PIN, LOW);
    ledActive = false;
  }
}