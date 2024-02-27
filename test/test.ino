#include <Wire.h>
#include "TLC59108.h"

// Replace these with the correct SDA and SCL pins for your ESP32
#define SDA_PIN 21
#define SCL_PIN 22

// Initialize TLC59108; assuming the default I2C address
TLC59108 ledDriver(TLC59108::I2C_ADDR::BASE);

void setup() {
  Serial.begin(9600); // Start serial for debugging
  Wire.begin(SDA_PIN, SCL_PIN); // Initialize I2C communication

  // Initialize the LED driver
  if (ledDriver.init() != 0) {
    Serial.println("Failed to initialize TLC59108");
    while (1); // Halt if initialization failed
  } else {
    Serial.println("TLC59108 Initialized successfully");
  }

  // Assuming all LEDs are to be set for individual PWM control
  // This might need to be adjusted based on your specific setup and the library's requirements
  for (int i = 0; i < 8; i++) {
    // Set each LED channel to PWM mode (might need adjustment based on your library)
    ledDriver.setLedOutputMode(TLC59108::LED_MODE::PWM_IND);
  }
}

void loop() {
  // Sweep LEDs on and off
  for (int i = 0; i < 8; i++) {
    ledDriver.setBrightness(i, 255); // Turn LED on (max brightness)
    delay(500); // Wait for half a second
    ledDriver.setBrightness(i, 0); // Turn LED off
  }

  // Optional: a delay here if you want a pause after sweeping through all LEDs
  delay(1000); // Wait for a second before starting the sweep again
}
