// Define the pin numbers
const int pwmPin = 6; // Assuming D6
const int gain22Pin = 3; // Specify the pin number
const int gain220Pin = 2; // Specify the pin number
const int offsetVoltagePin = A0; // Specify the pin number for offset voltage

// Constants for the temperature calculation
const float slope = 3.1223; // Update with your actual slope value
const float intercept = -35.24; // Update with your actual intercept value
const float offsetMultiplier = 22.08; // Constant to multiply with offset

int adjust_offset = 0;

void setup() {
  pinMode(pwmPin, OUTPUT); // Set D6 as an output pin
  pinMode(gain22Pin, OUTPUT); // Gain 22k
  pinMode(gain220Pin, OUTPUT); // Gain 220k

  analogWriteResolution(12);
  analogReadResolution(14);

  adjustGain(22); // Function definition needed for adjustGain
  analogWrite(offsetVoltagePin, 0);

  Serial.begin(115200); // Start serial communication at 115200 baud
}

void loop() {
  if (Serial.available() > 0) {
    int degrees = Serial.parseInt(); // Read number from serial
    startMeasuring(degrees);
  }
}

void startMeasuring(int degrees) {
  adjust_offset = 0;
  analogWrite(A0, (4.7 / offsetMultiplier) * adjust_offset * (4096 / 4.7));
  delay(50);

  while (true) {
    float outputValue = float(analogRead(A1)) * 4.7/16383; // Read the value from A1
    if (outputValue > 4.65) { // Assuming 4.65 is the saturation level
      adjust_offset++;
      analogWrite(A0, (4.7 / offsetMultiplier) * adjust_offset * (4096 / 4.7));
      delay(50);
    } else {
      break;
    }
  }

  float nonAmplifiedVoltage = float(analogRead(A2)) * 4.7/16383.0; // Read the value from A2
  float amplifiedVoltage = float(analogRead(A1)) * 4.7/16383.0; // Read the value from A1 again
  float offsetVoltage = (4.7 / offsetMultiplier) * adjust_offset;
  float calculatedTemp = calculateTemperature(amplifiedVoltage, offsetVoltage);

  // Print the values to the Serial
  Serial.print("received_temperature:");
  Serial.print(degrees);
  Serial.print(",output_temp:");
  Serial.println(calculatedTemp);

  flush_serial();
}

float calculateTemperature(float amplifiedVoltage, float offsetVoltage) {
  return slope * (amplifiedVoltage + (offsetVoltage*22.08)) + intercept;
}

void adjustGain(float gain){
  if (gain < 3) {
    digitalWrite(gain220Pin, LOW);
    digitalWrite(gain22Pin, HIGH);
  } else {
    digitalWrite(gain220Pin, HIGH);
    digitalWrite(gain22Pin, LOW);
  }
}

void flush_serial() {
  while (Serial.available() > 0) {
    Serial.read(); // Read and discard the incoming byte
  }
}
