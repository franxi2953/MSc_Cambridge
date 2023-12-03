void adjustGain(){
  if (gain < 3) {
    digitalWrite(gain220Pin, LOW);
    digitalWrite(gain22Pin, HIGH);
  } else {
    digitalWrite(gain220Pin, HIGH);
    digitalWrite(gain22Pin, LOW);
  }
}

void updateReadings() {
  // Measure the non-amplified and amplified voltage
  nonAmplifiedVoltage = (maxVoltage / 16383) * analogRead(nonAmplifiedVoltagePin);
  amplifiedVoltage = (maxVoltage / 16383) * analogRead(amplifiedVoltagePin);

  if (newSerialDataReceived && amplifiedVoltage > opAmpSaturationVoltage) {
    // Adjust offset based on the new readings
    while (amplifiedVoltage > opAmpSaturationVoltage)
    {
      offsetMultiplier++;
      adjustOffset();
      delay(100);
      amplifiedVoltage = (maxVoltage / 16383) * analogRead(amplifiedVoltagePin);
    }
    newSerialDataReceived = false; // Reset the flag after adjusting
  } else if (newSerialDataReceived && amplifiedVoltage <= 0.03) {
    while (amplifiedVoltage <=0.03 && offsetMultiplier > 0)
    {
      offsetMultiplier--;
      adjustOffset();
      delay(100);
      amplifiedVoltage = (maxVoltage / 16383) * analogRead(amplifiedVoltagePin);
    }
    newSerialDataReceived = false; // Reset the flag after adjusting
  }
}

void adjustOffset() {
  // Apply the calculated offset
  analogWrite(offsetVoltagePin, offsetMultiplier * (4095 / gain));

  // Store the offset for comparison in the next cycle
  previousOffset = offsetMultiplier;
}

void manageIncomingSerial() {
  if (Serial.available() > 0) {
    int receivedNumber = Serial.parseInt(); // Read the number from serial

    // Correct the value according to boundaries
    if (receivedNumber > 4096) receivedNumber = 4096;
    if (receivedNumber < 0) receivedNumber = 0;

    value = receivedNumber;          // Store the valid number
    analogWrite(pwmPin, value);      // Output PWM on pin D6

    newSerialDataReceived = true;    // Set flag to true as new data is received

    // Flush the serial buffer to discard any unread data
    while (Serial.available() > 0) {
      Serial.read();
    }
  }
}

void outputResults () {
  // Output the analog value and A0 voltage in CSV format
  Serial.print("Amplified:");
  Serial.print(amplifiedVoltage);
  Serial.print(",");
  Serial.print("Output:");
  Serial.print(nonAmplifiedVoltage);
  Serial.print(",");
  Serial.print("AppliedOffset:");
  Serial.println(previousOffset);
}

