#include<math.h>

//pins
int pwmPin = 6;   // PWM output pin D6
int gain22Pin = 3;
int gain220Pin = 2;
int offsetVoltagePin = A0;
int amplifiedVoltagePin = A1;
int nonAmplifiedVoltagePin = A2;


int value = 0;    // Variable to store the received PWM number

float maxVoltage = 4.7; // Max voltage that can be output by PWM
float opAmpSaturationVoltage = 4.65; // Op-amp saturation voltage

float offsetVoltage = 0; // Adjust A0 to maintain op-amp output at saturation voltage
int offsetMultiplier = 0;
const float hysteresisMargin = 0.1; // Margin around the op-amp saturation voltage for hysteresis
bool newSerialDataReceived = false; // Flag to indicate new serial data has been received


float amplifiedVoltage = 0;
float nonAmplifiedVoltage = 0;

int previousOffset = 0;

float gain = 22.1; //22.1

void setup()
{
  pinMode(pwmPin, OUTPUT); // Set D6 as an output pin
  pinMode(gain22Pin, OUTPUT);     // Gain 22k
  pinMode(gain220Pin, OUTPUT);     // Gain 220k

  analogWriteResolution(12);
  analogReadResolution(14);

  adjustGain();
  analogWrite(offsetVoltagePin,0);

  Serial.begin(115200);    // Start serial communication at 115200 baud

}

void loop() {
  manageIncomingSerial();

  updateReadings();
  outputResults();

  delay(100);
}
