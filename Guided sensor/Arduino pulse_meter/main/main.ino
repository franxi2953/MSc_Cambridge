#include <Arduino.h>
#include <WiFiS3.h>

const char* ssid = "linksys";
const char* password = "12345670";

WiFiServer server(80);

struct DataPoint {
    float value;        // Value from 0 to 1023
    float time;         // Time in hundredths of a second
};

DataPoint pulseData[500];
int currentIndex = 0;
bool dataReadyToSend = false;

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.print("Connecting to WiFi...");
    while (WiFi.status() != WL_CONNECTED) {
        Serial.print(".");
        WiFi.begin(ssid, password);
    }
    Serial.println("Connected to WiFi");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    server.begin();
}

void loop() {
    if (!dataReadyToSend) {
        pulseData[currentIndex].value = analogRead(A4);
        pulseData[currentIndex].time = millis() / 1000.0; // Convert to seconds
        currentIndex++;

        if (currentIndex >= 500) {
            currentIndex = 0;
            dataReadyToSend = true;
            Serial.println("Data ready to send");
        }
    } else {
        WiFiClient client = server.available();
        if (client && client.connected() && client.available()) {
            Serial.println("Client connected!");
            String request = client.readStringUntil('\r');
            client.flush();

            if (request.indexOf("/retrieve") != -1) {
                String dataString = "";
                for (int i = 0; i < 500; i++) {
                    dataString += String(pulseData[i].time, 2) + "," + String(pulseData[i].value) + "\n";
                }

                client.println("HTTP/1.1 200 OK");
                client.println("Content-Type: text/plain");
                client.println("Access-Control-Allow-Origin: *");
                client.println("Content-Length: " + String(dataString.length()));
                client.println();
                client.print(dataString);

                dataReadyToSend = false;
                Serial.println("Data sent!");
            }
            client.stop();
        }
    }

    delay(10);
}
