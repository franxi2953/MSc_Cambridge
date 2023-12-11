#include <Arduino.h>
#include <WiFiS3.h>
#include "WebPage.h"

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
    // Record data if not ready to send
    if (!dataReadyToSend) {
        pulseData[currentIndex].value = analogRead(A4);
        pulseData[currentIndex].time = millis() / 1000.0; // Convert to seconds
        currentIndex++;

        if (currentIndex >= 500) {
            currentIndex = 0;
            dataReadyToSend = true;
            Serial.println("Data ready to send");
        }
    }

    // Handle client requests
    WiFiClient client = server.available();
    if (client && client.connected() && dataReadyToSend) {
        Serial.println("Client connected!");
        String request = client.readStringUntil('\r');
        client.flush();

        if (request.indexOf("GET / ") >= 0) {
            // Serve the webpage
            String htmlContent = webpage;
            htmlContent.replace("%%IP_ADDRESS%%", WiFi.localIP().toString());

            client.println("HTTP/1.1 200 OK");
            client.println("Content-Type: text/html");
            client.println();
            client.print(htmlContent);
        } else if (request.indexOf("GET /retrieve") >= 0 && dataReadyToSend) {
            // Serve the sensor data
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

            dataReadyToSend = false; // Reset the flag after sending data
            Serial.println("Data sent!");
        }
        client.stop();
    }

    delay(10); // Small delay for stability
}

