#include <Arduino.h>
#include <WiFiS3.h>
#include "WebPage.h"  // Make sure this file contains your HTML content as a string

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
    // Rest of your existing code for collecting data...

    WiFiClient client = server.available();
    if (client && client.connected()) {
        String request = client.readStringUntil('\r');
        client.flush();

        // Check if the request is for the root ("/") or for data ("/retrieve")
        if (request.indexOf("GET / ") >= 0) {
            String htmlContent = webpage; // Copy the HTML content to a new string
            htmlContent.replace("%%IP_ADDRESS%%", WiFi.localIP().toString()); // Replace the placeholder

            client.println("HTTP/1.1 200 OK");
            client.println("Content-Type: text/html");
            client.println();
            client.print(htmlContent);
        } else if (request.indexOf("GET /retrieve") >= 0) {
            // Client requested data, serve the sensor data
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
