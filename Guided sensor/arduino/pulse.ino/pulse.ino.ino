#include <Arduino.h>
#include <WiFiS3.h>

const char* ssid = "linksys";
const char* password = "12345670";

WiFiServer server(80);

float pulseData[1000];
int currentIndex = 0;
int lastSentIndex = 0;

void setup() {
    Serial.begin(115200);
    while (WiFi.status() != WL_CONNECTED) {
        Serial.println("Connecting to WiFi...");
        WiFi.begin(ssid, password);
        delay(1000);
    }
    Serial.println("Connected to WiFi");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    server.begin();
}

void loop() {
    pulseData[currentIndex] = analogRead(A4);
    currentIndex = (currentIndex + 1) % 1000;

    WiFiClient client = server.available();
    if (client) {
        if (client.connected() && client.available()) {
            String request = client.readStringUntil('\r');
            client.flush();

            if (request.indexOf("/retrieve") != -1) {
                client.println("HTTP/1.1 200 OK");
                client.println("Content-Type: text/plain");
                client.println("Access-Control-Allow-Origin: *");
                client.println();

                int dataCount = (currentIndex >= lastSentIndex) ? (currentIndex - lastSentIndex) : (1000 - lastSentIndex + currentIndex);
                int dataIndex = lastSentIndex;
                for (int i = 0; i < dataCount; i++) {
                    client.println(pulseData[dataIndex]);
                    dataIndex = (dataIndex + 1) % 1000;
                }

                lastSentIndex = currentIndex;
            }
            client.stop();
        }
    }

    delay(10);
}
