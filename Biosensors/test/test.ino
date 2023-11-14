#include "WiFiS3.h"

char ssid[] = "linksys";
char pass[] = "12345670";
int keyIndex = 0;

int led =  LED_BUILTIN;
int status = WL_IDLE_STATUS;
WiFiServer server(80);

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  Serial.println("Access Point Web Server");

  pinMode(led, OUTPUT); 

  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    while (true);
  }

  String fv = WiFi.firmwareVersion();
  if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
    Serial.println("Please upgrade the firmware");
  }

  WiFi.config(IPAddress(192,48,56,2));
  
  Serial.print("Creating access point named: ");
  Serial.println(ssid);

  status = WiFi.beginAP(ssid, pass);
  if (status != WL_AP_LISTENING) {
    Serial.println("Creating access point failed");
    while (true);
  }

  delay(10000);
  server.begin();
  printWiFiStatus();
}

void loop() {
  if (status != WiFi.status()) {
    status = WiFi.status();

    if (status == WL_AP_CONNECTED) {
      Serial.println("Device connected to AP");
    } else {
      Serial.println("Device disconnected from AP");
    }
  }
  
  WiFiClient client = server.available();

  if (client) {
    Serial.println("new client"); 
    String request = client.readStringUntil('\r');
    client.flush();

    int textIndex = request.indexOf("/text?text=");
    if (textIndex != -1) {
      int startIdx = textIndex + 11;
      int endIdx = request.indexOf(" ", startIdx);
      String message = request.substring(startIdx, endIdx);

      // For now, just print the message to the Serial Monitor
      Serial.println("Received text: " + message);

      // Send response to the client
      client.println("HTTP/1.1 200 OK");
      client.println("Content-type:text/html");
      client.println();
      client.println("<html><body>");
      client.println("<p>Received text: " + message + "</p>");
      client.println("</body></html>");
      client.stop();
    }
  }
}

void printWiFiStatus() {
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  Serial.print("To see this page in action, open a browser to http://");
  Serial.println(ip);
}
