// ESP32 sensor node for the Explainable Edge AI Copilot
// Sensors: MQ-2 (smoke/LPG, analog), DHT22 (temperature), KY-037/MAX4466 (sound level)
// Posts JSON every second to the edge box:  POST http://<edge-ip>:8000/api/sensors
//   {"zone":"Lab Block A","smoke_ppm":412.0,"temp_c":31.2,"sound_db":71.5}
// The edge box timestamps on arrival and marks a sensor offline after 10 s of silence,
// so a dead node lowers that cue's reliability instead of freezing a stale value.
#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

const char* WIFI_SSID = "CAMPUS-IOT";
const char* WIFI_PASS = "change-me";
const char* EDGE_URL  = "http://192.168.1.50:8000/api/sensors";
const char* ZONE      = "Lab Block A";

const int MQ2_PIN = 34;      // ADC1 pin (ADC2 is unusable while WiFi is on)
const int SOUND_PIN = 35;
const int DHT_PIN = 4;
DHT dht(DHT_PIN, DHT22);

// MQ-2 calibration: R0 measured in clean air (run calibrate() once and paste the value)
float R0 = 9.83;
const float RL_KOHM = 5.0;

float mq2_ppm() {
  int raw = analogRead(MQ2_PIN);
  float v = raw * 3.3 / 4095.0;
  if (v < 0.01) return 0;
  float rs = (3.3 - v) / v * RL_KOHM;
  float ratio = rs / R0;
  // smoke curve from the MQ-2 datasheet (log-log fit): ppm = 10^((log10(ratio) - b) / m)
  const float m = -0.44, b = 1.62;
  return pow(10, (log10(ratio) - b) / m);
}

float sound_db() {
  // peak-to-peak amplitude over 50 ms mapped to an approximate dB(A) scale (calibrate with a meter)
  unsigned long start = millis(); int lo = 4095, hi = 0;
  while (millis() - start < 50) { int s = analogRead(SOUND_PIN); lo = min(lo, s); hi = max(hi, s); }
  float vpp = (hi - lo) * 3.3 / 4095.0;
  return 20.0 * log10(max(vpp, 0.001f) / 0.00631) ;  // 0.00631 V ~ 50 dB reference for MAX4466 gain 25
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nconnected");
}

void loop() {
  float t = dht.readTemperature();
  char body[200];
  snprintf(body, sizeof(body), "{\"zone\":\"%s\",\"smoke_ppm\":%.1f,\"temp_c\":%s,\"sound_db\":%.1f}",
           ZONE, mq2_ppm(), isnan(t) ? "null" : String(t, 2).c_str(), sound_db());
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http; http.begin(EDGE_URL); http.addHeader("Content-Type", "application/json");
    int code = http.POST(body); http.end();
    Serial.printf("%s -> %d\n", body, code);
  }
  delay(1000);
}
