// ESP32 sensor node for the Explainable Edge AI Copilot
// Sensors: MQ-2 (smoke/LPG, analog), DHT22 (temperature), MAX4466/KY-037 (sound level)
// Sends JSON once per second over HTTPS with a per-device API key:
//   POST https://<edge-host>/api/sensors   header  X-Api-Key: <device key>
//   {"zone":"Lab Block A","smoke_ppm":412.0,"temp_c":31.2,"sound_db":71.5}
// The server time-stamps readings on arrival and validates ranges; a node that stops
// reporting is marked offline after 10 s and its evidence is ignored (never trusted stale).
//
// Create a key on the edge box:  python -m edgecopilot.security keygen
// Put the printed key in DEVICE_KEY below and the digest in server.auth.sensor_api_keys.
// Wiring: MQ-2 AO -> GPIO34 (ADC1; ADC2 pins do not work while WiFi is on) via a 2:3 divider
//         (the MQ-2 module outputs up to 5 V, the ESP32 ADC tolerates 3.3 V), sound AO -> GPIO35, DHT22 -> GPIO4.
// Warm the MQ-2 for 24-48 h before calibrating R0 in clean air (calibrate()).
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <math.h>

const char* WIFI_SSID  = "CAMPUS-IOT";
const char* WIFI_PASS  = "change-me";
const char* EDGE_URL   = "https://copilot.campus.local/api/sensors";
const char* DEVICE_KEY = "paste-device-key-here";
const char* ZONE       = "Lab Block A";
// Campus CA certificate (PEM) used by the reverse proxy in front of the copilot
const char* CA_CERT = R"(-----BEGIN CERTIFICATE-----
...paste campus CA certificate...
-----END CERTIFICATE-----
)";

const int MQ2_PIN = 34;
const int SOUND_PIN = 35;
const int DHT_PIN = 4;
const float DIVIDER = 1.5f;     // 5 V module output -> 3.3 V ADC via 2:3 divider
DHT dht(DHT_PIN, DHT22);

float R0 = 9.83;                // kOhm in clean air, from calibrate()
const float RL_KOHM = 5.0;

float mq2_rs() {
  long acc = 0;
  for (int i = 0; i < 16; i++) { acc += analogRead(MQ2_PIN); delay(2); }   // average out ADC noise
  float v = (acc / 16.0f) * 3.3f / 4095.0f * DIVIDER;
  if (v < 0.05f) return NAN;                     // sensor unplugged or still cold
  return (5.0f - v) / v * RL_KOHM;
}

float mq2_ppm() {
  float rs = mq2_rs();
  if (isnan(rs) || rs <= 0) return NAN;
  // smoke curve from the MQ-2 datasheet (log-log fit): log10(ppm) = (log10(Rs/R0) - b) / m
  const float m = -0.44f, b = 1.62f;
  return powf(10.0f, (log10f(rs / R0) - b) / m);
}

void calibrate() {               // run once in clean air, then hard-code R0
  float sum = 0; int n = 0;
  for (int i = 0; i < 50; i++) { float rs = mq2_rs(); if (!isnan(rs)) { sum += rs; n++; } delay(100); }
  if (n) Serial.printf("R0 = %.2f kOhm (clean-air ratio 9.83)\n", (sum / n) / 9.83f);
}

float sound_db() {
  unsigned long start = millis(); int lo = 4095, hi = 0;
  while (millis() - start < 50) { int s = analogRead(SOUND_PIN); lo = min(lo, s); hi = max(hi, s); }
  float vpp = (hi - lo) * 3.3f / 4095.0f;
  if (vpp < 0.001f) vpp = 0.001f;                 // keep log10 finite in a silent room
  return 20.0f * log10f(vpp / 0.00631f);          // ~50 dB reference for MAX4466 at gain 25: calibrate with a meter
}

WiFiClientSecure tls;

void setup() {
  Serial.begin(115200);
  analogSetPinAttenuation(MQ2_PIN, ADC_11db);
  analogSetPinAttenuation(SOUND_PIN, ADC_11db);
  dht.begin();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  tls.setCACert(CA_CERT);
  Serial.println("\nconnected");
}

void appendNum(String& s, const char* key, float v, int digits) {
  if (isnan(v) || isinf(v)) return;               // omit unavailable readings instead of sending garbage
  if (s.length() > 1) s += ",";
  s += "\""; s += key; s += "\":"; s += String(v, digits);
}

void loop() {
  String body = "{\"zone\":\"" + String(ZONE) + "\"";
  appendNum(body, "smoke_ppm", mq2_ppm(), 1);
  appendNum(body, "temp_c", dht.readTemperature(), 2);
  appendNum(body, "sound_db", sound_db(), 1);
  body += "}";
  if (WiFi.status() != WL_CONNECTED) { WiFi.reconnect(); delay(1000); return; }
  HTTPClient http;
  http.setTimeout(3000);
  if (http.begin(tls, EDGE_URL)) {
    http.addHeader("Content-Type", "application/json");
    http.addHeader("X-Api-Key", DEVICE_KEY);
    int code = http.POST(body);
    Serial.printf("%s -> %d\n", body.c_str(), code);
    http.end();
  }
  delay(1000);
}
