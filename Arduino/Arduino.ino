#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "esp_camera.h"
#include <WebServer.h>

// Cấu hình WiFi
const char* ssid = "PTIT.HCM_CanBo";
const char* password = "";

// Server Django
const char* serverUrl = "http://10.252.3.209:8000/api/command/";

// Định nghĩa chân điều khiển
#define ENA 19
#define IN1 20
#define IN2 21
#define ENB 41
#define IN3 42
#define IN4 45

// Camera pins cho ESP32-S3-CAM
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 15
#define SIOD_GPIO_NUM 4
#define SIOC_GPIO_NUM 5

#define Y2_GPIO_NUM 11
#define Y3_GPIO_NUM 9
#define Y4_GPIO_NUM 8
#define Y5_GPIO_NUM 10
#define Y6_GPIO_NUM 12
#define Y7_GPIO_NUM 18
#define Y8_GPIO_NUM 17
#define Y9_GPIO_NUM 16

#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM 7
#define PCLK_GPIO_NUM 13

// Cấu hình PWM
const int freq = 1000;
const int pwmResolution = 8;

// Biến lưu trạng thái
int currentSpeed = 150;

WebServer streamServer(81);

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 10;
  config.fb_count = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
}

void handleStream() {
  WiFiClient client = streamServer.client();
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  client.print(response);

  while (client.connected()) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      break;
    }

    client.print("--frame\r\n");
    client.print("Content-Type: image/jpeg\r\n\r\n");
    client.write(fb->buf, fb->len);
    client.print("\r\n");

    esp_camera_fb_return(fb);

    if (!client.connected()) break;
  }
}

void setup() {
  Serial.begin(115200);

  // Cấu hình chân điều khiển
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Cấu hình PWM
  ledcAttach(ENA, freq, pwmResolution);
  ledcAttach(ENB, freq, pwmResolution);

  // Setup Camera
  setupCamera();

  // Kết nối WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Start camera stream server
  streamServer.on("/stream", handleStream);
  streamServer.begin();

  // Dừng xe ban đầu
  stopCar();
}

void loop() {
  // Handle camera stream
  streamServer.handleClient();

  // Check commands
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);

    int httpCode = http.GET();

    if (httpCode > 0) {
      String payload = http.getString();

      StaticJsonDocument<200> doc;
      DeserializationError error = deserializeJson(doc, payload);

      if (!error) {
        const char* command = doc["command"];
        int speed = doc["speed"] | currentSpeed;
        currentSpeed = speed;

        executeCommand(command, speed);
      }
    }

    http.end();
  }

  delay(100);
}

// Các hàm điều khiển giữ nguyên như code trên
void executeCommand(const char* command, int speed) {
  if (strcmp(command, "forward") == 0) {
    moveForward(speed);
  } else if (strcmp(command, "backward") == 0) {
    moveBackward(speed);
  } else if (strcmp(command, "left") == 0) {
    turnLeft(speed);
  } else if (strcmp(command, "right") == 0) {
    turnRight(speed);
  } else if (strcmp(command, "stop") == 0) {
    stopCar();
  }
}

void moveForward(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, speed);
  ledcWrite(ENB, speed);
  // Serial.println("Moving forward");
}

void moveBackward(int speed) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  ledcWrite(ENA, speed);
  ledcWrite(ENB, speed);
  // Serial.println("Moving backward");
}

void turnLeft(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  ledcWrite(ENA, speed);
  ledcWrite(ENB, speed);
  // Serial.println("Turning left");
}

void turnRight(int speed) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, speed);
  ledcWrite(ENB, speed);
  // Serial.println("Turning right");
}

void stopCar() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  ledcWrite(ENA, 0);
  ledcWrite(ENB, 0);
  // Serial.println("Stopped");
}