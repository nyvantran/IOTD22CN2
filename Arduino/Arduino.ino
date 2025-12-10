#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "esp_camera.h"
#include <WebServer.h>

// ======================= CẤU HÌNH WIFI + SERVER =======================

const char* ssid = "PTIT.HCM_SV";
const char* password = "";

// Hai server URL (chỉnh IP theo hệ thống của bạn)
const char* serverUrl1 = "http://10.251.5.141:8000/api/command/";
const char* serverUrl2 = "http://10.251.5.144:8000/api/command/";  // IP dự phòng, sửa nếu cần
const char* currentServerUrl = serverUrl1;

// ======================= CẤU HÌNH SPEED / MOTOR =======================

const int kickstartspeed = 200;
const int maxspeed = 255;
const int kickstarttime = 120;

// Tốc độ quay tại chỗ (rẽ nhẹ hơn)
const int turnBaseSpeed = 100;     // càng nhỏ quay càng nhẹ (thử 80–120)
const float leftTurnGain = 1.0f;   // bù motor trái (0.9–1.1)
const float rightTurnGain = 1.0f;  // bù motor phải

// chế độ hiện tại
const char* currenttask = "stop";

// Motor pins
#define ENA 46  // PWM Động cơ A (GPIO46)
#define IN1 38  // Động cơ A (GPIO38)
#define IN2 39  // Động cơ A (GPIO39)
#define ENB 40  // PWM Động cơ B (GPIO40)
#define IN3 41  // Động cơ B (GPIO41)
#define IN4 42  // Động cơ B (GPIO42)

// Camera pins cho ESP32-S3-CAM
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 15
#define SIOD_GPIO_NUM 4
#define SIOC_GPIO_NUM 5
#define Y9_GPIO_NUM 16
#define Y8_GPIO_NUM 17
#define Y7_GPIO_NUM 18
#define Y6_GPIO_NUM 12
#define Y5_GPIO_NUM 10
#define Y4_GPIO_NUM 8
#define Y3_GPIO_NUM 9
#define Y2_GPIO_NUM 11
#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM 7
#define PCLK_GPIO_NUM 13

WebServer server(80);

// Task handles
TaskHandle_t cameraTaskHandle;
TaskHandle_t motorTaskHandle;

// Shared variables
volatile int currentSpeed = 110;
volatile bool motorTaskRunning = false;

// PWM Configuration
const int pwmFreq = 5000;
const int pwmResolution = 8;

// ======================= CAMERA SETUP =======================

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
  config.frame_size = FRAMESIZE_VGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 15;
  config.fb_count = 1;

  if (psramFound()) {
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
  Serial.println("Camera initialized");
}

// ======================= MOTOR TASK (CORE 0) =======================

void kickStart(int speed);
void moveForward(int speed);
void moveBackward(int speed);
void turnLeft(int speed);
void turnRight(int speed);
void stopCar();

void motorTask(void* parameter) {
  motorTaskRunning = true;
  HTTPClient http;

  while (true) {
    if (WiFi.status() == WL_CONNECTED) {
      const char* primary = currentServerUrl;
      const char* secondary = (currentServerUrl == serverUrl1) ? serverUrl2 : serverUrl1;

      int httpCode = -1;
      String payload;

      // --- Thử server hiện tại ---
      http.begin(primary);
      http.setTimeout(1500);
      httpCode = http.GET();

      if (httpCode == HTTP_CODE_OK) {
        payload = http.getString();
      } else if (secondary && strlen(secondary) > 0) {
        // Thử server dự phòng
        http.end();
        http.begin(secondary);
        http.setTimeout(1500);
        int httpCode2 = http.GET();
        if (httpCode2 == HTTP_CODE_OK) {
          payload = http.getString();
          httpCode = httpCode2;
          currentServerUrl = secondary;  // switch sang server dự phòng
          Serial.println("Switched to backup server");
        }
      }

      if (httpCode == HTTP_CODE_OK) {
        StaticJsonDocument<200> doc;
        DeserializationError error = deserializeJson(doc, payload);

        if (!error) {
          const char* command = doc["command"];
          int speed = doc["speed"] | currentSpeed;
          currentSpeed = speed;

          // Kickstart khi chuyển từ stop sang lệnh chạy
          if (strcmp(command, currenttask) != 0 && strcmp(command, "forward") == 0 ) {
            kickStart(kickstartspeed);
          }

          if (strcmp(command, "forward") == 0) {
            currenttask = "forward";
            moveForward(speed);
          } else if (strcmp(command, "backward") == 0) {
            currenttask = "backward";
            moveBackward(speed);
          } else if (strcmp(command, "left") == 0) {
            currenttask = "left";
            turnLeft(speed);
          } else if (strcmp(command, "right") == 0) {
            currenttask = "right";
            turnRight(speed);
          } else if (strcmp(command, "stop") == 0) {
            currenttask = "stop";
            stopCar();
          }
        }
      }

      http.end();
    }

    vTaskDelay(100 / portTICK_PERIOD_MS);  // 100ms
  }
}

// ======================= CAMERA STREAM HANDLER =======================

void handleStream() {
  WiFiClient client = server.client();
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  client.print(response);

  const int64_t frameInterval = 150000;  // ~100ms ~ 10 FPS
  int64_t lastFrameTime = 0;

  while (client.connected()) {
    int64_t currentTime = esp_timer_get_time();

    if (currentTime - lastFrameTime > frameInterval) {
      camera_fb_t* fb = esp_camera_fb_get();
      if (!fb) {
        Serial.println("Camera capture failed");
        break;
      }

      client.print("--frame\r\n");
      client.print("Content-Type: image/jpeg\r\n");
      client.printf("Content-Length: %u\r\n\r\n", fb->len);
      client.write(fb->buf, fb->len);
      client.print("\r\n");

      esp_camera_fb_return(fb);
      lastFrameTime = currentTime;
    }

    vTaskDelay(1);
  }
}

// ======================= SETUP =======================

void setup() {
  Serial.begin(115200);
  Serial.println("ESP32-S3 Car Control Starting...");

  // Motor pins
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // PWM
  ledcAttach(ENA, pwmFreq, pwmResolution);
  ledcAttach(ENB, pwmFreq, pwmResolution);

  // WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Camera
  setupCamera();

  // Web server
  server.on("/stream", handleStream);
  server.on("/test", []() {
    server.send(200, "text/plain", "ESP32-S3 Car Control Active");
  });

  server.begin();
  Serial.println("HTTP server started");

  // Motor task on Core 0
  xTaskCreatePinnedToCore(
    motorTask,
    "MotorTask",
    8192,
    NULL,
    1,
    &motorTaskHandle,
    0);

  stopCar();

  Serial.println("Setup complete!");
  Serial.println("Camera stream: http://" + WiFi.localIP().toString() + "/stream");
}

// ======================= LOOP =======================

void loop() {
  server.handleClient();
  vTaskDelay(1);
}

// ======================= MOTOR HELPERS =======================

void kickStart(int speed) {
  // Cả 2 bánh tiến để "giật" xe chạy
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, speed);
  ledcWrite(ENB, speed);
  delay(kickstarttime);
}

// Quay phải tại chỗ (1 bánh tiến, 1 bánh lùi) nhưng nhẹ hơn
void turnRight(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);  // trái tiến
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);  // phải lùi

  ledcWrite(ENA, speed + 85);
  ledcWrite(ENB, speed - 10);
}

// Quay trái tại chỗ (1 bánh lùi, 1 bánh tiến) nhưng nhẹ hơn
void turnLeft(int speed) {

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);  // trái lùi
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);  // phải tiến

  ledcWrite(ENA, speed - 10);
  ledcWrite(ENB, speed + 85);
}

void moveBackward(int speed) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  ledcWrite(ENA, speed);
  ledcWrite(ENB, speed);
}

void moveForward(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, speed);
  ledcWrite(ENB, speed);
}

void stopCar() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  ledcWrite(ENA, 0);
  ledcWrite(ENB, 0);
}
