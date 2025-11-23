#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "esp_camera.h"
#include <WebServer.h>

// Cấu hình WiFi
const char* ssid = "Nha Tro Hanh Phuc_5G";
const char* password = "13681368";
const char* serverUrl = "http://192.168.1.11:8000/api/command/";

// Cấu hình speed
const int kickstartspeed = 150;
const int maxspeed = 255;
const int kickstarttime = 300;

// chế độ hiện tại
const char* currenttask = "stop";

// Motor pins
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
volatile int currentSpeed = 150;
volatile bool motorTaskRunning = false;

// PWM Configuration - đổi tên biến để tránh xung đột
const int pwmFreq = 5000;
const int pwmResolution = 8;

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

  // ESP32-S3 có PSRAM
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

// Motor control task - Core 0
void motorTask(void* parameter) {
  motorTaskRunning = true;
  HTTPClient http;

  while (true) {
    if (WiFi.status() == WL_CONNECTED) {
      http.begin(serverUrl);
      http.setTimeout(1500);

      int httpCode = http.GET();

      if (httpCode == HTTP_CODE_OK) {
        String payload = http.getString();

        StaticJsonDocument<200> doc;
        DeserializationError error = deserializeJson(doc, payload);

        if (!error) {
          const char* command = doc["command"];
          int speed = doc["speed"] | currentSpeed;
          currentSpeed = speed;

          // Execute command directly in this task
          if (strcmp(command, currenttask) != 0 and strcmp(command, "stop") != 0 and strcmp(currenttask, "stop") == 0) {
            kickStart(maxspeed);
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

    // Delay 100ms between checks
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}

// Camera streaming handler
void handleStream() {
  WiFiClient client = server.client();
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  client.print(response);

  const int64_t frameInterval = 100000;  // 100ms = 10 FPS
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

    // Small delay to prevent watchdog
    vTaskDelay(1);
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("ESP32-S3 Car Control Starting...");

  // Setup motors
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Configure PWM using new API
  ledcAttach(ENA, pwmFreq, pwmResolution);
  ledcAttach(ENB, pwmFreq, pwmResolution);

  // Connect WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Setup camera
  setupCamera();

  // Setup web server
  server.on("/stream", handleStream);

  // Test endpoint
  server.on("/test", []() {
    server.send(200, "text/plain", "ESP32-S3 Car Control Active");
  });

  server.begin();
  Serial.println("HTTP server started");

  // Create motor control task on Core 0
  xTaskCreatePinnedToCore(
    motorTask,         // Function
    "MotorTask",       // Name
    8192,              // Stack size
    NULL,              // Parameters
    1,                 // Priority
    &motorTaskHandle,  // Handle
    0                  // Core 0
  );

  // Initial stop
  stopCar();

  Serial.println("Setup complete!");
  Serial.println("Camera stream: http://" + WiFi.localIP().toString() + "/stream");
}

// Main loop runs on Core 1
void loop() {
  server.handleClient();
  vTaskDelay(1);
}

// Kickstart Motor functions
void kickStart(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, speed);
  ledcWrite(ENB, speed);
  delay(kickstarttime);
  // Serial.printf("Kickstart at speed %d\n", speed);
}

// Motor control functions
void turnRight(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, speed + 50);
  ledcWrite(ENB, speed - 25);
  // Serial.printf("Forward at speed %d\n", speed);
}

void turnLeft(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, speed - 25);
  ledcWrite(ENB, speed + 50);
  // Serial.printf("Backward at speed %d\n", speed);
}

void moveBackward(int speed) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  ledcWrite(ENA, speed);
  ledcWrite(ENB, speed);
  // Serial.printf("Left at speed %d\n", speed);
}

void moveForward(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, speed);
  ledcWrite(ENB, speed);
  // Serial.printf("Right at speed %d\n", speed);
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