#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "esp_camera.h"
#include <WebServer.h>

// ======================= CẤU HÌNH WIFI + SERVER =======================

const char* ssid = "PTIT.HCM_SV";
const char* password = "";

const char* serverUrl1 = "http://10.251.5.141:8000/api/command/";
const char* serverUrl2 = "http://10.251.5.144:8000/api/command/";

const char* currentServerUrl = serverUrl1;

// ======================= CẤU HÌNH US-100 =======================

#define TRIG_PIN 14  // GPIO14 cho Trigger
#define ECHO_PIN 21  // GPIO21 cho Echo

// Ngưỡng khoảng cách (cm)
#define DISTANCE_STOP 20  // Dừng khi < 20cm

volatile float currentDistance = 100.0;  // Khoảng cách hiện tại (cm)
volatile bool obstacleDetected = false;  // Cờ phát hiện vật cản

// ======================= CẤU HÌNH SPEED / MOTOR =======================

const int kickstartspeed = 200;
const int maxspeed = 255;
const int kickstarttime = 120;

const char* currenttask = "stop";
const char* lastServerCommand = "stop";  // Lưu lệnh cuối từ server

// Motor pins
#define ENA 46
#define IN1 38
#define IN2 39
#define ENB 40
#define IN3 41
#define IN4 42

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
TaskHandle_t ultrasonicTaskHandle;

// Shared variables
volatile int currentSpeed = 110;
volatile int lastServerSpeed = 110;  // Lưu speed cuối từ server
volatile bool motorTaskRunning = false;
volatile bool wasObstacleBlocking = false;  // Cờ theo dõi trạng thái vật cản trước đó

// PWM Configuration
const int pwmFreq = 5000;
const int pwmResolution = 8;

// ======================= US-100 FUNCTIONS =======================

float measureDistance() {
  // Gửi xung trigger
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Đọc thời gian phản hồi (timeout 30ms ~ 5m)
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);

  if (duration == 0) {
    return 999.0;  // Không có phản hồi = xa hoặc lỗi
  }

  // Tính khoảng cách (cm)
  float distance = duration * 0.034 / 2.0;

  return distance;
}

// Task đọc khoảng cách liên tục (Core 0)
void ultrasonicTask(void* parameter) {
  const int numReadings = 3;  // Số lần đọc để lọc nhiễu
  float readings[numReadings];
  int readIndex = 0;

  // Khởi tạo mảng
  for (int i = 0; i < numReadings; i++) {
    readings[i] = 100.0;
  }

  while (true) {
    // Đọc khoảng cách
    float distance = measureDistance();

    // Lọc nhiễu bằng trung bình động
    readings[readIndex] = distance;
    readIndex = (readIndex + 1) % numReadings;

    float sum = 0;
    for (int i = 0; i < numReadings; i++) {
      sum += readings[i];
    }
    currentDistance = sum / numReadings;

    // Cập nhật cờ phát hiện vật cản
    bool previousObstacle = obstacleDetected;
    obstacleDetected = (currentDistance < DISTANCE_STOP);

    // Debug output (chỉ in khi có thay đổi trạng thái)
    if (previousObstacle != obstacleDetected) {
      Serial.printf("🔔 Distance: %.1f cm | Obstacle: %s\n",
                    currentDistance,
                    obstacleDetected ? "⛔ YES - BLOCKING" : "✅ NO - CLEAR");
    }

    vTaskDelay(50 / portTICK_PERIOD_MS);  // Đọc mỗi 50ms
  }
}

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

// ======================= MOTOR FUNCTIONS =======================

void kickStart(int speed);
void moveForward(int speed);
void moveBackward(int speed);
void turnLeft(int speed);
void turnRight(int speed);
void stopCar();

// Hàm thực thi lệnh motor
void executeCommand(const char* command, int speed) {
  if (strcmp(command, currenttask) != 0 && strcmp(command, "forward") == 0) {
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

// ======================= MOTOR TASK (CORE 0) =======================

void motorTask(void* parameter) {
  motorTaskRunning = true;
  HTTPClient http;

  while (true) {
    // ========== KIỂM TRA VẬT CẢN TRƯỚC ==========
    if (obstacleDetected) {
      // Có vật cản -> DỪNG XE NGAY LẬP TỨC
      if (!wasObstacleBlocking) {
        Serial.println("⛔ OBSTACLE DETECTED! Emergency stop - Ignoring server commands");
        wasObstacleBlocking = true;
      }
      
      // Chỉ dừng nếu đang tiến về phía trước
      if (strcmp(currenttask, "forward") == 0 || strcmp(currenttask, "stop") != 0) {
        // Cho phép lùi và quay khi có vật cản phía trước
        if (strcmp(lastServerCommand, "backward") == 0) {
          executeCommand("backward", lastServerSpeed);
        } else if (strcmp(lastServerCommand, "left") == 0) {
          executeCommand("left", lastServerSpeed);
        } else if (strcmp(lastServerCommand, "right") == 0) {
          executeCommand("right", lastServerSpeed);
        } else {
          // Dừng nếu lệnh là forward hoặc stop
          currenttask = "stop";
          stopCar();
        }
      }
      
      vTaskDelay(100 / portTICK_PERIOD_MS);
      continue;  // Bỏ qua việc đọc lệnh từ server khi có vật cản (trừ lùi/quay)
    }

    // ========== KHÔNG CÓ VẬT CẢN -> XỬ LÝ LỆNH SERVER ==========
    
    // Kiểm tra nếu vừa thoát khỏi trạng thái vật cản
    if (wasObstacleBlocking) {
      Serial.println("✅ Obstacle cleared! Resuming server commands");
      wasObstacleBlocking = false;
      
      // Tiếp tục thực hiện lệnh cuối từ server
      Serial.printf("▶️ Resuming last command: %s at speed %d\n", lastServerCommand, lastServerSpeed);
      executeCommand(lastServerCommand, lastServerSpeed);
    }

    if (WiFi.status() == WL_CONNECTED) {
      const char* primary = currentServerUrl;
      const char* secondary = (currentServerUrl == serverUrl1) ? serverUrl2 : serverUrl1;

      int httpCode = -1;
      String payload;

      http.begin(primary);
      http.setTimeout(1500);
      httpCode = http.GET();

      if (httpCode == HTTP_CODE_OK) {
        payload = http.getString();
      } else if (secondary && strlen(secondary) > 0) {
        http.end();
        http.begin(secondary);
        http.setTimeout(1500);
        int httpCode2 = http.GET();
        if (httpCode2 == HTTP_CODE_OK) {
          payload = http.getString();
          httpCode = httpCode2;
          currentServerUrl = secondary;
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

          // Lưu lệnh và speed từ server
          lastServerCommand = command;
          lastServerSpeed = speed;

          // Thực thi lệnh
          executeCommand(command, speed);
        }
      }

      http.end();
    }

    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}

// ======================= CAMERA STREAM HANDLER =======================

void handleStream() {
  WiFiClient client = server.client();
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  client.print(response);

  const int64_t frameInterval = 150000;
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
  Serial.println("ESP32-S3 Car Control + US-100 Starting...");

  // US-100 pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

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

  // Web server endpoints
  server.on("/stream", handleStream);
  server.on("/test", []() {
    server.send(200, "text/plain", "ESP32-S3 Car Control + US-100 Active");
  });

  server.begin();
  Serial.println("HTTP server started");

  // US-100 Task on Core 0 (priority 2 - cao hơn motor task)
  xTaskCreatePinnedToCore(
    ultrasonicTask,
    "UltrasonicTask",
    4096,
    NULL,
    2,
    &ultrasonicTaskHandle,
    0);

  // Motor task on Core 0 (priority 1)
  xTaskCreatePinnedToCore(
    motorTask,
    "MotorTask",
    8192,
    NULL,
    1,
    &motorTaskHandle,
    0);

  stopCar();

  Serial.println("===========================================");
  Serial.println("Setup complete!");
  Serial.println("Camera stream: http://" + WiFi.localIP().toString() + "/stream");
  Serial.println("Obstacle detection: Active (< 20cm = STOP)");
  Serial.println("===========================================");
}

// ======================= LOOP =======================

void loop() {
  server.handleClient();
  vTaskDelay(1);
}

// ======================= MOTOR HELPERS =======================

void kickStart(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, speed);
  ledcWrite(ENB, speed);
  delay(kickstarttime);
}

void turnRight(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  ledcWrite(ENA, 185);
  ledcWrite(ENB, speed - 10);
}

void turnLeft(int speed) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, speed - 10);
  ledcWrite(ENB, 185);
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