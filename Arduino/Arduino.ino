#include <WiFi.h>
#include <WiFiUdp.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "esp_camera.h"

// ======================= CẤU HÌNH WIFI + SERVER =======================

const char* ssid = "Nokia 1280";
const char* password = "12345678";

const char* serverUrl1 = "http://172.20.10.6:8000/api/command/";
const char* serverUrl2 = "http://10.251.1.1:8000/api/command/";
const char* currentServerUrl = serverUrl1;

// ======================= UDP CONFIG =======================

WiFiUDP udp;
const int UDP_PORT = 8888;
const int CHUNK_SIZE = 1400;

IPAddress clientIP;
uint16_t clientPort = 0;
volatile bool hasClient = false;

// ======================= CẤU HÌNH US-100 =======================

#define TRIG_PIN 14
#define ECHO_PIN 21
#define DISTANCE_STOP 20

volatile float currentDistance = 100.0;
volatile bool obstacleDetected = false;

// ======================= CẤU HÌNH LINE SENSOR =======================

#define LINE_LEFT_PIN   47
#define LINE_RIGHT_PIN  48

// ⚠️ QUAN TRỌNG: Điều chỉnh theo loại cảm biến của bạn
// TRUE  = Sensor output LOW khi cán vạch đen (phổ biến nhất)
// FALSE = Sensor output HIGH khi cán vạch đen
#define LINE_ACTIVE_LOW false 

// Trạng thái line sensor
volatile bool leftLineDetected = false;
volatile bool rightLineDetected = false;
volatile bool lineOverrideActive = false;

#define LINE_TURN_DURATION 150
volatile unsigned long lineOverrideEndTime = 0;

enum LineCommand {
  LINE_CMD_NONE = 0,
  LINE_CMD_LEFT,
  LINE_CMD_RIGHT,
  LINE_CMD_BACKWARD
};
volatile LineCommand lineOverrideCmd = LINE_CMD_NONE;

// Debug counter
volatile unsigned long lineDebugCounter = 0;

// ======================= CẤU HÌNH MOTOR =======================

const int kickstartspeed = 200;
const int kickstarttime = 130;

const char* currenttask = "stop";
const char* lastServerCommand = "stop";

#define ENA 46
#define IN1 38
#define IN2 39
#define ENB 40
#define IN3 41  
#define IN4 42

// ======================= CAMERA PINS =======================

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

// ======================= TASK HANDLES =======================

TaskHandle_t motorTaskHandle = NULL;
TaskHandle_t ultrasonicTaskHandle = NULL;
TaskHandle_t cameraStreamTaskHandle = NULL;
TaskHandle_t lineSensorTaskHandle = NULL;

// ======================= SHARED VARIABLES =======================

volatile int currentSpeed = 110;
volatile int lastServerSpeed = 110;
volatile bool wasObstacleBlocking = false;

const int pwmFreq = 5000;
const int pwmResolution = 8;

portMUX_TYPE lineMux = portMUX_INITIALIZER_UNLOCKED;

// ======================= FORWARD DECLARATIONS =======================

void kickStart(int speed);
void moveForward(int speed);
void moveBackward(int speed);
void turnLeft(int speed);
void turnRight(int speed);
void stopCar();
void executeCommand(const char* command, int speed);
void executeDirectCommand(const char* command, int speed);

// ======================= US-100 =======================

float measureDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return 999.0;
  return duration * 0.034 / 2.0;
}

void ultrasonicTask(void* parameter) {
  Serial.println("🔊 Ultrasonic Task Started");
  
  float readings[3] = {100.0, 100.0, 100.0};
  int idx = 0;

  while (true) {
    readings[idx] = measureDistance();
    idx = (idx + 1) % 3;

    currentDistance = (readings[0] + readings[1] + readings[2]) / 3.0;
    
    bool prev = obstacleDetected;
    obstacleDetected = (currentDistance < DISTANCE_STOP);

    if (prev != obstacleDetected) {
      Serial.printf("🔔 %.1fcm | %s\n", currentDistance, obstacleDetected ? "⛔BLOCK" : "✅CLEAR");
    }

    vTaskDelay(pdMS_TO_TICKS(50));
  }
}

// ======================= LINE SENSOR TASK =======================

void lineSensorTask(void* parameter) {
  Serial.println("📏 Line Sensor Task Started");
  Serial.printf("   Config: ACTIVE_LOW=%s\n", LINE_ACTIVE_LOW ? "true" : "false");
  
  unsigned long lastDebugTime = 0;
  
  while (true) {
    // Đọc RAW values
    int leftRaw = digitalRead(LINE_LEFT_PIN);
    int rightRaw = digitalRead(LINE_RIGHT_PIN);
    
    // Xử lý logic theo cấu hình
    bool left, right;
    if (LINE_ACTIVE_LOW) {
      // LOW = cán vạch đen
      left = (leftRaw == LOW);
      right = (rightRaw == LOW);
    } else {
      // HIGH = cán vạch đen
      left = (leftRaw == HIGH);
      right = (rightRaw == HIGH);
    }
    
    unsigned long now = millis();
    
    // Debug mỗi 500ms
    if (now - lastDebugTime >= 500) {
      lastDebugTime = now;
      Serial.printf("📏 RAW: L=%d R=%d | Detected: L=%d R=%d | Override=%d\n",
                    leftRaw, rightRaw, left, right, lineOverrideActive);
    }
    
    portENTER_CRITICAL(&lineMux);
    leftLineDetected = left;
    rightLineDetected = right;
    
    if (left || right) {
      // Có cán vạch
      lineOverrideActive = true;
      lineOverrideEndTime = now + LINE_TURN_DURATION;
      
      if (left && right) {
        lineOverrideCmd = LINE_CMD_BACKWARD;
        lineOverrideEndTime = now + LINE_TURN_DURATION * 2;
      } else if (left) {
        lineOverrideCmd = LINE_CMD_RIGHT;
      } else {
        lineOverrideCmd = LINE_CMD_LEFT;
      }
    } else {
      // Không cán vạch - kiểm tra timeout
      if (lineOverrideActive) {
        if (now >= lineOverrideEndTime) {
          lineOverrideActive = false;
          lineOverrideCmd = LINE_CMD_NONE;
          Serial.println("✅ Line override ENDED - back to server control");
        }
      }
    }
    portEXIT_CRITICAL(&lineMux);
    
    vTaskDelay(pdMS_TO_TICKS(10));
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
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 30;
  config.fb_count = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed: 0x%x\n", err);
    return;
  }
  
  Serial.println("✅ Camera: VGA 640x480");
}

// ======================= UDP SEND FRAME =======================

void sendFrameUDP(camera_fb_t* fb, uint16_t frameId) {
  if (!hasClient || fb == NULL) return;
  
  uint16_t totalPackets = (fb->len + CHUNK_SIZE - 1) / CHUNK_SIZE;
  uint8_t header[8];
  
  size_t offset = 0;
  uint16_t packetIndex = 0;
  
  while (offset < fb->len) {
    size_t chunkLen = min((size_t)CHUNK_SIZE, fb->len - offset);
    
    header[0] = frameId & 0xFF;
    header[1] = (frameId >> 8) & 0xFF;
    header[2] = packetIndex & 0xFF;
    header[3] = (packetIndex >> 8) & 0xFF;
    header[4] = totalPackets & 0xFF;
    header[5] = (totalPackets >> 8) & 0xFF;
    header[6] = 0;
    header[7] = 0;
    
    udp.beginPacket(clientIP, clientPort);
    udp.write(header, 8);
    udp.write(fb->buf + offset, chunkLen);
    udp.endPacket();
    
    offset += chunkLen;
    packetIndex++;
    
    delayMicroseconds(50);
  }
}

// ======================= CAMERA STREAM TASK =======================

void cameraStreamTask(void* parameter) {
  Serial.println("📹 Camera Stream Task Started");
  
  uint32_t frameCount = 0;
  uint16_t frameId = 0;
  unsigned long lastFpsTime = millis();
  
  while (true) {
    int packetSize = udp.parsePacket();
    if (packetSize > 0) {
      clientIP = udp.remoteIP();
      clientPort = udp.remotePort();
      hasClient = true;
      
      char buffer[32];
      udp.read(buffer, min(packetSize, 31));
      buffer[min(packetSize, 31)] = '\0';
      
      Serial.printf("🔗 Client: %s:%d - %s\n", 
                    clientIP.toString().c_str(), clientPort, buffer);
    }
    
    if (hasClient) {
      camera_fb_t* fb = esp_camera_fb_get();
      
      if (fb) {
        sendFrameUDP(fb, frameId);
        frameId++;
        frameCount++;
        
        if (millis() - lastFpsTime >= 2000) {
          float fps = frameCount / 2.0;
          Serial.printf("📊 FPS: %.1f\n", fps);
          frameCount = 0;
          lastFpsTime = millis();
        }
        
        esp_camera_fb_return(fb);
      }
      
      vTaskDelay(pdMS_TO_TICKS(5));
    } else {
      vTaskDelay(pdMS_TO_TICKS(100));
    }
  }
}

// ======================= HELPER =======================

const char* getLineCmdString(LineCommand cmd) {
  switch (cmd) {
    case LINE_CMD_LEFT:     return "left";
    case LINE_CMD_RIGHT:    return "right";
    case LINE_CMD_BACKWARD: return "backward";
    default:                return "none";
  }
}

// ======================= MOTOR TASK =======================

void motorTask(void* parameter) {
  Serial.println("🚗 Motor Task Started");
  
  HTTPClient http;
  unsigned long lastHttpTime = 0;
  unsigned long lastMotorDebug = 0;
  const unsigned long HTTP_INTERVAL = 100;

  while (true) {
    unsigned long now = millis();
    
    // Debug motor state mỗi 1 giây
    if (now - lastMotorDebug >= 1000) {
      lastMotorDebug = now;
      
      portENTER_CRITICAL(&lineMux);
      bool la = lineOverrideActive;
      LineCommand lc = lineOverrideCmd;
      portEXIT_CRITICAL(&lineMux);
      
      Serial.printf("🚗 State: obstacle=%d, lineOverride=%d, lineCmd=%s, task=%s\n",
                    obstacleDetected, la, getLineCmdString(lc), currenttask);
    }
    
    // // ===== PRIORITY 1: OBSTACLE =====
    // if (obstacleDetected) {
    //   if (!wasObstacleBlocking) {
    //     Serial.println("⛔ OBSTACLE DETECTED!");
    //     wasObstacleBlocking = true;
    //   }
      
    //   if (strcmp(lastServerCommand, "backward") == 0) {
    //     executeDirectCommand("backward", lastServerSpeed);
    //   } else if (strcmp(lastServerCommand, "left") == 0) {
    //     executeDirectCommand("left", lastServerSpeed);
    //   } else if (strcmp(lastServerCommand, "right") == 0) {
    //     executeDirectCommand("right", lastServerSpeed);
    //   } else {
    //     stopCar();
    //     currenttask = "stop";
    //   }
      
    //   vTaskDelay(pdMS_TO_TICKS(50));
    //   continue;
    // }

    // if (wasObstacleBlocking) {
    //   Serial.println("✅ Obstacle Clear!");
    //   wasObstacleBlocking = false;
    // }

    // ===== PRIORITY 2: LINE SENSOR =====
    bool lineActive;
    LineCommand lineCmd;
    
    portENTER_CRITICAL(&lineMux);
    lineActive = lineOverrideActive;
    lineCmd = lineOverrideCmd;
    portEXIT_CRITICAL(&lineMux);
    
    if (lineActive && lineCmd != LINE_CMD_NONE) {
      const char* cmdStr = getLineCmdString(lineCmd);
      Serial.printf("📏 LINE OVERRIDE: %s\n", cmdStr);
      executeDirectCommand(cmdStr, 150);
      vTaskDelay(pdMS_TO_TICKS(20));
      continue;
    }

    // ===== PRIORITY 3: SERVER COMMAND =====
    if (WiFi.status() == WL_CONNECTED && (now - lastHttpTime >= HTTP_INTERVAL)) {
      lastHttpTime = now;
      
      http.begin(currentServerUrl);
      http.setTimeout(500);
      int httpCode = http.GET();

      if (httpCode == HTTP_CODE_OK) {
        String payload = http.getString();
        StaticJsonDocument<128> doc;
        
        if (!deserializeJson(doc, payload)) {
          const char* cmd = doc["command"];
          int spd = doc["speed"] | currentSpeed;
          currentSpeed = spd;
          lastServerCommand = cmd;
          lastServerSpeed = spd;
          
          Serial.printf("🌐 SERVER: cmd=%s, speed=%d\n", cmd, spd);
          executeCommand(cmd, spd);
        }
      } else if (httpCode < 0) {
        // Không log liên tục lỗi HTTP
      }
      http.end();
    }

    vTaskDelay(pdMS_TO_TICKS(20));
  }
}

// ======================= SETUP =======================

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n========================================");
  Serial.println("🚗 ESP32-S3 Car - DEBUG VERSION");
  Serial.println("========================================");

  // Ultrasonic
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  Serial.printf("✅ Ultrasonic: TRIG=%d, ECHO=%d\n", TRIG_PIN, ECHO_PIN);
  
  // Line sensor
  pinMode(LINE_LEFT_PIN, INPUT_PULLUP);
  pinMode(LINE_RIGHT_PIN, INPUT_PULLUP);
  Serial.printf("✅ Line Sensors: LEFT=GPIO%d, RIGHT=GPIO%d\n", LINE_LEFT_PIN, LINE_RIGHT_PIN);
  
  // Test đọc line sensor 5 lần
  Serial.println("📏 Testing Line Sensors (5 reads):");
  for (int i = 0; i < 5; i++) {
    delay(100);
    int l = digitalRead(LINE_LEFT_PIN);
    int r = digitalRead(LINE_RIGHT_PIN);
    Serial.printf("   Read %d: LEFT=%d, RIGHT=%d\n", i+1, l, r);
  }
  Serial.printf("   Expected when NO line: LEFT=1, RIGHT=1 (if ACTIVE_LOW)\n");
  Serial.printf("   Expected when ON line: LEFT=0, RIGHT=0 (if ACTIVE_LOW)\n");
  
  // Motor
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  ledcAttach(ENA, pwmFreq, pwmResolution);
  ledcAttach(ENB, pwmFreq, pwmResolution);
  Serial.println("✅ Motor pins configured");

  // WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("📶 WiFi connecting");
  
  int wifiTimeout = 0;
  while (WiFi.status() != WL_CONNECTED && wifiTimeout < 30) {
    delay(500);
    Serial.print(".");
    wifiTimeout++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n✅ WiFi connected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n⚠️ WiFi failed!");
  }

  setupCamera();

  udp.begin(UDP_PORT);
  Serial.printf("✅ UDP Port: %d\n", UDP_PORT);

  stopCar();

  // Tasks
  Serial.println("\n📋 Creating Tasks...");
  
  xTaskCreatePinnedToCore(lineSensorTask, "LineSensor", 4096, NULL, 4, &lineSensorTaskHandle, 0);
  delay(100);
  
  xTaskCreatePinnedToCore(ultrasonicTask, "Ultrasonic", 4096, NULL, 3, &ultrasonicTaskHandle, 0);
  delay(100);
  
  xTaskCreatePinnedToCore(motorTask, "Motor", 8192, NULL, 2, &motorTaskHandle, 0);
  delay(100);
  
  xTaskCreatePinnedToCore(cameraStreamTask, "CameraStream", 8192, NULL, 2, &cameraStreamTaskHandle, 1);

  Serial.println("\n========================================");
  Serial.println("🚗 System Ready - Watch Serial for DEBUG");
  Serial.println("========================================\n");
}

// ======================= LOOP =======================

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}

// ======================= MOTOR FUNCTIONS =======================

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

void executeDirectCommand(const char* command, int speed) {
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

void kickStart(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, speed);
  ledcWrite(ENB, speed);
  vTaskDelay(pdMS_TO_TICKS(kickstarttime));
}

void turnRight(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  ledcWrite(ENA, speed);
  ledcWrite(ENB, 90);
}

void turnLeft(int speed) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, 90);
  ledcWrite(ENB, speed);
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