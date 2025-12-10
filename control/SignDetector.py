import threading
import time
from ultralytics import YOLO
import cv2

class SignDetector:
    def __init__(self, model_path="../model/best.pt", conf_threshold=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.current_sign = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
        # Mapping các nhãn theo yêu cầu
        self.SPEED_LABELS = [
            "Speed Limit 100", "Speed Limit 110", "Speed Limit 120",
            "Speed Limit 20", "Speed Limit 30", "Speed Limit 40",
            "Speed Limit 50", "Speed Limit 60", "Speed Limit 70",
            "Speed Limit 80", "Speed Limit 90"
        ]
        self.STOP_LABELS = ["Red Light", "Stop"]
        self.GO_LABELS = ["Green Light"]

    def detect_loop(self, stream_manager):
        """Vòng lặp chạy ngầm để detect liên tục"""
        print("--- Bắt đầu luồng nhận diện biển báo ---")
        while self.running:
            frame = stream_manager.get_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            start_y = int(h * 0.5)

            roi_frame = frame[start_y:h, 0:w]
            # Chạy YOLO (verbose=False để đỡ spam log)
            try:
                results = self.model(roi_frame, verbose=False, conf=self.conf_threshold)
                
                detected = None
                highest_conf = 0

                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]
                        conf = float(box.conf[0])
                        
                        if conf > highest_conf:
                            highest_conf = conf
                            detected = label

                # Cập nhật kết quả (Thread-safe)
                with self.lock:
                    self.current_sign = detected
            
            except Exception as e:
                print(f"Lỗi YOLO detect: {e}")

            # Nghỉ một chút để giảm tải CPU
            time.sleep(0.02)

    def start(self, stream_manager):
        self.running = True
        self.thread = threading.Thread(target=self.detect_loop, args=(stream_manager,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def get_current_sign(self):
        with self.lock:
            return self.current_sign