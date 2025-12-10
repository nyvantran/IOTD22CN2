import cv2
import threading
import time


class StreamManager:
    """
    Một class Singleton để quản lý luồng lấy video stream.
    Nó chạy trong một background thread để liên tục lấy frame mới nhất.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(StreamManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, url="http://192.168.1.11/stream"):
        if not hasattr(self, 'is_initialized'):
            self.url = url
            self.latest_frame = None
            self.lock = threading.Lock()  # Rất quan trọng để đảm bảo thread-safe
            self.is_running = False
            self.thread = None
            self.is_initialized = True
            self.cap = None

    def _capture_loop(self):
        """Vòng lặp chạy trong thread để lấy và giải mã frame."""
        print(f"Bắt đầu thread lấy stream từ: {self.url}  ")
        try:
            self.cap = cv2.VideoCapture(self.url)
            if not self.cap.isOpened():
                print(f"Lỗi: Không thể mở stream từ {self.url}")
                self.is_running = False
                return
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Lỗi: Không thể đọc frame từ stream.")
                    time.sleep(1)  # Chờ một chút trước khi thử lại
                    continue

                with self.lock:
                    self.latest_frame = frame.copy()
        except Exception as e:
            print(f"Lỗi không xác định trong luồng: {e}")
        finally:
            print("Thread lấy stream đã dừng.")
            self.is_running = False

    def start(self):
        """Bắt đầu background thread."""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print("Stream manager đã khởi động.")
        while self.get_latest_frame() is None:
            time.sleep(0.1)  # Chờ cho đến khi có frame đầu tiên
        print("Frame đầu tiên đã sẵn sàng.")

    def stop(self):
        """Dừng background thread."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()  # Chờ thread kết thúc
        print("Stream manager đã dừng.")

    def get_latest_frame(self):
        """Lấy frame mới nhất một cách an toàn."""
        frame = None
        with self.lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
        return frame


# Tạo một instance duy nhất (singleton) để toàn bộ ứng dụng sử dụng
URL_STREAM = "http://10.251.5.145/stream"
stream_manager = StreamManager(url=URL_STREAM)
# stream_manager.start()
# cv2.imwrite("test.jpg", stream_manager.get_latest_frame())
# stream_manager.stop()
