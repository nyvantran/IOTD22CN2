import socket
import cv2
import numpy as np
import threading
from collections import defaultdict
import time

from control.stream_manager import stream_manager


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

    def __init__(self, esp_ip, port, chunk_size=1400, header_size=8):
        if not hasattr(self, 'is_initialized'):
            self.esp_ip = esp_ip
            self.port = port
            self.chunk_size = chunk_size
            self.header_size = header_size

            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB buffer
            self.sock.bind(('', port))
            self.sock.settimeout(0.1)

            self.frames = defaultdict(lambda: {'packets': {}, 'total': 0})
            self.current_frame = None
            self.frame_lock = threading.Lock()
            self.running = True

            # Stats
            self.frame_count = 0
            self.last_fps_time = time.time()
            self.fps = 0
            self.thread = None
            self.is_running = False

    def register_with_esp(self):
        """Gửi packet để đăng ký với ESP32"""
        self.sock.sendto(b"HELLO", (self.esp_ip, self.port))
        print(f"📡 Sent registration to {self.esp_ip}:{self.port}")

    def _capture_loop(self):
        """Thread nhận packets"""
        print(f"Bắt đầu thread lấy stream từ: {self.esp_ip}:{self.port}  ")
        while self.running:
            try:
                data, addr = self.sock.recvfrom(self.chunk_size + self.header_size + 100)

                if len(data) < 8:
                    continue

                # Parse header
                frame_id = data[0] | (data[1] << 8)
                packet_idx = data[2] | (data[3] << 8)
                total_packets = data[4] | (data[5] << 8)
                payload = data[8:]

                # Store packet
                frame_data = self.frames[frame_id]
                frame_data['packets'][packet_idx] = payload
                frame_data['total'] = total_packets

                # Check if frame complete
                if len(frame_data['packets']) == total_packets:
                    # Reassemble frame
                    frame_bytes = b''.join(frame_data['packets'][i] for i in range(total_packets))
                    np_arr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    with self.frame_lock:
                        self.current_frame = frame
                        self.frame_count += 1

                    # FPS calculation
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.last_fps_time = current_time

                    # Clear stored packets for this frame
                    del self.frames[frame_id]

            except socket.timeout:
                continue
            except Exception as e:
                print(f"Lỗi không xác định trong luồng: {e}")
        print("Thread lấy stream đã dừng.")
        self.is_running = False

    def start(self):
        """Bắt đầu background thread."""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print("Stream manager đã khởi động.")
            self.register_with_esp()
        while self.get_latest_frame() is None:
            time.sleep(0.1)  # Chờ cho đến khi có frame đầu tiên
        print("Frame đầu tiên đã sẵn sàng.")

    def stop(self):
        """Dừng background thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()  # Chờ thread kết thúc
        print("Stream manager đã dừng.")

    def get_fps(self):
        """Lấy FPS hiện tại."""
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now
        return self.fps

    def get_latest_frame(self):
        """Lấy frame mới nhất một cách an toàn."""
        frame = None
        with self.frame_lock:
            if self.current_frame is not None:
                frame = self.current_frame.copy()
        return frame


def main():
    esp_ip = "192.168.1.23"
    udp_port = 8888
    stream_manager = StreamManager(esp_ip, udp_port)
    stream_manager.start()
    # stream_manager.register_with_esp()
    print("🎥 Waiting for stream... Press 'q' to quit")
    try:
        while True:
            frame = stream_manager.get_latest_frame()
            cv2.putText(frame, f"FPS: {stream_manager.get_fps():.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if frame is not None:
                cv2.imshow("Video Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stream_manager.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

stream_manager = StreamManager(esp_ip="192.168.1.23", port=8888)
stream_manager.start()