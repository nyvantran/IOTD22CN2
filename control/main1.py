import socket
import cv2
import numpy as np
import threading
from collections import defaultdict
import time

ESP32_IP = "192.168.1.17"  # Thay IP ESP32
UDP_PORT = 8888
CHUNK_SIZE = 1400
HEADER_SIZE = 8


class UDPVideoReceiver:
    def __init__(self, esp_ip, port):
        self.esp_ip = esp_ip
        self.port = port
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

    def register_with_esp(self):
        """Gửi packet để đăng ký với ESP32"""
        self.sock.sendto(b"HELLO", (self.esp_ip, self.port))
        print(f"📡 Sent registration to {self.esp_ip}:{self.port}")

    def receive_loop(self):
        """Thread nhận packets"""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(CHUNK_SIZE + HEADER_SIZE + 100)

                if len(data) < HEADER_SIZE:
                    continue

                # Parse header
                frame_id = data[0] | (data[1] << 8)
                packet_idx = data[2] | (data[3] << 8)
                total_packets = data[4] | (data[5] << 8)
                payload = data[HEADER_SIZE:]

                # Store packet
                frame_data = self.frames[frame_id]
                frame_data['packets'][packet_idx] = payload
                frame_data['total'] = total_packets

                # Check if frame complete
                if len(frame_data['packets']) == total_packets:
                    # Reassemble frame
                    jpeg_data = b''
                    for i in range(total_packets):
                        if i in frame_data['packets']:
                            jpeg_data += frame_data['packets'][i]

                    # Decode JPEG
                    nparr = np.frombuffer(jpeg_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        with self.frame_lock:
                            self.current_frame = frame
                        self.frame_count += 1

                    # Cleanup old frames
                    old_frames = [fid for fid in self.frames if fid < frame_id - 5]
                    for fid in old_frames:
                        del self.frames[fid]

            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error: {e}")

    def get_frame(self):
        """Lấy frame mới nhất"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def get_fps(self):
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now
        return self.fps

    def stop(self):
        self.running = False
        self.sock.close()


def main():
    receiver = UDPVideoReceiver(ESP32_IP, UDP_PORT)

    # Start receive thread
    recv_thread = threading.Thread(target=receiver.receive_loop, daemon=True)
    recv_thread.start()

    # Register with ESP32
    receiver.register_with_esp()

    print("🎥 Waiting for stream... Press 'q' to quit")

    # Re-register periodically
    last_register = time.time()

    while True:
        # Re-register every 5 seconds
        if time.time() - last_register > 5:
            receiver.register_with_esp()
            last_register = time.time()

        frame = receiver.get_frame()

        if frame is not None:
            fps = receiver.get_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('ESP32 UDP Stream', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            receiver.register_with_esp()

    receiver.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()