import threading
import time
from enum import Enum
from typing import Tuple, Optional
import cv2
import numpy as np


class Command(Enum):
    """Enum định nghĩa các lệnh điều khiển xe."""
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STOP = "stop"


class CarControl:
    """
    Class quản lý điều khiển xe dựa trên phát hiện làn đường.
    Chạy trên một luồng riêng biệt, liên tục xử lý frame và cập nhật lệnh điều khiển.
    """

    def __init__(self, stream_manager, lane_nav, speed: int = 50):
        """
        Khởi tạo CarControl.

        Args:
            stream_manager: Đối tượng quản lý stream video, có phương thức get_latest_frame()
            lane_nav: Đối tượng LaneNavigator để phân tích làn đường
            speed: Tốc độ cố định của xe (0-100)
        """
        self.stream_manager = stream_manager
        self.lane_nav = lane_nav
        self.speed = speed

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Current state
        self._current_command = Command.STOP
        self._current_info = {}
        self._last_update_time = 0
        self._frame_count = 0
        self._fps = 0

        # Thông số điều khiển
        self.steering_threshold_soft = 0.15  # Ngưỡng rẽ nhẹ
        self.steering_threshold_hard = 0.4  # Ngưỡng rẽ mạnh
        self.lost_lane_timeout = 1.0  # Thời gian (s) mất làn trước khi dừng
        self._last_valid_detection_time = 0

        # Debug/Visualization
        self.enable_display = False
        self.latest_processed_frame = None

    def start(self):
        """Bắt đầu luồng điều khiển."""
        if self._running:
            print("[CarControl] Already running!")
            return

        self._running = True
        self._last_valid_detection_time = time.time()
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        print("[CarControl] Started control thread.")

    def stop(self):
        """Dừng luồng điều khiển."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            self._current_command = Command.STOP

        print("[CarControl] Stopped control thread.")

    def is_running(self) -> bool:
        """Kiểm tra luồng có đang chạy không."""
        return self._running

    def get_command(self) -> Tuple[str, int]:
        """
        Lấy lệnh điều khiển hiện tại.

        Returns:
            Tuple[str, int]: (command, speed)
                - command: 'forward', 'backward', 'left', 'right', 'stop'
                - speed: Tốc độ cố định
        """
        with self._lock:
            return (self._current_command.value, self.speed)

    def get_detailed_info(self) -> dict:
        """
        Lấy thông tin chi tiết về trạng thái điều khiển.

        Returns:
            dict: Thông tin chi tiết bao gồm command, speed, lane_info, fps, etc.
        """
        with self._lock:
            return {
                "command": self._current_command.value,
                "speed": self.speed,
                "info": self._current_info.copy(),
                "fps": self._fps,
                "frame_count": self._frame_count
            }

    def set_speed(self, speed: int):
        """Đặt tốc độ xe."""
        self.speed = max(0, min(100, speed))
        print(f"[CarControl] Speed set to: {self.speed}")

    def _control_loop(self):
        """Vòng lặp chính xử lý điều khiển xe."""
        fps_counter = 0
        fps_start_time = time.time()

        while self._running:
            try:
                loop_start = time.time()

                # Lấy frame mới nhất
                frame = self.stream_manager.get_latest_frame()

                if frame is None:
                    self._set_command(Command.STOP, {"status": "NO_FRAME"})
                    time.sleep(0.01)
                    continue

                # Xử lý frame với LaneNavigator
                processed_frame, info = self.lane_nav.process_frame(frame, debug=self.enable_display)

                # Chuyển đổi kết quả thành lệnh điều khiển
                command = self._process_lane_info(info)

                # Cập nhật lệnh
                self._set_command(command, info)

                # Lưu frame đã xử lý (cho visualization)
                if self.enable_display:
                    self.latest_processed_frame = processed_frame

                # Tính FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    with self._lock:
                        self._fps = fps_counter
                        self._frame_count += fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()

                # Giới hạn tốc độ xử lý (tối đa ~60 FPS)
                elapsed = time.time() - loop_start
                if elapsed < 0.016:  # ~60 FPS
                    time.sleep(0.016 - elapsed)

            except Exception as e:
                print(f"[CarControl] Error in control loop: {e}")
                self._set_command(Command.STOP, {"status": "ERROR", "error": str(e)})
                time.sleep(0.1)

    def _process_lane_info(self, info: dict) -> Command:
        """
        Chuyển đổi thông tin từ LaneNavigator thành lệnh điều khiển.

        Args:
            info: Dictionary chứa thông tin từ lane_nav.process_frame()

        Returns:
            Command: Lệnh điều khiển tương ứng
        """
        status = info.get("status", "")

        # Trường hợp mất làn
        if status == "LOST_LANE":
            time_since_valid = time.time() - self._last_valid_detection_time

            if time_since_valid > self.lost_lane_timeout:
                # Mất làn quá lâu -> dừng xe
                return Command.STOP
            else:
                # Mất làn tạm thời -> tiếp tục đi thẳng chậm
                return Command.FORWARD

        # Cập nhật thời gian phát hiện hợp lệ
        self._last_valid_detection_time = time.time()

        # Lấy action_code từ LaneNavigator
        action_code = info.get("action_code", 0)
        steering_score = info.get("steering_score", 0)
        confidence = info.get("confidence", 1.0)

        # Xử lý dựa trên action_code
        # action_code: -2 (sharp left), -1 (left), 0 (straight), 1 (right), 2 (sharp right)

        if action_code == 99:  # Mã lỗi đặc biệt
            return Command.STOP

        # Điều chỉnh quyết định dựa trên confidence
        if confidence < 0.3:
            # Confidence quá thấp -> đi thẳng cẩn thận
            return Command.FORWARD

        # Mapping action_code sang Command
        abs_score = abs(steering_score)

        if abs_score < self.steering_threshold_soft:
            # Đi thẳng
            return Command.FORWARD
        elif steering_score > 0:
            # Rẽ phải
            return Command.RIGHT
        else:
            # Rẽ trái
            return Command.LEFT

    def _set_command(self, command: Command, info: dict):
        """Thread-safe setter cho command và info."""
        with self._lock:
            self._current_command = command
            self._current_info = info
            self._last_update_time = time.time()


class CarControlAdvanced(CarControl):
    """
    Phiên bản nâng cao của CarControl với nhiều tính năng hơn:
    - Điều khiển tốc độ động dựa trên độ cong
    - Xử lý các tình huống đặc biệt
    - Logging và statistics
    """

    def __init__(self, stream_manager, lane_nav, base_speed: int = 50,
                 min_speed: int = 20, max_speed: int = 80):
        super().__init__(stream_manager, lane_nav, base_speed)

        self.base_speed = base_speed
        self.min_speed = min_speed
        self.max_speed = max_speed

        # Dynamic speed control
        self.enable_dynamic_speed = True

        # Statistics
        self._stats = {
            "total_frames": 0,
            "forward_count": 0,
            "left_count": 0,
            "right_count": 0,
            "stop_count": 0,
            "lost_lane_count": 0
        }

        # Command history (để làm mượt)
        self._command_history = []
        self._history_size = 5

    def get_command(self) -> Tuple[str, int]:
        """
        Lấy lệnh điều khiển với tốc độ động.

        Returns:
            Tuple[str, int]: (command, speed) với speed có thể thay đổi
        """
        with self._lock:
            command = self._current_command.value

            if self.enable_dynamic_speed:
                speed = self._calculate_dynamic_speed()
            else:
                speed = self.speed

            return (command, speed)

    def get_statistics(self) -> dict:
        """Lấy thống kê điều khiển."""
        with self._lock:
            return self._stats.copy()

    def reset_statistics(self):
        """Reset thống kê."""
        with self._lock:
            self._stats = {
                "total_frames": 0,
                "forward_count": 0,
                "left_count": 0,
                "right_count": 0,
                "stop_count": 0,
                "lost_lane_count": 0
            }

    def _calculate_dynamic_speed(self) -> int:
        """
        Tính tốc độ động dựa trên tình trạng đường.

        Returns:
            int: Tốc độ được tính toán
        """
        info = self._current_info

        # Mặc định là base_speed
        speed = self.base_speed

        # Lấy các thông số
        radius = info.get("raw_data", {}).get("radius_m", float('inf'))
        confidence = info.get("confidence", 1.0)
        steering_score = abs(info.get("steering_score", 0))

        # Giảm tốc khi cua gấp (bán kính nhỏ)
        if radius < 100:
            speed_factor = max(0.5, radius / 100)
            speed = int(speed * speed_factor)

        # Giảm tốc khi steering_score cao (cần rẽ nhiều)
        if steering_score > 0.3:
            speed_factor = max(0.6, 1 - steering_score * 0.5)
            speed = int(speed * speed_factor)

        # Giảm tốc khi confidence thấp
        if confidence < 0.7:
            speed = int(speed * confidence)

        # Clamp speed
        speed = max(self.min_speed, min(self.max_speed, speed))

        return speed

    def _set_command(self, command: Command, info: dict):
        """Override để thêm statistics và smoothing."""
        # Cập nhật history
        self._command_history.append(command)
        if len(self._command_history) > self._history_size:
            self._command_history.pop(0)

        # Smoothing: nếu command mới khác với majority trong history, giữ nguyên
        smoothed_command = self._smooth_command(command)

        with self._lock:
            self._current_command = smoothed_command
            self._current_info = info
            self._last_update_time = time.time()

            # Cập nhật statistics
            self._stats["total_frames"] += 1
            if smoothed_command == Command.FORWARD:
                self._stats["forward_count"] += 1
            elif smoothed_command == Command.LEFT:
                self._stats["left_count"] += 1
            elif smoothed_command == Command.RIGHT:
                self._stats["right_count"] += 1
            elif smoothed_command == Command.STOP:
                self._stats["stop_count"] += 1

            if info.get("status") == "LOST_LANE":
                self._stats["lost_lane_count"] += 1

    def _smooth_command(self, new_command: Command) -> Command:
        """
        Làm mượt lệnh điều khiển bằng cách xem xét history.
        Tránh việc lệnh thay đổi quá nhanh.
        """
        if len(self._command_history) < 3:
            return new_command

        # Đếm số lần xuất hiện của mỗi command trong history
        from collections import Counter
        counts = Counter(self._command_history[-3:])

        # Nếu command mới giống với majority, dùng nó
        most_common = counts.most_common(1)[0][0]

        # Nếu command mới là STOP, ưu tiên STOP (an toàn)
        if new_command == Command.STOP:
            return Command.STOP

        # Nếu đang chuyển từ LEFT/RIGHT sang FORWARD, cho phép ngay
        if new_command == Command.FORWARD and most_common in [Command.LEFT, Command.RIGHT]:
            return new_command

        # Nếu đang chuyển từ FORWARD sang LEFT/RIGHT, cần ít nhất 2/3 đồng ý
        if most_common == Command.FORWARD and new_command in [Command.LEFT, Command.RIGHT]:
            recent_turns = sum(1 for c in self._command_history[-3:]
                               if c in [Command.LEFT, Command.RIGHT])
            if recent_turns >= 2:
                return new_command
            return Command.FORWARD

        return new_command


# ========================================================
# VÍ DỤ SỬ DỤNG
# ========================================================

def main():
    """Ví dụ sử dụng CarControl."""
    from stream_manager import stream_manager
    from lane_navigator import LaneNavigator  # Import class LaneNavigator đã tạo ở trên

    # 1. Khởi tạo các components
    stream_manager.start()

    lane_nav = LaneNavigator()
    lane_nav.load_config("lane_nav_config.json")

    # 2. Tạo CarControl (hoặc CarControlAdvanced)
    car_control = CarControlAdvanced(
        stream_manager=stream_manager,
        lane_nav=lane_nav,
        base_speed=50,
        min_speed=20,
        max_speed=70
    )

    # Bật hiển thị debug
    car_control.enable_display = True

    # 3. Bắt đầu luồng điều khiển
    car_control.start()

    print("\n=== CAR CONTROL STARTED ===")
    print("Press 'q' to quit")
    print("Press 's' to show statistics")
    print("===========================\n")

    try:
        while True:
            # Lấy lệnh điều khiển
            command, speed = car_control.get_command()

            # Lấy thông tin chi tiết
            detail = car_control.get_detailed_info()

            # Hiển thị
            print(f"\r[CMD: {command:^10}] [SPD: {speed:3}] [FPS: {detail['fps']:2}]", end="")

            # Hiển thị frame nếu có
            if car_control.enable_display and car_control.latest_processed_frame is not None:
                frame = car_control.latest_processed_frame

                # Thêm thông tin command lên frame
                cv2.putText(frame, f"CMD: {command.upper()}", (10, frame.shape[0] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"SPEED: {speed}", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                cv2.imshow("Car Control View", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Hiển thị statistics
                stats = car_control.get_statistics()
                print(f"\n\n=== STATISTICS ===")
                print(f"Total frames: {stats['total_frames']}")
                print(
                    f"Forward: {stats['forward_count']} ({stats['forward_count'] / max(1, stats['total_frames']) * 100:.1f}%)")
                print(f"Left: {stats['left_count']} ({stats['left_count'] / max(1, stats['total_frames']) * 100:.1f}%)")
                print(
                    f"Right: {stats['right_count']} ({stats['right_count'] / max(1, stats['total_frames']) * 100:.1f}%)")
                print(f"Stop: {stats['stop_count']} ({stats['stop_count'] / max(1, stats['total_frames']) * 100:.1f}%)")
                print(f"Lost lane: {stats['lost_lane_count']}")
                print("==================\n")
            elif key == ord('+') or key == ord('='):
                car_control.set_speed(car_control.speed + 5)
            elif key == ord('-'):
                car_control.set_speed(car_control.speed - 5)

            # Gửi lệnh xuống xe (ví dụ)
            # send_to_car(command, speed)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # 4. Dừng
        car_control.stop()
        cv2.destroyAllWindows()
        print("\nCar control stopped.")


# ========================================================
# SIMPLE EXAMPLE - Chỉ lấy command
# ========================================================

def simple_example():
    """Ví dụ đơn giản nhất để sử dụng CarControl."""
    from stream_manager import stream_manager
    from lane_navigator import LaneNavigator  # Import class LaneNavigator đã tạo ở trên
    # Setup
    stream_manager.start()
    lane_nav = LaneNavigator()
    lane_nav.load_config("lane_nav_config.json")

    # Tạo controller với tốc độ cố định = 50
    car = CarControl(stream_manager, lane_nav, speed=50)
    car.start()

    # Vòng lặp chính
    try:
        while True:
            # Chỉ cần gọi get_command() để lấy lệnh
            command, speed = car.get_command()

            # Gửi xuống xe
            print(f"Command: {command}, Speed: {speed}")

            # Ví dụ gửi serial
            # serial.write(f"{command},{speed}\n".encode())

            time.sleep(0.05)  # 20Hz

    except KeyboardInterrupt:
        car.stop()


if __name__ == "__main__":
    main()
    # simple_example()
from stream_manager import stream_manager
from lane_navigator import lane_nav

car_control = CarControl(stream_manager=stream_manager, lane_nav=lane_nav, speed=105)
