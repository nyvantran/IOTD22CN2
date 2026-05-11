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


class ControlState(Enum):
    """Trạng thái của bộ điều khiển."""
    STOPPED = "stopped"  # Luồng đã dừng hoàn toàn
    RUNNING = "running"  # Đang chạy bình thường
    PAUSED = "paused"  # Tạm dừng (luồng vẫn chạy nhưng không điều khiển)


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

        # ===== PAUSE CONTROL (MỚI) =====
        self._paused = False
        self._pause_lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Mặc định không pause (event được set)

        # Current state
        self._current_command = Command.STOP
        self._current_info = {}
        self._last_update_time = 0
        self._frame_count = 0
        self._fps = 0
        self._state = ControlState.STOPPED

        # Lệnh trước khi pause (để có thể resume)
        self._command_before_pause = Command.STOP

        # Thông số điều khiển
        self.steering_threshold_soft = 0.2
        self.steering_threshold_hard = 0.5
        self.lost_lane_timeout = 1.0
        self._last_valid_detection_time = 0

        # Debug/Visualization
        self.enable_display = True
        self.latest_processed_frame = None

        # Debounce: cần X frame liên tiếp cùng hướng mới rẽ
        self.required_turn_frames = 1  # hoặc 4 nếu muốn chắc hơn
        self._last_decision_dir = 0  # -1 trái, 0 thẳng, 1 phải
        self._direction_consistency = 0

    # ================================================================
    # =============== PAUSE/RESUME METHODS (MỚI) =====================
    # ================================================================

    def pause(self):
        """
        Tạm dừng quá trình điều khiển xe.
        - Luồng vẫn chạy (để resume nhanh)
        - Xe sẽ nhận lệnh STOP
        - Vẫn tiếp tục xử lý frame (để hiển thị) nhưng không cập nhật command
        """
        with self._pause_lock:
            if self._paused:
                print("[CarControl] Already paused!")
                return

            if not self._running:
                print("[CarControl] Not running, cannot pause!")
                return

            # Lưu lại command trước khi pause
            with self._lock:
                self._command_before_pause = self._current_command
                self._current_command = Command.STOP
                self._state = ControlState.PAUSED

            self._paused = True
            self._pause_event.clear()  # Block waiting threads

        print("[CarControl] Paused. Command set to STOP.")

    def resume(self):
        """
        Tiếp tục quá trình điều khiển xe sau khi pause.
        """
        with self._pause_lock:
            if not self._paused:
                print("[CarControl] Not paused!")
                return

            if not self._running:
                print("[CarControl] Not running, cannot resume!")
                return

            self._paused = False
            self._pause_event.set()  # Release waiting threads

            with self._lock:
                self._state = ControlState.RUNNING
                self._last_valid_detection_time = time.time()  # Reset timeout

        print("[CarControl] Resumed.")

    def toggle_pause(self):
        """
        Chuyển đổi trạng thái pause/resume.

        Returns:
            bool: True nếu đang ở trạng thái paused sau khi toggle
        """
        with self._pause_lock:
            if self._paused:
                self.resume()
                return False
            else:
                self.pause()
                return True

    def is_paused(self) -> bool:
        """
        Kiểm tra xem đang ở trạng thái pause không.

        Returns:
            bool: True nếu đang pause
        """
        with self._pause_lock:
            return self._paused

    def get_state(self) -> ControlState:
        """
        Lấy trạng thái hiện tại của bộ điều khiển.

        Returns:
            ControlState: STOPPED, RUNNING, hoặc PAUSED
        """
        with self._lock:
            return self._state

    # ================================================================
    # =============== EXISTING METHODS (CẬP NHẬT) ====================
    # ================================================================

    def start(self):
        """Bắt đầu luồng điều khiển."""
        if self._running:
            print("[CarControl] Already running!")
            return

        self._running = True
        self._paused = False
        self._pause_event.set()

        with self._lock:
            self._state = ControlState.RUNNING

        self._last_valid_detection_time = time.time()
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        print("[CarControl] Started control thread.")

    def stop(self):
        """Dừng hoàn toàn luồng điều khiển."""
        # Đảm bảo không bị block ở pause
        self._pause_event.set()
        self._paused = False

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            self._current_command = Command.STOP
            self._state = ControlState.STOPPED

        print("[CarControl] Stopped control thread.")

    def is_running(self) -> bool:
        """Kiểm tra luồng có đang chạy không (bao gồm cả paused)."""
        return self._running

    def get_command(self) -> Tuple[str, int]:
        """
        Lấy lệnh điều khiển hiện tại.

        Returns:
            Tuple[str, int]: (command, speed)
                - command: 'forward', 'backward', 'left', 'right', 'stop'
                - speed: Tốc độ (0 nếu đang pause hoặc stop)
        """
        with self._lock:
            command = self._current_command.value

            # Nếu đang pause, trả về speed = 0
            if self._state == ControlState.PAUSED:
                return (Command.STOP.value, 0)
            elif self._current_command == Command.STOP:
                return (command, 0)
            else:
                return (command, self.speed)

    def get_detailed_info(self) -> dict:
        """
        Lấy thông tin chi tiết về trạng thái điều khiển.

        Returns:
            dict: Thông tin chi tiết
        """
        with self._lock:
            paused = self._state == ControlState.PAUSED
            return {
                "command": self._current_command.value,
                "speed": 0 if paused else self.speed,
                "state": self._state.value,
                "is_paused": paused,
                "info": self._current_info.copy(),
                "fps": self._fps,
                "frame_count": self._frame_count
            }

    def set_speed(self, speed: int):
        """Đặt tốc độ xe."""
        self.speed = max(100, min(225, speed))
        print(f"[CarControl] Speed set to: {self.speed}")

    def _control_loop(self):
        """Vòng lặp chính xử lý điều khiển xe."""
        fps_counter = 0
        fps_start_time = time.time()

        while self._running:
            try:
                # ===== KIỂM TRA PAUSE =====
                # Nếu đang pause, vẫn xử lý frame (để hiển thị) nhưng không cập nhật command
                is_paused = self.is_paused()

                loop_start = time.time()

                # Lấy frame mới nhất
                frame = self.stream_manager.get_latest_frame()

                if frame is None:
                    if not is_paused:
                        self._set_command(Command.STOP, {"status": "NO_FRAME"})
                    time.sleep(0.01)
                    continue

                # Xử lý frame với LaneNavigator (luôn xử lý để có frame hiển thị)
                processed_frame, info = self.lane_nav.process_frame(frame, debug=self.enable_display)

                # Lưu frame đã xử lý (cho visualization)
                if self.enable_display:
                    self.latest_processed_frame = processed_frame

                # ===== CHỈ CẬP NHẬT COMMAND KHI KHÔNG PAUSE =====
                if not is_paused:
                    # Chuyển đổi kết quả thành lệnh điều khiển
                    command = self._process_lane_info(info)
                    # Cập nhật lệnh
                    self._set_command(command, info)
                else:
                    # Đang pause - chỉ cập nhật info, giữ command STOP
                    with self._lock:
                        self._current_info = info

                # Tính FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    with self._lock:
                        self._fps = fps_counter
                        self._frame_count += fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()

                # Giới hạn tốc độ xử lý
                elapsed = time.time() - loop_start
                if elapsed < 0.016:
                    time.sleep(0.016 - elapsed)

            except Exception as e:
                print(f"[CarControl] Error in control loop: {e}")
                if not self.is_paused():
                    self._set_command(Command.STOP, {"status": "ERROR", "error": str(e)})
                time.sleep(0.1)

    def _process_lane_info(self, info: dict) -> Command:
        """Chuyển đổi thông tin từ LaneNavigator thành lệnh điều khiển."""
        status = info.get("status", "")

        if status == "LOST_LANE":
            time_since_valid = time.time() - self._last_valid_detection_time
            if time_since_valid > self.lost_lane_timeout:
                return Command.STOP
            else:
                return Command.FORWARD

        self._last_valid_detection_time = time.time()

        action_code = info.get("action_code", 0)
        steering_score = info.get("steering_score", 0)
        confidence = info.get("confidence", 1.0)

        if action_code == 99:
            return Command.STOP

        if confidence < 0.3:
            return Command.FORWARD

        abs_score = abs(steering_score)

        # 1. Xác định hướng AI muốn rẽ ở frame hiện tại
        if abs_score < self.steering_threshold_soft:
            desired_dir = 0  # đi thẳng
        elif steering_score > 0:
            desired_dir = 1  # phải
        else:
            desired_dir = -1  # trái

        # 2. Nếu muốn đi thẳng → reset bộ đếm và trả về FORWARD
        if desired_dir == 0:
            self._direction_consistency = 0
            self._last_decision_dir = 0
            return Command.FORWARD

        # 3. Nếu hướng giống frame trước → tăng độ “ổn định”
        if desired_dir == self._last_decision_dir:
            self._direction_consistency += 1
        else:
            # nếu đổi hướng → reset đếm
            self._direction_consistency = 1
            self._last_decision_dir = desired_dir

        # 4. Nếu chưa đủ số frame liên tiếp → vẫn đi thẳng
        if self._direction_consistency < self.required_turn_frames:
            return Command.FORWARD

        # 5. Đủ số frame → CHO RẼ
        if desired_dir == 1:
            return Command.RIGHT
        else:
            return Command.LEFT

    def _set_command(self, command: Command, info: dict):
        """Thread-safe setter cho command và info."""
        with self._lock:
            self._current_command = command
            self._current_info = info
            self._last_update_time = time.time()


class CarControlAdvanced(CarControl):
    """
    Phiên bản nâng cao của CarControl với nhiều tính năng hơn.
    """

    def __init__(self, stream_manager, lane_nav, sign_detector=None, base_speed: int = 50,
                 min_speed: int = 20, max_speed: int = 80):
        super().__init__(stream_manager, lane_nav, base_speed)

        self.base_speed = base_speed
        self.min_speed = min_speed
        self.max_speed = max_speed

        self._stats = {
            "total_frames": 0,
            "forward_count": 0,
            "left_count": 0,
            "right_count": 0,
            "stop_count": 0,
            "lost_lane_count": 0,
            "pause_count": 0,  # Số lần pause
            "total_pause_time": 0.0  # Tổng thời gian pause
        }

        self._command_history = []
        self._history_size = 5

        # Theo dõi thời gian pause
        self._pause_start_time = None

        # detect sign
        self.sign_detector = sign_detector  # Lưu instance detetor

        # Biến lưu trạng thái do biển báo tác động
        self.force_stop_by_sign = False
        self.override_speed = 110

    # ===== OVERRIDE PAUSE METHODS =====

    def pause(self):
        """Override pause để thêm statistics."""
        super().pause()
        with self._lock:
            self._stats["pause_count"] += 1
            self._pause_start_time = time.time()

    def resume(self):
        """Override resume để thêm statistics."""
        if self._pause_start_time is not None:
            with self._lock:
                pause_duration = time.time() - self._pause_start_time
                self._stats["total_pause_time"] += pause_duration
                self._pause_start_time = None
        super().resume()

    def get_command(self) -> Tuple[str, int]:
        """Lấy lệnh điều khiển với tốc độ động."""
        with self._lock:
            # Nếu đang pause, trả về STOP với speed = 0
            if self._state == ControlState.PAUSED:
                return (Command.STOP.value, 0)

            command = self._current_command.value

            if self._current_command == Command.STOP:
                return (command, 0)

            speed = self.speed

            return (command, speed)

    def get_statistics(self) -> dict:
        """Lấy thống kê điều khiển."""
        with self._lock:
            stats = self._stats.copy()
            # Tính thời gian pause hiện tại nếu đang pause
            if self._pause_start_time is not None:
                stats["current_pause_duration"] = time.time() - self._pause_start_time
            return stats

    def reset_statistics(self):
        """Reset thống kê."""
        with self._lock:
            self._stats = {
                "total_frames": 0,
                "forward_count": 0,
                "left_count": 0,
                "right_count": 0,
                "stop_count": 0,
                "lost_lane_count": 0,
                "pause_count": 0,
                "total_pause_time": 0.0
            }

    def set_base_speed(self, speed: int):
        """Đặt tốc độ cơ bản cho tính toán động."""
        self.base_speed = speed
        print(f"[CarControlAdvanced] Base speed set to: {self.base_speed}")

    def _set_command(self, command: Command, info: dict):
        """Override để thêm statistics và smoothing."""
        self._command_history.append(command)
        if len(self._command_history) > self._history_size:
            self._command_history.pop(0)

        smoothed_command = self._smooth_command(command)

        with self._lock:
            self._current_command = smoothed_command
            self._current_info = info
            self._last_update_time = time.time()

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
        """Làm mượt lệnh điều khiển."""
        if len(self._command_history) < 3:
            return new_command

        from collections import Counter
        counts = Counter(self._command_history[-3:])
        most_common = counts.most_common(1)[0][0]

        if new_command == Command.STOP:
            return Command.STOP

        if new_command == Command.FORWARD and most_common in [Command.LEFT, Command.RIGHT]:
            return new_command

        if most_common == Command.FORWARD and new_command in [Command.LEFT, Command.RIGHT]:
            recent_turns = sum(1 for c in self._command_history[-3:]
                               if c in [Command.LEFT, Command.RIGHT])
            if recent_turns >= 2:
                return new_command
            return Command.FORWARD

        return new_command

    # detect sign
    def _control_loop(self):
        """Vòng lặp điều khiển chính đã được nâng cấp"""
        while self._running:
            try:
                # 1. Lấy dữ liệu
                is_paused = self.is_paused()
                frame = self.stream_manager.get_latest_frame()

                if frame is None:
                    time.sleep(0.01)
                    continue

                # 2. Xử lý Làn đường (Luôn chạy để tính góc lái)
                processed_frame, lane_info = self.lane_nav.process_frame(frame, debug=self.enable_display)
                if self.enable_display:
                    self.latest_processed_frame = processed_frame

                # Nếu đang pause thủ công thì bỏ qua logic dưới
                if is_paused:
                    with self._lock:
                        self._current_info = lane_info
                    continue

                # Lấy biển báo mới nhất từ luồng YOLO
                # 3. LOGIC HỢP NHẤT: BIỂN BÁO + LÀN ĐƯỜNG
                # ====================================================

                current_sign = None
                if self.sign_detector:
                    current_sign = self.sign_detector.get_current_sign()

                # --- RESET KHI KHÔNG CÒN THẤY BIỂN NÀO ---
                if current_sign is None:
                    # Không còn thấy biển báo -> bỏ chế độ STOP do biển
                    self.force_stop_by_sign = False

                # --- XỬ LÝ TRẠNG THÁI TỪ BIỂN BÁO KHI CÓ current_sign ---
                else:
                    # Case 1: Gặp biển/đèn dừng (STOP, RED)
                    if current_sign in self.sign_detector.STOP_LABELS:
                        self.force_stop_by_sign = True
                        print(f"[SIGN] PHÁT HIỆN: {current_sign} -> DỪNG XE")

                    # Case 2: Gặp đèn xanh / biển CHO ĐI
                    elif current_sign in self.sign_detector.GO_LABELS:
                        self.force_stop_by_sign = False
                        print(f"[SIGN] PHÁT HIỆN: {current_sign} -> ĐI TIẾP")

                    # Case 3: Gặp biển TỐC ĐỘ
                    # elif current_sign in self.sign_detector.SPEED_LABELS:
                    #     self.override_speed = 110
                    #     # (Tuỳ bạn có dùng override_speed hay không)
                    #     print(f"[SIGN] PHÁT HIỆN: {current_sign} -> SET TỐC ĐỘ 110")

                    # Case 4: Biển khác (TURN LEFT/RIGHT, WARNING, …)
                    else:
                        # Ở đây mình cũng bỏ chế độ STOP, tuỳ ý bạn
                        self.force_stop_by_sign = False

                # --- RA QUYẾT ĐỊNH CUỐI CÙNG ---
                if self.force_stop_by_sign:
                    # Ưu tiên cao nhất: Dừng do biển báo
                    final_command = Command.STOP
                    lane_info['warning'] = f"STOP BY SIGN"
                else:
                    # Không bị dừng -> Lái theo làn đường
                    final_command = self._process_lane_info(lane_info)

                # Cập nhật lệnh xuống ESP32/Database
                self._set_command_with_sign_logic(final_command, self.speed, lane_info)

                # ====================================================
                time.sleep(0.0)


            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.1)

    def _set_command_with_sign_logic(self, command, speed, info):
        """Hàm cập nhật lệnh thay thế cho _set_command cũ để hỗ trợ speed tùy chỉnh"""
        # Logic smoothing lệnh giữ nguyên...
        smoothed_command = self._smooth_command(command)

        with self._lock:
            self._current_command = smoothed_command
            self.speed = speed  # Cập nhật tốc độ thực tế sẽ gửi đi
            self._current_info = info
            self._last_update_time = time.time()


# ========================================================
# VÍ DỤ SỬ DỤNG
# ========================================================

def main():
    """Ví dụ sử dụng CarControl với pause/resume."""
    from stream_manager import stream_manager
    from lane_navigator import LaneNavigator
    # 1. Khởi tạo
    stream_manager.start()

    lane_nav = LaneNavigator()
    lane_nav.load_config("lane_nav_config.json")

    # 2. Tạo CarControl
    car_control = CarControlAdvanced(
        stream_manager=stream_manager,
        lane_nav=lane_nav,
        base_speed=50,
        min_speed=20,
        max_speed=70
    )

    car_control.enable_display = True
    car_control.start()

    print("\n" + "=" * 50)
    print("           CAR CONTROL STARTED")
    print("=" * 50)
    print("  Q      : Quit")
    print("  SPACE  : Pause/Resume")
    print("  P      : Pause")
    print("  R      : Resume")
    print("  S      : Show statistics")
    print("  +/-    : Increase/Decrease speed")
    print("=" * 50 + "\n")

    try:
        while True:
            # Lấy lệnh điều khiển
            command, speed = car_control.get_command()

            # Lấy thông tin chi tiết
            detail = car_control.get_detailed_info()
            state = detail['state']

            # Hiển thị trạng thái
            state_str = f"[{state.upper():^8}]" if state == "paused" else f"[{state.upper():^8}]"
            print(f"\r{state_str} [CMD: {command:^10}] [SPD: {speed:3}] [FPS: {detail['fps']:2}]", end="")

            # Hiển thị frame
            if car_control.enable_display and car_control.latest_processed_frame is not None:
                frame = car_control.latest_processed_frame.copy()

                # Thêm overlay khi pause
                if car_control.is_paused():
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 100), -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                    # Text PAUSED
                    text = "PAUSED"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_size = cv2.getTextSize(text, font, 2, 3)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = (frame.shape[0] + text_size[1]) // 2
                    cv2.putText(frame, text, (text_x, text_y), font, 2, (0, 0, 255), 3)
                    cv2.putText(frame, "Press SPACE to resume", (text_x - 50, text_y + 50),
                                font, 0.7, (255, 255, 255), 2)

                # Thông tin command
                color = (0, 255, 255) if not car_control.is_paused() else (0, 0, 255)
                cv2.putText(frame, f"CMD: {command.upper()}", (10, frame.shape[0] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"SPEED: {speed}", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Trạng thái
                state_color = (0, 255, 0) if state == "running" else (
                    (0, 165, 255) if state == "paused" else (0, 0, 255))
                cv2.putText(frame, f"STATE: {state.upper()}", (frame.shape[1] - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

                cv2.imshow("Car Control View", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('i'):  # SPACE - Toggle pause
                is_paused = car_control.toggle_pause()
                print(f"\n{'PAUSED' if is_paused else 'RESUMED'}")
            elif key == ord('p'):  # P - Pause
                car_control.pause()
            elif key == ord('r'):  # R - Resume
                car_control.resume()
            elif key == ord('s'):  # S - Statistics
                stats = car_control.get_statistics()
                print(f"\n\n{'=' * 40}")
                print("           STATISTICS")
                print('=' * 40)
                print(f"State: {car_control.get_state().value}")
                print(f"Total frames: {stats['total_frames']}")
                print(
                    f"Forward: {stats['forward_count']} ({stats['forward_count'] / max(1, stats['total_frames']) * 100:.1f}%)")
                print(f"Left: {stats['left_count']} ({stats['left_count'] / max(1, stats['total_frames']) * 100:.1f}%)")
                print(
                    f"Right: {stats['right_count']} ({stats['right_count'] / max(1, stats['total_frames']) * 100:.1f}%)")
                print(f"Stop: {stats['stop_count']} ({stats['stop_count'] / max(1, stats['total_frames']) * 100:.1f}%)")
                print(f"Lost lane: {stats['lost_lane_count']}")
                print(f"Pause count: {stats['pause_count']}")
                print(f"Total pause time: {stats['total_pause_time']:.1f}s")
                print('=' * 40 + "\n")
            elif key == ord('+') or key == ord('='):
                car_control.set_speed(car_control.speed + 5)
            elif key == ord('-'):
                car_control.set_speed(car_control.speed - 5)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        car_control.stop()
        cv2.destroyAllWindows()
        print("\nCar control stopped.")


if __name__ == "__main__":
    main()

from .stream_manager import stream_manager
from .lane_navigator import lane_nav

car_control = CarControlAdvanced(stream_manager, lane_nav, base_speed=110, min_speed=100, max_speed=255)