import cv2
import numpy as np
import math


class LaneNavigator:
    def __init__(self, blind_spot_height=50):
        """
        Khởi tạo LaneNavigator.
        :param blind_spot_height: Chiều cao (pixel) tính từ đáy ảnh lên sẽ bị che đi (điểm mù).
        """
        self.roi_rect = None  # Lưu tọa độ vùng ROI (x, y, w, h)
        self.blind_spot_h = blind_spot_height

        # Các tham số điều khiển ngưỡng rẽ
        self.center_threshold = 20  # Độ lệch cho phép để coi là "Đi thẳng"

        # Lưu trữ trạng thái trước đó để làm mượt (smoothing)
        self.prev_left = None
        self.prev_right = None

    def select_roi_interactive(self, frame):
        """
        Hiển thị frame đầu tiên để người dùng chọn vùng quan tâm (ROI).
        """
        print(">>> Vui lòng dùng chuột chọn vùng đường (ROI) trên cửa sổ hiển thị, sau đó nhấn ENTER hoặc SPACE.")
        # cv2.selectROI trả về (x, y, w, h)
        self.roi_rect = cv2.selectROI("Chon ROI (Nhan Enter de xac nhan)", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Chon ROI (Nhan Enter de xac nhan)")
        print(f">>> Đã chọn ROI: {self.roi_rect}")

    def _mask_blind_spot(self, img):
        """Che vùng điểm mù trước mũi xe"""
        h, w = img.shape[:2]
        # Vẽ hình chữ nhật đen ở đáy ảnh
        cv2.rectangle(img, (0, h - self.blind_spot_h), (w, h), (0, 0, 0), -1)
        return img

    def _make_points(self, image, line_params):
        """Chuyển đổi slope và intercept thành tọa độ pixel"""
        if line_params is None:
            return None
        slope, intercept = line_params
        y1 = image.shape[0]  # Đáy ảnh
        y2 = int(y1 * (3 / 5))  # Vẽ line dài lên 3/5 chiều cao ảnh

        # y = mx + b -> x = (y - b) / m
        if slope == 0: slope = 0.001  # Tránh chia cho 0
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [[x1, y1, x2, y2]]

    def _average_slope_intercept(self, image, lines):
        """Tính trung bình các đoạn thẳng để tìm ra 1 line trái và 1 line phải"""
        left_fit = []
        right_fit = []

        if lines is None:
            return None, None

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # Fit polynomial degree 1 (đường thẳng): y = mx + b
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            # Phân loại trái phải dựa vào hệ số góc (slope)
            # Lưu ý: Trục y hướng xuống dưới nên slope âm là bên trái, dương là bên phải (tùy góc nhìn)
            # Với góc nhìn camera gắn xe thông thường:
            if slope < -0.4:  # Line trái (nghiêng về bên trái)
                left_fit.append((slope, intercept))
            elif slope > 0.4:  # Line phải (nghiêng về bên phải)
                right_fit.append((slope, intercept))

        # Tính trung bình
        left_fit_average = np.average(left_fit, axis=0) if left_fit else self.prev_left
        right_fit_average = np.average(right_fit, axis=0) if right_fit else self.prev_right

        # Cập nhật trạng thái cũ nếu tìm thấy line mới
        if left_fit: self.prev_left = left_fit_average
        if right_fit: self.prev_right = right_fit_average

        left_line = self._make_points(image, left_fit_average)
        right_line = self._make_points(image, right_fit_average)

        return left_line, right_line

    def process_frame(self, frame):
        """
        Hàm xử lý chính.
        Input: Frame hình ảnh gốc.
        Output: (decision_text, steering_angle, processed_frame)
        """
        # 1. Cắt vùng ROI (nếu đã chọn)
        if self.roi_rect is not None and self.roi_rect[2] > 0 and self.roi_rect[3] > 0:
            x, y, w, h = self.roi_rect
            roi = frame[y:y + h, x:x + w].copy()
        else:
            roi = frame.copy()

        # 2. Xử lý ảnh (Tiền xử lý)
        # Che điểm mù
        roi = self._mask_blind_spot(roi)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny Edge Detection
        canny = cv2.Canny(blur, 50, 150)

        # 3. Tìm đường thẳng (Hough Transform)
        lines = cv2.HoughLinesP(canny, 2, np.pi / 180, 50, np.array([]), minLineLength=20, maxLineGap=100)

        # 4. Tính toán làn đường trung bình
        left_line, right_line = self._average_slope_intercept(roi, lines)

        # 5. Vẽ lại lên ảnh để trực quan
        line_image = np.zeros_like(roi)
        lane_center_x = None
        roi_center_x = roi.shape[1] // 2

        valid_lines = []
        if left_line is not None:
            valid_lines.append(left_line)
        if right_line is not None:
            valid_lines.append(right_line)

        # Vẽ các line phát hiện được
        if valid_lines:
            for line in valid_lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Màu xanh lá

        # 6. Logic điều khiển (Steering)
        decision = "Stop/Lost"
        deviation = 0

        # Nếu thấy cả 2 line -> đi vào giữa
        if left_line is not None and right_line is not None:
            l_x1, _, l_x2, _ = left_line[0]
            r_x1, _, r_x2, _ = right_line[0]
            # Tính trung điểm tại đáy ảnh (nơi gần xe nhất)
            lane_center_x = int((l_x1 + r_x1) / 2)

        # Nếu chỉ thấy line trái -> bám theo bên trái
        elif left_line is not None:
            l_x1, _, l_x2, _ = left_line[0]
            lane_center_x = l_x1 + 250  # Giả sử độ rộng làn là 500px, lệch phải 250

        # Nếu chỉ thấy line phải -> bám theo bên phải
        elif right_line is not None:
            r_x1, _, r_x2, _ = right_line[0]
            lane_center_x = r_x1 - 250

        # Tính toán quyết định
        if lane_center_x is not None:
            cv2.circle(line_image, (lane_center_x, roi.shape[0]), 10, (0, 0, 255), -1)  # Chấm đỏ tâm làn
            deviation = lane_center_x - roi_center_x

            if abs(deviation) < self.center_threshold:
                decision = "DI THANG"
            elif deviation < 0:
                decision = f"RE TRAI ({abs(deviation)})"
            else:
                decision = f"RE PHAI ({abs(deviation)})"

        # Gộp ảnh visualize vào ảnh gốc
        combined_roi = cv2.addWeighted(roi, 0.8, line_image, 1, 1)

        # Đưa ROI trở lại khung hình lớn (để hiển thị full)
        final_frame = frame.copy()
        if self.roi_rect is not None:
            final_frame[y:y + h, x:x + w] = combined_roi
            # Vẽ khung chữ nhật ROI trên ảnh gốc
            cv2.rectangle(final_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        else:
            final_frame = combined_roi

        # Hiển thị thông số lên màn hình
        cv2.putText(final_frame, f"Decision: {decision}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(final_frame, f"Deviation: {deviation}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)

        return decision, deviation, final_frame


# --- HÀM MAIN ĐỂ CHẠY DEMO ---
def main():
    # Nguồn video: Có thể là đường dẫn file hoặc số (0) cho webcam
    # video_source = 0 # Webcam
    from stream_manager import stream_manager
    stream_manager.start()
    # CHẾ ĐỘ VIDEO/CAMERA
    navigator = LaneNavigator(blind_spot_height=60)
    first_frame = stream_manager.get_latest_frame()
    # Đọc frame đầu tiên để chọn ROI

    navigator.select_roi_interactive(first_frame)


    while True:
        frame = stream_manager.get_latest_frame()


        # Xử lý frame
        decision, deviation, result_frame = navigator.process_frame(frame)

        # Hiển thị
        cv2.imshow("Lane Navigator System", result_frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()