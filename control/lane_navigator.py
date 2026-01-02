import cv2
import numpy as np
import json


class LaneNavigator:
    def __init__(self):
        self.roi_points = []
        self.bev_src_points = None
        self.bev_dst_points = None
        self.M = None
        self.Minv = None

        # Thông số kỹ thuật (override cho track giấy 20cm, cam thấp ~13–15cm)
        # (ước lượng: ROI cao ~1/3 ảnh ~160 px)
        self.ym_per_pix = 0.20 / 160.0  # ~0.00125 m/pixel dọc
        self.xm_per_pix = 0.20 / 130.0  # lane 20cm ~130 px trong BEV

        # Thông số điều khiển - GIẢM NHẠY để xe ổn định hơn
        self.k_offset = 0.25  # trước là 1.0
        self.k_angle = 0.6  # giữ
        self.k_curvature = 0.1  # trước là 0.5

        self.steering_threshold = 0.2  # trước 0.35
        self.sharp_turn_threshold = 0.5

        self.prev_steering_score = 0
        self.smoothing_factor = 0.5  # trước 0.3 => mượt hơn

        # ===== THÔNG SỐ MỚI CHO XỬ LÝ 1 LÀN =====
        # Track giấy: đường rộng 20cm, trong BEV ~120–140 px
        self.standard_lane_width_pixels = 130  # trước 500, quá to
        self.lane_width_history = []
        self.max_history = 30

        # Lưu lại fit trước đó
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.frames_since_both_lanes = 0
        self.max_frames_without_both = 15

        # Track nhỏ -> hạ ngưỡng điểm tối thiểu
        self.min_lane_points = 50  # trước 100

    def load_config(self, filepath="lane_nav_config.json"):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.roi_points = data["roi_points"]
            self.bev_src_points = np.array(data["bev_src_points"], dtype=np.float32)
            self.bev_dst_points = np.array(data["bev_dst_points"], dtype=np.float32)
            self.M = np.array(data["M"], dtype=np.float32)
            self.Minv = np.array(data["Minv"], dtype=np.float32)
            self.standard_lane_width_pixels = data["standard_lane_width_pixels"]
            # Load lane width nếu có

            if "standard_lane_width" in data:
                self.standard_lane_width_pixels = data["standard_lane_width"]
        print("Cấu hình đã được tải từ", filepath)

    def save_config(self):
        json_str = json.dumps({
            "roi_points": self.roi_points,
            "bev_src_points": self.bev_src_points.tolist(),
            "bev_dst_points": self.bev_dst_points.tolist(),
            "M": self.M.tolist(),
            "Minv": self.Minv.tolist(),
            # "ym_per_pix": self.ym_per_pix,
            # "xm_per_pix": self.xm_per_pix,
            # "k_offset": self.k_offset,
            # "k_angle": self.k_angle,
            # "k_curvature": self.k_curvature,
            # "steering_threshold": self.steering_threshold,
            "standard_lane_width_pixels": self.standard_lane_width_pixels,
            "standard_lane_width": self.standard_lane_width_pixels
        })
        with open("lane_nav_config.json", "w") as f:
            f.write(json_str)
        print("Cấu hình đã được lưu vào lane_nav_config.json")

    def select_points_interactive(self, frame):
        """Hàm mở cửa sổ để người dùng click chọn 4 điểm ROI/BEV."""
        print("HƯỚNG DẪN: Click 4 điểm theo thứ tự:\n1. Dưới-Trái\n2. Dưới-Phải\n3. Trên-Phải\n4. Trên-Trái")

        temp_img = frame.copy()
        self.roi_points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.roi_points) < 4:
                    self.roi_points.append((x, y))
                    cv2.circle(temp_img, (x, y), 5, (0, 0, 255), -1)
                    if len(self.roi_points) > 1:
                        cv2.line(temp_img, self.roi_points[-2], self.roi_points[-1], (0, 255, 0), 2)
                    if len(self.roi_points) == 4:
                        cv2.line(temp_img, self.roi_points[-1], self.roi_points[0], (0, 255, 0), 2)
                    cv2.imshow("Config ROI", temp_img)

        cv2.namedWindow("Config ROI")
        cv2.setMouseCallback("Config ROI", mouse_callback)
        cv2.imshow("Config ROI", temp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(self.roi_points) != 4:
            raise Exception("Bạn chưa chọn đủ 4 điểm!")

        h, w = frame.shape[:2]
        self.bev_src_points = np.float32(self.roi_points)
        offset = 100
        self.bev_dst_points = np.float32([
            [offset, h], [w - offset, h], [w - offset, 0], [offset, 0]
        ])

        self.M = cv2.getPerspectiveTransform(self.bev_src_points, self.bev_dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.bev_dst_points, self.bev_src_points)

        # Ước tính độ rộng làn ban đầu từ ROI
        bottom_width = abs(self.roi_points[1][0] - self.roi_points[0][0])
        self.standard_lane_width_pixels = int(bottom_width * 0.8)

        print(f"Cấu hình hoàn tất. Độ rộng làn ước tính: {self.standard_lane_width_pixels}px")
        self.save_config()

    # ==== ROI & BEV AUTO CHO GÓC CAM NÀY ====

    def _compute_auto_roi_points(self, h, w):
        """
        ROI tự động:
        - 1/3 ảnh từ dưới lên
        - 5/7 bề ngang tính từ giữa ảnh
        """
        y_bottom = h - 1
        y_top = int(h * (2.0 / 3.0))  # giữ 1/3 dưới

        roi_width = int(w * 5.0 / 7.0)
        half = roi_width // 2
        cx = w // 2
        x_left = max(0, cx - half)
        x_right = min(w - 1, cx + half)

        return np.array([
            [x_left, y_bottom],
            [x_right, y_bottom],
            [x_right, y_top],
            [x_left, y_top]
        ], dtype=np.float32)

    def _ensure_auto_bev(self, frame_shape):
        """Khởi tạo src/dst cho BEV dựa trên ROI tự động (ghi đè config cũ)."""
        h, w = frame_shape[:2]
        roi_pts = self._compute_auto_roi_points(h, w)

        # Lưu lại ROI để debug
        self.roi_points = roi_pts.tolist()
        self.bev_src_points = roi_pts

        # Destination: hình chữ nhật gần full frame, chừa mép 15%
        offset = int(w * 0.15)
        self.bev_dst_points = np.float32([
            [offset, h],
            [w - offset, h],
            [w - offset, 0],
            [offset, 0]
        ])

        self.M = cv2.getPerspectiveTransform(self.bev_src_points, self.bev_dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.bev_dst_points, self.bev_src_points)

    def preprocess_advanced(self, img):
        """
        Tiền xử lý cho track giấy trắng + băng keo đen (~1.5–2cm):

        - ROI: 1/3 ảnh phía dưới + 5/7 bề ngang quanh tâm.
        - HLS, lấy kênh L (Lightness).
        - Tìm pixel TỐI (vạch đen) + cạnh (Sobel).
        - Áp ROI mask: ngoài vùng chạy = 0.
        - Trả về ảnh nhị phân 0/1.
        """
        h, w = img.shape[:2]

        # 1. ROI mask
        roi_poly = self._compute_auto_roi_points(h, w).astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [roi_poly], 1)

        # 2. HLS -> kênh L
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]

        # 3. Gradient theo x (giữ logic cũ, nhưng ngưỡng mềm hơn)
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-6))

        sxbinary = np.zeros_like(scaled_sobel, dtype=np.uint8)
        sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 150)] = 1

        # 4. Ngưỡng L thấp -> vùng ĐEN
        l_binary = np.zeros_like(l_channel, dtype=np.uint8)
        # nếu vạch hơi xám thì có thể tăng 80 -> 100
        l_binary[l_channel < 60] = 1

        # 5. Kết hợp & áp ROI
        combined_binary = np.zeros_like(sxbinary, dtype=np.uint8)
        combined_binary[(sxbinary == 1) | (l_binary == 1)] = 1

        # Chỉ giữ trong ROI
        combined_binary = combined_binary * mask

        return combined_binary

    # ================================================================
    # =============== PHẦN MỚI: PHÁT HIỆN LÀN CẢI TIẾN ===============
    # ================================================================

    def detect_lanes_sliding_window(self, binary_warped):
        """
        Tìm làn đường bằng phương pháp cửa sổ trượt.
        CẢI TIẾN: Trả về thông tin chi tiết về việc phát hiện được bao nhiêu làn.

        Returns:
            left_fit: Hệ số đa thức làn trái (hoặc None)
            right_fit: Hệ số đa thức làn phải (hoặc None)
            lane_data: Tuple (leftx, lefty, rightx, righty)
            detection_info: Dict chứa thông tin phát hiện
        """
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = np.int64(histogram.shape[0] / 2)

        # Tìm đỉnh histogram cho mỗi nửa
        left_half = histogram[:midpoint]
        right_half = histogram[midpoint:]

        leftx_base = np.argmax(left_half)
        rightx_base = np.argmax(right_half) + midpoint

        # Kiểm tra xem có đủ tín hiệu không
        left_peak = left_half[leftx_base] if len(left_half) > 0 else 0
        right_peak = right_half[rightx_base - midpoint] if len(right_half) > 0 else 0

        # Ngưỡng tối thiểu cho peak
        min_peak_threshold = 20

        nwindows = 9
        window_height = np.int64(binary_warped.shape[0] / nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 60
        minpix = 30
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            # Cửa sổ làn trái
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin

            # Cửa sổ làn phải
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Xác định làn nào được phát hiện
        left_detected = len(left_lane_inds) >= self.min_lane_points and left_peak >= min_peak_threshold
        right_detected = len(right_lane_inds) >= self.min_lane_points and right_peak >= min_peak_threshold

        left_fit = None
        right_fit = None
        leftx, lefty, rightx, righty = [], [], [], []

        if left_detected:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            left_fit = np.polyfit(lefty, leftx, 2)

        if right_detected:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            right_fit = np.polyfit(righty, rightx, 2)

        detection_info = {
            "left_detected": left_detected,
            "right_detected": right_detected,
            "left_points": len(left_lane_inds),
            "right_points": len(right_lane_inds),
            "left_peak": left_peak,
            "right_peak": right_peak,
            "both_detected": left_detected and right_detected,
            "single_lane": (left_detected and not right_detected) or (not left_detected and right_detected),
            "no_lane": not left_detected and not right_detected
        }

        return left_fit, right_fit, (leftx, lefty, rightx, righty), detection_info

    def estimate_missing_lane(self, detected_fit, is_left_detected, img_height):
        """
        Ước tính làn đường còn thiếu dựa trên làn đã phát hiện và độ rộng làn chuẩn.

        Args:
            detected_fit: Hệ số đa thức của làn đã phát hiện
            is_left_detected: True nếu làn đã phát hiện là làn trái
            img_height: Chiều cao ảnh

        Returns:
            estimated_fit: Hệ số đa thức ước tính cho làn còn thiếu
        """
        # Tạo các điểm y
        ploty = np.linspace(0, img_height - 1, img_height)

        # Tính x của làn đã phát hiện
        detected_x = detected_fit[0] * ploty ** 2 + detected_fit[1] * ploty + detected_fit[2]

        # Dịch chuyển theo độ rộng làn
        if is_left_detected:
            # Làn trái đã phát hiện -> ước tính làn phải
            estimated_x = detected_x + self.standard_lane_width_pixels
        else:
            # Làn phải đã phát hiện -> ước tính làn trái
            estimated_x = detected_x - self.standard_lane_width_pixels

        # Fit lại đa thức cho làn ước tính
        estimated_fit = np.polyfit(ploty, estimated_x, 2)

        return estimated_fit

    def update_lane_width(self, left_fit, right_fit, img_height):
        """Cập nhật độ rộng làn đường chuẩn dựa trên làn phát hiện được."""
        y_bottom = img_height - 1
        left_x = left_fit[0] * y_bottom ** 2 + left_fit[1] * y_bottom + left_fit[2]
        right_x = right_fit[0] * y_bottom ** 2 + right_fit[1] * y_bottom + right_fit[2]

        current_width = right_x - left_x

        # Chỉ cập nhật nếu độ rộng hợp lý (200-800 pixels)
        if 250 < current_width < 450:
            self.lane_width_history.append(current_width)
            if len(self.lane_width_history) > self.max_history:
                self.lane_width_history.pop(0)

            # Cập nhật độ rộng chuẩn bằng trung bình
            self.standard_lane_width_pixels = int(np.mean(self.lane_width_history))

    def get_lanes_with_estimation(self, binary_warped, img_shape):
        """
        Phát hiện làn đường với khả năng ước tính làn bị mất.

        Returns:
            left_fit, right_fit: Hệ số đa thức (thực tế hoặc ước tính)
            lane_status: Dict chứa trạng thái chi tiết
        """
        h = img_shape[0]

        # Phát hiện làn đường
        left_fit, right_fit, lane_data, detection_info = self.detect_lanes_sliding_window(binary_warped)

        lane_status = {
            "detection": detection_info,
            "left_source": "none",
            "right_source": "none",
            "confidence": 0.0,
            "warning": None
        }

        # === TRƯỜNG HỢP 1: CẢ HAI LÀN ĐƯỢC PHÁT HIỆN ===
        if detection_info["both_detected"]:
            # Cập nhật độ rộng làn chuẩn
            self.update_lane_width(left_fit, right_fit, h)

            # Lưu lại fit để dùng sau
            self.prev_left_fit = left_fit.copy()
            self.prev_right_fit = right_fit.copy()
            self.frames_since_both_lanes = 0

            lane_status["left_source"] = "detected"
            lane_status["right_source"] = "detected"
            lane_status["confidence"] = 1.0

            return left_fit, right_fit, lane_status

        # === TRƯỜNG HỢP 2: CHỈ CÓ LÀN TRÁI ===
        elif detection_info["left_detected"] and not detection_info["right_detected"]:
            self.frames_since_both_lanes += 1
            self.prev_left_fit = left_fit.copy()

            lane_status["left_source"] = "detected"
            lane_status["warning"] = "RIGHT_LANE_MISSING"

            # Ước tính làn phải
            if self.frames_since_both_lanes <= self.max_frames_without_both:
                if self.prev_right_fit is not None:
                    # Sử dụng fit trước đó với điều chỉnh
                    right_fit = self._blend_with_previous(
                        self.estimate_missing_lane(left_fit, True, h),
                        self.prev_right_fit,
                        self.frames_since_both_lanes
                    )
                    lane_status["right_source"] = "estimated_from_previous"
                else:
                    right_fit = self.estimate_missing_lane(left_fit, True, h)
                    lane_status["right_source"] = "estimated_from_left"

                lane_status["confidence"] = max(0.3, 1.0 - self.frames_since_both_lanes * 0.05)
            else:
                right_fit = self.estimate_missing_lane(left_fit, True, h)
                lane_status["right_source"] = "estimated_from_left"
                lane_status["confidence"] = 0.3

            return left_fit, right_fit, lane_status

        # === TRƯỜNG HỢP 3: CHỈ CÓ LÀN PHẢI ===
        elif detection_info["right_detected"] and not detection_info["left_detected"]:
            self.frames_since_both_lanes += 1
            self.prev_right_fit = right_fit.copy()

            lane_status["right_source"] = "detected"
            lane_status["warning"] = "LEFT_LANE_MISSING"

            # Ước tính làn trái
            if self.frames_since_both_lanes <= self.max_frames_without_both:
                if self.prev_left_fit is not None:
                    left_fit = self._blend_with_previous(
                        self.estimate_missing_lane(right_fit, False, h),
                        self.prev_left_fit,
                        self.frames_since_both_lanes
                    )
                    lane_status["left_source"] = "estimated_from_previous"
                else:
                    left_fit = self.estimate_missing_lane(right_fit, False, h)
                    lane_status["left_source"] = "estimated_from_right"

                lane_status["confidence"] = max(0.3, 1.0 - self.frames_since_both_lanes * 0.05)
            else:
                left_fit = self.estimate_missing_lane(right_fit, False, h)
                lane_status["left_source"] = "estimated_from_right"
                lane_status["confidence"] = 0.3

            return left_fit, right_fit, lane_status

        # === TRƯỜNG HỢP 4: KHÔNG PHÁT HIỆN ĐƯỢC LÀN NÀO ===
        else:
            self.frames_since_both_lanes += 1
            lane_status["warning"] = "NO_LANES_DETECTED"

            # Sử dụng fit trước đó nếu có và còn trong thời hạn
            if (self.frames_since_both_lanes <= self.max_frames_without_both and
                    self.prev_left_fit is not None and self.prev_right_fit is not None):
                left_fit = self.prev_left_fit
                right_fit = self.prev_right_fit
                lane_status["left_source"] = "previous"
                lane_status["right_source"] = "previous"
                lane_status["confidence"] = max(0.1, 0.5 - self.frames_since_both_lanes * 0.03)

                return left_fit, right_fit, lane_status

            # Hoàn toàn mất dấu
            lane_status["confidence"] = 0.0
            return None, None, lane_status

    def _blend_with_previous(self, estimated_fit, previous_fit, frames_elapsed):
        """Kết hợp fit ước tính với fit trước đó để mượt hơn."""
        # Trọng số cho fit trước đó giảm dần theo thời gian
        prev_weight = max(0.2, 1.0 - frames_elapsed * 0.1)
        est_weight = 1.0 - prev_weight

        blended_fit = prev_weight * previous_fit + est_weight * estimated_fit
        return blended_fit

    # ================================================================
    # =============== PHÂN TÍCH HÌNH HỌC LÀN ĐƯỜNG ===================
    # ================================================================

    def analyze_lane_geometry(self, img_shape, left_fit, right_fit):
        """Phân tích hình học làn đường chi tiết."""
        h, w = img_shape[:2]

        y_bottom = h - 1
        y_middle = int(h * 0.6)
        y_ahead = int(h * 0.3)

        def get_lane_center(y):
            left_x = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
            right_x = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
            return (left_x + right_x) / 2, left_x, right_x

        center_bottom, left_bottom, right_bottom = get_lane_center(y_bottom)
        center_middle, left_middle, right_middle = get_lane_center(y_middle)
        center_ahead, left_ahead, right_ahead = get_lane_center(y_ahead)

        # Offset
        image_center = w / 2
        offset_pixels = image_center - center_bottom
        offset_meters = offset_pixels * self.xm_per_pix

        # Góc nghiêng
        dx = center_ahead - center_bottom
        dy = y_bottom - y_ahead
        lane_angle_deg = np.degrees(np.arctan2(dx, dy))

        dx_near = center_middle - center_bottom
        dy_near = y_bottom - y_middle
        lane_angle_near = np.degrees(np.arctan2(dx_near, dy_near))

        # Độ cong
        avg_a = (left_fit[0] + right_fit[0]) / 2
        avg_b = (left_fit[1] + right_fit[1]) / 2

        curvature_strength = abs(avg_a) * 100000
        curvature_direction = -np.sign(avg_a) if abs(avg_a) > 1e-6 else 0

        y_eval = y_bottom
        left_deriv = 2 * left_fit[0] * y_eval + left_fit[1]
        right_deriv = 2 * right_fit[0] * y_eval + right_fit[1]

        left_curverad = ((1 + left_deriv ** 2) ** 1.5) / (np.abs(2 * left_fit[0]) + 1e-6)
        right_curverad = ((1 + right_deriv ** 2) ** 1.5) / (np.abs(2 * right_fit[0]) + 1e-6)
        avg_radius = (left_curverad + right_curverad) / 2 * self.xm_per_pix / self.ym_per_pix

        lane_width_bottom = right_bottom - left_bottom
        lane_width_ahead = right_ahead - left_ahead

        return {
            "center_bottom": center_bottom,
            "center_middle": center_middle,
            "center_ahead": center_ahead,
            "image_center": image_center,
            "offset_pixels": offset_pixels,
            "offset_meters": offset_meters,
            "lane_angle_deg": lane_angle_deg,
            "lane_angle_near": lane_angle_near,
            "curvature_strength": curvature_strength,
            "curvature_direction": curvature_direction,
            "radius_meters": avg_radius,
            "lane_width_bottom": lane_width_bottom,
            "lane_width_ahead": lane_width_ahead,
            "poly_coef_a": avg_a,
            "poly_coef_b": avg_b
        }

    def calculate_steering_decision(self, lane_geometry, lane_status):
        """
        Tính toán quyết định điều khiển.

        - 2 làn thật: offset + góc + cong (như cũ)
        - 1 làn (một lane bị mất, lane kia ước lượng): bỏ offset,
          chỉ dùng góc gần + cong cùng chiều với góc.
        """
        offset = lane_geometry["offset_meters"]
        angle = lane_geometry["lane_angle_deg"]
        angle_near = lane_geometry["lane_angle_near"]
        curve_dir = lane_geometry["curvature_direction"]
        curve_strength = lane_geometry["curvature_strength"]
        radius = lane_geometry["radius_meters"]

        # Độ tin cậy & trạng thái làn
        confidence = lane_status.get("confidence", 1.0)
        warning = lane_status.get("warning")
        detection = lane_status.get("detection", {})
        left_src = lane_status.get("left_source", "none")
        right_src = lane_status.get("right_source", "none")

        # 2 làn thật, tin cậy
        both_lanes_reliable = (
                detection.get("both_detected", False)
                and warning is None
                and left_src == "detected"
                and right_src == "detected"
                and confidence >= 0.6
        )
        # Chỉ còn 1 làn (bên kia mất/ước lượng)
        single_lane_mode = (
                detection.get("single_lane", False)
                or warning in ("LEFT_LANE_MISSING", "RIGHT_LANE_MISSING")
        )

        # Trọng số cơ bản theo confidence
        effective_k_offset = self.k_offset * confidence
        effective_k_angle = self.k_angle * confidence
        effective_k_curvature = self.k_curvature * confidence

        # Nếu chỉ 1 làn: bỏ offset, tăng vai trò góc + cong
        if single_lane_mode:
            effective_k_offset *= 0.2
            effective_k_angle *= 1.2
            effective_k_curvature *= 1.5

        # ===== TÍNH CÁC THÀNH PHẦN ĐÓNG GÓP =====
        offset_contribution = 0.0
        angle_contribution = 0.0
        curve_contribution = 0.0

        # 2 làn thật hoặc mode "bình thường" -> logic cũ
        if both_lanes_reliable or not single_lane_mode:
            # Offset: offset > 0 => làn lệch trái => cần quẹo trái => dấu "-"
            offset_norm = np.clip(offset / 0.5, -1, 1)
            offset_contribution = -effective_k_offset * offset_norm

            # Góc: kết hợp xa + gần, ưu tiên góc gần
            angle_norm = np.clip(angle / 25.0, -1, 1)
            angle_near_norm = np.clip(angle_near / 30.0, -1, 1)
            angle_combined = 0.4 * angle_norm + 0.6 * angle_near_norm
            angle_contribution = effective_k_angle * angle_combined

            # Độ cong từ coef a
            curve_norm = curve_dir * min(curve_strength / 50.0, 1.0)
            curve_contribution = effective_k_curvature * curve_norm

        # ===== CHẾ ĐỘ CHỈ 1 LÀN =====
        else:
            # Chỉ tin góc gần (ROI ~ 20cm trước xe)
            angle_near_norm = np.clip(angle_near / 25.0, -1, 1)
            angle_contribution = effective_k_angle * angle_near_norm

            # Hướng cong: cho cùng chiều với góc gần
            if abs(angle_near_norm) > 0.05:
                curve_dir_single = np.sign(angle_near_norm)
            else:
                # Nếu góc rất nhỏ thì giữ hướng như frame trước (nếu có)
                curve_dir_single = np.sign(self.prev_steering_score) if self.prev_steering_score != 0 else 0.0

            # Độ mạnh cong: scale theo độ lớn góc gần (0..1)
            curve_strength_single = min(abs(angle_near) / 20.0, 1.0)
            curve_contribution = effective_k_curvature * curve_dir_single * curve_strength_single

        # ===== TỔNG HỢP =====
        raw_steering_score = offset_contribution + angle_contribution + curve_contribution

        # Van an toàn: trong chế độ 1 làn, không cho đổi dấu "gấp" khi tín hiệu yếu
        if single_lane_mode and self.prev_steering_score != 0:
            if raw_steering_score * self.prev_steering_score < 0 and abs(raw_steering_score) < 0.6:
                raw_steering_score = np.sign(self.prev_steering_score) * abs(raw_steering_score)

        # Làm mượt - tăng smoothing khi confidence thấp
        # Làm mượt - tăng smoothing khi confidence thấp (0.7–0.95)
        base = self.smoothing_factor  # vd: 0.9
        effective_smoothing = np.clip(
            base + (1 - confidence) * 0.05,  # confidence thấp -> nhích thêm một chút
            0.3,  # tối thiểu
            0.9  # tối đa để khỏi quá ì
        )

        # Nếu hướng raw khác hướng trước đó và raw khá mạnh -> tin tín hiệu mới hơn
        if self.prev_steering_score != 0 and raw_steering_score * self.prev_steering_score < 0 \
                and abs(raw_steering_score) > 0.3:
            effective_smoothing = min(effective_smoothing, 0.3)

        steering_score = (effective_smoothing * self.prev_steering_score +
                          (1 - effective_smoothing) * raw_steering_score)

        self.prev_steering_score = steering_score

        # Quyết định hành động
        abs_score = abs(steering_score)

        if abs_score < self.steering_threshold:
            action = "GO STRAIGHT"
            action_code = 0
            direction = "center"
        elif steering_score > 0:
            if abs_score > self.sharp_turn_threshold:
                action = "SHARP RIGHT >>>"
                action_code = 2
            else:
                action = "TURN RIGHT -->"
                action_code = 1
            direction = "right"
        else:
            if abs_score > self.sharp_turn_threshold:
                action = "<<< SHARP LEFT"
                action_code = -2
            else:
                action = "<-- TURN LEFT"
                action_code = -1
            direction = "left"

        # Gắn nhãn ước tính nếu confidence thấp
        if confidence < 0.7:
            action = f"[EST] {action}"

        intensity = min(abs_score / 1.0, 1.0) * 100
        suggested_steering_angle = np.clip(steering_score * 45, -45, 45)

        return {
            "action": action,
            "action_code": action_code,
            "direction": direction,
            "steering_score": steering_score,
            "intensity_percent": intensity,
            "suggested_angle": suggested_steering_angle,
            "confidence": confidence,
            "contributions": {
                "offset": offset_contribution,
                "angle": angle_contribution,
                "curvature": curve_contribution,
            },
            "raw_data": {
                "offset_m": offset,
                "lane_angle": angle,
                "lane_angle_near": angle_near,
                "curve_direction": "left" if curve_dir < 0 else ("right" if curve_dir > 0 else "straight"),
                "radius_m": radius,
                "mode": (
                    "two_lanes" if both_lanes_reliable
                    else ("single_lane" if single_lane_mode else "mixed")
                ),
                "warning": warning,
            },
        }

    # ================================================================
    # =============== PROCESS FRAME CHÍNH ============================
    # ================================================================

    def process_frame(self, frame, debug=False):
        """Hàm xử lý chính cho 1 frame ảnh."""
        self._ensure_auto_bev(frame.shape)

        if self.M is None:
            raise Exception("Vui lòng chạy select_points_interactive() hoặc load_config() trước!")

        # 1. Preprocessing & Warp
        binary_img = self.preprocess_advanced(frame)
        img_size = (frame.shape[1], frame.shape[0])
        warped = cv2.warpPerspective(binary_img, self.M, img_size, flags=cv2.INTER_LINEAR)

        # 2. Detect Lanes với ước tính (CẢI TIẾN)
        left_fit, right_fit, lane_status = self.get_lanes_with_estimation(warped, frame.shape)

        # 3. Xử lý khi không phát hiện được làn
        if left_fit is None or right_fit is None:
            self.prev_steering_score = 0
            return frame, {
                "status": "LOST_LANE",
                "action": "!!! SLOW DOWN !!!",
                "action_code": 99,
                "lane_status": lane_status
            }

        # 4. Phân tích hình học
        lane_geometry = self.analyze_lane_geometry(frame.shape, left_fit, right_fit)

        # 5. Tính toán quyết định
        control_decision = self.calculate_steering_decision(lane_geometry, lane_status)

        # 6. Vẽ lên ảnh
        result = self._draw_lanes(frame, warped, left_fit, right_fit,
                                  lane_geometry, control_decision, lane_status)

        # DEBUG MODE
        if debug:
            self._draw_debug_view(result, warped, lane_geometry, lane_status)

        # 7. Output
        output_info = {
            "status": "TRACKING",
            **control_decision,
            "lane_status": lane_status,
            "geometry": lane_geometry
        }

        return result, output_info

    def _draw_lanes(self, frame, warped, left_fit, right_fit, geometry, decision, lane_status):
        """Vẽ làn đường lên ảnh."""
        h, w = frame.shape[:2]

        ploty = np.linspace(0, h - 1, h)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Màu vùng làn theo hướng và confidence
        confidence = lane_status.get("confidence", 1.0)

        if decision["direction"] == "left":
            base_color = (255, 100, 0)  # Xanh dương
        elif decision["direction"] == "right":
            base_color = (0, 100, 255)  # Cam
        else:
            base_color = (0, 255, 0)  # Xanh lá

        # Giảm độ sáng nếu confidence thấp
        lane_color = tuple(int(c * confidence) for c in base_color)

        cv2.fillPoly(color_warp, np.int_([pts]), lane_color)

        # Vẽ đường làn với màu khác nhau cho detected vs estimated
        left_color = (0, 255, 0) if lane_status["left_source"] == "detected" else (0, 255, 255)  # Vàng nếu ước tính
        right_color = (0, 255, 0) if lane_status["right_source"] == "detected" else (0, 255, 255)

        for i in range(len(ploty) - 1):
            pt1_l = (int(left_fitx[i]), int(ploty[i]))
            pt2_l = (int(left_fitx[i + 1]), int(ploty[i + 1]))
            cv2.line(color_warp, pt1_l, pt2_l, left_color, 3)

            pt1_r = (int(right_fitx[i]), int(ploty[i]))
            pt2_r = (int(right_fitx[i + 1]), int(ploty[i + 1]))
            cv2.line(color_warp, pt1_r, pt2_r, right_color, 3)

        # Vẽ đường tâm
        center_fitx = (left_fitx + right_fitx) / 2
        for i in range(0, len(ploty) - 10, 10):
            pt1 = (int(center_fitx[i]), int(ploty[i]))
            pt2 = (int(center_fitx[i + 10]), int(ploty[i + 10]))
            cv2.line(color_warp, pt1, pt2, (255, 255, 0), 3)

        # Unwarp
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (w, h))
        result = cv2.addWeighted(frame, 1, newwarp, 0.4, 0)

        # Vẽ info panel
        self._draw_info_panel(result, geometry, decision, lane_status)

        return result

    def _draw_info_panel(self, img, geometry, decision, lane_status):
        """Vẽ bảng thông tin điều khiển."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = img.shape[:2]

        # Background panel
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (420, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # Màu theo hướng
        if decision["direction"] == "left":
            cmd_color = (255, 150, 0)
        elif decision["direction"] == "right":
            cmd_color = (0, 150, 255)
        else:
            cmd_color = (0, 255, 0)

        y_offset = 35
        line_height = 26

        # === LANE STATUS ===
        confidence = lane_status.get("confidence", 1.0)
        warning = lane_status.get("warning")

        # Hiển thị trạng thái phát hiện làn
        left_src = lane_status.get("left_source", "none")
        right_src = lane_status.get("right_source", "none")

        status_text = f"L:{left_src[:3].upper()} | R:{right_src[:3].upper()} | Conf:{confidence:.0%}"
        status_color = (0, 255, 0) if confidence > 0.7 else ((0, 255, 255) if confidence > 0.3 else (0, 0, 255))
        cv2.putText(img, status_text, (20, y_offset), font, 0.5, status_color, 1)
        y_offset += line_height

        # Cảnh báo nếu có
        if warning:
            warning_color = (0, 165, 255)  # Cam
            if warning == "NO_LANES_DETECTED":
                warning_color = (0, 0, 255)  # Đỏ
            cv2.putText(img, f"! {warning}", (20, y_offset), font, 0.5, warning_color, 2)
            y_offset += line_height

        # === LỆNH ĐIỀU KHIỂN ===
        cv2.putText(img, f"CMD: {decision['action']}", (20, y_offset),
                    font, 0.7, cmd_color, 2)
        y_offset += line_height + 5

        # Thông số chi tiết
        cv2.putText(img, f"Offset: {geometry['offset_meters']:+.2f}m", (20, y_offset),
                    font, 0.55, (255, 255, 255), 1)
        y_offset += line_height

        cv2.putText(img, f"Lane Angle: {geometry['lane_angle_deg']:+.1f}deg", (20, y_offset),
                    font, 0.55, (255, 255, 255), 1)
        y_offset += line_height

        curve_text = geometry['curvature_direction']
        curve_str = "LEFT" if curve_text == -1 else ("RIGHT" if curve_text == 1 else "STRAIGHT")
        cv2.putText(img, f"Curve: {curve_str} (R={geometry['radius_meters']:.0f}m)", (20, y_offset),
                    font, 0.55, (255, 255, 255), 1)
        y_offset += line_height

        cv2.putText(img, f"Lane Width: {self.standard_lane_width_pixels}px", (20, y_offset),
                    font, 0.55, (200, 200, 200), 1)
        y_offset += line_height

        cv2.putText(img, f"Steer Score: {decision['steering_score']:+.2f}", (20, y_offset),
                    font, 0.55, (255, 255, 255), 1)
        y_offset += line_height

        # Steering bar
        bar_center = 210
        bar_width = 160
        bar_y = y_offset
        score = decision['steering_score']

        cv2.rectangle(img, (bar_center - bar_width // 2, bar_y),
                      (bar_center + bar_width // 2, bar_y + 15), (100, 100, 100), -1)
        cv2.line(img, (bar_center, bar_y), (bar_center, bar_y + 15), (255, 255, 255), 2)

        indicator_x = int(bar_center + score * (bar_width // 2))
        indicator_x = np.clip(indicator_x, bar_center - bar_width // 2, bar_center + bar_width // 2)
        cv2.circle(img, (indicator_x, bar_y + 7), 8, cmd_color, -1)

    def _draw_debug_view(self, result, warped, geometry, lane_status):
        """Vẽ view debug."""
        h, w = result.shape[:2]

        # BEV view
        debug_view = np.dstack((warped, warped, warped)) * 255
        debug_view = debug_view.astype(np.uint8)

        # Vẽ điểm tâm
        scale_y = warped.shape[0] / h
        cv2.circle(debug_view, (int(geometry['center_bottom']), int((h - 1) * scale_y)),
                   8, (0, 0, 255), -1)
        cv2.circle(debug_view, (int(geometry['center_ahead']), int(h * 0.3 * scale_y)),
                   8, (0, 255, 0), -1)
        cv2.circle(debug_view, (int(geometry['image_center']), int((h - 1) * scale_y)),
                   8, (255, 0, 0), -1)

        debug_view = cv2.resize(debug_view, (320, 180))
        result[0:180, w - 320:w] = debug_view

        # Legend
        cv2.putText(result, "RED: Lane Center", (w - 310, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(result, "BLUE: Image Center", (w - 310, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Detection info
        det = lane_status.get("detection", {})
        cv2.putText(result, f"L:{det.get('left_points', 0)} R:{det.get('right_points', 0)}",
                    (w - 310, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def set_control_weights(self, k_offset=None, k_angle=None, k_curvature=None):
        """Điều chỉnh trọng số."""
        if k_offset is not None:
            self.k_offset = k_offset
        if k_angle is not None:
            self.k_angle = k_angle
        if k_curvature is not None:
            self.k_curvature = k_curvature
        print(f"Weights: offset={self.k_offset}, angle={self.k_angle}, curve={self.k_curvature}")

    def set_lane_width(self, width_pixels):
        """Đặt độ rộng làn đường chuẩn thủ công."""
        self.standard_lane_width_pixels = width_pixels
        print(f"Lane width set to: {width_pixels}px")


# ========================================================
# MAIN PROGRAM
# ========================================================
def main():
    from stream_manager import stream_manager

    stream_manager.start()
    lane_nav = LaneNavigator()

    first_frame = stream_manager.get_latest_frame()

    # Chọn cấu hình
    lane_nav.select_points_interactive(first_frame)
    # lane_nav.load_config("lane_nav_config.json")

    print("\n=== ĐIỀU KHIỂN ===")
    print("Q: Thoát")
    print("1/2/3: Tăng trọng số Offset/Angle/Curve")
    print("W: Tăng độ rộng làn | S: Giảm độ rộng làn")
    print("==================\n")

    while True:
        frame = stream_manager.get_latest_frame()

        try:
            processed_frame, info = lane_nav.process_frame(frame, debug=True)

            cv2.imshow("Lane Tracking", processed_frame)

            # In thông tin
            if info["status"] == "TRACKING":
                lane_st = info.get("lane_status", {})
                conf = lane_st.get("confidence", 1.0)
                warn = lane_st.get("warning", "")

                print(f"[{info['action']:^25}] "
                      f"Score:{info['steering_score']:+.2f} "
                      f"Off:{info['raw_data']['offset_m']:+.2f}m "
                      f"Ang:{info['raw_data']['lane_angle']:+.1f}° "
                      f"Conf:{conf:.0%} {warn}")
            else:
                print(f"[{info['action']}] - {info.get('lane_status', {}).get('warning', '')}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            lane_nav.set_control_weights(k_offset=lane_nav.k_offset + 0.1)
        elif key == ord('2'):
            lane_nav.set_control_weights(k_angle=lane_nav.k_angle + 0.1)
        elif key == ord('3'):
            lane_nav.set_control_weights(k_curvature=lane_nav.k_curvature + 0.1)
        elif key == ord('w'):
            lane_nav.set_lane_width(lane_nav.standard_lane_width_pixels + 20)
        elif key == ord('s'):
            lane_nav.set_lane_width(lane_nav.standard_lane_width_pixels - 20)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

lane_nav = LaneNavigator()
lane_nav.load_config("lane_nav_config.json")