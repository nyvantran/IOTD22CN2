import cv2
import numpy as np
import json


class LaneNavigator:
    def __init__(self):
        # === ROI & BEV ===
        self.roi_points = []
        self.bev_src_points = None
        self.bev_dst_points = None
        self.M = None
        self.Minv = None

        # === FLAGS ===
        self.config_loaded = False
        self.use_auto_roi = False

        # === THÔNG SỐ KỸ THUẬT ===
        self.ym_per_pix = 0.20 / 160.0  # m/pixel dọc
        self.xm_per_pix = 0.20 / 130.0  # m/pixel ngang

        # === THÔNG SỐ ĐIỀU KHIỂN ===
        self.k_offset = 0.4
        self.k_angle = 0.8
        self.k_curvature = 0.2

        self.steering_threshold = 0.2
        self.sharp_turn_threshold = 0.5

        self.prev_steering_score = 0
        self.smoothing_factor = 0.8

        # === THÔNG SỐ LÀN ĐƯỜNG ===
        self.standard_lane_width_pixels = 130
        self.lane_width_history = []
        self.max_history = 30

        self.prev_left_fit = None
        self.prev_right_fit = None
        self.frames_since_both_lanes = 0
        self.max_frames_without_both = 15
        self.min_lane_points = 50

        # === THÔNG SỐ TIỀN XỬ LÝ ===
        self.sobel_thresh_min = 20
        self.sobel_thresh_max = 150
        self.lightness_thresh = 60  # Ngưỡng L cho vạch đen

        # === CROP INFO (tính từ ROI) ===
        self.crop_rect = None  # (x, y, w, h)

    # ================================================================
    # =============== LOAD / SAVE CONFIG =============================
    # ================================================================

    def load_config(self, filepath="lane_nav_config.json"):
        """Load toàn bộ cấu hình từ file JSON."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # === ROI & BEV ===
            if "roi_points" in data:
                self.roi_points = data["roi_points"]
            if "bev_src_points" in data:
                self.bev_src_points = np.array(data["bev_src_points"], dtype=np.float32)
            if "bev_dst_points" in data:
                self.bev_dst_points = np.array(data["bev_dst_points"], dtype=np.float32)
            if "M" in data:
                self.M = np.array(data["M"], dtype=np.float32)
            if "Minv" in data:
                self.Minv = np.array(data["Minv"], dtype=np.float32)
            if "crop_rect" in data:
                self.crop_rect = tuple(data["crop_rect"]) if data["crop_rect"] else None

            # === THÔNG SỐ KỸ THUẬT ===
            if "ym_per_pix" in data:
                self.ym_per_pix = data["ym_per_pix"]
            if "xm_per_pix" in data:
                self.xm_per_pix = data["xm_per_pix"]

            # === THÔNG SỐ ĐIỀU KHIỂN ===
            if "k_offset" in data:
                self.k_offset = data["k_offset"]
            if "k_angle" in data:
                self.k_angle = data["k_angle"]
            if "k_curvature" in data:
                self.k_curvature = data["k_curvature"]
            if "steering_threshold" in data:
                self.steering_threshold = data["steering_threshold"]
            if "sharp_turn_threshold" in data:
                self.sharp_turn_threshold = data["sharp_turn_threshold"]
            if "smoothing_factor" in data:
                self.smoothing_factor = data["smoothing_factor"]

            # === THÔNG SỐ LÀN ĐƯỜNG ===
            if "standard_lane_width_pixels" in data:
                self.standard_lane_width_pixels = data["standard_lane_width_pixels"]
            if "max_history" in data:
                self.max_history = data["max_history"]
            if "max_frames_without_both" in data:
                self.max_frames_without_both = data["max_frames_without_both"]
            if "min_lane_points" in data:
                self.min_lane_points = data["min_lane_points"]

            # === THÔNG SỐ TIỀN XỬ LÝ ===
            if "sobel_thresh_min" in data:
                self.sobel_thresh_min = data["sobel_thresh_min"]
            if "sobel_thresh_max" in data:
                self.sobel_thresh_max = data["sobel_thresh_max"]
            if "lightness_thresh" in data:
                self.lightness_thresh = data["lightness_thresh"]

            self.config_loaded = True
            self.use_auto_roi = False

            print(f"✓ Cấu hình đã được tải từ {filepath}")
            self._print_config_summary()
            return True

        except FileNotFoundError:
            print(f"✗ Không tìm thấy file {filepath}")
            return False
        except Exception as e:
            print(f"✗ Lỗi khi load config: {e}")
            return False

    def save_config(self, filepath="lane_nav_config.json"):
        """Lưu toàn bộ cấu hình vào file JSON."""
        if not self.roi_points:
            print("✗ Chưa có ROI để lưu!")
            return False

        config_data = {
            # === ROI & BEV ===
            "roi_points": self.roi_points if isinstance(self.roi_points, list) else self.roi_points.tolist(),
            "bev_src_points": self.bev_src_points.tolist() if self.bev_src_points is not None else None,
            "bev_dst_points": self.bev_dst_points.tolist() if self.bev_dst_points is not None else None,
            "M": self.M.tolist() if self.M is not None else None,
            "Minv": self.Minv.tolist() if self.Minv is not None else None,
            "crop_rect": list(self.crop_rect) if self.crop_rect else None,

            # === THÔNG SỐ KỸ THUẬT ===
            "ym_per_pix": self.ym_per_pix,
            "xm_per_pix": self.xm_per_pix,

            # === THÔNG SỐ ĐIỀU KHIỂN ===
            "k_offset": self.k_offset,
            "k_angle": self.k_angle,
            "k_curvature": self.k_curvature,
            "steering_threshold": self.steering_threshold,
            "sharp_turn_threshold": self.sharp_turn_threshold,
            "smoothing_factor": self.smoothing_factor,

            # === THÔNG SỐ LÀN ĐƯỜNG ===
            "standard_lane_width_pixels": self.standard_lane_width_pixels,
            "max_history": self.max_history,
            "max_frames_without_both": self.max_frames_without_both,
            "min_lane_points": self.min_lane_points,

            # === THÔNG SỐ TIỀN XỬ LÝ ===
            "sobel_thresh_min": self.sobel_thresh_min,
            "sobel_thresh_max": self.sobel_thresh_max,
            "lightness_thresh": self.lightness_thresh,
        }

        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=2)

        print(f"✓ Cấu hình đã được lưu vào {filepath}")
        return True

    def _print_config_summary(self):
        """In tóm tắt cấu hình hiện tại."""
        print("  --- Tóm tắt Config ---")
        print(f"  ROI: {len(self.roi_points)} điểm")
        print(f"  Crop: {self.crop_rect}")
        print(f"  Control: k_off={self.k_offset}, k_ang={self.k_angle}, k_curv={self.k_curvature}")
        print(f"  Thresholds: steer={self.steering_threshold}, sharp={self.sharp_turn_threshold}")
        print(f"  Lane width: {self.standard_lane_width_pixels}px")
        print(
            f"  Preprocess: sobel=[{self.sobel_thresh_min},{self.sobel_thresh_max}], L_thresh={self.lightness_thresh}")
        print("  ----------------------")

    # ================================================================
    # =============== CẤU HÌNH ROI ===================================
    # ================================================================

    def select_points_interactive(self, frame, save_path="lane_nav_config.json"):
        """Mở cửa sổ để người dùng click chọn 4 điểm ROI."""
        print("\n" + "=" * 50)
        print("HƯỚNG DẪN CHỌN ROI:")
        print("  1. Click điểm Dưới-Trái")
        print("  2. Click điểm Dưới-Phải")
        print("  3. Click điểm Trên-Phải")
        print("  4. Click điểm Trên-Trái")
        print("  ESC: Hủy | R: Reset | ENTER: Xác nhận")
        print("=" * 50 + "\n")

        temp_img = frame.copy()
        points = []

        def mouse_callback(event, x, y, flags, param):
            nonlocal temp_img, points

            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 4:
                    points.append((x, y))
                    cv2.circle(temp_img, (x, y), 6, (0, 0, 255), -1)
                    cv2.putText(temp_img, str(len(points)), (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if len(points) > 1:
                        cv2.line(temp_img, points[-2], points[-1], (0, 255, 0), 2)
                    if len(points) == 4:
                        cv2.line(temp_img, points[-1], points[0], (0, 255, 0), 2)
                        overlay = temp_img.copy()
                        cv2.fillPoly(overlay, [np.array(points)], (0, 255, 0))
                        cv2.addWeighted(overlay, 0.3, temp_img, 0.7, 0, temp_img)

                    cv2.imshow("Config ROI", temp_img)

        cv2.namedWindow("Config ROI")
        cv2.setMouseCallback("Config ROI", mouse_callback)
        cv2.imshow("Config ROI", temp_img)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                cv2.destroyWindow("Config ROI")
                print("✗ Đã hủy chọn ROI")
                return False

            elif key == ord('r'):
                temp_img = frame.copy()
                points = []
                cv2.imshow("Config ROI", temp_img)
                print("  Reset - chọn lại từ đầu")

            elif key == 13 or key == 10:
                if len(points) == 4:
                    break
                else:
                    print(f"  Cần chọn đủ 4 điểm (hiện có {len(points)})")

        cv2.destroyWindow("Config ROI")

        if len(points) != 4:
            print("✗ Chưa chọn đủ 4 điểm!")
            return False

        # Thiết lập ROI
        self._setup_roi_from_points(points, frame.shape)

        self.config_loaded = True
        self.use_auto_roi = False

        print(f"✓ Cấu hình ROI hoàn tất!")
        self._print_config_summary()

        self.save_config(save_path)
        return True

    def set_roi_programmatically(self, roi_points, frame_shape):
        """Đặt ROI bằng code."""
        if len(roi_points) != 4:
            raise ValueError("Cần đúng 4 điểm ROI!")

        self._setup_roi_from_points(list(roi_points), frame_shape)

        self.config_loaded = True
        self.use_auto_roi = False

        print(f"✓ ROI đã được đặt: {roi_points}")

    def _setup_roi_from_points(self, points, frame_shape):
        """Thiết lập ROI, BEV transform và crop rect từ các điểm."""
        h, w = frame_shape[:2]

        self.roi_points = points
        self.bev_src_points = np.float32(points)

        # Tính crop rectangle từ ROI polygon
        pts_array = np.array(points)
        x_min = max(0, int(np.min(pts_array[:, 0])))
        x_max = min(w, int(np.max(pts_array[:, 0])))
        y_min = max(0, int(np.min(pts_array[:, 1])))
        y_max = min(h, int(np.max(pts_array[:, 1])))

        self.crop_rect = (x_min, y_min, x_max - x_min, y_max - y_min)

        # BEV destination
        offset = 100
        self.bev_dst_points = np.float32([
            [offset, h],
            [w - offset, h],
            [w - offset, 0],
            [offset, 0]
        ])

        self.M = cv2.getPerspectiveTransform(self.bev_src_points, self.bev_dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.bev_dst_points, self.bev_src_points)

        # Ước tính độ rộng làn
        bottom_width = abs(points[1][0] - points[0][0])
        self.standard_lane_width_pixels = int(bottom_width * 0.8)

    def enable_auto_roi(self, enable=True):
        """Bật/tắt chế độ ROI tự động."""
        self.use_auto_roi = enable
        if enable:
            self.crop_rect = None  # Reset crop khi dùng auto
            print("✓ Đã bật chế độ ROI tự động")
        else:
            print("✓ Đã tắt chế độ ROI tự động")

    # ================================================================
    # =============== CROP & ROI HELPERS =============================
    # ================================================================

    def _compute_auto_roi_points(self, h, w):
        """ROI tự động: 2/5 ảnh từ dưới, 5/7 bề ngang."""
        y_bottom = h - 1
        y_top = int(h * (3.0 / 5.0))

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

    def _crop_frame(self, frame):
        """
        Crop frame theo ROI bounding box.
        Trả về: (cropped_frame, offset_x, offset_y)
        """
        if self.use_auto_roi or self.crop_rect is None:
            # Tính auto crop từ auto ROI
            h, w = frame.shape[:2]
            roi_pts = self._compute_auto_roi_points(h, w)
            x_min = int(np.min(roi_pts[:, 0]))
            x_max = int(np.max(roi_pts[:, 0]))
            y_min = int(np.min(roi_pts[:, 1]))
            y_max = int(np.max(roi_pts[:, 1]))
        else:
            x_min, y_min, crop_w, crop_h = self.crop_rect
            x_max = x_min + crop_w
            y_max = y_min + crop_h

        # Đảm bảo trong bounds
        h, w = frame.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        cropped = frame[y_min:y_max, x_min:x_max].copy()

        return cropped, x_min, y_min

    def _get_roi_mask_for_crop(self, crop_shape, offset_x, offset_y):
        """
        Tạo ROI mask cho ảnh đã crop.
        Điều chỉnh tọa độ ROI theo offset crop.
        """
        h, w = crop_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if self.use_auto_roi or not self.config_loaded:
            # Auto ROI - full crop area
            cv2.rectangle(mask, (0, 0), (w - 1, h - 1), 1, -1)
        else:
            # Điều chỉnh ROI points theo crop offset
            adjusted_pts = []
            for pt in self.roi_points:
                new_x = pt[0] - offset_x
                new_y = pt[1] - offset_y
                adjusted_pts.append([new_x, new_y])

            roi_poly = np.array(adjusted_pts, dtype=np.int32)
            cv2.fillPoly(mask, [roi_poly], 1)

        return mask

    def _ensure_bev_transform(self, frame_shape):
        """Đảm bảo có ma trận BEV transform."""
        if self.M is not None and self.Minv is not None:
            return

        if self.use_auto_roi or not self.config_loaded:
            h, w = frame_shape[:2]
            roi_pts = self._compute_auto_roi_points(h, w)

            self.roi_points = roi_pts.tolist()
            self.bev_src_points = roi_pts

            # Tính crop rect
            x_min = int(np.min(roi_pts[:, 0]))
            x_max = int(np.max(roi_pts[:, 0]))
            y_min = int(np.min(roi_pts[:, 1]))
            y_max = int(np.max(roi_pts[:, 1]))
            self.crop_rect = (x_min, y_min, x_max - x_min, y_max - y_min)

            offset = int(w * 0.15)
            self.bev_dst_points = np.float32([
                [offset, h],
                [w - offset, h],
                [w - offset, 0],
                [offset, 0]
            ])

            self.M = cv2.getPerspectiveTransform(self.bev_src_points, self.bev_dst_points)
            self.Minv = cv2.getPerspectiveTransform(self.bev_dst_points, self.bev_src_points)
            print("⚠ Đang dùng ROI tự động")

    # ================================================================
    # =============== TIỀN XỬ LÝ ẢNH =================================
    # ================================================================

    def preprocess_advanced(self, cropped_img, roi_mask):
        """
        Tiền xử lý cho track giấy trắng + băng keo đen.
        Input: ảnh đã crop và mask ROI tương ứng.
        """
        # HLS -> kênh L
        hls = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]

        # Gradient theo x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-6))

        sxbinary = np.zeros_like(scaled_sobel, dtype=np.uint8)
        sxbinary[(scaled_sobel >= self.sobel_thresh_min) &
                 (scaled_sobel <= self.sobel_thresh_max)] = 1

        # Ngưỡng L thấp -> vùng ĐEN
        l_binary = np.zeros_like(l_channel, dtype=np.uint8)
        l_binary[l_channel < self.lightness_thresh] = 1

        # Kết hợp & áp ROI mask
        combined_binary = np.zeros_like(sxbinary, dtype=np.uint8)
        combined_binary[(sxbinary == 1) | (l_binary == 1)] = 1
        combined_binary = combined_binary * roi_mask

        return combined_binary

    # ================================================================
    # =============== PHÁT HIỆN LÀN ĐƯỜNG ============================
    # ================================================================

    def detect_lanes_sliding_window(self, binary_warped):
        """Tìm làn đường bằng phương pháp cửa sổ trượt."""
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = np.int64(histogram.shape[0] / 2)

        left_half = histogram[:midpoint]
        right_half = histogram[midpoint:]

        leftx_base = np.argmax(left_half)
        rightx_base = np.argmax(right_half) + midpoint

        left_peak = left_half[leftx_base] if len(left_half) > 0 else 0
        right_peak = right_half[rightx_base - midpoint] if len(right_half) > 0 else 0

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

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
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
        """Ước tính làn đường còn thiếu."""
        ploty = np.linspace(0, img_height - 1, img_height)
        detected_x = detected_fit[0] * ploty ** 2 + detected_fit[1] * ploty + detected_fit[2]

        if is_left_detected:
            estimated_x = detected_x + self.standard_lane_width_pixels
        else:
            estimated_x = detected_x - self.standard_lane_width_pixels

        estimated_fit = np.polyfit(ploty, estimated_x, 2)
        return estimated_fit

    def update_lane_width(self, left_fit, right_fit, img_height):
        """Cập nhật độ rộng làn đường chuẩn."""
        y_bottom = img_height - 1
        left_x = left_fit[0] * y_bottom ** 2 + left_fit[1] * y_bottom + left_fit[2]
        right_x = right_fit[0] * y_bottom ** 2 + right_fit[1] * y_bottom + right_fit[2]

        current_width = right_x - left_x

        if 80 < current_width < 400:
            self.lane_width_history.append(current_width)
            if len(self.lane_width_history) > self.max_history:
                self.lane_width_history.pop(0)
            self.standard_lane_width_pixels = int(np.mean(self.lane_width_history))

    def get_lanes_with_estimation(self, binary_warped, img_shape):
        """Phát hiện làn đường với khả năng ước tính làn bị mất."""
        h = img_shape[0]

        left_fit, right_fit, lane_data, detection_info = self.detect_lanes_sliding_window(binary_warped)

        lane_status = {
            "detection": detection_info,
            "left_source": "none",
            "right_source": "none",
            "confidence": 0.0,
            "warning": None
        }

        # TRƯỜNG HỢP 1: CẢ HAI LÀN ĐƯỢC PHÁT HIỆN
        if detection_info["both_detected"]:
            self.update_lane_width(left_fit, right_fit, h)
            self.prev_left_fit = left_fit.copy()
            self.prev_right_fit = right_fit.copy()
            self.frames_since_both_lanes = 0

            lane_status["left_source"] = "detected"
            lane_status["right_source"] = "detected"
            lane_status["confidence"] = 1.0

            return left_fit, right_fit, lane_status

        # TRƯỜNG HỢP 2: CHỈ CÓ LÀN TRÁI
        elif detection_info["left_detected"] and not detection_info["right_detected"]:
            self.frames_since_both_lanes += 1
            self.prev_left_fit = left_fit.copy()

            lane_status["left_source"] = "detected"
            lane_status["warning"] = "RIGHT_LANE_MISSING"

            if self.frames_since_both_lanes <= self.max_frames_without_both:
                if self.prev_right_fit is not None:
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

        # TRƯỜNG HỢP 3: CHỈ CÓ LÀN PHẢI
        elif detection_info["right_detected"] and not detection_info["left_detected"]:
            self.frames_since_both_lanes += 1
            self.prev_right_fit = right_fit.copy()

            lane_status["right_source"] = "detected"
            lane_status["warning"] = "LEFT_LANE_MISSING"

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

        # TRƯỜNG HỢP 4: KHÔNG PHÁT HIỆN ĐƯỢC LÀN NÀO
        else:
            self.frames_since_both_lanes += 1
            lane_status["warning"] = "NO_LANES_DETECTED"

            if (self.frames_since_both_lanes <= self.max_frames_without_both and
                    self.prev_left_fit is not None and self.prev_right_fit is not None):
                left_fit = self.prev_left_fit
                right_fit = self.prev_right_fit
                lane_status["left_source"] = "previous"
                lane_status["right_source"] = "previous"
                lane_status["confidence"] = max(0.1, 0.5 - self.frames_since_both_lanes * 0.03)

                return left_fit, right_fit, lane_status

            lane_status["confidence"] = 0.0
            return None, None, lane_status

    def _blend_with_previous(self, estimated_fit, previous_fit, frames_elapsed):
        """Kết hợp fit ước tính với fit trước đó."""
        prev_weight = max(0.2, 1.0 - frames_elapsed * 0.1)
        est_weight = 1.0 - prev_weight
        blended_fit = prev_weight * previous_fit + est_weight * estimated_fit
        return blended_fit

    # ================================================================
    # =============== PHÂN TÍCH HÌNH HỌC =============================
    # ================================================================

    def analyze_lane_geometry(self, img_shape, left_fit, right_fit):
        """Phân tích hình học làn đường."""
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

        image_center = w / 2
        offset_pixels = image_center - center_bottom
        offset_meters = offset_pixels * self.xm_per_pix

        dx = center_ahead - center_bottom
        dy = y_bottom - y_ahead
        lane_angle_deg = np.degrees(np.arctan2(dx, dy))

        dx_near = center_middle - center_bottom
        dy_near = y_bottom - y_middle
        lane_angle_near = np.degrees(np.arctan2(dx_near, dy_near))

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
        """Tính toán quyết định điều khiển."""
        offset = lane_geometry["offset_meters"]
        angle = lane_geometry["lane_angle_deg"]
        angle_near = lane_geometry["lane_angle_near"]
        curve_dir = lane_geometry["curvature_direction"]
        curve_strength = lane_geometry["curvature_strength"]
        radius = lane_geometry["radius_meters"]

        confidence = lane_status.get("confidence", 1.0)
        warning = lane_status.get("warning")
        detection = lane_status.get("detection", {})
        left_src = lane_status.get("left_source", "none")
        right_src = lane_status.get("right_source", "none")

        both_lanes_reliable = (
                detection.get("both_detected", False)
                and warning is None
                and left_src == "detected"
                and right_src == "detected"
                and confidence >= 0.6
        )

        single_lane_mode = (
                detection.get("single_lane", False)
                or warning in ("LEFT_LANE_MISSING", "RIGHT_LANE_MISSING")
        )

        effective_k_offset = self.k_offset * confidence
        effective_k_angle = self.k_angle * confidence
        effective_k_curvature = self.k_curvature * confidence

        if single_lane_mode:
            effective_k_offset = 0.0
            effective_k_angle *= 1.2
            effective_k_curvature *= 1.5

        offset_contribution = 0.0
        angle_contribution = 0.0
        curve_contribution = 0.0

        if both_lanes_reliable or not single_lane_mode:
            offset_norm = np.clip(offset / 0.5, -1, 1)
            offset_contribution = -effective_k_offset * offset_norm

            angle_norm = np.clip(angle / 25.0, -1, 1)
            angle_near_norm = np.clip(angle_near / 30.0, -1, 1)
            angle_combined = 0.4 * angle_norm + 0.6 * angle_near_norm
            angle_contribution = effective_k_angle * angle_combined

            curve_norm = curve_dir * min(curve_strength / 50.0, 1.0)
            curve_contribution = effective_k_curvature * curve_norm
        else:
            angle_near_norm = np.clip(angle_near / 25.0, -1, 1)
            angle_contribution = effective_k_angle * angle_near_norm

            if abs(angle_near_norm) > 0.05:
                curve_dir_single = np.sign(angle_near_norm)
            else:
                curve_dir_single = np.sign(self.prev_steering_score) if self.prev_steering_score != 0 else 0.0

            curve_strength_single = min(abs(angle_near) / 20.0, 1.0)
            curve_contribution = effective_k_curvature * curve_dir_single * curve_strength_single

        raw_steering_score = offset_contribution + angle_contribution + curve_contribution

        if single_lane_mode and self.prev_steering_score != 0:
            if raw_steering_score * self.prev_steering_score < 0 and abs(raw_steering_score) < 0.6:
                raw_steering_score = np.sign(self.prev_steering_score) * abs(raw_steering_score)

        effective_smoothing = min(0.6, self.smoothing_factor + (1 - confidence) * 0.2)
        steering_score = (effective_smoothing * self.prev_steering_score +
                          (1 - effective_smoothing) * raw_steering_score)
        self.prev_steering_score = steering_score

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
    # =============== PROCESS FRAME ==================================
    # ================================================================

    def process_frame(self, frame, debug=False):
        """Hàm xử lý chính cho 1 frame ảnh."""

        # Đảm bảo có BEV transform
        self._ensure_bev_transform(frame.shape)

        if self.M is None:
            raise Exception("Chưa có config! Chạy select_points_interactive() hoặc load_config()")

        # === 1. CROP FRAME THEO ROI ===
        cropped_frame, offset_x, offset_y = self._crop_frame(frame)

        # === 2. TẠO ROI MASK CHO VÙNG CROP ===
        roi_mask = self._get_roi_mask_for_crop(cropped_frame.shape, offset_x, offset_y)

        # === 3. PREPROCESSING TRÊN ẢNH ĐÃ CROP ===
        binary_cropped = self.preprocess_advanced(cropped_frame, roi_mask)

        # === 4. ĐƯA VỀ KÍCH THƯỚC GỐC ĐỂ WARP ===
        h, w = frame.shape[:2]
        binary_full = np.zeros((h, w), dtype=np.uint8)

        crop_h, crop_w = binary_cropped.shape[:2]
        y_end = min(offset_y + crop_h, h)
        x_end = min(offset_x + crop_w, w)
        binary_full[offset_y:y_end, offset_x:x_end] = binary_cropped[:y_end - offset_y, :x_end - offset_x]

        # === 5. WARP TO BEV ===
        img_size = (w, h)
        warped = cv2.warpPerspective(binary_full, self.M, img_size, flags=cv2.INTER_LINEAR)

        # === 6. DETECT LANES ===
        left_fit, right_fit, lane_status = self.get_lanes_with_estimation(warped, frame.shape)

        # === 7. XỬ LÝ KHI MẤT LÀN ===
        if left_fit is None or right_fit is None:
            self.prev_steering_score = 0
            result = frame.copy()
            if debug:
                self._draw_crop_overlay(result, offset_x, offset_y, crop_w, crop_h)
            return result, {
                "status": "LOST_LANE",
                "action": "!!! SLOW DOWN !!!",
                "action_code": 99,
                "lane_status": lane_status
            }

        # === 8. PHÂN TÍCH HÌNH HỌC ===
        lane_geometry = self.analyze_lane_geometry(frame.shape, left_fit, right_fit)

        # === 9. TÍNH TOÁN QUYẾT ĐỊNH ===
        control_decision = self.calculate_steering_decision(lane_geometry, lane_status)

        # === 10. VẼ KẾT QUẢ ===
        result = self._draw_lanes(frame, warped, left_fit, right_fit,
                                  lane_geometry, control_decision, lane_status)

        # === DEBUG MODE ===
        if debug:
            self._draw_debug_view(result, warped, binary_cropped, lane_geometry, lane_status)
            self._draw_crop_overlay(result, offset_x, offset_y, crop_w, crop_h)

        # === OUTPUT ===
        output_info = {
            "status": "TRACKING",
            **control_decision,
            "lane_status": lane_status,
            "geometry": lane_geometry,
            "crop_info": {
                "offset_x": offset_x,
                "offset_y": offset_y,
                "width": crop_w,
                "height": crop_h
            }
        }

        return result, output_info

    def _draw_crop_overlay(self, img, offset_x, offset_y, crop_w, crop_h):
        """Vẽ đường viền vùng crop và ROI lên ảnh."""
        # Vẽ vùng crop (màu xanh dương)
        cv2.rectangle(img,
                      (offset_x, offset_y),
                      (offset_x + crop_w, offset_y + crop_h),
                      (255, 0, 0), 2)

        # Vẽ ROI polygon (màu tím)
        if self.roi_points:
            pts = np.array(self.roi_points, dtype=np.int32)
            cv2.polylines(img, [pts], True, (255, 0, 255), 2)

        # Hiển thị thông tin
        roi_source = "AUTO" if self.use_auto_roi else "CONFIG"
        cv2.putText(img, f"ROI: {roi_source} | Crop: {crop_w}x{crop_h}",
                    (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

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

        confidence = lane_status.get("confidence", 1.0)

        if decision["direction"] == "left":
            base_color = (255, 100, 0)
        elif decision["direction"] == "right":
            base_color = (0, 100, 255)
        else:
            base_color = (0, 255, 0)

        lane_color = tuple(int(c * confidence) for c in base_color)
        cv2.fillPoly(color_warp, np.int_([pts]), lane_color)

        left_color = (0, 255, 0) if lane_status["left_source"] == "detected" else (0, 255, 255)
        right_color = (0, 255, 0) if lane_status["right_source"] == "detected" else (0, 255, 255)

        for i in range(len(ploty) - 1):
            pt1_l = (int(left_fitx[i]), int(ploty[i]))
            pt2_l = (int(left_fitx[i + 1]), int(ploty[i + 1]))
            cv2.line(color_warp, pt1_l, pt2_l, left_color, 3)

            pt1_r = (int(right_fitx[i]), int(ploty[i]))
            pt2_r = (int(right_fitx[i + 1]), int(ploty[i + 1]))
            cv2.line(color_warp, pt1_r, pt2_r, right_color, 3)

        center_fitx = (left_fitx + right_fitx) / 2
        for i in range(0, len(ploty) - 10, 10):
            pt1 = (int(center_fitx[i]), int(ploty[i]))
            pt2 = (int(center_fitx[i + 10]), int(ploty[i + 10]))
            cv2.line(color_warp, pt1, pt2, (255, 255, 0), 3)

        newwarp = cv2.warpPerspective(color_warp, self.Minv, (w, h))
        result = cv2.addWeighted(frame, 1, newwarp, 0.4, 0)

        self._draw_info_panel(result, geometry, decision, lane_status)

        return result

    def _draw_info_panel(self, img, geometry, decision, lane_status):
        """Vẽ bảng thông tin điều khiển."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = img.shape[:2]

        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (420, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        if decision["direction"] == "left":
            cmd_color = (255, 150, 0)
        elif decision["direction"] == "right":
            cmd_color = (0, 150, 255)
        else:
            cmd_color = (0, 255, 0)

        y_offset = 35
        line_height = 26

        confidence = lane_status.get("confidence", 1.0)
        warning = lane_status.get("warning")

        left_src = lane_status.get("left_source", "none")
        right_src = lane_status.get("right_source", "none")

        status_text = f"L:{left_src[:3].upper()} | R:{right_src[:3].upper()} | Conf:{confidence:.0%}"
        status_color = (0, 255, 0) if confidence > 0.7 else ((0, 255, 255) if confidence > 0.3 else (0, 0, 255))
        cv2.putText(img, status_text, (20, y_offset), font, 0.5, status_color, 1)
        y_offset += line_height

        if warning:
            warning_color = (0, 165, 255)
            if warning == "NO_LANES_DETECTED":
                warning_color = (0, 0, 255)
            cv2.putText(img, f"! {warning}", (20, y_offset), font, 0.5, warning_color, 2)
            y_offset += line_height

        cv2.putText(img, f"CMD: {decision['action']}", (20, y_offset),
                    font, 0.7, cmd_color, 2)
        y_offset += line_height + 5

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

    def _draw_debug_view(self, result, warped, binary_cropped, geometry, lane_status):
        """Vẽ view debug với ảnh binary đã crop."""
        h, w = result.shape[:2]

        # BEV warped view
        debug_warp = np.dstack((warped, warped, warped)) * 255
        debug_warp = debug_warp.astype(np.uint8)

        scale_y = warped.shape[0] / h
        cv2.circle(debug_warp, (int(geometry['center_bottom']), int((h - 1) * scale_y)),
                   8, (0, 0, 255), -1)
        cv2.circle(debug_warp, (int(geometry['center_ahead']), int(h * 0.3 * scale_y)),
                   8, (0, 255, 0), -1)
        cv2.circle(debug_warp, (int(geometry['image_center']), int((h - 1) * scale_y)),
                   8, (255, 0, 0), -1)

        debug_warp = cv2.resize(debug_warp, (200, 150))
        result[0:150, w - 200:w] = debug_warp

        # Binary cropped view
        binary_view = np.dstack((binary_cropped, binary_cropped, binary_cropped)) * 255
        binary_view = binary_view.astype(np.uint8)
        binary_view = cv2.resize(binary_view, (200, 150))
        result[0:150, w - 410:w - 210] = binary_view

        # Labels
        cv2.putText(result, "BEV", (w - 195, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(result, "Binary Crop", (w - 405, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        det = lane_status.get("detection", {})
        cv2.putText(result, f"L:{det.get('left_points', 0)} R:{det.get('right_points', 0)}",
                    (w - 195, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ================================================================
    # =============== UTILITY METHODS ================================
    # ================================================================

    def set_control_weights(self, k_offset=None, k_angle=None, k_curvature=None):
        """Điều chỉnh trọng số điều khiển."""
        if k_offset is not None:
            self.k_offset = k_offset
        if k_angle is not None:
            self.k_angle = k_angle
        if k_curvature is not None:
            self.k_curvature = k_curvature
        print(f"Weights: offset={self.k_offset:.2f}, angle={self.k_angle:.2f}, curve={self.k_curvature:.2f}")

    def set_lane_width(self, width_pixels):
        """Đặt độ rộng làn đường chuẩn."""
        self.standard_lane_width_pixels = width_pixels
        print(f"Lane width: {width_pixels}px")

    def set_preprocess_params(self, sobel_min=None, sobel_max=None, lightness=None):
        """Điều chỉnh thông số tiền xử lý."""
        if sobel_min is not None:
            self.sobel_thresh_min = sobel_min
        if sobel_max is not None:
            self.sobel_thresh_max = sobel_max
        if lightness is not None:
            self.lightness_thresh = lightness
        print(f"Preprocess: sobel=[{self.sobel_thresh_min},{self.sobel_thresh_max}], L={self.lightness_thresh}")

    def set_steering_thresholds(self, steer=None, sharp=None):
        """Điều chỉnh ngưỡng steering."""
        if steer is not None:
            self.steering_threshold = steer
        if sharp is not None:
            self.sharp_turn_threshold = sharp
        print(f"Thresholds: steer={self.steering_threshold:.2f}, sharp={self.sharp_turn_threshold:.2f}")

    def get_config_dict(self):
        """Trả về dict chứa toàn bộ config hiện tại."""
        return {
            "roi_points": self.roi_points,
            "crop_rect": self.crop_rect,
            "use_auto_roi": self.use_auto_roi,
            "config_loaded": self.config_loaded,
            "control": {
                "k_offset": self.k_offset,
                "k_angle": self.k_angle,
                "k_curvature": self.k_curvature,
                "steering_threshold": self.steering_threshold,
                "sharp_turn_threshold": self.sharp_turn_threshold,
                "smoothing_factor": self.smoothing_factor,
            },
            "lane": {
                "standard_width": self.standard_lane_width_pixels,
                "min_points": self.min_lane_points,
            },
            "preprocess": {
                "sobel_min": self.sobel_thresh_min,
                "sobel_max": self.sobel_thresh_max,
                "lightness_thresh": self.lightness_thresh,
            }
        }


# ========================================================
# MAIN PROGRAM
# ========================================================
def main():
    from stream_manager import stream_manager
    stream_manager.start()
    lane_nav = LaneNavigator()

    # Đọc ảnh test
    first_frame = stream_manager.get_latest_frame()
    if first_frame is None:
        print("Không tìm thấy ảnh img_1.png")
        return

    # === Load hoặc tạo config ===
    if not lane_nav.load_config("lane_nav_config.json"):
        print("\nKhông có config, mở giao diện chọn ROI...")
        lane_nav.select_points_interactive(first_frame)

    print("\n=== ĐIỀU KHIỂN ===")
    print("Q: Thoát")
    print("R: Chọn lại ROI")
    print("A: Bật/tắt ROI tự động")
    print("S: Lưu config")
    print("1/2/3: +/- trọng số Offset/Angle/Curve (Shift để giảm)")
    print("4/5: +/- ngưỡng steering/sharp")
    print("6/7/8: +/- sobel_min/sobel_max/lightness")
    print("W/X: +/- độ rộng làn")
    print("==================\n")

    while True:
        frame = stream_manager.get_latest_frame()

        try:
            processed_frame, info = lane_nav.process_frame(frame, debug=True)

            cv2.imshow("Lane Tracking", processed_frame)

            if info["status"] == "TRACKING":
                lane_st = info.get("lane_status", {})
                conf = lane_st.get("confidence", 1.0)
                warn = lane_st.get("warning", "")

                print(f"\r[{info['action']:^25}] "
                      f"Score:{info['steering_score']:+.2f} "
                      f"Off:{info['raw_data']['offset_m']:+.2f}m "
                      f"Ang:{info['raw_data']['lane_angle']:+.1f}° "
                      f"Conf:{conf:.0%} {warn}", end="")
            else:
                print(f"\r[{info['action']}] - {info.get('lane_status', {}).get('warning', '')}", end="")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            continue

        key = cv2.waitKey(100) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            cv2.destroyAllWindows()
            lane_nav.select_points_interactive(first_frame)
        elif key == ord('a'):
            lane_nav.enable_auto_roi(not lane_nav.use_auto_roi)
        elif key == ord('s'):
            lane_nav.save_config()

        # Control weights
        elif key == ord('1'):
            lane_nav.set_control_weights(k_offset=lane_nav.k_offset + 0.05)
        elif key == ord('!'):
            lane_nav.set_control_weights(k_offset=max(0, lane_nav.k_offset - 0.05))
        elif key == ord('2'):
            lane_nav.set_control_weights(k_angle=lane_nav.k_angle + 0.05)
        elif key == ord('@'):
            lane_nav.set_control_weights(k_angle=max(0, lane_nav.k_angle - 0.05))
        elif key == ord('3'):
            lane_nav.set_control_weights(k_curvature=lane_nav.k_curvature + 0.05)
        elif key == ord('#'):
            lane_nav.set_control_weights(k_curvature=max(0, lane_nav.k_curvature - 0.05))

        # Steering thresholds
        elif key == ord('4'):
            lane_nav.set_steering_thresholds(steer=lane_nav.steering_threshold + 0.05)
        elif key == ord('$'):
            lane_nav.set_steering_thresholds(steer=max(0, lane_nav.steering_threshold - 0.05))
        elif key == ord('5'):
            lane_nav.set_steering_thresholds(sharp=lane_nav.sharp_turn_threshold + 0.05)
        elif key == ord('%'):
            lane_nav.set_steering_thresholds(sharp=max(0, lane_nav.sharp_turn_threshold - 0.05))

        # Preprocess params
        elif key == ord('6'):
            lane_nav.set_preprocess_params(sobel_min=lane_nav.sobel_thresh_min + 5)
        elif key == ord('^'):
            lane_nav.set_preprocess_params(sobel_min=max(0, lane_nav.sobel_thresh_min - 5))
        elif key == ord('7'):
            lane_nav.set_preprocess_params(sobel_max=lane_nav.sobel_thresh_max + 5)
        elif key == ord('&'):
            lane_nav.set_preprocess_params(sobel_max=max(0, lane_nav.sobel_thresh_max - 5))
        elif key == ord('8'):
            lane_nav.set_preprocess_params(lightness=lane_nav.lightness_thresh + 5)
        elif key == ord('*'):
            lane_nav.set_preprocess_params(lightness=max(0, lane_nav.lightness_thresh - 5))

        # Lane width
        elif key == ord('w'):
            lane_nav.set_lane_width(lane_nav.standard_lane_width_pixels + 10)
        elif key == ord('x'):
            lane_nav.set_lane_width(max(50, lane_nav.standard_lane_width_pixels - 10))

    cv2.destroyAllWindows()
    stream_manager.stop()


if __name__ == "__main__":
    main()

lane_nav = LaneNavigator()
lane_nav.load_config("control/lane_nav_config.json")
