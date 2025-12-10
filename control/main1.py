import cv2
import numpy as np
import json


class LaneNavigator:
    def __init__(self):
        # === CẤU HÌNH HÌNH HỌC ===
        self.roi_points = []
        self.roi_mask = None  # Mask cho ROI
        self.bev_src_points = None
        self.bev_dst_points = None
        self.M = None
        self.Minv = None

        # === THÔNG SỐ CHUYỂN ĐỔI TỌA ĐỘ ===
        self.ym_per_pix = 30 / 720
        self.xm_per_pix = 3.7 / 700

        # === THÔNG SỐ ĐIỀU KHIỂN ===
        self.k_offset = 1.0
        self.k_angle = 0.8
        self.k_curvature = 0.5

        self.steering_threshold = 0.35
        self.sharp_turn_threshold = 0.5

        self.prev_steering_score = 0
        self.smoothing_factor = 0.3

        # === THÔNG SỐ XỬ LÝ LÀN ĐƯỜNG ===
        self.standard_lane_width_pixels = 500
        self.lane_width_history = []
        self.max_history = 30
        self.min_lane_points = 100

        # === LƯU TRỮ FIT TRƯỚC ĐÓ ===
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.frames_since_both_lanes = 0
        self.max_frames_without_both = 15

        # === LOẠI LÀN ĐƯỜNG ===
        self.lane_type = "dark_on_light"

        # === THÔNG SỐ PREPROCESSING ===
        self.adaptive_block_size = 51
        self.adaptive_c = 15
        self.dark_threshold = 80
        self.light_threshold = 200
        self.sobel_low = 30
        self.sobel_high = 200
        self.min_blob_area = 100

        # === THÔNG SỐ SLIDING WINDOW ===
        self.n_windows = 9
        self.window_margin = 80
        self.min_pixels_recenter = 30
        self.histogram_smooth_window = 30

        # === ROI SETTINGS (MỚI) ===
        self.use_roi_crop = True  # Bật/tắt cắt ROI
        self.roi_expand_ratio = 0.1  # Mở rộng ROI thêm 10%

    # ================================================================
    # =============== CẤU HÌNH VÀ LƯU/TẢI ============================
    # ================================================================

    def load_config(self, filepath="lane_nav_config.json"):
        """Tải cấu hình từ file JSON."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            self.roi_points = data.get("roi_points", [])

            if "bev_src_points" in data:
                self.bev_src_points = np.array(data["bev_src_points"], dtype=np.float32)
            if "bev_dst_points" in data:
                self.bev_dst_points = np.array(data["bev_dst_points"], dtype=np.float32)
            if "M" in data:
                self.M = np.array(data["M"], dtype=np.float32)
            if "Minv" in data:
                self.Minv = np.array(data["Minv"], dtype=np.float32)

            self.standard_lane_width_pixels = data.get("standard_lane_width_pixels", 500)
            self.lane_type = data.get("lane_type", "dark_on_light")

            self.adaptive_block_size = data.get("adaptive_block_size", 51)
            self.adaptive_c = data.get("adaptive_c", 15)
            self.dark_threshold = data.get("dark_threshold", 80)

            # Tạo ROI mask sau khi load
            if self.roi_points:
                self._create_roi_mask()

            print(f"Cấu hình đã được tải từ {filepath}")
            print(f"  - Lane type: {self.lane_type}")
            print(f"  - Lane width: {self.standard_lane_width_pixels}px")
            print(f"  - ROI points: {len(self.roi_points)} điểm")

        except FileNotFoundError:
            print(f"Không tìm thấy file {filepath}")
        except Exception as e:
            print(f"Lỗi khi tải cấu hình: {e}")

    def save_config(self, filepath="lane_nav_config.json"):
        """Lưu cấu hình vào file JSON."""
        config = {
            "roi_points": self.roi_points,
            "bev_src_points": self.bev_src_points.tolist() if self.bev_src_points is not None else [],
            "bev_dst_points": self.bev_dst_points.tolist() if self.bev_dst_points is not None else [],
            "M": self.M.tolist() if self.M is not None else [],
            "Minv": self.Minv.tolist() if self.Minv is not None else [],
            "standard_lane_width_pixels": self.standard_lane_width_pixels,
            "lane_type": self.lane_type,
            "adaptive_block_size": self.adaptive_block_size,
            "adaptive_c": self.adaptive_c,
            "dark_threshold": self.dark_threshold,
        }

        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Cấu hình đã được lưu vào {filepath}")

    def select_points_interactive(self, frame):
        """Mở cửa sổ để người dùng click chọn 4 điểm ROI/BEV."""
        print("\n" + "=" * 50)
        print("HƯỚNG DẪN CHỌN ĐIỂM ROI:")
        print("  Click 4 điểm theo thứ tự:")
        print("  1. Dưới-Trái (góc dưới trái của làn)")
        print("  2. Dưới-Phải (góc dưới phải của làn)")
        print("  3. Trên-Phải (góc trên phải của làn)")
        print("  4. Trên-Trái (góc trên trái của làn)")
        print("  Nhấn phím bất kỳ sau khi chọn xong")
        print("=" * 50 + "\n")

        temp_img = frame.copy()
        self.roi_points = []
        point_names = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]

        def mouse_callback(event, x, y, flags, param):
            nonlocal temp_img
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.roi_points) < 4:
                    self.roi_points.append((x, y))

                    cv2.circle(temp_img, (x, y), 8, (0, 0, 255), -1)
                    cv2.putText(temp_img, f"{len(self.roi_points)}: {point_names[len(self.roi_points) - 1]}",
                                (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if len(self.roi_points) > 1:
                        cv2.line(temp_img, self.roi_points[-2], self.roi_points[-1], (0, 255, 0), 2)
                    if len(self.roi_points) == 4:
                        cv2.line(temp_img, self.roi_points[-1], self.roi_points[0], (0, 255, 0), 2)
                        cv2.putText(temp_img, "Press any key to confirm",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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
            [offset, h],
            [w - offset, h],
            [w - offset, 0],
            [offset, 0]
        ])

        self.M = cv2.getPerspectiveTransform(self.bev_src_points, self.bev_dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.bev_dst_points, self.bev_src_points)

        bottom_width = abs(self.roi_points[1][0] - self.roi_points[0][0])
        self.standard_lane_width_pixels = int(bottom_width * 0.8)

        # Tạo ROI mask
        self._create_roi_mask(frame.shape[:2])

        print(f"Cấu hình hoàn tất!")
        print(f"  - Độ rộng làn ước tính: {self.standard_lane_width_pixels}px")

        self.save_config()
        return True

    def _create_roi_mask(self, img_shape=None):
        """
        Tạo mask cho vùng ROI.
        Mask này sẽ được dùng để loại bỏ vùng ngoài ROI.
        """
        if not self.roi_points or len(self.roi_points) != 4:
            self.roi_mask = None
            return

        if img_shape is None:
            # Ước tính kích thước từ ROI points
            max_x = max(p[0] for p in self.roi_points) + 100
            max_y = max(p[1] for p in self.roi_points) + 100
            img_shape = (max_y, max_x)

        # Mở rộng ROI một chút để không bị cắt mất biên
        roi_np = np.array(self.roi_points, dtype=np.float32)
        centroid = np.mean(roi_np, axis=0)

        expanded_roi = []
        for pt in roi_np:
            direction = pt - centroid
            expanded_pt = pt + direction * self.roi_expand_ratio
            expanded_roi.append(expanded_pt)

        expanded_roi = np.array(expanded_roi, dtype=np.int32)

        # Tạo mask
        self.roi_mask = np.zeros(img_shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.roi_mask, [expanded_roi], 255)

        # Lưu bounding box của ROI để crop
        x_coords = [p[0] for p in expanded_roi]
        y_coords = [p[1] for p in expanded_roi]

        self.roi_bbox = {
            'x_min': max(0, min(x_coords)),
            'x_max': min(img_shape[1] if len(img_shape) > 1 else 9999, max(x_coords)),
            'y_min': max(0, min(y_coords)),
            'y_max': min(img_shape[0], max(y_coords))
        }

        print(f"ROI mask created: bbox = {self.roi_bbox}")

    # ================================================================
    # =============== ROI OPERATIONS (MỚI) ===========================
    # ================================================================

    def apply_roi_mask(self, img):
        """
        Áp dụng ROI mask lên ảnh.
        Vùng ngoài ROI sẽ bị đen.
        """
        if self.roi_mask is None:
            return img

        # Resize mask nếu cần
        if self.roi_mask.shape[:2] != img.shape[:2]:
            self._create_roi_mask(img.shape[:2])

        if len(img.shape) == 3:
            # Ảnh màu
            mask_3ch = cv2.merge([self.roi_mask, self.roi_mask, self.roi_mask])
            return cv2.bitwise_and(img, mask_3ch)
        else:
            # Ảnh grayscale hoặc binary
            return cv2.bitwise_and(img, self.roi_mask)

    def crop_to_roi(self, img):
        """
        Cắt ảnh theo bounding box của ROI.
        Trả về ảnh đã cắt và thông tin offset.
        """
        if not hasattr(self, 'roi_bbox') or self.roi_bbox is None:
            return img, {'x_offset': 0, 'y_offset': 0, 'cropped': False}

        bbox = self.roi_bbox
        h, w = img.shape[:2]

        # Đảm bảo bbox trong phạm vi ảnh
        x_min = max(0, bbox['x_min'])
        x_max = min(w, bbox['x_max'])
        y_min = max(0, bbox['y_min'])
        y_max = min(h, bbox['y_max'])

        cropped = img[y_min:y_max, x_min:x_max]

        offset_info = {
            'x_offset': x_min,
            'y_offset': y_min,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'cropped': True
        }

        return cropped, offset_info

    def get_roi_cropped_frame(self, frame):
        """
        Lấy frame đã được cắt theo ROI và apply mask.
        Đây là bước đầu tiên trước khi preprocessing.
        """
        if not self.use_roi_crop:
            return frame, {'cropped': False, 'x_offset': 0, 'y_offset': 0}

        # Bước 1: Apply mask để loại bỏ vùng ngoài ROI
        masked = self.apply_roi_mask(frame)

        # Bước 2: Crop theo bounding box
        cropped, offset_info = self.crop_to_roi(masked)

        return cropped, offset_info

    # ================================================================
    # =============== SETTINGS =======================================
    # ================================================================

    def set_lane_type(self, lane_type):
        """Đặt loại làn đường."""
        valid_types = ["dark_on_light", "light_on_dark", "auto"]
        if lane_type not in valid_types:
            raise ValueError(f"lane_type phải là một trong {valid_types}")
        self.lane_type = lane_type
        print(f"Lane type: {lane_type}")

    def set_preprocessing_params(self, **kwargs):
        """Điều chỉnh thông số preprocessing."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"  {key} = {value}")

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

    def set_roi_crop(self, enabled, expand_ratio=None):
        """Bật/tắt cắt ROI và điều chỉnh expand ratio."""
        self.use_roi_crop = enabled
        if expand_ratio is not None:
            self.roi_expand_ratio = expand_ratio
            if self.roi_points:
                self._create_roi_mask()
        print(f"ROI crop: {'ON' if enabled else 'OFF'}, expand_ratio: {self.roi_expand_ratio}")

    # ================================================================
    # =============== PREPROCESSING ==================================
    # ================================================================

    def preprocess_advanced(self, img, lane_type=None):
        """
        Tiền xử lý ảnh để phát hiện làn đường.
        LƯU Ý: img ở đây đã được cắt ROI nếu use_roi_crop=True
        """
        if lane_type is None:
            lane_type = self.lane_type

        if lane_type == "auto":
            lane_type = self._detect_lane_type(img)

        if lane_type == "dark_on_light":
            return self._preprocess_dark_lane(img)
        else:
            return self._preprocess_light_lane(img)

    def _detect_lane_type(self, img):
        """Tự động phát hiện loại làn đường."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi = gray[int(h * 0.5):h, int(w * 0.2):int(w * 0.8)]
        mean_val = np.mean(roi)

        if mean_val > 140:
            return "dark_on_light"
        else:
            return "light_on_dark"

    def _preprocess_dark_lane(self, img):
        """
        Phát hiện làn đường TỐI trên nền SÁNG.
        """
        # === 1. CHUYỂN ĐỔI KHÔNG GIAN MÀU ===
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE để tăng contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        # LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_lab = lab[:, :, 0]

        # === 2. ADAPTIVE THRESHOLDING ===
        # Đảm bảo block size là số lẻ
        block_size = self.adaptive_block_size
        if block_size % 2 == 0:
            block_size += 1

        adaptive_thresh = cv2.adaptiveThreshold(
            gray_enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=self.adaptive_c
        )

        # === 3. NGƯỠNG TUYỆT ĐỐI ===
        dark_binary = np.zeros_like(gray)
        dark_binary[gray < self.dark_threshold] = 255

        l_lab_binary = np.zeros_like(l_lab)
        l_lab_binary[l_lab < self.dark_threshold + 20] = 255

        # === 4. SOBEL GRADIENT ===
        sobelx = cv2.Sobel(gray_enhanced, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)

        if np.max(abs_sobelx) > 0:
            scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        else:
            scaled_sobel = np.zeros_like(gray)

        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= self.sobel_low) & (scaled_sobel <= self.sobel_high)] = 255

        # === 5. MORPHOLOGICAL OPERATIONS ===
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_line = np.ones((7, 1), np.uint8)

        adaptive_clean = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel_small)
        adaptive_clean = cv2.morphologyEx(adaptive_clean, cv2.MORPH_CLOSE, kernel_line)

        # === 6. KẾT HỢP ===
        combined = np.zeros_like(gray, dtype=np.uint8)

        condition1 = (adaptive_clean > 0) & ((dark_binary > 0) | (l_lab_binary > 0))
        condition2 = (sobel_binary > 0) & (l_lab_binary > 0)

        combined[condition1 | condition2] = 255

        if np.sum(combined > 0) < 500:
            combined = adaptive_clean

        # === 7. LỌC BLOB NHỎ ===
        combined = self._filter_small_blobs(combined)

        result = np.zeros_like(combined, dtype=np.uint8)
        result[combined > 0] = 1

        return result

    def _preprocess_light_lane(self, img):
        """Phát hiện làn đường SÁNG trên nền TỐI."""
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)

        if np.max(abs_sobelx) > 0:
            scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        else:
            scaled_sobel = np.zeros_like(l_channel)

        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= self.light_threshold) & (l_channel <= 255)] = 1

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1

        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1) | (l_binary == 1)] = 1

        return combined_binary

    def _filter_small_blobs(self, binary_img):
        """Loại bỏ các blob nhỏ (noise)."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_img, connectivity=8
        )

        filtered = np.zeros_like(binary_img)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_blob_area:
                filtered[labels == i] = 255

        return filtered

    def preprocess_with_visualization(self, img):
        """Preprocessing với hiển thị các bước trung gian."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_lab = lab[:, :, 0]

        block_size = self.adaptive_block_size if self.adaptive_block_size % 2 == 1 else self.adaptive_block_size + 1
        adaptive = cv2.adaptiveThreshold(
            gray_enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=self.adaptive_c
        )

        dark = np.zeros_like(gray)
        dark[gray < self.dark_threshold] = 255

        sobelx = cv2.Sobel(gray_enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobel_abs = np.absolute(sobelx)
        if np.max(sobel_abs) > 0:
            sobel_vis = np.uint8(255 * sobel_abs / np.max(sobel_abs))
        else:
            sobel_vis = np.zeros_like(gray)

        final = self._preprocess_dark_lane(img)
        final_vis = (final * 255).astype(np.uint8)

        h, w = gray.shape
        grid = np.zeros((h * 2, w * 3), dtype=np.uint8)

        grid[0:h, 0:w] = gray
        grid[0:h, w:2 * w] = gray_enhanced
        grid[0:h, 2 * w:3 * w] = l_lab
        grid[h:2 * h, 0:w] = adaptive
        grid[h:2 * h, w:2 * w] = dark
        grid[h:2 * h, 2 * w:3 * w] = final_vis

        font = cv2.FONT_HERSHEY_SIMPLEX
        labels_text = ["Gray", "CLAHE", "LAB-L", "Adaptive", "Dark Mask", "Final"]
        positions = [(10, 25), (w + 10, 25), (2 * w + 10, 25), (10, h + 25), (w + 10, h + 25), (2 * w + 10, h + 25)]

        for text, pos in zip(labels_text, positions):
            cv2.putText(grid, text, pos, font, 0.6, 255, 2)

        return grid, final

    # ================================================================
    # =============== PHÁT HIỆN LÀN ĐƯỜNG ============================
    # ================================================================

    def detect_lanes_sliding_window(self, binary_warped):
        """Tìm làn đường bằng phương pháp cửa sổ trượt."""
        h, w = binary_warped.shape

        histogram = np.sum(binary_warped[h // 2:, :], axis=0)
        histogram = self._smooth_histogram(histogram, self.histogram_smooth_window)

        midpoint = w // 2
        center_margin = 50

        left_half = histogram[:midpoint - center_margin]
        right_half = histogram[midpoint + center_margin:]

        max_hist = np.max(histogram)
        min_peak_height = max_hist * 0.1 if max_hist > 0 else 0

        leftx_base = self._find_peak_with_threshold(left_half, min_peak_height)
        rightx_base = self._find_peak_with_threshold(right_half, min_peak_height)

        if rightx_base is not None:
            rightx_base += midpoint + center_margin

        if leftx_base is None:
            leftx_base = midpoint // 2
        if rightx_base is None:
            rightx_base = midpoint + midpoint // 2

        left_peak = histogram[leftx_base] if 0 <= leftx_base < len(histogram) else 0
        right_peak = histogram[rightx_base] if 0 <= rightx_base < len(histogram) else 0

        window_height = h // self.n_windows

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []
        window_rects = []

        for window in range(self.n_windows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            win_xleft_low = max(0, leftx_current - self.window_margin)
            win_xleft_high = min(w, leftx_current + self.window_margin)

            win_xright_low = max(0, rightx_current - self.window_margin)
            win_xright_high = min(w, rightx_current + self.window_margin)

            window_rects.append({
                'left': (win_xleft_low, win_y_low, win_xleft_high, win_y_high),
                'right': (win_xright_low, win_y_low, win_xright_high, win_y_high)
            })

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.min_pixels_recenter:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.min_pixels_recenter:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        if left_lane_inds:
            left_lane_inds = np.concatenate(left_lane_inds)
        else:
            left_lane_inds = np.array([])

        if right_lane_inds:
            right_lane_inds = np.concatenate(right_lane_inds)
        else:
            right_lane_inds = np.array([])

        min_peak_threshold = 50

        left_detected = (len(left_lane_inds) >= self.min_lane_points and
                         left_peak >= min_peak_threshold)
        right_detected = (len(right_lane_inds) >= self.min_lane_points and
                          right_peak >= min_peak_threshold)

        left_fit = None
        right_fit = None
        leftx, lefty, rightx, righty = [], [], [], []

        if left_detected and len(left_lane_inds) > 0:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            try:
                left_fit = np.polyfit(lefty, leftx, 2)
                if not self._validate_lane_fit(left_fit, binary_warped.shape):
                    left_fit = None
                    left_detected = False
            except Exception:
                left_fit = None
                left_detected = False

        if right_detected and len(right_lane_inds) > 0:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            try:
                right_fit = np.polyfit(righty, rightx, 2)
                if not self._validate_lane_fit(right_fit, binary_warped.shape):
                    right_fit = None
                    right_detected = False
            except Exception:
                right_fit = None
                right_detected = False

        detection_info = {
            "left_detected": left_detected,
            "right_detected": right_detected,
            "left_points": len(left_lane_inds),
            "right_points": len(right_lane_inds),
            "left_peak": float(left_peak),
            "right_peak": float(right_peak),
            "both_detected": left_detected and right_detected,
            "single_lane": (left_detected and not right_detected) or
                           (not left_detected and right_detected),
            "no_lane": not left_detected and not right_detected,
            "histogram_quality": self._assess_histogram_quality(histogram),
            "window_rects": window_rects
        }

        return left_fit, right_fit, (leftx, lefty, rightx, righty), detection_info

    def _smooth_histogram(self, histogram, window_size=30):
        """Làm mượt histogram."""
        if window_size <= 0 or len(histogram) < window_size:
            return histogram
        kernel = np.ones(window_size) / window_size
        return np.convolve(histogram, kernel, mode='same')

    def _find_peak_with_threshold(self, arr, min_height):
        """Tìm peak trong array."""
        if len(arr) == 0:
            return None
        max_idx = np.argmax(arr)
        if arr[max_idx] >= min_height:
            return max_idx
        return None

    def _validate_lane_fit(self, fit, img_shape):
        """Kiểm tra lane fit có hợp lệ không."""
        if fit is None:
            return False

        h, w = img_shape

        if abs(fit[0]) > 0.003:
            return False

        y_bottom = h - 1
        y_top = 0

        x_bottom = fit[0] * y_bottom ** 2 + fit[1] * y_bottom + fit[2]
        x_top = fit[0] * y_top ** 2 + fit[1] * y_top + fit[2]

        if x_bottom < -w * 0.5 or x_bottom > w * 1.5:
            return False
        if x_top < -w * 0.5 or x_top > w * 1.5:
            return False

        return True

    def _assess_histogram_quality(self, histogram):
        """Đánh giá chất lượng histogram."""
        max_val = np.max(histogram)
        mean_val = np.mean(histogram)

        if max_val < 100:
            return "poor"
        elif mean_val > 0 and max_val / mean_val > 5:
            return "good"
        else:
            return "moderate"

    # ================================================================
    # =============== ƯỚC TÍNH LÀN BỊ MẤT ============================
    # ================================================================

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

        if 200 < current_width < 800:
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

        # TRƯỜNG HỢP 1: CẢ HAI LÀN
        if detection_info["both_detected"]:
            self.update_lane_width(left_fit, right_fit, h)
            self.prev_left_fit = left_fit.copy()
            self.prev_right_fit = right_fit.copy()
            self.frames_since_both_lanes = 0

            lane_status["left_source"] = "detected"
            lane_status["right_source"] = "detected"
            lane_status["confidence"] = 1.0

            return left_fit, right_fit, lane_status

        # TRƯỜNG HỢP 2: CHỈ LÀN TRÁI
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
                    lane_status["right_source"] = "estimated_blended"
                else:
                    right_fit = self.estimate_missing_lane(left_fit, True, h)
                    lane_status["right_source"] = "estimated"

                lane_status["confidence"] = max(0.3, 1.0 - self.frames_since_both_lanes * 0.05)
            else:
                right_fit = self.estimate_missing_lane(left_fit, True, h)
                lane_status["right_source"] = "estimated"
                lane_status["confidence"] = 0.3

            return left_fit, right_fit, lane_status

        # TRƯỜNG HỢP 3: CHỈ LÀN PHẢI
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
                    lane_status["left_source"] = "estimated_blended"
                else:
                    left_fit = self.estimate_missing_lane(right_fit, False, h)
                    lane_status["left_source"] = "estimated"

                lane_status["confidence"] = max(0.3, 1.0 - self.frames_since_both_lanes * 0.05)
            else:
                left_fit = self.estimate_missing_lane(right_fit, False, h)
                lane_status["left_source"] = "estimated"
                lane_status["confidence"] = 0.3

            return left_fit, right_fit, lane_status

        # TRƯỜNG HỢP 4: KHÔNG CÓ LÀN NÀO
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
        return prev_weight * previous_fit + est_weight * estimated_fit

    # ================================================================
    # =============== PHÂN TÍCH HÌNH HỌC =============================
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

    # ================================================================
    # =============== TÍNH TOÁN ĐIỀU KHIỂN ===========================
    # ================================================================

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
                detection.get("both_detected", False) and
                warning is None and
                left_src == "detected" and
                right_src == "detected" and
                confidence >= 0.6
        )

        single_lane_mode = (
                detection.get("single_lane", False) or
                warning in ("LEFT_LANE_MISSING", "RIGHT_LANE_MISSING")
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
        suggested_angle = np.clip(steering_score * 45, -45, 45)

        return {
            "action": action,
            "action_code": action_code,
            "direction": direction,
            "steering_score": steering_score,
            "intensity_percent": intensity,
            "suggested_angle": suggested_angle,
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
                "mode": "two_lanes" if both_lanes_reliable else ("single_lane" if single_lane_mode else "mixed"),
                "warning": warning,
            }
        }

    # ================================================================
    # =============== PROCESS FRAME ==================================
    # ================================================================

    def process_frame(self, frame, debug=False):
        """
        Hàm xử lý chính cho 1 frame ảnh.

        Pipeline:
        1. Cắt ROI (nếu bật)
        2. Preprocessing (detect lane pixels)
        3. Warp to Bird's Eye View
        4. Detect lanes (sliding window)
        5. Estimate missing lane (nếu cần)
        6. Analyze geometry
        7. Calculate steering
        8. Visualization
        """
        if self.M is None:
            raise Exception("Chưa cấu hình! Chạy select_points_interactive() hoặc load_config() trước!")

        original_frame = frame.copy()
        h_orig, w_orig = frame.shape[:2]

        # === 1. CẮT ROI (MỚI) ===
        if self.use_roi_crop and self.roi_mask is not None:
            # Tạo mask nếu chưa có hoặc kích thước khác
            if self.roi_mask.shape[:2] != frame.shape[:2]:
                self._create_roi_mask(frame.shape[:2])

            # Apply mask (vùng ngoài ROI thành đen)
            roi_frame = self.apply_roi_mask(frame)

            # if debug:
            #     cv2.imshow("1. ROI Masked", roi_frame)
        else:
            roi_frame = frame

        # === 2. PREPROCESSING ===
        binary_img = self.preprocess_advanced(roi_frame)

        # if debug:
        #     binary_display = (binary_img * 255).astype(np.uint8)
        #     cv2.imshow("2. Binary", binary_display)

        # === 3. WARP TO BEV ===
        img_size = (w_orig, h_orig)
        warped = cv2.warpPerspective(
            binary_img.astype(np.uint8),
            self.M,
            img_size,
            flags=cv2.INTER_LINEAR
        )

        # if debug:
        #     warped_display = (warped * 255).astype(np.uint8)
        #     cv2.imshow("3. Warped BEV", warped_display)

        # === 4 & 5. DETECT LANES + ESTIMATION ===
        left_fit, right_fit, lane_status = self.get_lanes_with_estimation(warped, frame.shape)

        # === XỬ LÝ MẤT LÀN HOÀN TOÀN ===
        if left_fit is None or right_fit is None:
            self.prev_steering_score = 0
            result = self._draw_lost_lane_warning(original_frame)
            return result, {
                "status": "LOST_LANE",
                "action": "!!! SLOW DOWN !!!",
                "action_code": 99,
                "lane_status": lane_status
            }

        # === 6. PHÂN TÍCH HÌNH HỌC ===
        lane_geometry = self.analyze_lane_geometry(frame.shape, left_fit, right_fit)

        # === 7. TÍNH TOÁN ĐIỀU KHIỂN ===
        control_decision = self.calculate_steering_decision(lane_geometry, lane_status)

        # === 8. VISUALIZATION ===
        result = self._draw_lanes(original_frame, warped, left_fit, right_fit,
                                  lane_geometry, control_decision, lane_status)

        if debug:
            self._draw_debug_view(result, warped, lane_geometry, lane_status)

        # === OUTPUT ===
        output_info = {
            "status": "TRACKING",
            **control_decision,
            "lane_status": lane_status,
            "geometry": lane_geometry
        }

        return result, output_info

    def _draw_lost_lane_warning(self, frame):
        """Vẽ cảnh báo khi mất làn hoàn toàn."""
        result = frame.copy()
        h, w = result.shape[:2]

        # Overlay đỏ
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 100), -1)
        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)

        # Text cảnh báo
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "!!! LANE LOST !!!"
        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2

        cv2.putText(result, text, (text_x, text_y), font, 1.5, (0, 0, 255), 3)
        cv2.putText(result, "SLOW DOWN", (text_x + 50, text_y + 50), font, 1.0, (0, 255, 255), 2)

        return result

    # ================================================================
    # =============== VISUALIZATION ==================================
    # ================================================================

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

        # Vẽ ROI outline
        if self.roi_points and len(self.roi_points) == 4:
            roi_pts = np.array(self.roi_points, dtype=np.int32)
            cv2.polylines(result, [roi_pts], True, (255, 0, 255), 2)

        self._draw_info_panel(result, geometry, decision, lane_status)

        return result

    def _draw_info_panel(self, img, geometry, decision, lane_status):
        """Vẽ bảng thông tin điều khiển."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = img.shape[:2]

        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (420, 320), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        if decision["direction"] == "left":
            cmd_color = (255, 150, 0)
        elif decision["direction"] == "right":
            cmd_color = (0, 150, 255)
        else:
            cmd_color = (0, 255, 0)

        y = 35
        dy = 24

        # Lane status
        confidence = lane_status.get("confidence", 1.0)
        warning = lane_status.get("warning")
        left_src = lane_status.get("left_source", "none")[:3].upper()
        right_src = lane_status.get("right_source", "none")[:3].upper()

        status_color = (0, 255, 0) if confidence > 0.7 else ((0, 255, 255) if confidence > 0.3 else (0, 0, 255))
        cv2.putText(img, f"L:{left_src} | R:{right_src} | Conf:{confidence:.0%}",
                    (20, y), font, 0.5, status_color, 1)
        y += dy

        if warning:
            warn_color = (0, 0, 255) if "NO_LANES" in warning else (0, 165, 255)
            cv2.putText(img, f"! {warning}", (20, y), font, 0.5, warn_color, 2)
            y += dy

        cv2.putText(img, f"CMD: {decision['action']}", (20, y), font, 0.7, cmd_color, 2)
        y += dy + 5

        cv2.putText(img, f"Offset: {geometry['offset_meters']:+.2f}m", (20, y), font, 0.5, (255, 255, 255), 1)
        y += dy
        cv2.putText(img, f"Lane Angle: {geometry['lane_angle_deg']:+.1f}deg", (20, y), font, 0.5, (255, 255, 255), 1)
        y += dy

        curve_str = {-1: "LEFT", 1: "RIGHT", 0: "STRAIGHT"}.get(int(geometry['curvature_direction']), "?")
        cv2.putText(img, f"Curve: {curve_str} (R={geometry['radius_meters']:.0f}m)", (20, y), font, 0.5,
                    (255, 255, 255), 1)
        y += dy

        cv2.putText(img, f"Lane Width: {self.standard_lane_width_pixels}px", (20, y), font, 0.5, (200, 200, 200), 1)
        y += dy
        cv2.putText(img, f"Steer Score: {decision['steering_score']:+.2f}", (20, y), font, 0.5, (255, 255, 255), 1)
        y += dy
        cv2.putText(img, f"Type: {self.lane_type} | ROI: {'ON' if self.use_roi_crop else 'OFF'}",
                    (20, y), font, 0.5, (200, 200, 200), 1)
        y += dy + 5

        # Steering bar
        bar_center = 210
        bar_width = 160
        bar_y = y
        score = decision['steering_score']

        cv2.rectangle(img, (bar_center - bar_width // 2, bar_y),
                      (bar_center + bar_width // 2, bar_y + 15), (100, 100, 100), -1)
        cv2.line(img, (bar_center, bar_y), (bar_center, bar_y + 15), (255, 255, 255), 2)

        indicator_x = int(bar_center + score * (bar_width // 2))
        indicator_x = np.clip(indicator_x, bar_center - bar_width // 2 + 5, bar_center + bar_width // 2 - 5)
        cv2.circle(img, (indicator_x, bar_y + 7), 8, cmd_color, -1)

    def _draw_debug_view(self, result, warped, geometry, lane_status):
        """Vẽ debug view."""
        h, w = result.shape[:2]

        debug_view = np.dstack((warped, warped, warped)) * 255
        debug_view = debug_view.astype(np.uint8)

        scale_y = warped.shape[0] / h
        cv2.circle(debug_view, (int(geometry['center_bottom']), int((h - 1) * scale_y)), 8, (0, 0, 255), -1)
        cv2.circle(debug_view, (int(geometry['center_ahead']), int(h * 0.3 * scale_y)), 8, (0, 255, 0), -1)
        cv2.circle(debug_view, (int(geometry['image_center']), int((h - 1) * scale_y)), 8, (255, 0, 0), -1)

        debug_view = cv2.resize(debug_view, (320, 180))
        result[0:180, w - 320:w] = debug_view

        cv2.putText(result, "RED: Lane Center", (w - 310, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(result, "BLUE: Image Center", (w - 310, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        det = lane_status.get("detection", {})
        cv2.putText(result, f"L:{det.get('left_points', 0)} R:{det.get('right_points', 0)}",
                    (w - 310, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# ========================================================
# MAIN PROGRAM
# ========================================================

def main():
    """Main function với demo và keyboard controls."""
    from stream_manager import stream_manager
    stream_manager.start()
    lane_nav = LaneNavigator()

    # Đọc ảnh test
    first_frame = stream_manager.get_latest_frame()

    # Load hoặc tạo config
    # lane_nav.load_config("lane_nav_config.json")
    lane_nav.select_points_interactive(first_frame)

    # Tạo ROI mask với kích thước ảnh thực tế
    if lane_nav.roi_points:
        lane_nav._create_roi_mask(first_frame.shape[:2])

    lane_nav.set_lane_type("dark_on_light")
    lane_nav.set_roi_crop(True, expand_ratio=0.1)

    print("\n" + "=" * 60)
    print("ĐIỀU KHIỂN:")
    print("  Q: Thoát")
    print("  R: Chọn lại ROI")
    print("  T: Chuyển đổi lane type (dark/light/auto)")
    print("  D: Bật/tắt debug view")
    print("  V: Hiển thị preprocessing steps")
    print("  C: Bật/tắt ROI crop")
    print("  1/2/3: Tăng trọng số Offset/Angle/Curve")
    print("  !/@/#: Giảm trọng số Offset/Angle/Curve")
    print("  W/S: Tăng/giảm độ rộng làn")
    print("  +/-: Tăng/giảm adaptive_c")
    print("  [/]: Tăng/giảm dark_threshold")
    print("=" * 60 + "\n")

    debug_mode = True
    show_preprocessing = False

    while True:
        frame = stream_manager.get_latest_frame()

        try:
            if show_preprocessing:
                # Crop ROI trước khi hiển thị preprocessing
                if lane_nav.use_roi_crop:
                    roi_frame = lane_nav.apply_roi_mask(frame)
                else:
                    roi_frame = frame
                prep_grid, _ = lane_nav.preprocess_with_visualization(roi_frame)
                cv2.imshow("Preprocessing Steps", prep_grid)

            processed_frame, info = lane_nav.process_frame(frame, debug=debug_mode)
            cv2.imshow("Lane Tracking", processed_frame)

            if info["status"] == "TRACKING":
                lane_st = info.get("lane_status", {})
                conf = lane_st.get("confidence", 1.0)
                warn = lane_st.get("warning", "") or ""

                print(f"[{info['action']:^25}] "
                      f"Score:{info['steering_score']:+.2f} "
                      f"Off:{info['raw_data']['offset_m']:+.2f}m "
                      f"Ang:{info['raw_data']['lane_angle']:+.1f}° "
                      f"Conf:{conf:.0%} {warn}")
            else:
                warn = info.get('lane_status', {}).get('warning', '')
                print(f"[{info['action']}] - {warn}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        key = cv2.waitKey(100) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            lane_nav.select_points_interactive(first_frame)
            lane_nav._create_roi_mask(first_frame.shape[:2])
        elif key == ord('t'):
            types = ["dark_on_light", "light_on_dark", "auto"]
            current_idx = types.index(lane_nav.lane_type) if lane_nav.lane_type in types else 0
            lane_nav.set_lane_type(types[(current_idx + 1) % len(types)])
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {debug_mode}")
            if not debug_mode:
                for win in ["1. ROI Masked", "2. Binary", "3. Warped BEV"]:
                    try:
                        cv2.destroyWindow(win)
                    except:
                        pass
        elif key == ord('v'):
            show_preprocessing = not show_preprocessing
            print(f"Show preprocessing: {show_preprocessing}")
            if not show_preprocessing:
                try:
                    cv2.destroyWindow("Preprocessing Steps")
                except:
                    pass
        elif key == ord('c'):
            lane_nav.set_roi_crop(not lane_nav.use_roi_crop)
        elif key == ord('1'):
            lane_nav.set_control_weights(k_offset=lane_nav.k_offset + 0.1)
        elif key == ord('!'):
            lane_nav.set_control_weights(k_offset=max(0, lane_nav.k_offset - 0.1))
        elif key == ord('2'):
            lane_nav.set_control_weights(k_angle=lane_nav.k_angle + 0.1)
        elif key == ord('@'):
            lane_nav.set_control_weights(k_angle=max(0, lane_nav.k_angle - 0.1))
        elif key == ord('3'):
            lane_nav.set_control_weights(k_curvature=lane_nav.k_curvature + 0.1)
        elif key == ord('#'):
            lane_nav.set_control_weights(k_curvature=max(0, lane_nav.k_curvature - 0.1))
        elif key == ord('w'):
            lane_nav.set_lane_width(lane_nav.standard_lane_width_pixels + 20)
        elif key == ord('s'):
            lane_nav.set_lane_width(max(100, lane_nav.standard_lane_width_pixels - 20))
        elif key == ord('+') or key == ord('='):
            lane_nav.adaptive_c += 2
            print(f"adaptive_c = {lane_nav.adaptive_c}")
        elif key == ord('-'):
            lane_nav.adaptive_c = max(1, lane_nav.adaptive_c - 2)
            print(f"adaptive_c = {lane_nav.adaptive_c}")
        elif key == ord(']'):
            lane_nav.dark_threshold += 5
            print(f"dark_threshold = {lane_nav.dark_threshold}")
        elif key == ord('['):
            lane_nav.dark_threshold = max(10, lane_nav.dark_threshold - 5)
            print(f"dark_threshold = {lane_nav.dark_threshold}")

    cv2.destroyAllWindows()
    print("Đã thoát.")


if __name__ == "__main__":
    main()

lane_nav = LaneNavigator()
lane_nav.load_config("control/lane_nav_config.json")