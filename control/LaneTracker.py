import cv2
import numpy as np
import json


class LaneNavigator:
    def __init__(self):
        self.roi_points = []
        self.bev_src_points = None
        self.bev_dst_points = None
        self.M = None  # Ma trận biến đổi perspective
        self.Minv = None  # Ma trận nghịch đảo

        # Thông số kỹ thuật (giả định camera ô tô chuẩn)
        self.ym_per_pix = 30 / 720  # mét trên pixel theo chiều dọc
        self.xm_per_pix = 3.7 / 700  # mét trên pixel theo chiều ngang (làn đường chuẩn rộng 3.7m)

    def load_config(self, filepath="lane_nav_config.json"):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.roi_points = data["roi_points"]
            self.bev_src_points = np.array(data["bev_src_points"], dtype=np.float32)
            self.bev_dst_points = np.array(data["bev_dst_points"], dtype=np.float32)
            self.M = np.array(data["M"], dtype=np.float32)
            self.Minv = np.array(data["Minv"], dtype=np.float32)
        print("Cấu hình đã được tải từ lane_nav_config.json")

    def save_config(self):
        json_str = json.dumps({
            "roi_points": self.roi_points,
            "bev_src_points": self.bev_src_points.tolist(),
            "bev_dst_points": self.bev_dst_points.tolist(),
            "M": self.M.tolist(),
            "Minv": self.Minv.tolist()
        })
        with open("lane_nav_config.json", "w") as f:
            f.write(json_str)
        print("Cấu hình đã được lưu vào lane_nav_config.json")

    def select_points_interactive(self, frame):
        """
        Hàm mở cửa sổ để người dùng click chọn 4 điểm ROI/BEV.
        Thứ tự click: Dưới-Trái -> Dưới-Phải -> Trên-Phải -> Trên-Trái
        """
        print(
            "HƯỚNG DẪN: Click 4 điểm tạo hình thang theo thứ tự: \n1. Dưới-Trái\n2. Dưới-Phải\n3. Trên-Phải\n4. Trên-Trái\nNhấn phím bất kỳ để xác nhận sau khi chọn xong.")

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

        # Thiết lập source và destination cho BEV
        h, w = frame.shape[:2]
        self.bev_src_points = np.float32(self.roi_points)

        # Điểm đích (Destination) tạo ra hình chữ nhật nhìn từ trên cao
        offset = 100  # Khoảng cách lề
        self.bev_dst_points = np.float32([
            [offset, h],  # Dưới-Trái
            [w - offset, h],  # Dưới-Phải
            [w - offset, 0],  # Trên-Phải
            [offset, 0]  # Trên-Trái
        ])

        # Tính ma trận biến đổi một lần duy nhất
        self.M = cv2.getPerspectiveTransform(self.bev_src_points, self.bev_dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.bev_dst_points, self.bev_src_points)
        print("Cấu hình hoàn tất.")
        self.save_config()

    def preprocess_advanced(self, img):
        """
        Kết hợp Color Threshold (HLS) và Gradient (Sobel) để chịu được ánh sáng thay đổi.
        """
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # 1. Sobel x: Phát hiện cạnh dọc (tốt cho vạch kẻ đường) bất kể màu sắc
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold cho Sobel
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

        # 2. Color Threshold (S channel cho vàng, L channel cho trắng)
        # Lọc màu trắng (Lightness cao)
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= 200) & (l_channel <= 255)] = 1

        # Lọc màu vàng (Saturation cao)
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1

        # Kết hợp
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1) | (l_binary == 1)] = 1

        return combined_binary

    def detect_lanes_sliding_window(self, binary_warped):
        """Tìm làn đường bằng phương pháp cửa sổ trượt (Sliding Window)"""
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = np.int64(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = np.int64(binary_warped.shape[0] / nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
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

        if len(left_lane_inds) == 0 or len(right_lane_inds) == 0:
            return None, None, None  # Không tìm thấy làn

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit, (leftx, lefty, rightx, righty)

    def calculate_steering_info(self, img_shape, left_fit, right_fit):
        """Tính toán độ cong và độ lệch khỏi tâm làn"""
        y_eval = img_shape[0] - 1

        # Tính bán kính cong (Radius of Curvature)
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
        avg_curverad = (left_curverad + right_curverad) / 2

        # Tính độ lệch tâm (Offset from Center)
        # Tìm vị trí x của làn trái và phải tại đáy ảnh
        left_lane_bottom_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        right_lane_bottom_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

        lane_center = (left_lane_bottom_x + right_lane_bottom_x) / 2
        image_center = img_shape[1] / 2

        # Độ lệch tính bằng mét (dương là xe lệch sang phải, âm là lệch sang trái)
        offset_meters = (image_center - lane_center) * self.xm_per_pix

        return avg_curverad, offset_meters

    def process_frame(self, frame, debug=False):
        """
        Hàm xử lý chính cho 1 frame ảnh.
        Trả về: Ảnh kết quả, Thông tin điều khiển (Dictionary)
        """
        if self.M is None:
            raise Exception("Vui lòng chạy calibrate_roi() trước khi xử lý!")

        # 1. Preprocessing & Warp
        binary_img = self.preprocess_advanced(frame)
        img_size = (frame.shape[1], frame.shape[0])
        warped = cv2.warpPerspective(binary_img, self.M, img_size, flags=cv2.INTER_LINEAR)

        # 2. Detect Lanes
        detection = self.detect_lanes_sliding_window(warped)

        if detection[0] is None:  # Nếu mất dấu làn
            return frame, {"status": "Lost Lane"}

        left_fit, right_fit, _ = detection

        # 3. Tính toán thông số điều khiển
        radius, offset = self.calculate_steering_info(frame.shape, left_fit, right_fit)

        # 4. Vẽ lại làn đường lên ảnh gốc
        ploty = np.linspace(0, frame.shape[0] - 1, frame.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Tạo các điểm polygon để vẽ
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))  # Vẽ vùng xanh lá

        # Unwarp (Biến đổi ngược về góc nhìn gốc)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (frame.shape[1], frame.shape[0]))
        result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)

        # 5. Logic điều khiển bám làn
        control_info = {
            "status": "Tracking",
            "radius": radius,
            "offset": offset,  # Đơn vị mét
            "action": ""
        }

        if offset > 0.1:
            control_info["action"] = "Turn Left (<--)"
        elif offset < -0.1:
            control_info["action"] = "Turn Right (-->)"
        else:
            control_info["action"] = "Center (Straight)"

        # Hiển thị thông tin lên ảnh
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, f"Radius: {int(radius)}m", (50, 50), font, 1, (255, 255, 255), 2)
        cv2.putText(result, f"Offset: {offset:.2f}m", (50, 100), font, 1, (255, 255, 255), 2)
        cv2.putText(result, f"Cmd: {control_info['action']}", (50, 150), font, 1, (0, 255, 255), 2)

        # DEBUG MODE: Trả về ảnh nhị phân biến đổi nếu cần kiểm tra lỗi
        if debug:
            # Tạo ảnh debug nhỏ gắn vào góc
            debug_view = np.dstack((warped, warped, warped)) * 255
            debug_view = cv2.resize(debug_view, (320, 180))
            result[0:180, frame.shape[1] - 320:frame.shape[1]] = debug_view

        return result, control_info


# ========================================================
# VÍ DỤ CÁCH SỬ DỤNG (MAIN PROGRAM)
# ========================================================
if __name__ == "__main__":
    from stream_manager import stream_manager

    stream_manager.start()
    # 1. Khởi tạo
    lane_nav = LaneNavigator()

    # Mở video
    first_frame = stream_manager.get_latest_frame()

    lane_nav.select_points_interactive(first_frame)
    lane_nav.load_config("lane_nav_config.json")

    # Setup Video Writer (nếu muốn lưu)

    print("Đang xử lý video...")
    while True:
        frame = stream_manager.get_latest_frame()

        try:
            # 3. Xử lý từng frame bằng phương thức duy nhất
            # Bật debug=True để xem ảnh BEV ở góc phải trên
            processed_frame, info = lane_nav.process_frame(frame, debug=True)

            # In thông tin điều khiển ra console (nếu cần gửi xuống vi điều khiển)
            # print(f"Steer: {info['action']} | Offset: {info['offset']}")

            cv2.imshow("Advanced Lane Tracking", processed_frame)
            print(info["action"])

        except Exception as e:
            print(f"Error: {e}")
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
