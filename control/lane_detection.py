import cv2
import numpy as np
import sys


# ==============================================================================
# LỚP TRỢ GIÚP CẤU HÌNH (Không thay đổi)
# ==============================================================================
class ConfigHelper:
    """
    Lớp trợ giúp để cấu hình các điểm trên một hình ảnh bằng cách nhấp chuột.
    """

    def __init__(self, image, num_points, window_name="Config Helper"):
        self.image = image.copy()
        self.display_image = image.copy()
        self.num_points = num_points
        self.window_name = window_name
        self.points = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < self.num_points:
                self.points.append((x, y))
                cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.display_image, str(len(self.points)), (x + 10, y - 10),
                            self.font, 0.7, (0, 0, 255), 2)
                cv2.imshow(self.window_name, self.display_image)

    def get_points(self):
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, self.display_image)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print(f"Vui lòng nhấp chuột để chọn {self.num_points} điểm trên cửa sổ '{self.window_name}'.")
        print("Nhấn 'r' để chọn lại, 'q' để thoát.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Đã hủy cấu hình.")
                cv2.destroyWindow(self.window_name)
                return None
            if key == ord('r'):
                print("Đặt lại các điểm. Vui lòng chọn lại.")
                self.points = []
                self.display_image = self.image.copy()
                cv2.imshow(self.window_name, self.display_image)
            if len(self.points) == self.num_points:
                print(f"Đã chọn đủ {self.num_points} điểm.")
                break

        cv2.destroyWindow(self.window_name)
        return self.points


# ==============================================================================
# LỚP LANE DETECTOR HOÀN CHỈNH
# ==============================================================================
class LaneDetector:
    """
    Lớp phát hiện làn đường, tính toán độ cong và đưa ra gợi ý lái xe.
    """

    def __init__(self, roi_poly, warp_src_pts, warp_dst_size=(400, 960)):
        self.roi_poly = np.array([roi_poly], dtype=np.int32)
        self.warp_src = np.float32(warp_src_pts)
        self.warp_dst_width, self.warp_dst_height = warp_dst_size

        self.warp_dst = np.float32([
            [0, 0],
            [self.warp_dst_width, 0],
            [0, self.warp_dst_height],
            [self.warp_dst_width, self.warp_dst_height]
        ])

        self.M = cv2.getPerspectiveTransform(self.warp_src, self.warp_dst)
        self.Minv = cv2.getPerspectiveTransform(self.warp_dst, self.warp_src)

        self.nwindows = 50
        self.margin = 100
        self.minpix = 50

        self.left_fit = None
        self.right_fit = None

    def _preprocess(self, img):
        """
        Tiền xử lý ảnh để chỉ phát hiện làn đường màu trắng, hoạt động tốt trong
        các điều kiện ánh sáng khác nhau.
        """
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        L_channel = hls[:, :, 1]

        L_channel_blurred = cv2.GaussianBlur(L_channel, (5, 5), 0)

        white_mask = cv2.adaptiveThreshold(L_channel_blurred, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 15, -2)

        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        return final_mask

    def _region_of_interest(self, img):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, self.roi_poly, 255)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    def _fit_curve(self, binary_img, M):
        """
        Tìm các điểm pixel của làn đường trên ảnh gốc, warp các điểm này sang không gian BEV,
        và sau đó khớp đa thức.
        """
        y_coords, x_coords = binary_img.nonzero()

        if len(x_coords) == 0:
            return self.left_fit, self.right_fit, None, None

        pts = np.float32(np.column_stack((x_coords, y_coords))).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, M)

        warped_x = warped_pts[:, 0, 0]
        warped_y = warped_pts[:, 0, 1]

        # Lọc ra các điểm nằm ngoài ranh giới của ảnh BEV để tránh lỗi IndexError
        valid_indices = (warped_x >= 0) & (warped_x < self.warp_dst_width) & \
                        (warped_y >= 0) & (warped_y < self.warp_dst_height)

        warped_x = warped_x[valid_indices]
        warped_y = warped_y[valid_indices]

        if len(warped_x) == 0:
            return self.left_fit, self.right_fit, None, None

        # Sliding Window trên không gian BEV
        warped_midpoint = self.warp_dst_width / 2
        leftx_base_candidates = warped_x[warped_x < warped_midpoint]
        rightx_base_candidates = warped_x[warped_x >= warped_midpoint]

        if len(leftx_base_candidates) == 0 or len(rightx_base_candidates) == 0:
            return self.left_fit, self.right_fit, None, None

        leftx_current = np.mean(leftx_base_candidates)
        rightx_current = np.mean(rightx_base_candidates)

        left_lane_indices = []
        right_lane_indices = []

        window_height = int(self.warp_dst_height / self.nwindows)

        for window in range(self.nwindows):
            win_y_low = self.warp_dst_height - (window + 1) * window_height
            win_y_high = self.warp_dst_height - window * window_height

            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            good_left_inds = ((warped_y >= win_y_low) & (warped_y < win_y_high) &
                              (warped_x >= win_xleft_low) & (warped_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((warped_y >= win_y_low) & (warped_y < win_y_high) &
                               (warped_x >= win_xright_low) & (warped_x < win_xright_high)).nonzero()[0]

            left_lane_indices.append(good_left_inds)
            right_lane_indices.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(warped_x[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(warped_x[good_right_inds]))

        if not any(map(len, left_lane_indices)) or not any(map(len, right_lane_indices)):
            return self.left_fit, self.right_fit, None, None

        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

        leftx, lefty = warped_x[left_lane_indices], warped_y[left_lane_indices]
        rightx, righty = warped_x[right_lane_indices], warped_y[right_lane_indices]

        if len(leftx) > 0 and len(rightx) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            self.left_fit = left_fit
            self.right_fit = right_fit

        return self.left_fit, self.right_fit, (leftx, lefty), (rightx, righty)

    def _draw_lane_area(self, original_img, left_fit, right_fit, Minv):
        """
        Tạo các đường cong trong không gian BEV, unwarp chúng về ảnh gốc và vẽ vùng làn đường.
        """
        if left_fit is None or right_fit is None:
            return original_img

        ploty = np.linspace(0, self.warp_dst_height - 1, self.warp_dst_height)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

        unwarped_pts_left = cv2.perspectiveTransform(pts_left, Minv)
        unwarped_pts_right = cv2.perspectiveTransform(pts_right, Minv)

        pts = np.hstack((unwarped_pts_left, unwarped_pts_right))

        overlay = np.zeros_like(original_img)
        cv2.fillPoly(overlay, np.int_([pts]), (0, 255, 0))

        result = cv2.addWeighted(original_img, 1, overlay, 0.3, 0)

        cv2.polylines(result, np.int32([unwarped_pts_left]), isClosed=False, color=(255, 0, 0), thickness=10)
        cv2.polylines(result, np.int32([unwarped_pts_right]), isClosed=False, color=(0, 0, 255), thickness=10)

        return result

    def _calculate_curvature_and_offset(self, bev_shape, left_fit, right_fit):
        if left_fit is None or right_fit is None:
            return 0, 0

        ym_per_pix = 30 / 720
        xm_per_pix = 3.7 / 700

        ploty = np.linspace(0, bev_shape[0] - 1, bev_shape[0])
        y_eval = np.max(ploty)

        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        avg_radius = (left_curverad + right_curverad) / 2

        lane_center_pos = (leftx[-1] + rightx[-1]) / 2
        car_center_pos = bev_shape[1] / 2
        offset = (car_center_pos - lane_center_pos) * xm_per_pix

        return avg_radius, offset

    def _add_turn_info(self, img, radius, offset):
        command = "straight"
        if radius > 4000:  # Ngưỡng đi thẳng lớn hơn
            turn = "Di Thang"
            command = "straight"
        elif offset > 0.15:  # Lệch trái nhiều
            turn = f"Re Phai Nhieu (R={int(radius)}m)"
            command = "hard_right"
        elif offset < -0.15:  # Lệch phải nhiều
            turn = f"Re Trai Nhieu (R={int(abs(radius))}m)"
            command = "hard_left"
        elif radius > 0:  # Bán kính dương -> cua phải
            turn = f"Re Phai (R={int(radius)}m)"
            command = "right"
        else:  # Bán kính âm -> cua trái
            turn = f"Re Trai (R={int(abs(radius))}m)"
            command = "left"

        if abs(offset) < 0.1:
            pos_text = "Vi tri: Giua Lan"
        elif offset > 0:
            pos_text = f"Lech Trai: {offset:.2f}m"
        else:
            pos_text = f"Lech Phai: {abs(offset):.2f}m"

        cv2.putText(img, turn, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, pos_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return img, command

    def _create_dashboard(self, original, processed, masked, left_pts_bev, right_pts_bev):
        h, w = 240, 320

        processed_3ch = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        masked_3ch = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)

        bev_canvas = np.zeros((self.warp_dst_height, self.warp_dst_width, 3), dtype=np.uint8)
        if left_pts_bev is not None and right_pts_bev is not None:
            leftx_bev, lefty_bev = left_pts_bev
            rightx_bev, righty_bev = right_pts_bev
            if len(leftx_bev) > 0:
                bev_canvas[lefty_bev.astype(int), leftx_bev.astype(int)] = [255, 0, 0]
            if len(rightx_bev) > 0:
                bev_canvas[righty_bev.astype(int), rightx_bev.astype(int)] = [0, 0, 255]

        original_small = cv2.resize(original, (w, h))
        processed_small = cv2.resize(processed_3ch, (w, h))
        masked_small = cv2.resize(masked_3ch, (w, h))
        bev_small = cv2.resize(bev_canvas, (w, h))

        cv2.putText(original_small, 'Original', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(processed_small, 'Preprocessed Mask', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(masked_small, 'ROI Masked', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(bev_small, 'Warped Lane Points', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        top_row = np.hstack((original_small, processed_small))
        bottom_row = np.hstack((masked_small, bev_small))
        dashboard = np.vstack((top_row, bottom_row))

        return dashboard

    def process_frame(self, frame):
        """
        Phương thức chính xử lý một khung hình video.
        """
        processed_img = self._preprocess(frame)
        masked_img = self._region_of_interest(processed_img)

        left_fit, right_fit, left_pts_bev, right_pts_bev = self._fit_curve(masked_img, self.M)

        bev_shape = (self.warp_dst_height, self.warp_dst_width)
        avg_rad, offset = self._calculate_curvature_and_offset(bev_shape, left_fit, right_fit)

        final_image = self._draw_lane_area(frame, left_fit, right_fit, self.Minv)
        final_image, command = self._add_turn_info(final_image, avg_rad, offset)

        dashboard = self._create_dashboard(frame, processed_img, masked_img, left_pts_bev, right_pts_bev)

        return final_image, dashboard, command


# ==============================================================================
# CHƯƠNG TRÌNH CHÍNH ĐỂ CHẠY
# ==============================================================================
def main():
    from stream_manager import stream_manager
    stream_manager.start()
    first_frame = stream_manager.get_latest_frame()

    frame_h, frame_w = first_frame.shape[:2]

    # ----- CẤU HÌNH TƯƠNG TÁC -----
    print("--- BẮT ĐẦU CẤU HÌNH ---")
    roi_helper = ConfigHelper(first_frame, 4, "Cau hinh ROI")
    print("Thu tu goi y cho ROI: duoi-trai -> tren-trai -> tren-phai -> duoi-phai")
    roi_points = roi_helper.get_points()
    if roi_points is None: sys.exit()

    frame_with_roi = first_frame.copy()
    cv2.polylines(frame_with_roi, [np.array(roi_points, np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
    warp_helper = ConfigHelper(frame_with_roi, 4, "Cau hinh Warp")
    print("\nThu tu goi y cho diem Warp: tren-trai -> tren-phai -> duoi-trai -> duoi-phai")
    warp_points = warp_helper.get_points()
    if warp_points is None: sys.exit()

    print("\n--- CẤU HÌNH HOÀN TẤT ---")
    print(f"Cac diem ROI da chon: {roi_points}")
    print(f"Cac diem Warp da chon: {warp_points}")

    # ----- KHỞI TẠO VÀ XỬ LÝ VIDEO -----
    detector = LaneDetector(roi_poly=roi_points, warp_src_pts=warp_points, warp_dst_size=(400, 960))

    print("\nDang xu ly video... Nhan 'q' de dung lai.")
    while True:
        frame = stream_manager.get_latest_frame()

        final_result, dashboard, command = detector.process_frame(frame)
        print(f"Command: {command}")  # In ra lệnh điều khiển

        cv2.imshow("Ket qua Phat hien Lan duong", final_result)
        cv2.imshow("Dashboard", dashboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
