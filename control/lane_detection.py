import cv2
import numpy as np
import sys
import json


class ConfigHelper:
    """
    Lớp trợ giúp để cấu hình các điểm trên một hình ảnh bằng cách nhấp chuột.
    - Nhấp chuột trái để chọn một điểm.
    - Nhấn 'r' để đặt lại (reset) các điểm đã chọn.
    - Nhấn 'q' để thoát mà không lưu.
    Khi đủ số điểm được chọn, cửa sổ sẽ tự động đóng.
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
                # Vẽ điểm và số thứ tự lên ảnh
                cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.display_image, str(len(self.points)), (x + 10, y - 10),
                            self.font, 0.7, (0, 0, 255), 2)
                cv2.imshow(self.window_name, self.display_image)

    def get_points(self):
        """
        Hiển thị cửa sổ và thu thập các điểm từ người dùng.
        Trả về danh sách các điểm hoặc None nếu người dùng thoát.
        """
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

    def save_config(self, filename, **kargs):
        """ Lưu cấu hình điểm vào file """
        with open(filename, 'w') as f:
            json.dump(kargs, f, indent=4)

        print(f"Cấu hình đã được lưu vào '{filename}'.")

    def load_config(self, filename):
        """ Tải cấu hình điểm từ file """
        with open(filename, 'r') as f:
            config = json.load(f)
        print(f"Cấu hình đã được tải từ '{filename}'.")
        return config


class LaneDetector:
    """
    Lớp phát hiện làn đường, tính toán độ cong và đưa ra gợi ý lái xe.
    """

    def __init__(self, roi_poly, warp_src_pts, warp_dst_size=(400, 960)):
        """
        Khởi tạo đối tượng LaneDetector.
        :param roi_poly: Danh sách các điểm (polygon) xác định vùng quan tâm (ROI).
        :param warp_src_pts: 4 điểm nguồn cho phép biến đổi phối cảnh.
        :param warp_dst_size: Tuple (width, height) của hình ảnh đích sau khi biến đổi.
        """
        self.roi_poly = np.array([roi_poly], dtype=np.int32)
        self.warp_src = np.float32(warp_src_pts)
        self.warp_dst_width, self.warp_dst_height = warp_dst_size

        self.warp_dst = np.float32([
            [0, 0],
            [self.warp_dst_width, 0],
            [0, self.warp_dst_height],
            [self.warp_dst_width, self.warp_dst_height]
        ])

        # Tính toán ma trận biến đổi phối cảnh một lần
        self.M = cv2.getPerspectiveTransform(self.warp_src, self.warp_dst)
        self.Minv = cv2.getPerspectiveTransform(self.warp_dst, self.warp_src)

        # Tham số cho việc tìm làn
        self.nwindows = 50
        self.margin = 100
        self.minpix = 50

        # Các biến để lưu trữ kết quả của khung hình trước (cho sự ổn định)
        self.left_fit = None
        self.right_fit = None

    def _preprocess(self, img):
        """ Tiền xử lý ảnh để phát hiện làn màu vàng và trắng """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Làn trắng
        gblur = cv2.GaussianBlur(gray, (5, 5), 0)
        white_mask = cv2.threshold(gblur, 200, 255, cv2.THRESH_BINARY)[1]

        # Làn vàng
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask = cv2.bitwise_or(white_mask, yellow_mask)
        return mask

    def _region_of_interest(self, img):
        """ Áp dụng mặt nạ vùng quan tâm (ROI) """
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, self.roi_poly, 255)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    def _warp(self, img):
        """ Biến đổi phối cảnh (bird's-eye view) """
        return cv2.warpPerspective(img, self.M, (self.warp_dst_width, self.warp_dst_height))

    def _unwarp(self, img, src_size):
        """ Biến đổi phối cảnh ngược lại """
        return cv2.warpPerspective(img, self.Minv, src_size)

    def _fit_curve(self, warped_img):
        """ Tìm và khớp đa thức cho các vạch làn đường """
        histogram = np.sum(warped_img[warped_img.shape[0] // 2:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = int(warped_img.shape[0] / self.nwindows)
        y, x = warped_img.nonzero()
        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_indices = []
        right_lane_indices = []

        for window in range(self.nwindows):
            win_y_low = warped_img.shape[0] - (window + 1) * window_height
            win_y_high = warped_img.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            good_left_indices = \
                ((y >= win_y_low) & (y < win_y_high) & (x >= win_xleft_low) & (x < win_xleft_high)).nonzero()[0]
            good_right_indices = \
                ((y >= win_y_low) & (y < win_y_high) & (x >= win_xright_low) & (x < win_xright_high)).nonzero()[0]

            left_lane_indices.append(good_left_indices)
            right_lane_indices.append(good_right_indices)

            if len(good_left_indices) > self.minpix:
                leftx_current = int(np.mean(x[good_left_indices]))
            if len(good_right_indices) > self.minpix:
                rightx_current = int(np.mean(x[good_right_indices]))

        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

        leftx, lefty = x[left_lane_indices], y[left_lane_indices]
        rightx, righty = x[right_lane_indices], y[right_lane_indices]

        # Chỉ khớp đa thức nếu tìm thấy đủ điểm
        if len(leftx) > 0 and len(rightx) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            self.left_fit = left_fit
            self.right_fit = right_fit

        return self.left_fit, self.right_fit

    def _draw_lane_area(self, original_img, warped_img, left_fit, right_fit):
        """ Vẽ vùng làn đường lên ảnh gốc """
        if left_fit is None or right_fit is None:
            return original_img

        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Tạo ảnh để vẽ các làn đường
        warp_zero = np.zeros_like(warped_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Vẽ vùng giữa hai làn
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Biến đổi ngược lại và kết hợp với ảnh gốc
        new_warp = self._unwarp(color_warp, (original_img.shape[1], original_img.shape[0]))
        result = cv2.addWeighted(original_img, 1, new_warp, 0.3, 0)
        return result

    def _calculate_curvature_and_offset(self, img_shape, left_fit, right_fit):
        """ Tính toán độ cong và độ lệch so với tâm làn đường """
        if left_fit is None or right_fit is None:
            return 0, 0, 0, 0

        # Định nghĩa hệ số chuyển đổi từ pixel sang mét
        ym_per_pix = 30 / 720  # mét trên mỗi pixel theo chiều y
        xm_per_pix = 3.7 / 700  # mét trên mỗi pixel theo chiều x

        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        y_eval = np.max(ploty)

        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Khớp đa thức mới trên hệ mét
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        # Tính bán kính độ cong
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        avg_radius = (left_curverad + right_curverad) / 2

        # Tính độ lệch của xe
        lane_center_pos = (leftx[-1] + rightx[-1]) / 2
        car_center_pos = img_shape[1] / 2
        offset = (car_center_pos - lane_center_pos) * xm_per_pix

        return left_curverad, right_curverad, avg_radius, offset

    def _add_turn_info(self, img, radius, offset):
        """ Thêm thông tin về độ cong và hướng rẽ lên ảnh """

        if radius > 5000:
            turn = "Di Thang"
            command = "forward"
        elif radius > 0:
            turn = f"Re Phai (R={int(radius)}m)"
            command = "left"
        else:
            turn = f"Re Trai (R={int(abs(radius))}m)"
            command = "right"

        if abs(offset) < 0.1:
            pos_text = "Vi tri: Giua Lan"
        elif offset > 0:
            pos_text = f"Lech Trai: {offset:.2f}m"
        else:
            pos_text = f"Lech Phai: {abs(offset):.2f}m"

        cv2.putText(img, turn, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, pos_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return img, command

    def process_frame(self, frame):
        """
        Phương thức chính xử lý một khung hình video.
        :param frame: Khung hình đầu vào (BGR).
        :return: Tuple (final_image, dashboard_image)
        """
        # 1. Tiền xử lý
        processed_img = self._preprocess(frame)

        # 2. Áp dụng ROI
        masked_img = self._region_of_interest(processed_img)
        # 3. Biến đổi phối cảnh
        warped_img = self._warp(masked_img)

        # 4. Tìm và khớp đường cong
        left_fit, right_fit = self._fit_curve(warped_img)

        # 5. Vẽ vùng làn đường
        final_image = self._draw_lane_area(frame, warped_img, left_fit, right_fit)

        # 6. Tính toán độ cong và độ lệch
        l_rad, r_rad, avg_rad, offset = self._calculate_curvature_and_offset(warped_img.shape, left_fit, right_fit)

        # 7. Thêm thông tin lên ảnh
        final_image, command = self._add_turn_info(final_image, avg_rad, offset)

        # 8. Tạo dashboard để gỡ lỗi
        dashboard = self._create_dashboard(frame, processed_img, masked_img, warped_img)

        return final_image, dashboard, command

    def _create_dashboard(self, original, processed, masked, warped):
        """ Tạo một hình ảnh tổng hợp các bước xử lý """
        h, w = 240, 320  # Kích thước cho mỗi ảnh nhỏ

        # Chuyển các ảnh 1 kênh sang 3 kênh để ghép
        processed_3ch = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        masked_3ch = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
        warped_3ch = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

        # Thay đổi kích thước
        original_small = cv2.resize(original, (w, h))
        processed_small = cv2.resize(processed_3ch, (w, h))
        masked_small = cv2.resize(masked_3ch, (w, h))
        warped_small = cv2.resize(warped_3ch, (w, h))

        # Thêm tiêu đề cho mỗi ảnh
        cv2.putText(original_small, 'Original', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(processed_small, 'Preprocessed', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(masked_small, 'ROI Masked', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(warped_small, 'Warped', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Ghép các ảnh lại
        top_row = np.hstack((original_small, processed_small))
        bottom_row = np.hstack((masked_small, warped_small))
        dashboard = np.vstack((top_row, bottom_row))

        return dashboard


def main():
    from stream_manager import stream_manager
    stream_manager.start()
    fist_frame = stream_manager.latest_frame
    # fist_frame = cv2.imread(r"D:\Project\IOT\IOTD22CN2\0.jpg")
    config_helper = ConfigHelper(fist_frame, 4)
    roi_point = config_helper.get_points()
    warp_helper = ConfigHelper(fist_frame, 4, window_name="Warp Point Configuration")
    warp_point = warp_helper.get_points()
    print("nhập (width, height) của hình ảnh đích")
    width = int(input("width: "))
    height = int(input("hegiht: "))
    config_path = "lane_detection.json"

    config_helper.save_config(config_path, roi_point=roi_point, warp_point=warp_point, warp_dst_size=(width, height))


def main1():
    config_path = "lane_detection.json"
    config = ConfigHelper(np.array([0]), 4).load_config(config_path)

    lane_detector = LaneDetector(roi_poly=config["roi_point"], warp_src_pts=config["warp_point"],
                                 warp_dst_size=config["warp_dst_size"])
    from stream_manager import stream_manager
    stream_manager.start()

    while True:
        frame = stream_manager.get_latest_frame()
        if frame is not None:
            final_image, dashboard, command = lane_detector.process_frame(frame)

            cv2.imshow("Lane Detection", final_image)
            cv2.imshow("Dashboard", dashboard)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    # main()
    main1()
    pass
