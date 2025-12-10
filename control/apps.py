from django.apps import AppConfig
import os  # <-- Thêm dòng này


class ControlConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "control"

    def ready(self):
        # Kiểm tra biến môi trường 'RUN_MAIN'
        # Code này sẽ CHỈ chạy trong quy trình con (worker)
        if os.environ.get('RUN_MAIN') == 'true':
            # Import ở đây để tránh lỗi
            from .stream_manager import stream_manager
            from .car_control import car_control
            from .SignDetector import SignDetector
            print("Khởi động Stream Manager (Chỉ trong process worker)...")
            sign_detector_instance = SignDetector(model_path="../model/best.pt")
            car_control.sign_detector = sign_detector_instance
            stream_manager.start()
            sign_detector_instance.start(stream_manager)
            car_control.start()

        else:
            print("Bỏ qua khởi động Stream Manager (Trong process reloader)...")
