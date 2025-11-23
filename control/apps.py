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
            print("Khởi động Stream Manager (Chỉ trong process worker)...")
            stream_manager.start()
        else:
            print("Bỏ qua khởi động Stream Manager (Trong process reloader)...")
