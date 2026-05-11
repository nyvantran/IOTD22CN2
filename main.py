import subprocess
import re
import sys


def get_wifi_ip_from_ipconfig():
    """Chạy ipconfig và lọc lấy IPv4 của Wireless LAN adapter Wi-Fi"""
    try:
        # Chạy lệnh ipconfig và lấy kết quả đầu ra
        # encoding='mbcs' giúp đọc đúng font trên Windows (tiếng Anh/Việt)
        result = subprocess.run(['ipconfig'], capture_output=True, text=True)
        output = result.stdout

        # Tìm phần Wireless LAN adapter Wi-Fi và lấy IPv4 bên trong đó
        # Regex giải thích:
        # 1. Tìm cụm "Wireless LAN adapter Wi-Fi:"
        # 2. (.*?) lấy tất cả ký tự (bao gồm xuống dòng nhờ re.DOTALL) cho đến khi gặp...
        # 3. "IPv4 Address" (hoặc Địa chỉ IPv4) và các dấu chấm
        # 4. Lấy nhóm số ([0-9.]+) là địa chỉ IP

        # Lưu ý: Nếu Windows của bạn là Tiếng Việt, hãy đổi "IPv4 Address" thành "Địa chỉ IPv4"
        pattern = re.compile(r"Wireless LAN adapter Wi-Fi:.*?IPv4 Address[ .]*: ([\d.]+)", re.DOTALL)

        match = pattern.search(output)

        if match:
            return match.group(1)
        else:
            return None

    except Exception as e:
        print(f"Lỗi khi chạy ipconfig: {e}")
        return None


def run_django_server(ip):
    """Chạy lệnh runserver với IP tìm được"""
    if not ip:
        print("❌ Không tìm thấy IP của Wi-Fi. Hãy kiểm tra lại kết nối mạng.")
        return

    print(f"✅ Đã tìm thấy IP: {ip}")
    print(f"🚀 Đang khởi động Server: python .\\manage.py runserver {ip}:8000")
    print("👉 Nhấn Ctrl+C để dừng server.\n")

    cmd = ['python', 'manage.py', 'runserver', f'{ip}:8000']

    try:
        # Chạy lệnh runserver, chờ user tương tác
        subprocess.run(cmd)
    except KeyboardInterrupt:
        # Xử lý khi nhấn Ctrl+C để không hiện lỗi loằng ngoằng
        print("\n🛑 Server đã được ngắt bởi người dùng (Ctrl+C).")
        sys.exit(0)


if __name__ == "__main__":
    wifi_ip = get_wifi_ip_from_ipconfig()
    run_django_server(wifi_ip)