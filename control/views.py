from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Command, DetectionResult
from .stream_manager import stream_manager
from .car_control import car_control
import base64
from django.http import StreamingHttpResponse

from .car_control import car_control

from ultralytics import YOLO
import cv2
import numpy as np

try:
    yolo_model = YOLO("../model/best.pt")
    print("====== MODEL YOLO ĐÃ TẢI THÀNH CÔNG ======")
except Exception as e:
    yolo_model = None
    print(f"====== LỖI KHI TẢI MODEL YOLO: {e} ======")

MIN_CONFIDENCE = 0.40

# Biến lưu trạng thái hiện tại
current_command = {'command': 'stop', 'speed': 150}
last_analysis_result = {"detections": [], "status": "idle"}


def index(request):
    return render(request, 'control/index.html')


@api_view(['GET'])
def get_command(request):
    """API endpoint để ESP32 lấy lệnh"""
    cmd, speed = car_control.get_command()
    print("GET COMMAND:", cmd, speed)
    # global current_command
    current_command = {'command': cmd, 'speed': speed}
    return Response(current_command)


@csrf_exempt
@api_view(['POST'])
def set_command(request):
    """API endpoint để gửi lệnh điều khiển"""
    global current_command

    command = request.data.get('command', 'stop')
    speed = request.data.get('speed', 110)

    # Lưu lệnh vào database (optional)
    Command.objects.create(command=command, speed=speed)
    car_control.set_speed(speed)
    car_control.set_base_speed(speed)
    # Cập nhật lệnh hiện tại
    current_command = {'command': command, 'speed': speed}
    if command == 'stop':
        car_control.pause()
    if command == 'forward':
        car_control.resume()

    return Response({'status': 'success', 'command': command, 'speed': speed})


@api_view(['GET'])
def command_history(request):
    """Lấy lịch sử lệnh"""
    commands = Command.objects.all()[:20]
    data = [{'command': c.command, 'speed': c.speed, 'timestamp': c.timestamp}
            for c in commands]
    return Response(data)


@api_view(['GET'])
def analyze_stream_once(request):
    global last_analysis_result

    if not yolo_model:
        last_analysis_result = {"detections": [], "status": "error", "error_msg": "Model not loaded"}
        DetectionResult.objects.create(status="error", error_msg="Model not loaded")
        return Response(
            last_analysis_result,
            status=500
        )

    # stream_manager.start()
    # frame = None
    try:
        frame = stream_manager.get_latest_frame()

        if frame is None:
            DetectionResult.objects.create(status="error", error_msg="Stream not ready or failed to get frame")
            return Response({"detections": [], "status": "error", "error_msg": "Stream not ready"}, status=503)

        results = yolo_model(frame, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = yolo_model.names[class_id]
                confidence = float(box.conf)

                if confidence > MIN_CONFIDENCE:
                    detections.append({
                        "bien_bao": class_name,
                        "do_tin_cay": round(confidence, 2)
                    })

        if detections:
            result_obj = DetectionResult.objects.create(
                detections=detections,
                status="success"
            )
            last_analysis_result = {"detections": detections, "status": "success"}
        else:
            result_obj = DetectionResult.objects.create(
                detections=[],
                status="no_detection"
            )
            last_analysis_result = {"detections": [], "status": "no_detection"}

        return Response(last_analysis_result)

    except Exception as e:
        DetectionResult.objects.create(status="error", error_msg=str(e))
        last_analysis_result = {"detections": [], "status": "error", "error_msg": str(e)}
        return Response(last_analysis_result, status=500)



@csrf_exempt
@api_view(['POST'])
def detect_uploaded_image(request):
    """
    API cho phép người dùng tải ảnh lên để YOLO phân tích.
    KẾT QUẢ SẼ ĐƯỢC CẬP NHẬT VÀO BIẾN TOÀN CỤC `last_analysis_result`.
    """
    global last_analysis_result  # <--- Sử dụng biến toàn cục

    if not yolo_model:
        error_msg = "Model YOLO không khả dụng"
        last_analysis_result = {"detections": [], "status": "error", "error_msg": error_msg}
        return Response({"error": error_msg}, status=500)

    try:
        image = None

        # 1️⃣ Nếu gửi file qua form-data
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            file_bytes = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 2️⃣ Nếu gửi base64 qua JSON
        elif 'image' in request.data:
            img_data = request.data['image']
            if img_data.startswith("data:image"):
                img_data = img_data.split(",")[1]
            img_bytes = base64.b64decode(img_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            last_analysis_result = {"detections": [], "status": "error", "error_msg": "Thiếu ảnh"}
            return Response({"error": "Thiếu ảnh đầu vào"}, status=400)

        if image is None:
            last_analysis_result = {"detections": [], "status": "error", "error_msg": "Không thể đọc ảnh"}
            return Response({"error": "Không thể đọc ảnh"}, status=400)

        # 3️⃣ Chạy YOLO
        results = yolo_model(image, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = yolo_model.names[class_id]
                confidence = float(box.conf)
                if confidence > MIN_CONFIDENCE:
                    detections.append({
                        "bien_bao": class_name,
                        "do_tin_cay": round(confidence, 2)
                    })

        # 4️⃣ (Tuỳ chọn) Vẽ kết quả lên ảnh
        annotated_frame = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        base64_result = base64.b64encode(buffer).decode('utf-8')
        annotated_image_data = f"data:image/jpeg;base64,{base64_result}"

        # 5️⃣ 🔹 CẬP NHẬT BIẾN TOÀN CỤC 🔹
        if detections:
            last_analysis_result = {
                "detections": detections,
                "status": "success",
                "annotated_image": annotated_image_data  # Gửi cả ảnh đã vẽ
            }
        else:
            last_analysis_result = {
                "detections": [],
                "status": "no_detection",
                "annotated_image": annotated_image_data
            }

        if detections:
            status = "success"
        else:
            status = "no_detection"

        # DetectionResult.objects.create(
        #     detections=detections,
        #     status=status
        # )
        # Trả về kết quả cho người tải lên
        return Response({
            "status": "success",
            "detections": detections,
            "annotated_image": annotated_image_data
        })

    except Exception as e:
        error_msg = f"Lỗi xử lý: {str(e)}"
        last_analysis_result = {"detections": [], "status": "error", "error_msg": error_msg}
        return Response({"error": error_msg}, status=500)


def generate_processed_frames():
    while True:
        try:
            frame = car_control.latest_processed_frame.copy()
            _, buffer = cv2.imencode('.jpg', frame)
            processed_jpg = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + processed_jpg + b'\r\n')
        except Exception as e:
            print(f"Processing error: {e}")


def stream_live_feed(request):
    return StreamingHttpResponse(generate_processed_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


@api_view(['GET'])
def get_control_info(request):
    """
    Trả về thông tin chi tiết từ CarControl:
    - command hiện tại (do detect làn + logic biển báo quyết định)
    - speed, state, fps, frame_count
    - info chi tiết từ LaneNavigator (offset, góc, confidence, warning, ...)
    - current_sign: biển báo mới nhất từ SignDetector (nếu có)
    """
    try:
        detail = car_control.get_detailed_info()
        # Gắn thêm thông tin biển báo từ SignDetector (nếu đã gán vào car_control)
        current_sign = None
        try:
            sign_detector = getattr(car_control, "sign_detector", None)
            if sign_detector is not None:
                current_sign = sign_detector.get_current_sign()
        except Exception as e:
            current_sign = None

        detail["current_sign"] = current_sign
        return Response(detail)
    except Exception as e:
        return Response({"error": str(e)}, status=500)