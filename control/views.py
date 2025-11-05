from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Command
import json
import base64

from ultralytics import YOLO
import cv2
import numpy as np

try:
    yolo_model = YOLO("../model/best.pt") 
    print("====== MODEL YOLO ĐÃ TẢI THÀNH CÔNG ======")
except Exception as e:
    yolo_model = None
    print(f"====== LỖI KHI TẢI MODEL YOLO: {e} ======")

ESP32_STREAM_URL = "http://192.168.1.50/stream"
MIN_CONFIDENCE = 0.50

# Biến lưu trạng thái hiện tại
current_command = {'command': 'stop', 'speed': 150}


def index(request):
    return render(request, 'control/index.html')


@api_view(['GET'])
def get_command(request):
    """API endpoint để ESP32 lấy lệnh"""
    return Response(current_command)


@csrf_exempt
@api_view(['POST'])
def set_command(request):
    """API endpoint để gửi lệnh điều khiển"""
    global current_command

    command = request.data.get('command', 'stop')
    speed = request.data.get('speed', 150)

    # Lưu lệnh vào database (optional)
    Command.objects.create(command=command, speed=speed)

    # Cập nhật lệnh hiện tại
    current_command = {'command': command, 'speed': speed}

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
    if not yolo_model:
        return Response(
            {"error": "Model YOLO không khả dụng"}, 
            status=500
        )

    cap = None
    try:
        cap = cv2.VideoCapture(ESP32_STREAM_URL)
        
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000) 

        if not cap.isOpened():
            return Response(
                {"error": f"Không thể kết nối đến stream: {ESP32_STREAM_URL}"}, 
                status=504 
            )

        # 2. Đọc một khung hình
        ret, frame = cap.read()
        
        # 3. Ngắt kết nối ngay lập tức để giải phóng tài nguyên
        cap.release()

        if not ret or frame is None:
            return Response(
                {"error": "Không thể đọc khung hình từ stream"}, 
                status=500
            )

        # 4. Chạy YOLO trên khung hình
        results = yolo_model(frame, verbose=False)

        # 5. Xử lý kết quả
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

        # 6. Trả về kết quả
        return Response({"detections": detections, "status": "success"})

    except Exception as e:
        if cap:
            cap.release() # Đảm bảo giải phóng nếu có lỗi
        return Response({"error": f"Lỗi xử lý: {str(e)}"}, status=500)
    

@csrf_exempt
@api_view(['POST'])
def detect_uploaded_image(request):
    """
    API cho phép người dùng tải ảnh lên để YOLO phân tích.
    Có thể gửi:
    - file: multipart/form-data, key='image'
    - hoặc JSON: {"image": "<base64 string>"}
    """
    if not yolo_model:
        return Response({"error": "Model YOLO không khả dụng"}, status=500)

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
            # Loại bỏ tiền tố nếu có
            if img_data.startswith("data:image"):
                img_data = img_data.split(",")[1]
            img_bytes = base64.b64decode(img_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        else:
            return Response({"error": "Thiếu ảnh đầu vào"}, status=400)

        if image is None:
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

        return Response({
            "status": "success",
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{base64_result}"
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)