from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Command
import json

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
