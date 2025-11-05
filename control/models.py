from django.db import models


class Command(models.Model):
    COMMAND_CHOICES = [
        ('forward', 'Forward'),
        ('backward', 'Backward'),
        ('left', 'Left'),
        ('right', 'Right'),
        ('stop', 'Stop'),
    ]

    command = models.CharField(max_length=20, choices=COMMAND_CHOICES, default='stop')
    speed = models.IntegerField(default=150)
    timestamp = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-timestamp']

class DetectionResult(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='idle')
    
    detections = models.JSONField(default=list) 
    
    error_msg = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        ordering = ['-timestamp']