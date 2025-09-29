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