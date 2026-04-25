from django.db import models
from django.utils import timezone


class ResumeUpload(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('valid', 'Valid'),
        ('invalid', 'Invalid'),
    ]

    file_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    uploaded_at = models.DateTimeField(default=timezone.now)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    category = models.CharField(max_length=100, blank=True, null=True)
    score = models.FloatField(default=0.0)
    extracted_text = models.TextField(blank=True, null=True)
    skills_found = models.JSONField(default=list)
    suggestions = models.JSONField(default=list)
    jd_match_percent = models.FloatField(default=0.0)
    matching_skills = models.JSONField(default=list)
    missing_skills = models.JSONField(default=list)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.file_name} - {self.status}"
